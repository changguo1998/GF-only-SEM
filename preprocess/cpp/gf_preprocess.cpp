/// gf_preprocess.cpp — unified preprocess entry point
///
/// Dispatches to stage1, stage2, or run.  The run subcommand executes the
/// full preprocessing pipeline:
///
///   stage1 (GLL geometry + PML damping + CFL h_min)
///     → material evaluation (user model or HDF5 fallback)
///     → stage2 (λ/μ + solver_dt)
///     → C-PML (κ/d/α profiles)
///     → STF (source time function)
///     → source location (Newton + Lagrange weights)
///     → METIS partition (k-way via C API)
///     → global node numbering (topology-driven)
///     → config.h5 writer
///
/// Usage:
///   gf_preprocess stage1 <model.h5> [--N N] [--cfl-safety val] ...
///   gf_preprocess stage2 <model.h5>
///   gf_preprocess run    <model.h5> [--N N] [--cfl-safety val] ...

#include <hdf5.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gf_config.h"
#include "gf_hdf5_helpers.hpp"

#ifdef GF_HAS_USER_MATERIAL
#include "gf_material_user.h"
#endif

extern int stage1_main(int argc, char** argv);
extern int stage2_main(int argc, char** argv);

// ═════════════════════════════════════════════════════════════════════════════
//  run subcommand  —  full pipeline: stage1 → material → stage2 → ... → config.h5
// ═════════════════════════════════════════════════════════════════════════════

static int run_main(int argc, char** argv) {
    if (argc < 1) {
        fprintf(stderr,
                "Usage: gf_preprocess run <model.h5> [--N N] [--cfl-safety val] "
                "[--nx N] [--ny N] [--pml-* THICK]\n");
        return 1;
    }

    // ── Collect stage1 arguments (model.h5 + --option pairs) ──

    std::vector<char*> stage1_args;
    stage1_args.push_back(argv[0]);  // program name placeholder
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "run") == 0)
            continue;
        stage1_args.push_back(argv[i]);
    }

    // ── Stage 1: GLL geometry, PML damping, CFL h_min ──

    int rc = stage1_main(static_cast<int>(stage1_args.size()), stage1_args.data());
    if (rc != 0) {
        fprintf(stderr, "ERROR: stage1 failed with code %d\n", rc);
        return rc;
    }

    // Locate model.h5 path (first non-option argument after "run")

    const char* model_path = nullptr;
    for (int i = 0; i < argc; ++i) {
        if (argv[i][0] == '-')
            continue;
        if (std::strcmp(argv[i], "run") == 0)
            continue;
        model_path = argv[i];
        break;
    }
    if (!model_path) {
        fprintf(stderr, "ERROR: model.h5 path required\n");
        return 1;
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Material evaluation
    // ═════════════════════════════════════════════════════════════════════════

#ifdef GF_HAS_USER_MATERIAL
    fprintf(stderr, "=== Material evaluation (user C++ model) ===\n");

    hid_t material_fid = gf::h5::open_or_fail(model_path, H5F_ACC_RDWR);

    std::vector<double> gll_coords_local =
        gf::h5::read_double(material_fid, "field/element/coords");
    std::size_t n_points = gll_coords_local.size() / 3;
    fprintf(stderr, "  GLL nodes: %zu\n", n_points);

    // Extract coordinate arrays
    std::vector<double> x_arr(n_points), y_arr(n_points), z_arr(n_points);
    for (std::size_t i = 0; i < n_points; ++i) {
        x_arr[i] = gll_coords_local[i * 3 + 0];
        y_arr[i] = gll_coords_local[i * 3 + 1];
        z_arr[i] = gll_coords_local[i * 3 + 2];
    }

    // Evaluate user material model
    std::vector<double> vp_flat_local(n_points), vs_flat_local(n_points),
        density_flat_local(n_points);
    gf::evaluate_vp(n_points, x_arr.data(), y_arr.data(), z_arr.data(), vp_flat_local.data());
    gf::evaluate_vs(n_points, x_arr.data(), y_arr.data(), z_arr.data(), vs_flat_local.data());
    gf::evaluate_density(n_points, x_arr.data(), y_arr.data(), z_arr.data(),
                         density_flat_local.data());

    // Shape is [n_cell, NGLL, NGLL, NGLL, 3] → field dims = first ndims-1
    std::vector<hsize_t> coord_dims_local = gf::h5::get_dims(material_fid, "field/element/coords");
    std::vector<hsize_t> field_dims_local(coord_dims_local.begin(), coord_dims_local.end() - 1);

    gf::h5::write_double(material_fid, "field/element/vp", vp_flat_local, field_dims_local);
    gf::h5::write_double(material_fid, "field/element/vs", vs_flat_local, field_dims_local);
    gf::h5::write_double(material_fid, "field/element/density", density_flat_local,
                         field_dims_local);

    H5Fclose(material_fid);
    fprintf(stderr, "  vp/vs/density written to model.h5\n");
#else
    fprintf(stderr,
            "=== Material evaluation ===\n"
            "  No user material compiled in (GF_HAS_USER_MATERIAL not defined).\n"
            "  Looking for vp/vs/density in HDF5 (Python path)...\n");
#endif

    // Write /config group attributes before stage2 (stage2 reads them from HDF5)
    {
        hid_t config_fid = gf::h5::open_or_fail(model_path, H5F_ACC_RDWR);
        hid_t cfg_grp = H5Gcreate2(config_fid, "config", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        gf::Config stage2_cfg = gf::get_config();

        auto write_dbl = [&](const char* name, double v) {
            hid_t sp = H5Screate(H5S_SCALAR);
            hid_t a = H5Acreate2(cfg_grp, name, H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(a, H5T_NATIVE_DOUBLE, &v);
            H5Aclose(a);
            H5Sclose(sp);
        };
        auto write_i64 = [&](const char* name, int64_t v) {
            hid_t sp = H5Screate(H5S_SCALAR);
            hid_t a = H5Acreate2(cfg_grp, name, H5T_NATIVE_INT64, sp, H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(a, H5T_NATIVE_INT64, &v);
            H5Aclose(a);
            H5Sclose(sp);
        };
        auto write_str = [&](const char* name, const char* v) {
            hid_t sp = H5Screate(H5S_SCALAR);
            hid_t tp = H5Tcopy(H5T_C_S1);
            H5Tset_size(tp, std::strlen(v));
            H5Tset_strpad(tp, H5T_STR_NULLTERM);
            hid_t a = H5Acreate2(cfg_grp, name, tp, sp, H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(a, tp, v);
            H5Aclose(a);
            H5Tclose(tp);
            H5Sclose(sp);
        };

        write_dbl("cfl_safety", stage2_cfg.cfl_safety);
        write_dbl("output_dt_s", stage2_cfg.output_dt_s);
        write_dbl("total_duration_s", stage2_cfg.total_duration_s);
        write_dbl("storage_limit_gb", stage2_cfg.storage_limit_gb);
        write_dbl("record_depth_max_m", stage2_cfg.record_depth_max_m);
        write_i64("n_ranks", stage2_cfg.n_ranks);
        write_i64("nx_elements", stage2_cfg.nx_elements);
        write_i64("ny_elements", stage2_cfg.ny_elements);
        write_i64("NGLL", stage2_cfg.polynomial_order + 1);
        write_str("snapshot_precision",
                  stage2_cfg.snapshot_precision_bytes == 4 ? "float32" : "float64");

        H5Gclose(cfg_grp);
        H5Fclose(config_fid);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  Stage 2: λ/μ, solver_dt
    // ═════════════════════════════════════════════════════════════════════════

    fprintf(stderr, "=== Stage 2 (λ/μ + CFL) ===\n");
    char* stage2_args[] = {argv[0], const_cast<char*>(model_path), nullptr};
    rc = stage2_main(2, stage2_args);
    if (rc != 0) {
        fprintf(stderr, "ERROR: stage2 failed with code %d\n", rc);
        return rc;
    }

    // ═════════════════════════════════════════════════════════════════════════
    //  C-PML + STF + source location
    // ═════════════════════════════════════════════════════════════════════════

    fprintf(stderr, "=== Post-stage2 steps (C-PML + source + STF) ===\n");

    hid_t model_fid = gf::h5::open_or_fail(model_path, H5F_ACC_RDWR);

    gf::Config cfg = gf::get_config();

    // Read coords for element/NGLL counts
    std::vector<double> gll_coords = gf::h5::read_double(model_fid, "field/element/coords");
    std::vector<hsize_t> coord_dims = gf::h5::get_dims(model_fid, "field/element/coords");
    int n_cell = static_cast<int>(coord_dims[0]);
    int ngll = static_cast<int>(coord_dims[1]);

    // Read vp and identify PML elements
    std::vector<double> vp_flat = gf::h5::read_double(model_fid, "field/element/vp");
    std::vector<double> damping = gf::h5::read_double(model_fid, "field/element/damping");

    std::vector<int> is_pml(n_cell, 0);
    for (int e = 0; e < n_cell; ++e) {
        for (int i = 0; i < ngll * ngll * ngll; ++i) {
            if (damping[e * ngll * ngll * ngll + i] > 0.0) {
                is_pml[e] = 1;
                break;
            }
        }
    }

    // Read domain bounds from attributes
    double domain_bounds[6] = {};
    hid_t dom_grp = H5Gopen2(model_fid, "domain", H5P_DEFAULT);
    if (dom_grp >= 0) {
        auto read_attr = [&](const char* name, double& v) {
            if (H5Aexists(dom_grp, name)) {
                hid_t a = H5Aopen(dom_grp, name, H5P_DEFAULT);
                H5Aread(a, H5T_NATIVE_DOUBLE, &v);
                H5Aclose(a);
            }
        };
        read_attr("xmin", domain_bounds[0]);
        read_attr("xmax", domain_bounds[1]);
        read_attr("ymin", domain_bounds[2]);
        read_attr("ymax", domain_bounds[3]);
        read_attr("zmin", domain_bounds[4]);
        read_attr("zmax", domain_bounds[5]);
        H5Gclose(dom_grp);
    }

    // PML element widths per direction
    double dx = (domain_bounds[1] - domain_bounds[0]) / std::max(cfg.nx_elements, 1);
    double dy = (domain_bounds[3] - domain_bounds[2]) / std::max(cfg.ny_elements, 1);
    int n_z_elements = n_cell / std::max(cfg.nx_elements * cfg.ny_elements, 1);
    double dz = (domain_bounds[5] - domain_bounds[4]) / std::max(n_z_elements, 1);

    double pml_widths_m[6] = {
        cfg.pml_xmin * dx, cfg.pml_xmax * dx, cfg.pml_ymin * dy,
        cfg.pml_ymax * dy, cfg.pml_zmin * dz, cfg.pml_zmax * dz,
    };

    // Classify PML regions: 1=X, 2=Y, 3=Z, combinations are sums
    std::vector<int> pml_regions(n_cell, 0);
    for (int e = 0; e < n_cell; ++e) {
        if (!is_pml[e])
            continue;
        const double* elem_coords = gll_coords.data() + e * ngll * ngll * ngll * 3;
        double xmin = 1e30, xmax = -1e30, ymin = 1e30, ymax = -1e30, zmin = 1e30, zmax = -1e30;
        for (int i = 0; i < ngll * ngll * ngll; ++i) {
            xmin = std::min(xmin, elem_coords[i * 3]);
            xmax = std::max(xmax, elem_coords[i * 3]);
            ymin = std::min(ymin, elem_coords[i * 3 + 1]);
            ymax = std::max(ymax, elem_coords[i * 3 + 1]);
            zmin = std::min(zmin, elem_coords[i * 3 + 2]);
            zmax = std::max(zmax, elem_coords[i * 3 + 2]);
        }
        int rx = (pml_widths_m[0] > 0 && xmin <= domain_bounds[0] + pml_widths_m[0])   ? 1
                 : (pml_widths_m[1] > 0 && xmax >= domain_bounds[1] - pml_widths_m[1]) ? 1
                                                                                       : 0;
        int ry = (pml_widths_m[2] > 0 && ymin <= domain_bounds[2] + pml_widths_m[2])   ? 2
                 : (pml_widths_m[3] > 0 && ymax >= domain_bounds[3] - pml_widths_m[3]) ? 2
                                                                                       : 0;
        int rz = (pml_widths_m[4] > 0 && zmin <= domain_bounds[4] + pml_widths_m[4])   ? 3
                 : (pml_widths_m[5] > 0 && zmax >= domain_bounds[5] - pml_widths_m[5]) ? 3
                                                                                       : 0;
        pml_regions[e] = rx + ry + rz;
    }

    // Compute C-PML κ/d/α profiles
    std::vector<double> cpml_K, cpml_d, cpml_alpha;
    gf::compute_cpml_profiles(gll_coords.data(), n_cell, ngll, is_pml.data(), pml_regions.data(),
                              domain_bounds, pml_widths_m, vp_flat.data(), cfg.f0_for_pml_hz,
                              cpml_K, cpml_d, cpml_alpha);

    std::vector<hsize_t> pml_dims = {static_cast<hsize_t>(n_cell),
                                     static_cast<hsize_t>(ngll * ngll * ngll), 3};
    gf::h5::write_double(model_fid, "field/element/cpml_K", cpml_K, pml_dims);
    gf::h5::write_double(model_fid, "field/element/cpml_d", cpml_d, pml_dims);
    gf::h5::write_double(model_fid, "field/element/cpml_alpha", cpml_alpha, pml_dims);
    fprintf(stderr, "  C-PML: %zu K/d/alpha values written\n", cpml_K.size() / 3);

    // STF
    int nsteps = static_cast<int>(cfg.total_duration_s / cfg.output_dt_s) + 1;
    double output_dt_s = cfg.output_dt_s;
    std::vector<double> stf_times, stf_values;
    gf::evaluate_stf_array(output_dt_s, nsteps, stf_times, stf_values);

    {
        std::vector<hsize_t> stf_shape = {static_cast<hsize_t>(nsteps)};
        gf::h5::write_double(model_fid, "config/stf_t", stf_times, stf_shape);
        gf::h5::write_double(model_fid, "config/stf_values", stf_values, stf_shape);
    }
    fprintf(stderr, "  STF: %d steps, dt=%g\n", nsteps, output_dt_s);

    // Source location
    fprintf(stderr, "=== Source location ===\n");

    double source_point_m[3] = {cfg.source_x_m, cfg.source_y_m, cfg.source_z_m};

    std::vector<int64_t> cell_to_surface =
        gf::h5::read_int64(model_fid, "topology/cell_to_surface");
    std::vector<hsize_t> c2s_dims = gf::h5::get_dims(model_fid, "topology/cell_to_surface");

    // Number of surfaces from topology/surface_to_edge
    std::vector<hsize_t> s2e_dims = gf::h5::get_dims(model_fid, "topology/surface_to_edge");

    std::vector<int64_t> boundary_tag =
        gf::h5::read_int64(model_fid, "field/surface/boundary_tag");

    gf::SourceResult source_result =
        gf::locate_source(cfg, gll_coords.data(), n_cell, ngll, cell_to_surface.data(),
                          static_cast<int>(s2e_dims[0]), boundary_tag.data(), is_pml.data());

    fprintf(stderr, "  Source in %d element(s)\n", source_result.n_src_cell);
    if (!source_result.cell_ids.empty()) {
        fprintf(stderr, "  Cell %d, xi=(%g,%g,%g)\n", source_result.cell_ids[0],
                source_result.xi[0], source_result.eta[0], source_result.zeta[0]);
    // Write source cell count to source attrs
    if (!source_result.cell_ids.empty()) {
        hid_t src_grp = H5Gopen2(model_fid, "source", H5P_DEFAULT);
        if (src_grp >= 0) {
            hsize_t one = 1;
            hid_t spc = H5Screate(H5S_SCALAR);
            hid_t attr = H5Acreate2(src_grp, "n_src_cell", H5T_NATIVE_INT, spc,
                                   H5P_DEFAULT, H5P_DEFAULT);
            int n_src = source_result.n_src_cell;
            H5Awrite(attr, H5T_NATIVE_INT, &n_src);
            H5Aclose(attr);
            H5Sclose(spc);
            H5Gclose(src_grp);
        }
    }
    }

    H5Fclose(model_fid);

    // ═════════════════════════════════════════════════════════════════════════
    //  METIS partition + global node numbering + config.h5
    // ═════════════════════════════════════════════════════════════════════════

    gf::partition_metis(model_path, cfg.n_ranks);
    gf::compute_global_node_ids(model_path, ngll);

    // Read solver_dt from stage2 output (/info group)
    model_fid = gf::h5::open_or_fail(model_path, H5F_ACC_RDWR);
    double solver_dt = 0.0;
    {
        hid_t info_gid = H5Gopen2(model_fid, "info", H5P_DEFAULT);
        if (info_gid >= 0) {
            if (H5Aexists(info_gid, "solver_dt")) {
                hid_t attr = H5Aopen(info_gid, "solver_dt", H5P_DEFAULT);
                H5Aread(attr, H5T_NATIVE_DOUBLE, &solver_dt);
                H5Aclose(attr);
            }
            H5Gclose(info_gid);
        }
    }
    if (solver_dt <= 0.0)
        solver_dt = cfg.output_dt_s / 10.0;

    int snapshot_stride = static_cast<int>(cfg.output_dt_s / solver_dt);
    double log_dt_s = cfg.log_stride * cfg.output_dt_s;

    std::vector<double> source_point_vec = {source_point_m[0], source_point_m[1],
                                            source_point_m[2]};

    // Read element-to-rank from partition group written by partition_metis
    std::vector<int32_t> element_to_rank;
    if (H5Lexists(model_fid, "partition", H5P_DEFAULT)) {
        hid_t part_grp = H5Gopen2(model_fid, "partition", H5P_DEFAULT);
        if (part_grp >= 0) {
            if (H5Lexists(part_grp, "element_to_rank", H5P_DEFAULT)) {
                // Read as int64_t (hsize_t), convert to int32_t
                std::vector<int64_t> etr64 = gf::h5::read_int64(part_grp, "element_to_rank");
                element_to_rank.resize(etr64.size());
                for (std::size_t i = 0; i < etr64.size(); ++i)
                    element_to_rank[i] = static_cast<int32_t>(etr64[i]);
            }
            H5Gclose(part_grp);
        }
    }

    H5Fclose(model_fid);

    gf::write_config_h5("config.h5", cfg, solver_dt, snapshot_stride, nsteps, stf_times,
                        stf_values, source_point_vec, source_result, cfg.record_depth_max_m,
                        element_to_rank, cfg.n_ranks, log_dt_s);

    fprintf(stderr, "=== Preprocess complete ===\n");
    return 0;
}

// ═════════════════════════════════════════════════════════════════════════════
//  usage
// ═════════════════════════════════════════════════════════════════════════════

static void usage() {
    fprintf(stderr,
            "Usage: gf_preprocess <stage1|stage2|run> [args...]\n"
            "\n"
            "  stage1  Compute GLL geometry, PML damping, CFL h_min.\n"
            "          gf_preprocess stage1 --help\n"
            "  stage2  Compute λ/μ, solver_dt, pre-flight statistics.\n"
            "          gf_preprocess stage2 <model.h5>\n"
            "  run     Full pipeline: stage1 → material → stage2 → C-PML → source\n"
            "          → STF → METIS → global IDs → config.h5\n"
            "          gf_preprocess run <model.h5> --N N --cfl-safety val ...\n"
            "\n"
            "Material model:\n"
            "  Compiled-in (static link):  cmake -DGF_MATERIAL_USER_SOURCE=my_model.cpp\n"
            "  Python fallback:            python -m preprocess\n");
}

// ═════════════════════════════════════════════════════════════════════════════
//  main
// ═════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    const char* subcommand = argv[1];

    if (std::strcmp(subcommand, "stage1") == 0)
        return stage1_main(argc - 1, argv + 1);
    if (std::strcmp(subcommand, "stage2") == 0)
        return stage2_main(argc - 1, argv + 1);
    if (std::strcmp(subcommand, "run") == 0)
        return run_main(argc - 1, argv + 1);
    if (std::strcmp(subcommand, "--help") == 0 || std::strcmp(subcommand, "-h") == 0) {
        usage();
        return 0;
    }
    if (std::strcmp(subcommand, "--version") == 0 || std::strcmp(subcommand, "-V") == 0) {
        fprintf(stdout, "gf_preprocess v1.0.0\n");
        return 0;
    }

    fprintf(stderr, "ERROR: unknown subcommand: %s\n", subcommand);
    usage();
    return 1;
}