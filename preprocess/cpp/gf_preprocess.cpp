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
#include <filesystem>
#include <string>
#include <vector>

#include "gf_config.h"
#include "gf_hdf5_helpers.hpp"

#ifdef GF_HAS_USER_MATERIAL
#include "gf_material_user.h"
#endif

/// C++ preprocessor stage 1: compute GLL geometry and model arrays.
extern int stage1_main(int argc, char** argv);
/// C++ preprocessor stage 2: partition mesh and write per-rank HDF5.
extern int stage2_main(int argc, char** argv);

// ═════════════════════════════════════════════════════════════════════════════
//  run subcommand  —  full pipeline: stage1 → material → stage2 → ... → config.h5
// ═════════════════════════════════════════════════════════════════════════════

static int run_main(int argc, char** argv) {
    if (argc < 1) {
        fprintf(stderr,
                "Usage: gf_preprocess run <model.h5> [--N N] [--cfl-safety val] "
                "[--nx N] [--ny N] [--n-ranks N] [--pml-* THICK]\n");
        return 1;
    }

    // ── Collect stage1 arguments (model.h5 + --option pairs) ──
    // `--n-ranks N` is consumed here (NOT forwarded to stage1, which rejects
    // unknown options); it overrides cfg.n_ranks for the METIS partition so the
    // CLI (config.py:n_ranks) is the single source of truth for rank count.

    int n_ranks_override = -1;
    const char* model_path = nullptr;
    std::vector<char*> stage1_args;
    stage1_args.push_back(argv[0]);  // program name placeholder
    for (int i = 0; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "run") == 0)
            continue;
        if (std::strcmp(arg, "--n-ranks") == 0) {
            if (i + 1 < argc) {
                n_ranks_override = std::atoi(argv[++i]);
                continue;
            }
        }
        if (model_path == nullptr && arg[0] != '-')
            model_path = arg;
        stage1_args.push_back(argv[i]);
    }

    // ── Stage 1: GLL geometry, PML damping, CFL h_min ──

    int rc = stage1_main(static_cast<int>(stage1_args.size()), stage1_args.data());
    if (rc != 0) {
        fprintf(stderr, "ERROR: stage1 failed with code %d\n", rc);
        return rc;
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
        if (H5Lexists(config_fid, "config", H5P_DEFAULT) > 0)
            H5Ldelete(config_fid, "config", H5P_DEFAULT);
        hid_t cfg_grp = H5Gcreate2(config_fid, "config", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        gf::Config stage2_cfg = gf::get_config();
        if (n_ranks_override > 0)
            stage2_cfg.n_ranks = n_ranks_override;

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
    if (n_ranks_override > 0)
        cfg.n_ranks = n_ranks_override;

    // Read coords for element/NGLL counts
    std::vector<double> gll_coords = gf::h5::read_double(model_fid, "field/element/coords");
    std::vector<hsize_t> coord_dims = gf::h5::get_dims(model_fid, "field/element/coords");
    int n_cell = static_cast<int>(coord_dims[0]);
    int ngll = static_cast<int>(coord_dims[1]);

    // Stage2 is authoritative for the solver timestep and total step count.
    double solver_dt = 0.0;
    int snapshot_stride = 0;
    int64_t nsteps64 = 0;
    {
        hid_t info_group = H5Gopen2(model_fid, "info", H5P_DEFAULT);
        if (info_group < 0) {
            fprintf(stderr, "ERROR: stage2 did not create /info\n");
            return 1;
        }
        hid_t attribute = H5Aopen(info_group, "solver_dt", H5P_DEFAULT);
        H5Aread(attribute, H5T_NATIVE_DOUBLE, &solver_dt);
        H5Aclose(attribute);
        attribute = H5Aopen(info_group, "snapshot_stride", H5P_DEFAULT);
        H5Aread(attribute, H5T_NATIVE_INT, &snapshot_stride);
        H5Aclose(attribute);
        attribute = H5Aopen(info_group, "nsteps", H5P_DEFAULT);
        H5Aread(attribute, H5T_NATIVE_INT64, &nsteps64);
        H5Aclose(attribute);
        H5Gclose(info_group);
    }
    if (solver_dt <= 0.0 || snapshot_stride <= 0 || nsteps64 <= 0) {
        fprintf(stderr, "ERROR: invalid stage2 timing metadata\n");
        H5Fclose(model_fid);
        return 1;
    }
    int nsteps = static_cast<int>(nsteps64);

    // Read material and the authoritative PML element mask from stage1.
    std::vector<double> vp_flat = gf::h5::read_double(model_fid, "field/element/vp");
    std::vector<int64_t> is_pml_i64 = gf::h5::read_int64(model_fid, "field/element/is_pml");
    std::vector<int> is_pml(is_pml_i64.begin(), is_pml_i64.end());

    // Read domain bounds, or derive and persist them for topology-only input files.
    double domain_bounds[6] = {};
    hid_t dom_grp = -1;
    if (H5Lexists(model_fid, "domain", H5P_DEFAULT) > 0)
        dom_grp = H5Gopen2(model_fid, "domain", H5P_DEFAULT);
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
    } else {
        domain_bounds[0] = domain_bounds[2] = domain_bounds[4] = 1.0e300;
        domain_bounds[1] = domain_bounds[3] = domain_bounds[5] = -1.0e300;
        for (size_t node = 0; node < gll_coords.size() / 3; ++node) {
            for (int axis = 0; axis < 3; ++axis) {
                domain_bounds[axis * 2] =
                    std::min(domain_bounds[axis * 2], gll_coords[node * 3 + axis]);
                domain_bounds[axis * 2 + 1] =
                    std::max(domain_bounds[axis * 2 + 1], gll_coords[node * 3 + axis]);
            }
        }
        dom_grp = H5Gcreate2(model_fid, "domain", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        const char* bound_names[] = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
        for (int bound = 0; bound < 6; ++bound) {
            hid_t space = H5Screate(H5S_SCALAR);
            hid_t attribute = H5Acreate2(dom_grp, bound_names[bound], H5T_NATIVE_DOUBLE, space,
                                         H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(attribute, H5T_NATIVE_DOUBLE, &domain_bounds[bound]);
            H5Aclose(attribute);
            H5Sclose(space);
        }
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
        double center_x = 0.0, center_y = 0.0, center_z = 0.0;
        for (int i = 0; i < ngll * ngll * ngll; ++i) {
            center_x += elem_coords[i * 3];
            center_y += elem_coords[i * 3 + 1];
            center_z += elem_coords[i * 3 + 2];
        }
        double inverse_node_count = 1.0 / (ngll * ngll * ngll);
        center_x *= inverse_node_count;
        center_y *= inverse_node_count;
        center_z *= inverse_node_count;
        constexpr double relative_tolerance = 1.0e-6;
        bool active_x =
            (pml_widths_m[0] > 0 && center_x < domain_bounds[0] + pml_widths_m[0] +
                                                   relative_tolerance * pml_widths_m[0]) ||
            (pml_widths_m[1] > 0 &&
             center_x > domain_bounds[1] - pml_widths_m[1] - relative_tolerance * pml_widths_m[1]);
        bool active_y =
            (pml_widths_m[2] > 0 && center_y < domain_bounds[2] + pml_widths_m[2] +
                                                   relative_tolerance * pml_widths_m[2]) ||
            (pml_widths_m[3] > 0 &&
             center_y > domain_bounds[3] - pml_widths_m[3] - relative_tolerance * pml_widths_m[3]);
        bool active_z =
            (pml_widths_m[4] > 0 && center_z < domain_bounds[4] + pml_widths_m[4] +
                                                   relative_tolerance * pml_widths_m[4]) ||
            (pml_widths_m[5] > 0 &&
             center_z > domain_bounds[5] - pml_widths_m[5] - relative_tolerance * pml_widths_m[5]);
        if (active_x && active_y && active_z)
            pml_regions[e] = 7;
        else if (active_y && active_z)
            pml_regions[e] = 6;
        else if (active_x && active_z)
            pml_regions[e] = 5;
        else if (active_x && active_y)
            pml_regions[e] = 4;
        else if (active_z)
            pml_regions[e] = 3;
        else if (active_y)
            pml_regions[e] = 2;
        else if (active_x)
            pml_regions[e] = 1;
    }

    // Compute the complete C-PML profiles and recursive-convolution coefficients.
    std::vector<double> cpml_K, cpml_d, cpml_alpha;
    gf::compute_cpml_profiles(gll_coords.data(), n_cell, ngll, is_pml.data(), pml_regions.data(),
                              domain_bounds, pml_widths_m, vp_flat.data(), cfg.f0_for_pml_hz,
                              cpml_K, cpml_d, cpml_alpha);
    std::vector<double> pml_coefficient_alpha, pml_coefficient_beta, pml_coefficient_acceleration,
        pml_coefficient_strain;
    gf::compute_cpml_coefficients(n_cell, ngll, pml_regions.data(), solver_dt, cpml_K, cpml_d,
                                  cpml_alpha, pml_coefficient_alpha, pml_coefficient_beta,
                                  pml_coefficient_acceleration, pml_coefficient_strain);

    std::vector<double> mass = gf::h5::read_double(model_fid, "field/element/mass");
    gf::apply_cpml_mass_correction(n_cell, ngll, pml_regions.data(), solver_dt, cpml_K, cpml_d,
                                   mass);
    gf::h5::write_double(model_fid, "field/element/mass", mass,
                         {static_cast<hsize_t>(n_cell), static_cast<hsize_t>(ngll),
                          static_cast<hsize_t>(ngll), static_cast<hsize_t>(ngll)});

    std::vector<hsize_t> pml_dims = {static_cast<hsize_t>(n_cell),
                                     static_cast<hsize_t>(ngll * ngll * ngll), 3};
    gf::h5::write_double(model_fid, "field/element/cpml_K", cpml_K, pml_dims);
    gf::h5::write_double(model_fid, "field/element/cpml_d", cpml_d, pml_dims);
    gf::h5::write_double(model_fid, "field/element/cpml_alpha", cpml_alpha, pml_dims);
    std::vector<hsize_t> convolution_dimensions = {static_cast<hsize_t>(n_cell),
                                                   static_cast<hsize_t>(ngll * ngll * ngll), 9};
    gf::h5::write_double(model_fid, "field/element/pml_coef_alpha", pml_coefficient_alpha,
                         convolution_dimensions);
    gf::h5::write_double(model_fid, "field/element/pml_coef_beta", pml_coefficient_beta,
                         convolution_dimensions);
    gf::h5::write_double(
        model_fid, "field/element/pml_coef_abar", pml_coefficient_acceleration,
        {static_cast<hsize_t>(n_cell), static_cast<hsize_t>(ngll * ngll * ngll), 5});
    gf::h5::write_double(
        model_fid, "field/element/pml_coef_strain", pml_coefficient_strain,
        {static_cast<hsize_t>(n_cell), static_cast<hsize_t>(ngll * ngll * ngll), 18});
    std::vector<int32_t> pml_regions_i32(pml_regions.begin(), pml_regions.end());
    gf::h5::write_int32(model_fid, "field/element/pml_region", pml_regions_i32,
                        {static_cast<hsize_t>(n_cell)});
    fprintf(stderr, "  C-PML: %zu nodes with complete convolution coefficients written\n",
            cpml_K.size() / 3);

    // STF
    std::vector<double> stf_times, stf_values;
    gf::evaluate_stf_array(solver_dt, nsteps, stf_times, stf_values);

    {
        std::vector<hsize_t> stf_shape = {static_cast<hsize_t>(nsteps)};
        gf::h5::write_double(model_fid, "config/stf_t", stf_times, stf_shape);
        gf::h5::write_double(model_fid, "config/stf_values", stf_values, stf_shape);
    }
    fprintf(stderr, "  STF: %d steps, dt=%g\n", nsteps, solver_dt);

    // Source location
    fprintf(stderr, "=== Source location ===\n");

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
            hid_t src_grp = -1;
            if (H5Lexists(model_fid, "source", H5P_DEFAULT) > 0)
                src_grp = H5Gopen2(model_fid, "source", H5P_DEFAULT);
            if (src_grp >= 0) {
                hsize_t one = 1;
                hid_t spc = H5Screate(H5S_SCALAR);
                hid_t attr = H5Acreate2(src_grp, "n_src_cell", H5T_NATIVE_INT, spc, H5P_DEFAULT,
                                        H5P_DEFAULT);
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

    // Read element-to-rank and write complete solver partition files.
    model_fid = gf::h5::open_or_fail(model_path, H5F_ACC_RDWR);
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

    double record_depth_actual_m = gf::write_partition_files(model_path, cfg, element_to_rank);
    int nz_elements = n_cell / std::max(cfg.nx_elements * cfg.ny_elements, 1);
    std::filesystem::path config_path =
        std::filesystem::path(model_path).parent_path() / "config.h5";
    gf::write_config_h5(config_path.c_str(), cfg, solver_dt, snapshot_stride, nsteps, stf_times,
                        stf_values, source_result, domain_bounds, nz_elements,
                        record_depth_actual_m);

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
            "          → STF → METIS → partitions + recording map → config.h5\n"
            "          gf_preprocess run <model.h5> --N N --cfl-safety val ...\n"
            "\n"
            "C++ configuration:\n"
            "  Compiled-in:  cmake -DGF_USER_CONFIG=/absolute/path/to/config_user.cpp\n"
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
