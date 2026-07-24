/* gf_preprocess.cpp — unified preprocess entry point
 *
 * Dispatches to stage1, stage2, or run (stage1 → material → stage2).
 *
 * Usage:
 *   gf_preprocess stage1 <model.h5> [--N N] [--cfl-safety val] ...
 *   gf_preprocess stage2 <model.h5>
 *   gf_preprocess run <model.h5> [--N N] [--cfl-safety val] ...
 *
 * The 'run' subcommand chains stage1 and stage2, inserting user-provided
 * material evaluation in between.  When compiled with GF_HAS_USER_MATERIAL
 * (i.e. -DGF_MATERIAL_USER_SOURCE=... was passed to CMake), it calls the
 * user's gf::evaluate_vp/vs/density directly.  Otherwise it falls back to
 * reading pre-computed arrays from HDF5 (the Python path).
 */

#include <hdf5.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gf_config.h"

#ifdef GF_HAS_USER_MATERIAL
#include "gf_material_user.h"
#endif

extern int stage1_main(int argc, char** argv);
extern int stage2_main(int argc, char** argv);

// ── helpers ────────────────────────────────────────────────────────────────

static hid_t open_or_fail(const char* path, unsigned flags) {
    hid_t fid = H5Fopen(path, flags, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "ERROR: cannot open HDF5 file: %s\n", path);
        std::exit(1);
    }
    return fid;
}

static std::vector<double> read_dataset_double(hid_t fid, const char* name) {
    hid_t ds = H5Dopen2(fid, name, H5P_DEFAULT);
    if (ds < 0) {
        fprintf(stderr, "ERROR: dataset not found: %s\n", name);
        std::exit(1);
    }
    hid_t space = H5Dget_space(ds);
    hsize_t dims[8];
    int ndims = H5Sget_simple_extent_ndims(space);
    H5Sget_simple_extent_dims(space, dims, nullptr);
    hsize_t total = 1;
    for (int i = 0; i < ndims; ++i)
        total *= dims[i];
    std::vector<double> buf(total);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    return buf;
}

static void write_dataset_double(hid_t fid, const char* name, const std::vector<double>& data,
                                 const std::vector<hsize_t>& dims) {
    hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
    hid_t ds =
        H5Dcreate2(fid, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(ds);
    H5Sclose(space);
}

// ── run subcommand ─────────────────────────────────────────────────────────

static int run_main(int argc, char** argv) {
    if (argc < 1) {
        fprintf(stderr,
                "Usage: gf_preprocess run <model.h5> [--N N] [--cfl-safety val] "
                "[--nx N] [--ny N] [--pml-* THICK]\n");
        return 1;
    }

    // Collect stage1 arguments (model.h5 + --option pairs)
    std::vector<char*> stage1_args;
    stage1_args.push_back(argv[0]);  // program name placeholder
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "run") == 0)
            continue;
        stage1_args.push_back(argv[i]);
    }

    // stage1_main expects model path at argv[1] (argv[0] = program name)
    int rc = stage1_main(static_cast<int>(stage1_args.size()), stage1_args.data());
    if (rc != 0) {
        fprintf(stderr, "ERROR: stage1 failed with code %d\n", rc);
        return rc;
    }

    // Find model.h5 path (first non-option arg after "run")
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

#ifdef GF_HAS_USER_MATERIAL
    // ── User material evaluation (direct C++ call) ──
    fprintf(stderr, "=== Material evaluation (user C++ model) ===\n");

    hid_t fid = open_or_fail(model_path, H5F_ACC_RDWR);

    // Read GLL coordinates
    std::vector<double> coords_flat = read_dataset_double(fid, "field/element/coords");
    std::size_t n_points = coords_flat.size() / 3;
    fprintf(stderr, "  GLL nodes: %zu\n", n_points);

    // Extract x, y, z arrays
    std::vector<double> x_arr(n_points), y_arr(n_points), z_arr(n_points);
    for (std::size_t i = 0; i < n_points; ++i) {
        x_arr[i] = coords_flat[i * 3 + 0];
        y_arr[i] = coords_flat[i * 3 + 1];
        z_arr[i] = coords_flat[i * 3 + 2];
    }

    // Evaluate user material model
    std::vector<double> vp(n_points), vs(n_points), density(n_points);
    gf::evaluate_vp(n_points, x_arr.data(), y_arr.data(), z_arr.data(), vp.data());
    gf::evaluate_vs(n_points, x_arr.data(), y_arr.data(), z_arr.data(), vs.data());
    gf::evaluate_density(n_points, x_arr.data(), y_arr.data(), z_arr.data(), density.data());

    // Determine field shape from coords dataset
    hid_t ds = H5Dopen2(fid, "field/element/coords", H5P_DEFAULT);
    hid_t space = H5Dget_space(ds);
    hsize_t dims[8];
    int ndims = H5Sget_simple_extent_ndims(space);
    H5Sget_simple_extent_dims(space, dims, nullptr);
    H5Dclose(ds);
    H5Sclose(space);

    // shape is [n_cell, NGLL, NGLL, NGLL, 3] → field shape = first ndims-1 dims
    std::vector<hsize_t> field_dims(dims, dims + ndims - 1);

    // Delete existing datasets if present
    H5Ldelete(fid, "field/element/vp", H5P_DEFAULT);
    H5Ldelete(fid, "field/element/vs", H5P_DEFAULT);
    H5Ldelete(fid, "field/element/density", H5P_DEFAULT);

    write_dataset_double(fid, "field/element/vp", vp, field_dims);
    write_dataset_double(fid, "field/element/vs", vs, field_dims);
    write_dataset_double(fid, "field/element/density", density, field_dims);

    H5Fclose(fid);
    fprintf(stderr, "  vp/vs/density written to model.h5\n");
#else
    fprintf(stderr,
            "=== Material evaluation ===\n"
            "  No user material compiled in (GF_HAS_USER_MATERIAL not defined).\n"
            "  Looking for vp/vs/density in HDF5 (Python path)...\n");
#endif

    // ── Post-stage2: C-PML, STF ──
    fprintf(stderr, "=== Post-stage2 steps (C-PML + STF) ===\n");

    hid_t fid2 = open_or_fail(model_path, H5F_ACC_RDWR);

    // Get config (available regardless of GF_HAS_USER_MATERIAL)
    gf::Config cfg = gf::get_config();

    // Read coords for shape info
    std::vector<double> cm = read_dataset_double(fid2, "field/element/coords");
    hid_t dds = H5Dopen2(fid2, "field/element/coords", H5P_DEFAULT);
    hid_t ssp = H5Dget_space(dds);
    hsize_t cdims[8];
    int cnd = H5Sget_simple_extent_ndims(ssp);
    H5Sget_simple_extent_dims(ssp, cdims, nullptr);
    int nc = static_cast<int>(cdims[0]);
    int ng = static_cast<int>(cdims[1]);
    H5Dclose(dds);
    H5Sclose(ssp);

    // Read vp, is_pml, domain bounds
    std::vector<double> vp = read_dataset_double(fid2, "field/element/vp");
    std::vector<double> damp = read_dataset_double(fid2, "field/element/damping");
    std::vector<int> isp(nc, false);
    for (int e = 0; e < nc; ++e)
        for (int i = 0; i < ng * ng * ng; ++i)
            if (damp[e * ng * ng * ng + i] > 0.0) {
                isp[e] = 1;
                break;
            }

    double db[6] = {};
    hid_t dom = H5Gopen2(fid2, "domain", H5P_DEFAULT);
    if (dom >= 0) {
        auto rd = [&](const char* n, double& v) {
            if (H5Aexists(dom, n)) {
                hid_t a = H5Aopen(dom, n, H5P_DEFAULT);
                H5Aread(a, H5T_NATIVE_DOUBLE, &v);
                H5Aclose(a);
            }
        };
        rd("xmin", db[0]);
        rd("xmax", db[1]);
        rd("ymin", db[2]);
        rd("ymax", db[3]);
        rd("zmin", db[4]);
        rd("zmax", db[5]);
        H5Gclose(dom);
    }

    double dx = (db[1] - db[0]) / std::max(cfg.nx_elements, 1);
    double dy = (db[3] - db[2]) / std::max(cfg.ny_elements, 1);
    int nz_el = nc / std::max(cfg.nx_elements * cfg.ny_elements, 1);
    double dz = (db[5] - db[4]) / std::max(nz_el, 1);
    double pw[6] = {cfg.pml_xmin * dx, cfg.pml_xmax * dx, cfg.pml_ymin * dy,
                    cfg.pml_ymax * dy, cfg.pml_zmin * dz, cfg.pml_zmax * dz};

    // PML regions: classify from is_pml + element position
    std::vector<int> pmr(nc, 0);
    for (int e = 0; e < nc; ++e) {
        if (!isp[e])
            continue;
        const double* ec = cm.data() + e * ng * ng * ng * 3;
        double xmin = 1e30, xmax = -1e30, ymin = 1e30, ymax = -1e30, zmin = 1e30, zmax = -1e30;
        for (int i = 0; i < ng * ng * ng; ++i) {
            xmin = std::min(xmin, ec[i * 3]);
            xmax = std::max(xmax, ec[i * 3]);
            ymin = std::min(ymin, ec[i * 3 + 1]);
            ymax = std::max(ymax, ec[i * 3 + 1]);
            zmin = std::min(zmin, ec[i * 3 + 2]);
            zmax = std::max(zmax, ec[i * 3 + 2]);
        }
        int rx = (pw[0] > 0 && xmin <= db[0] + pw[0])   ? 1
                 : (pw[1] > 0 && xmax >= db[1] - pw[1]) ? 1
                                                        : 0;
        int ry = (pw[2] > 0 && ymin <= db[2] + pw[2])   ? 2
                 : (pw[3] > 0 && ymax >= db[3] - pw[3]) ? 2
                                                        : 0;
        int rz = (pw[4] > 0 && zmin <= db[4] + pw[4])   ? 3
                 : (pw[5] > 0 && zmax >= db[5] - pw[5]) ? 3
                                                        : 0;
        pmr[e] = rx + ry + rz;
    }

    // C-PML profiles
    std::vector<double> K_store, d_store, alpha_store;
    gf::compute_cpml_profiles(cm.data(), nc, ng, isp.data(), pmr.data(), db, pw, vp.data(),
                              cfg.f0_for_pml_hz, K_store, d_store, alpha_store);
    std::vector<hsize_t> p3d = {static_cast<hsize_t>(nc), static_cast<hsize_t>(ng * ng * ng), 3};
    H5Ldelete(fid2, "field/element/cpml_K", H5P_DEFAULT);
    H5Ldelete(fid2, "field/element/cpml_d", H5P_DEFAULT);
    H5Ldelete(fid2, "field/element/cpml_alpha", H5P_DEFAULT);
    write_dataset_double(fid2, "field/element/cpml_K", K_store, p3d);
    write_dataset_double(fid2, "field/element/cpml_d", d_store, p3d);
    write_dataset_double(fid2, "field/element/cpml_alpha", alpha_store, p3d);
    fprintf(stderr, "  C-PML: %zu K/d/alpha values written\n", K_store.size() / 3);

    // STF
    int nsteps = static_cast<int>(cfg.total_duration_s / cfg.output_dt_s) + 1;
    double sdt = cfg.output_dt_s;
    std::vector<double> st, sv;
    gf::evaluate_stf_array(sdt, nsteps, st, sv);
    std::vector<hsize_t> sdim = {static_cast<hsize_t>(nsteps)};
    H5Ldelete(fid2, "config/stf_t", H5P_DEFAULT);
    H5Ldelete(fid2, "config/stf_values", H5P_DEFAULT);
    write_dataset_double(fid2, "config/stf_t", st, sdim);
    write_dataset_double(fid2, "config/stf_values", sv, sdim);
    fprintf(stderr, "  STF: %d timesteps, dt=%g\n", nsteps, sdt);

    H5Fclose(fid2);
    fprintf(stderr, "=== Preprocess complete ===\n");
    return 0;
}

// ── usage ──────────────────────────────────────────────────────────────────

static void usage() {
    fprintf(stderr,
            "Usage: gf_preprocess <stage1|stage2|run> [args...]\n"
            "\n"
            "  stage1  Compute GLL geometry, PML damping, CFL h_min.\n"
            "          gf_preprocess stage1 --help\n"
            "  stage2  Compute λ/μ, solver_dt, pre-flight statistics.\n"
            "          gf_preprocess stage2 <model.h5>\n"
            "  run     Run stage1 → material → stage2 in one invocation.\n"
            "          gf_preprocess run <model.h5> --N N --cfl-safety val ...\n"
            "\n"
            "Material model:\n"
            "  Compiled-in (static link):  cmake -DGF_MATERIAL_USER_SOURCE=my_model.cpp\n"
            "  Python fallback:            python -m preprocess\n");
}

// ── main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    const char* subcommand = argv[1];

    if (std::strcmp(subcommand, "stage1") == 0) {
        return stage1_main(argc - 1, argv + 1);
    }
    if (std::strcmp(subcommand, "stage2") == 0) {
        return stage2_main(argc - 1, argv + 1);
    }
    if (std::strcmp(subcommand, "run") == 0) {
        return run_main(argc - 1, argv + 1);
    }

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