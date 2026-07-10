/* postprocess/cpp/main.cpp — C++ accelerated Green's function postprocessor
 *
 * CLI:
 *   gf_postprocess <model.h5> <config.h5> \
 *       --fx <dir> --fy <dir> --fz <dir> -o <output_dir>
 *
 * Pipeline:
 *   1. Read config.h5 + model.h5
 *   2. Discover per-step record files in each direction dir
 *   3. Per-step: merge strain by vertex_id across ranks
 *   4. Assemble Green's tensor [nt, n_vertex, 6, 3]
 *   5. Bin recorded vertices into tiles (element-count or spatial)
 *   6. Write tile_x{i}_y{j}.h5 files
 *
 * Output matches Python gf_post.writer.GFWriter byte-for-byte equivalent.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "reader.hh"
#include "writer.hh"

// -----------------------------------------------------------------------
// CLI argument parsing
// -----------------------------------------------------------------------

struct Args {
    std::string model_path;
    std::string config_path;
    std::string fx_dir;
    std::string fy_dir;
    std::string fz_dir;
    std::string output_dir = "greenfun";
};

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s <model.h5> <config.h5> --fx <dir> --fy <dir> --fz <dir> [-o <dir>]\n"
            "\n"
            "Extract strain Green's functions from SEM record files.\n"
            "Reads per-step record_{r}_{step}.h5 files from three force-direction\n"
            "forward runs, merges per-rank records, assembles the full 3x6 Green's\n"
            "tensor at every recorded mesh vertex, and writes tiled HDF5 output.\n"
            "\n"
            "Arguments:\n"
            "  model.h5   Mesh file with /topology/vertex_to_coord + /domain/ bounds\n"
            "  config.h5  Simulation config with /simulation/ attrs and tile arrays\n"
            "  --fx dir   Directory with x-direction record files\n"
            "  --fy dir   Directory with y-direction record files\n"
            "  --fz dir   Directory with z-direction record files\n"
            "  -o dir     Output directory (default: greenfun/)\n",
            prog);
}

static Args parse_args(int argc, char** argv) {
    if (argc < 7) {
        print_usage(argv[0]);
        exit(1);
    }

    Args args;
    args.model_path = argv[1];
    args.config_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fx") == 0 && i + 1 < argc) {
            args.fx_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--fy") == 0 && i + 1 < argc) {
            args.fy_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--fz") == 0 && i + 1 < argc) {
            args.fz_dir = argv[++i];
        } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            args.output_dir = argv[++i];
        }
    }

    if (args.fx_dir.empty() || args.fy_dir.empty() || args.fz_dir.empty()) {
        fprintf(stderr, "ERROR: --fx, --fy, and --fz are required\n");
        print_usage(argv[0]);
        exit(1);
    }

    return args;
}

// -----------------------------------------------------------------------
// Merge records for one direction: returns [n_steps, n_vertex, 6] float32
// -----------------------------------------------------------------------

struct MergedDirection {
    std::vector<float> strain;        // [n_steps, n_vertex, 6]
    std::vector<float> displacement;  // [n_steps, n_vertex, 3]
    std::vector<bool> vertex_mask;    // [n_vertex] — which vertices were recorded
    int64_t n_steps = 0;
    bool has_displacement = false;
};

static MergedDirection merge_direction(const char* dir_path, int64_t n_vertex) {
    MergedDirection result;
    fprintf(stderr, "[postprocess] Merging records from %s...\n", dir_path);

    auto files = discover_records(dir_path);
    if (files.empty()) {
        fprintf(stderr, "ERROR: no record files found in %s\n", dir_path);
        exit(1);
    }

    // Check for legacy format
    bool has_legacy = false;
    for (auto& f : files) {
        // Check if filename has single number (legacy)
        std::string bn = f.path.substr(f.path.find_last_of('/') + 1);
        int r, s;
        bool leg;
        parse_record_filename(bn, r, s, leg);
        if (leg)
            has_legacy = true;
    }
    if (has_legacy) {
        fprintf(stderr,
                "[postprocess] Detected legacy monolithic record files (record_{r}.h5).\n");
    }

    auto groups = group_by_step(files);
    result.n_steps = (int64_t)groups.size();

    fprintf(stderr, "[postprocess]   Found %lld steps, %lld files\n", (long long)result.n_steps,
            (long long)files.size());

    // Allocate merged arrays
    result.strain.resize((size_t)result.n_steps * (size_t)n_vertex * 6, 0.0f);
    result.displacement.resize((size_t)result.n_steps * (size_t)n_vertex * 3, 0.0f);
    result.vertex_mask.resize((size_t)n_vertex, false);

    // Check first file for displacement dataset presence
    if (!files.empty()) {
        hid_t probe = H5Fopen(files[0].path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (probe >= 0) {
            hid_t ds = H5Dopen2(probe, "displacement", H5P_DEFAULT);
            result.has_displacement = (ds >= 0);
            if (ds >= 0)
                H5Dclose(ds);
            H5Fclose(probe);
        }
    }

    for (int64_t snap_idx = 0; snap_idx < result.n_steps; ++snap_idx) {
        auto& group = groups[(size_t)snap_idx];
        float* step_data = result.strain.data() + snap_idx * n_vertex * 6;
        float* step_disp = result.displacement.data() + snap_idx * n_vertex * 3;

        // Per-rank scratch buffers for this step
        std::vector<float> step_scratch((size_t)n_vertex * 6, 0.0f);
        std::vector<float> step_disp_scratch((size_t)n_vertex * 3, 0.0f);
        std::vector<bool> step_mask((size_t)n_vertex, false);

        for (auto& fi : group.files) {
            read_record_into(fi, n_vertex, step_scratch, step_mask, step_disp_scratch);
        }

        // Copy to merged array and accumulate global mask
        for (int64_t vi = 0; vi < n_vertex; ++vi) {
            if (step_mask[(size_t)vi]) {
                float* src = step_scratch.data() + vi * 6;
                float* dst = step_data + vi * 6;
                for (int c = 0; c < 6; ++c)
                    dst[c] = src[c];
                result.vertex_mask[(size_t)vi] = true;

                // Copy displacement if available
                if (result.has_displacement) {
                    float* dsrc = step_disp_scratch.data() + vi * 3;
                    float* ddst = step_disp + vi * 3;
                    for (int c = 0; c < 3; ++c)
                        ddst[c] = dsrc[c];
                }
            }
        }
    }

    return result;
}

// (assembly done inline in main to subset to recorded vertices)
// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------

int main(int argc, char** argv) {
    double start = 0.0;
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        start = ts.tv_sec + ts.tv_nsec * 1e-9;
    }

    fprintf(stderr, "[postprocess] Starting...\n");

    Args args = parse_args(argc, argv);

    // ---- Read config ----
    fprintf(stderr, "[postprocess] Reading config from %s\n", args.config_path.c_str());
    ConfigParams cfg = read_config(args.config_path.c_str());

    // ---- Read mesh ----
    fprintf(stderr, "[postprocess] Reading mesh geometry from %s\n", args.model_path.c_str());
    ModelData model = read_model(args.model_path.c_str());
    int64_t n_vertex = model.n_vertex;
    fprintf(stderr, "[postprocess]   n_vertex = %lld\n", (long long)n_vertex);

    // ---- Merge records for each direction ----
    MergedDirection fx = merge_direction(args.fx_dir.c_str(), n_vertex);
    MergedDirection fy = merge_direction(args.fy_dir.c_str(), n_vertex);
    MergedDirection fz = merge_direction(args.fz_dir.c_str(), n_vertex);

    // Consistency: same number of steps
    if (fx.n_steps != fy.n_steps || fx.n_steps != fz.n_steps) {
        fprintf(stderr,
                "ERROR: mismatched number of steps across directions "
                "(%lld, %lld, %lld)\n",
                (long long)fx.n_steps, (long long)fy.n_steps, (long long)fz.n_steps);
        exit(1);
    }
    int64_t n_steps = fx.n_steps;

    // Combined vertex mask
    std::vector<bool> vertex_mask((size_t)n_vertex, false);
    for (int64_t i = 0; i < n_vertex; ++i) {
        vertex_mask[(size_t)i] =
            fx.vertex_mask[(size_t)i] && fy.vertex_mask[(size_t)i] && fz.vertex_mask[(size_t)i];
    }

    // Check consistency
    bool masks_match = true;
    for (int64_t i = 0; i < n_vertex; ++i) {
        if (fx.vertex_mask[(size_t)i] != fy.vertex_mask[(size_t)i] ||
            fx.vertex_mask[(size_t)i] != fz.vertex_mask[(size_t)i]) {
            masks_match = false;
            break;
        }
    }
    if (!masks_match) {
        fprintf(stderr, "[postprocess] WARNING: recorded vertex sets differ across directions\n");
    }

    // Build recorded vertex list (1-based)
    std::vector<int64_t> recorded_ids;
    for (int64_t i = 0; i < n_vertex; ++i) {
        if (vertex_mask[(size_t)i]) {
            recorded_ids.push_back(i + 1);  // 1-based
        }
    }
    int64_t n_recorded = (int64_t)recorded_ids.size();
    fprintf(stderr, "[postprocess] %lld/%lld vertices recorded\n", (long long)n_recorded,
            (long long)n_vertex);

    if (n_recorded == 0) {
        fprintf(stderr, "ERROR: no vertices recorded\n");
        return 1;
    }

    // ---- Build time array ----
    std::vector<double> time_arr((size_t)n_steps);
    for (int64_t s = 0; s < n_steps; ++s) {
        time_arr[(size_t)s] = (double)s * cfg.output_dt_s;
    }

    // ---- Subset strain to recorded vertices and assemble Green's tensor ----
    // First, subset each direction to recorded vertices
    // strain_fx: [n_steps, n_vertex, 6] → subset to [n_steps, n_recorded, 6]
    // We'll do this during assembly

    // Build recorded index map: global vertex 0-based → recorded index
    std::vector<int64_t> global_to_recorded((size_t)n_vertex, -1);
    for (int64_t ri = 0; ri < n_recorded; ++ri) {
        int64_t gid = recorded_ids[(size_t)ri] - 1;
        global_to_recorded[(size_t)gid] = ri;
    }

    // Allocate subset strain arrays
    std::vector<float> fx_subset((size_t)n_steps * (size_t)n_recorded * 6, 0.0f);
    std::vector<float> fy_subset((size_t)n_steps * (size_t)n_recorded * 6, 0.0f);
    std::vector<float> fz_subset((size_t)n_steps * (size_t)n_recorded * 6, 0.0f);

    // Check if displacement data is available across all directions
    bool has_displacement = fx.has_displacement && fy.has_displacement && fz.has_displacement;

    // Allocate subset displacement arrays [n_steps, n_recorded, 3]
    std::vector<float> fx_disp_subset;
    std::vector<float> fy_disp_subset;
    std::vector<float> fz_disp_subset;
    if (has_displacement) {
        fx_disp_subset.resize((size_t)n_steps * (size_t)n_recorded * 3, 0.0f);
        fy_disp_subset.resize((size_t)n_steps * (size_t)n_recorded * 3, 0.0f);
        fz_disp_subset.resize((size_t)n_steps * (size_t)n_recorded * 3, 0.0f);
    }

    for (int64_t s = 0; s < n_steps; ++s) {
        for (int64_t gv = 0; gv < n_vertex; ++gv) {
            int64_t ri = global_to_recorded[(size_t)gv];
            if (ri < 0)
                continue;

            size_t src_base = ((size_t)s * (size_t)n_vertex + (size_t)gv) * 6;
            size_t dst_base = ((size_t)s * (size_t)n_recorded + (size_t)ri) * 6;

            float* src_fx = fx.strain.data() + src_base;
            float* src_fy = fy.strain.data() + src_base;
            float* src_fz = fz.strain.data() + src_base;
            float* dst_fx = fx_subset.data() + dst_base;
            float* dst_fy = fy_subset.data() + dst_base;
            float* dst_fz = fz_subset.data() + dst_base;

            for (int c = 0; c < 6; ++c) {
                dst_fx[c] = src_fx[c];
                dst_fy[c] = src_fy[c];
                dst_fz[c] = src_fz[c];
            }

            // Subset displacement if available
            if (has_displacement) {
                size_t dsrc_base = ((size_t)s * (size_t)n_vertex + (size_t)gv) * 3;
                size_t ddst_base = ((size_t)s * (size_t)n_recorded + (size_t)ri) * 3;
                float* dsrc_fx = fx.displacement.data() + dsrc_base;
                float* dsrc_fy = fy.displacement.data() + dsrc_base;
                float* dsrc_fz = fz.displacement.data() + dsrc_base;
                float* ddst_fx = fx_disp_subset.data() + ddst_base;
                float* ddst_fy = fy_disp_subset.data() + ddst_base;
                float* ddst_fz = fz_disp_subset.data() + ddst_base;
                for (int c = 0; c < 3; ++c) {
                    ddst_fx[c] = dsrc_fx[c];
                    ddst_fy[c] = dsrc_fy[c];
                    ddst_fz[c] = dsrc_fz[c];
                }
            }
        }
    }

    // ---- Assemble Green's tensor at recorded vertices ----
    fprintf(stderr, "[postprocess] Assembling Green's tensor...\n");
    // greens_subset: [n_steps, n_recorded, 6, 3]
    std::vector<float> greens_subset((size_t)n_steps * (size_t)n_recorded * 6 * 3, 0.0f);

    for (int64_t s = 0; s < n_steps; ++s) {
        for (int64_t ri = 0; ri < n_recorded; ++ri) {
            size_t base = ((size_t)s * (size_t)n_recorded + (size_t)ri) * 6;
            size_t g_base = ((size_t)s * (size_t)n_recorded + (size_t)ri) * 6 * 3;

            // fx → dir 0
            float* src_fx = fx_subset.data() + base;
            float* d0 = greens_subset.data() + g_base + 0 * 6;
            for (int c = 0; c < 6; ++c)
                d0[c] = src_fx[c];

            // fy → dir 1
            float* src_fy = fy_subset.data() + base;
            float* d1 = greens_subset.data() + g_base + 1 * 6;
            for (int c = 0; c < 6; ++c)
                d1[c] = src_fy[c];

            // fz → dir 2
            float* src_fz = fz_subset.data() + base;
            float* d2 = greens_subset.data() + g_base + 2 * 6;
            for (int c = 0; c < 6; ++c)
                d2[c] = src_fz[c];
        }
    }

    // ---- Assemble displacement tensor at recorded vertices ----
    // disp_subset: [n_steps, n_recorded, 3, 3]
    // Force-x direction → column 0, force-y → column 1, force-z → column 2
    std::vector<float> disp_subset;
    if (has_displacement) {
        disp_subset.resize((size_t)n_steps * (size_t)n_recorded * 3 * 3, 0.0f);
        for (int64_t s = 0; s < n_steps; ++s) {
            for (int64_t ri = 0; ri < n_recorded; ++ri) {
                size_t base = ((size_t)s * (size_t)n_recorded + (size_t)ri) * 3;
                size_t d_base = ((size_t)s * (size_t)n_recorded + (size_t)ri) * 3 * 3;

                // fx → dir 0
                float* src_fx = fx_disp_subset.data() + base;
                float* d0 = disp_subset.data() + d_base + 0 * 3;
                for (int c = 0; c < 3; ++c)
                    d0[c] = src_fx[c];

                // fy → dir 1
                float* src_fy = fy_disp_subset.data() + base;
                float* d1 = disp_subset.data() + d_base + 1 * 3;
                for (int c = 0; c < 3; ++c)
                    d1[c] = src_fy[c];

                // fz → dir 2
                float* src_fz = fz_disp_subset.data() + base;
                float* d2 = disp_subset.data() + d_base + 2 * 3;
                for (int c = 0; c < 3; ++c)
                    d2[c] = src_fz[c];
            }
        }
    }

    // Free large per-direction arrays to save memory
    fx.strain.clear();
    fx.strain.shrink_to_fit();
    fy.strain.clear();
    fy.strain.shrink_to_fit();
    fz.strain.clear();
    fz.strain.shrink_to_fit();
    fx_subset.clear();
    fx_subset.shrink_to_fit();
    fy_subset.clear();
    fy_subset.shrink_to_fit();
    fz_subset.clear();
    fz_subset.shrink_to_fit();
    fx.displacement.clear();
    fx.displacement.shrink_to_fit();
    fy.displacement.clear();
    fy.displacement.shrink_to_fit();
    fz.displacement.clear();
    fz.displacement.shrink_to_fit();
    if (has_displacement) {
        fx_disp_subset.clear();
        fx_disp_subset.shrink_to_fit();
        fy_disp_subset.clear();
        fy_disp_subset.shrink_to_fit();
        fz_disp_subset.clear();
        fz_disp_subset.shrink_to_fit();
    }

    // ---- Bin recorded vertices into tiles ----
    fprintf(stderr, "[postprocess] Binning vertices into tiles...\n");
    TileBins bins = bin_vertices(cfg, model, recorded_ids, n_recorded);

    // ---- Write tiles ----
    fprintf(stderr, "[postprocess] Writing Green's function tiles to %s...\n",
            args.output_dir.c_str());

    // Create output directory
    std::string mkdir_cmd = "mkdir -p " + args.output_dir;
    if (system(mkdir_cmd.c_str()) != 0) {
        fprintf(stderr, "WARNING: could not create output directory %s\n",
                args.output_dir.c_str());
    }

    double xmin = model.xmin, ymin = model.ymin;
    double xmax = model.xmax, ymax = model.ymax;
    double zmin = model.zmin, zmax = model.zmax;
    bool use_spatial = (cfg.green_tile_size_m > 0);

    int64_t n_tiles = (int64_t)bins.keys.size();

    // Pre-compute tile bound lambda for each tile
    auto compute_tile_bounds = [&](const TileKey& key, double& tx_min, double& tx_max,
                                   double& ty_min, double& ty_max) {
        if (use_spatial) {
            double gts = cfg.green_tile_size_m;
            tx_min = xmin + key.tx * gts;
            tx_max = xmin + (key.tx + 1) * gts;
            ty_min = ymin + key.ty * gts;
            ty_max = ymin + (key.ty + 1) * gts;
        } else {
            // Element-count tiling
            int64_t tile_x_cum = 0;
            int64_t tile_y_cum = 0;
            for (int t = 0; t < key.tx; ++t)
                tile_x_cum += cfg.tilex_elements[(size_t)t];
            for (int t = 0; t < key.ty; ++t)
                tile_y_cum += cfg.tiley_elements[(size_t)t];

            double dx = (xmax - xmin) / cfg.nx_elements;
            double dy = (ymax - ymin) / cfg.ny_elements;

            int64_t i_start = cfg.pml_xmin + tile_x_cum;
            int64_t i_end = cfg.pml_xmin + tile_x_cum + cfg.tilex_elements[(size_t)key.tx];
            int64_t j_start = cfg.pml_ymin + tile_y_cum;
            int64_t j_end = cfg.pml_ymin + tile_y_cum + cfg.tiley_elements[(size_t)key.ty];

            tx_min = xmin + i_start * dx;
            tx_max = xmin + i_end * dx;
            ty_min = ymin + j_start * dy;
            ty_max = ymin + j_end * dy;
        }
    };

    // Write tiles (could be OpenMP parallel, but HDF5 C library is not thread-safe
    // for file creation — serialize writes)
    for (int64_t ti = 0; ti < n_tiles; ++ti) {
        const TileKey& key = bins.keys[(size_t)ti];
        const auto& vert_indices = bins.bins.at(key);
        int64_t n_local = (int64_t)vert_indices.size();

        // Build tile vertex_ids (1-based)
        std::vector<int64_t> tile_vertex_ids((size_t)n_local);
        for (int64_t i = 0; i < n_local; ++i) {
            tile_vertex_ids[(size_t)i] = recorded_ids[(size_t)vert_indices[(size_t)i]];
        }

        // Build tile greens: [n_steps, n_local, 6, 3]
        std::vector<float> tile_greens((size_t)n_steps * (size_t)n_local * 6 * 3);
        for (int64_t s = 0; s < n_steps; ++s) {
            for (int64_t li = 0; li < n_local; ++li) {
                int64_t ri = vert_indices[(size_t)li];  // recorded index
                size_t src_base = ((size_t)s * (size_t)n_recorded + (size_t)ri) * 6 * 3;
                size_t dst_base = ((size_t)s * (size_t)n_local + (size_t)li) * 6 * 3;
                float* src = greens_subset.data() + src_base;
                float* dst = tile_greens.data() + dst_base;
                for (size_t k = 0; k < (size_t)(6 * 3); ++k)
                    dst[k] = src[k];
            }
        }

        // Build tile displacement: [n_steps, n_local, 3, 3] (nullable)
        std::vector<float> tile_displacement;
        if (has_displacement) {
            tile_displacement.resize((size_t)n_steps * (size_t)n_local * 3 * 3);
            for (int64_t s = 0; s < n_steps; ++s) {
                for (int64_t li = 0; li < n_local; ++li) {
                    int64_t ri = vert_indices[(size_t)li];  // recorded index
                    size_t src_base = ((size_t)s * (size_t)n_recorded + (size_t)ri) * 3 * 3;
                    size_t dst_base = ((size_t)s * (size_t)n_local + (size_t)li) * 3 * 3;
                    float* src = disp_subset.data() + src_base;
                    float* dst = tile_displacement.data() + dst_base;
                    for (size_t k = 0; k < (size_t)(3 * 3); ++k)
                        dst[k] = src[k];
                }
            }
        }

        // Build tile vertex coords [n_local, 3] from model
        std::vector<double> tile_vertex_coords((size_t)n_local * 3);
        for (int64_t i = 0; i < n_local; ++i) {
            int64_t gid = recorded_ids[(size_t)vert_indices[(size_t)i]] - 1;
            tile_vertex_coords[(size_t)i * 3 + 0] = model.vertex_coords[(size_t)gid * 3 + 0];
            tile_vertex_coords[(size_t)i * 3 + 1] = model.vertex_coords[(size_t)gid * 3 + 1];
            tile_vertex_coords[(size_t)i * 3 + 2] = model.vertex_coords[(size_t)gid * 3 + 2];
        }

        // Source position
        double source_xyz_m[3] = {cfg.source_x_m, cfg.source_y_m, cfg.source_z_m};

        // Compute tile bounds
        double tx_min, tx_max, ty_min, ty_max;
        compute_tile_bounds(key, tx_min, tx_max, ty_min, ty_max);

        // Build filename
        char fname[256];
        std::snprintf(fname, sizeof(fname), "%s/tile_x%03d_y%03d.h5", args.output_dir.c_str(),
                      key.tx, key.ty);

        write_tile(fname, key.tx, key.ty, tx_min, tx_max, ty_min, ty_max, zmin, zmax,
                   cfg.record_depth_max_m, cfg.record_depth_actual_m, tile_vertex_ids, time_arr,
                   cfg.solver_dt, tile_greens, source_xyz_m, tile_vertex_coords,
                   has_displacement ? tile_displacement.data() : nullptr);
    }

    // ---- Print machine-parseable stats ----
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double elapsed = (ts.tv_sec + ts.tv_nsec * 1e-9) - start;

        printf("STAT_NSTEPS=%lld\n", (long long)n_steps);
        printf("STAT_NVERTEX=%lld\n", (long long)n_vertex);
        printf("STAT_NRECORDED=%lld\n", (long long)n_recorded);
        printf("STAT_NTILES=%lld\n", (long long)n_tiles);
        printf("STAT_ELAPSED_S=%.1f\n", elapsed);
        fflush(stdout);

        fprintf(stderr, "[postprocess] Done in %.1fs — %lld tile(s), %lld recorded vertex(ices)\n",
                elapsed, (long long)n_tiles, (long long)n_recorded);
    }

    return 0;
}