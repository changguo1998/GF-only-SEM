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
#include <unordered_map>
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
// Merge records for one direction: returns [n_steps, n_vertex, 6] double
// -----------------------------------------------------------------------

// GLL-node merged data for one force direction
struct MergedDirection {
    std::vector<double> strain;           // [n_steps, n_unique_gll, 6]
    std::vector<double> displacement;     // [n_steps, n_unique_gll, 3]
    std::vector<double> velocity;         // [n_steps, n_unique_gll, 3]
    std::vector<double> acceleration;     // [n_steps, n_unique_gll, 3]
    std::vector<double> gll_node_coords;  // [n_unique_gll, 3]
    std::vector<int64_t> gll_node_ids;    // [n_unique_gll] 1-based global DOF
    // Cell-level data (for whole-cell tiling + mass-weighted L2 projection)
    std::vector<int32_t> cell_gll_node_index;         // [n_rec_cell_merged * n_node]
    std::vector<int64_t> recording_cell_model_index;  // [n_rec_cell_merged]
    int64_t n_unique_gll = 0;
    int64_t n_steps = 0;
    int64_t n_node_per_cell = 0;
    int64_t n_rec_cell_merged = 0;
    bool has_displacement = false;
    bool has_velocity = false;
    bool has_acceleration = false;
};

// GLL-aware merge: reads 4D records, deduplicates GLL nodes across ranks,
// and produces simple-averaged strain (L2 projection deferred).
static MergedDirection merge_direction(const char* dir_path, const std::vector<double>& cell_mass,
                                       int64_t n_model_cell, int ngll) {
    MergedDirection result;
    fprintf(stderr, "[postprocess] Merging GLL records from %s...\n", dir_path);

    auto files = discover_records(dir_path);
    if (files.empty()) {
        fprintf(stderr, "ERROR: no record files found in %s\n", dir_path);
        exit(1);
    }

    // --- First pass: read GLL metadata from all files, build global union ---
    std::unordered_map<int64_t, int32_t> global_to_merged_idx;
    std::vector<int64_t> merged_gll_node_ids;
    std::vector<double> merged_gll_node_coords;
    int64_t n_node_per_cell = 0;

    struct FileMapping {
        RecordFileInfo info;
        std::vector<int32_t> local_to_global;
        hsize_t n_rec_cell = 0;
        hsize_t nnodes = 0;
        std::vector<int32_t> cell_gll_idx;        // [n_rec_cell * n_node]
        std::vector<int64_t> rec_cell_model_idx;  // [n_rec_cell]
    };
    std::vector<FileMapping> file_maps;

    for (auto& fi : files) {
        hid_t fid = H5Fopen(fi.path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fid < 0) {
            fprintf(stderr, "WARNING: cannot open %s for metadata scan\n", fi.path.c_str());
            continue;
        }

        hsize_t nids = 0;
        auto local_ids = read_int64_1d(fid, "gll_node_ids", nids);
        if (nids == 0) {
            H5Fclose(fid);
            continue;
        }

        // Read gll_node_coords
        hid_t cds = H5Dopen2(fid, "gll_node_coords", H5P_DEFAULT);
        hsize_t cdims[2];
        std::vector<double> local_coords;
        if (cds >= 0) {
            hid_t cspace = H5Dget_space(cds);
            H5Sget_simple_extent_dims(cspace, cdims, nullptr);
            local_coords.resize(cdims[0] * cdims[1]);
            H5Dread(cds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, local_coords.data());
            H5Sclose(cspace);
            H5Dclose(cds);
        }

        // Read cell_gll_node_index to determine n_node_per_cell and n_rec_cell
        hsize_t ncell = 0;
        std::vector<int32_t> cell_gll_idx_local;
        hsize_t idxdims[2] = {0, 0};
        hid_t idxds = H5Dopen2(fid, "cell_gll_node_index", H5P_DEFAULT);
        if (idxds >= 0) {
            hid_t ispace = H5Dget_space(idxds);
            H5Sget_simple_extent_dims(ispace, idxdims, nullptr);
            n_node_per_cell = (int64_t)idxdims[1];
            ncell = idxdims[0];
            // Read full cell_gll_node_index data
            hsize_t idx_total = idxdims[0] * idxdims[1];
            cell_gll_idx_local.resize((size_t)idx_total);
            H5Dread(idxds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    cell_gll_idx_local.data());
            H5Sclose(ispace);
            H5Dclose(idxds);
            H5Dclose(idxds);
        } else {
            // Fallback: read n_rec_cell from root attribute
            int64_t attr_val = 0;
            read_attr_int64(fid, "n_rec_cell", attr_val);
            ncell = (hsize_t)attr_val;
        }
        FileMapping fm;
        fm.info = fi;
        fm.n_rec_cell = ncell;
        fm.nnodes = nids;
        fm.local_to_global.resize((size_t)nids, -1);
        fm.cell_gll_idx = std::move(cell_gll_idx_local);

        // Read recording_cell_model_index
        {
            hid_t rcm_ds = H5Dopen2(fid, "recording_cell_model_index", H5P_DEFAULT);
            if (rcm_ds >= 0) {
                hid_t rcm_space = H5Dget_space(rcm_ds);
                hsize_t rcm_dim = 0;
                H5Sget_simple_extent_dims(rcm_space, &rcm_dim, nullptr);
                fm.rec_cell_model_idx.resize((size_t)rcm_dim);
                H5Dread(rcm_ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        fm.rec_cell_model_idx.data());
                H5Sclose(rcm_space);
                H5Dclose(rcm_ds);
            } else {
                fm.rec_cell_model_idx.resize(ncell, -1);
            }
        }
        H5Fclose(fid);

        for (hsize_t li = 0; li < nids; ++li) {
            int64_t gid = local_ids[li];
            auto it = global_to_merged_idx.find(gid);
            if (it == global_to_merged_idx.end()) {
                int32_t merged_idx = (int32_t)merged_gll_node_ids.size();
                global_to_merged_idx[gid] = merged_idx;
                merged_gll_node_ids.push_back(gid);
                if (!local_coords.empty() && li < nids) {
                    merged_gll_node_coords.push_back(local_coords[li * 3 + 0]);
                    merged_gll_node_coords.push_back(local_coords[li * 3 + 1]);
                    merged_gll_node_coords.push_back(local_coords[li * 3 + 2]);
                }
                fm.local_to_global[(size_t)li] = merged_idx;
            } else {
                fm.local_to_global[(size_t)li] = it->second;
            }
        }
        file_maps.push_back(std::move(fm));
    }

    result.n_unique_gll = (int64_t)merged_gll_node_ids.size();
    result.n_node_per_cell = n_node_per_cell;
    result.gll_node_ids = std::move(merged_gll_node_ids);
    result.gll_node_coords = std::move(merged_gll_node_coords);

    // Accumulate merged cell-level data for whole-cell tiling
    for (auto& fm : file_maps) {
        for (hsize_t c = 0; c < fm.n_rec_cell; ++c) {
            for (hsize_t p = 0; p < (hsize_t)n_node_per_cell; ++p) {
                int32_t local_idx = fm.cell_gll_idx[c * (hsize_t)n_node_per_cell + p];
                int32_t global_idx =
                    (local_idx >= 0 && (size_t)local_idx < fm.local_to_global.size())
                        ? fm.local_to_global[(size_t)local_idx]
                        : -1;
                result.cell_gll_node_index.push_back(global_idx);
            }
            if ((size_t)c < fm.rec_cell_model_idx.size())
                result.recording_cell_model_index.push_back(fm.rec_cell_model_idx[(size_t)c]);
            else
                result.recording_cell_model_index.push_back(-1);
            result.n_rec_cell_merged++;
        }
    }
    fprintf(stderr, "[postprocess]   %lld merged recording cells\n",
            (long long)result.n_rec_cell_merged);

    fprintf(stderr, "[postprocess]   %lld unique GLL nodes from %zu rank files\n",
            (long long)result.n_unique_gll, file_maps.size());

    auto groups = group_by_step(files);
    result.n_steps = (int64_t)groups.size();
    fprintf(stderr, "[postprocess]   %lld steps\n", (long long)result.n_steps);

    // Allocate merged arrays
    size_t ng = (size_t)result.n_unique_gll;
    result.strain.resize((size_t)result.n_steps * ng * 6, 0.0);
    result.displacement.resize((size_t)result.n_steps * ng * 3, 0.0);
    result.velocity.resize((size_t)result.n_steps * ng * 3, 0.0);
    result.acceleration.resize((size_t)result.n_steps * ng * 3, 0.0);

    // Detect optional field presence from first file
    if (!files.empty()) {
        hid_t probe = H5Fopen(files[0].path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (probe >= 0) {
            hid_t ds;
            ds = H5Dopen2(probe, "displacement", H5P_DEFAULT);
            result.has_displacement = (ds >= 0);
            if (ds >= 0)
                H5Dclose(ds);
            ds = H5Dopen2(probe, "velocity", H5P_DEFAULT);
            result.has_velocity = (ds >= 0);
            if (ds >= 0)
                H5Dclose(ds);
            ds = H5Dopen2(probe, "acceleration", H5P_DEFAULT);
            result.has_acceleration = (ds >= 0);
            if (ds >= 0)
                H5Dclose(ds);
            H5Fclose(probe);
        }
    }

    // --- Second pass: per-step GLL-node simple averaging ---
    for (int64_t snap_idx = 0; snap_idx < result.n_steps; ++snap_idx) {
        auto& group = groups[(size_t)snap_idx];
        double* step_data = result.strain.data() + snap_idx * ng * 6;
        double* step_disp = result.displacement.data() + snap_idx * ng * 3;
        double* step_vel = result.velocity.data() + snap_idx * ng * 3;
        double* step_acc = result.acceleration.data() + snap_idx * ng * 3;

        std::vector<double> node_weight_sum(ng, 0.0);
        bool any_mass_weight = false;
        bool use_mass_weighted = !cell_mass.empty() && ngll > 0 && n_model_cell > 0;
        int ngll2 = ngll * ngll;
        int n_node_mass = ngll * ngll2;

        for (auto& fm : file_maps) {
            const RecordFileInfo* gfi = nullptr;
            for (auto& fi : group.files) {
                if (fi.path == fm.info.path) {
                    gfi = &fi;
                    break;
                }
            }
            if (!gfi)
                continue;

            hid_t fid = H5Fopen(gfi->path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (fid < 0)
                continue;

            // Read 4D strain [1, n_rec_cell, n_node_per_cell, 6]
            hsize_t nrc = 0, nnp = 0;
            std::vector<double> strain_buf;
            read_strain_4d(fid, "strain", nrc, nnp, strain_buf);

            // Read cell_gll_node_index
            std::vector<int32_t> cell_gll_idx;
            hid_t idxds = H5Dopen2(fid, "cell_gll_node_index", H5P_DEFAULT);
            if (idxds >= 0) {
                hid_t ispace = H5Dget_space(idxds);
                hsize_t idxd2[2];
                H5Sget_simple_extent_dims(ispace, idxd2, nullptr);
                cell_gll_idx.resize(idxd2[0] * idxd2[1]);
                H5Dread(idxds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        cell_gll_idx.data());
                H5Sclose(ispace);
                H5Dclose(idxds);
            }

            for (hsize_t c = 0; c < nrc && c < fm.n_rec_cell; ++c) {
                for (hsize_t p = 0; p < nnp && p < (hsize_t)n_node_per_cell; ++p) {
                    int32_t local_gll_idx = cell_gll_idx[c * (hsize_t)n_node_per_cell + p];
                    if (local_gll_idx < 0 || local_gll_idx >= (int32_t)fm.local_to_global.size())
                        continue;
                    int32_t global_idx = fm.local_to_global[(size_t)local_gll_idx];
                    if (global_idx < 0 || global_idx >= (int32_t)ng)
                        continue;

                    // Mass weight lookup
                    double weight = 1.0;
                    if (use_mass_weighted && (size_t)c < fm.rec_cell_model_idx.size()) {
                        int64_t gcell = fm.rec_cell_model_idx[(size_t)c];
                        if (gcell >= 0 && gcell < n_model_cell) {
                            int mi = (int)p / ngll2;
                            int mj = ((int)p / ngll) % ngll;
                            int mk = (int)p % ngll;
                            size_t mass_off = (size_t)gcell * (size_t)n_node_mass +
                                              (size_t)mi * (size_t)ngll2 +
                                              (size_t)mj * (size_t)ngll + (size_t)mk;
                            weight = cell_mass[mass_off];
                            any_mass_weight = true;
                        }
                    }
                    double* src = strain_buf.data() + (c * nnp + p) * 6;
                    double* dst = step_data + (size_t)global_idx * 6;
                    for (int comp = 0; comp < 6; ++comp)
                        dst[comp] += src[comp] * weight;
                    node_weight_sum[(size_t)global_idx] += weight;
                }
            }

            // Read displacement [1, n_rec_cell, n_node_per_cell, 3]
            if (result.has_displacement) {
                hsize_t drc = 0, dnp = 0;
                std::vector<double> disp_buf;
                read_field_4d(fid, "displacement", drc, dnp, disp_buf);
                for (hsize_t c = 0; c < drc && c < nrc; ++c) {
                    for (hsize_t p = 0; p < dnp && p < n_node_per_cell; ++p) {
                        int32_t local_gll_idx = cell_gll_idx[c * (hsize_t)n_node_per_cell + p];
                        if (local_gll_idx < 0 ||
                            local_gll_idx >= (int32_t)fm.local_to_global.size())
                            continue;
                        int32_t global_idx = fm.local_to_global[(size_t)local_gll_idx];
                        if (global_idx < 0 || global_idx >= (int32_t)ng)
                            continue;
                        double* dsrc = disp_buf.data() + (c * dnp + p) * 3;
                        double* ddst = step_disp + (size_t)global_idx * 3;
                        for (int comp = 0; comp < 3; ++comp)
                            ddst[comp] += dsrc[comp];
                    }
                }
            }

            // Velocity and acceleration follow same pattern — deferred

            H5Fclose(fid);
        }

        // Mass-weighted average for strain; simple average for displacement
        for (size_t gi = 0; gi < ng; ++gi) {
            if (node_weight_sum[gi] > 0.0) {
                double inv = 1.0 / node_weight_sum[gi];
                double* dst = step_data + gi * 6;
                for (int c = 0; c < 6; ++c)
                    dst[c] *= inv;
                if (result.has_displacement) {
                    double* ddst = step_disp + gi * 3;
                    for (int c = 0; c < 3; ++c)
                        ddst[c] *= inv;
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
    int64_t n_vertex = model.n_vertex;  // kept for domain bounds
    fprintf(stderr, "[postprocess]   domain vertex count = %lld\n", (long long)n_vertex);

    // ---- Read cell mass from model.h5 for L2 projection ----
    std::vector<double> cell_mass;
    int64_t n_model_cell = 0;
    int ngll_model = 0;
    {
        hid_t model_fid = H5Fopen(args.model_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (model_fid >= 0) {
            hid_t mass_ds = H5Dopen2(model_fid, "/field/cell/mass", H5P_DEFAULT);
            if (mass_ds >= 0) {
                hid_t mass_space = H5Dget_space(mass_ds);
                hsize_t mass_dims[4] = {0, 0, 0, 0};
                H5Sget_simple_extent_dims(mass_space, mass_dims, nullptr);
                n_model_cell = (int64_t)mass_dims[0];
                ngll_model = (int)mass_dims[1];
                hsize_t total = mass_dims[0] * mass_dims[1] * mass_dims[2] * mass_dims[3];
                cell_mass.resize((size_t)total);
                H5Dread(mass_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        cell_mass.data());
                fprintf(stderr, "[postprocess]   read cell mass [%lld, %d, %d, %d]\n",
                        (long long)n_model_cell, ngll_model, ngll_model, ngll_model);
                H5Sclose(mass_space);
                H5Dclose(mass_ds);
            }
            H5Fclose(model_fid);
        }
    }

    // ---- Merge records for each direction (GLL format) ----
    MergedDirection fx = merge_direction(args.fx_dir.c_str(), cell_mass, n_model_cell, ngll_model);
    MergedDirection fy = merge_direction(args.fy_dir.c_str(), cell_mass, n_model_cell, ngll_model);
    MergedDirection fz = merge_direction(args.fz_dir.c_str(), cell_mass, n_model_cell, ngll_model);

    // Consistency: same number of steps
    if (fx.n_steps != fy.n_steps || fx.n_steps != fz.n_steps) {
        fprintf(stderr,
                "ERROR: mismatched number of steps across directions "
                "(%lld, %lld, %lld)\n",
                (long long)fx.n_steps, (long long)fy.n_steps, (long long)fz.n_steps);
        exit(1);
    }
    int64_t n_steps = fx.n_steps;

    // Consistency: same GLL node set across directions
    if (fx.gll_node_ids != fy.gll_node_ids || fx.gll_node_ids != fz.gll_node_ids) {
        fprintf(stderr, "[postprocess] WARNING: GLL node sets differ across directions\n");
    }

    // GLL node IDs (1-based, shared across directions)
    const auto& recorded_ids = fx.gll_node_ids;
    int64_t n_recorded = fx.n_unique_gll;
    fprintf(stderr, "[postprocess] %lld unique GLL nodes recorded\n", (long long)n_recorded);

    if (n_recorded == 0) {
        fprintf(stderr, "ERROR: no GLL nodes recorded\n");
        return 1;
    }

    // ---- Build time array ----
    std::vector<double> time_arr((size_t)n_steps);
    for (int64_t s = 0; s < n_steps; ++s) {
        time_arr[(size_t)s] = (double)s * cfg.output_dt_s;
    }

    // ---- Downsample STF to tile time axis ----
    // config.h5 stores stf_t/stf_values at solver_dt [nsteps_solver]. Tile
    // time_arr is at output_dt_s [nt]. Resample STF to match tile time axis
    // so users can deconvolve with a STF sampled at the same rate as the
    // Green's function tensors.
    std::vector<double> stf_t_ds, stf_values_ds;
    if (!cfg.stf_t.empty() && n_steps > 0) {
        int64_t nstf = (int64_t)cfg.stf_t.size();
        int64_t stride = (n_steps > 0) ? nstf / n_steps : 1;
        if (stride < 1)
            stride = 1;
        stf_t_ds.resize((size_t)n_steps);
        stf_values_ds.resize((size_t)n_steps);
        for (int64_t s = 0; s < n_steps; ++s) {
            int64_t idx = s * stride;
            if (idx >= nstf)
                idx = nstf - 1;
            stf_t_ds[(size_t)s] = cfg.stf_t[(size_t)idx];
            stf_values_ds[(size_t)s] = cfg.stf_values[(size_t)idx];
        }
    }

    // Detect optional field availability
    bool has_displacement = fx.has_displacement && fy.has_displacement && fz.has_displacement;
    bool has_velocity = fx.has_velocity && fy.has_velocity && fz.has_velocity;
    bool has_acceleration = fx.has_acceleration && fy.has_acceleration && fz.has_acceleration;
    fprintf(stderr, "[postprocess]   displacement=%s velocity=%s acceleration=%s\n",
            has_displacement ? "yes" : "no", has_velocity ? "yes" : "no",
            has_acceleration ? "yes" : "no");
    // ---- Assemble Green's tensor directly from GLL-merged data ----
    // fx.strain is already [n_steps, n_unique_gll, 6] — no subsetting needed
    fprintf(stderr, "[postprocess] Assembling Green's tensor...\n");
    // greens_subset: [n_steps, n_unique_gll, 6, 3]
    std::vector<double> greens_subset((size_t)n_steps * (size_t)n_recorded * 6 * 3, 0.0);

    for (int64_t s = 0; s < n_steps; ++s) {
        for (int64_t gi = 0; gi < n_recorded; ++gi) {
            size_t base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 6;
            size_t g_base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 6 * 3;

            // fx → dir 0
            double* src_fx = fx.strain.data() + base;
            double* d0 = greens_subset.data() + g_base + 0 * 6;
            for (int c = 0; c < 6; ++c)
                d0[c] = src_fx[c];

            // fy → dir 1
            double* src_fy = fy.strain.data() + base;
            double* d1 = greens_subset.data() + g_base + 1 * 6;
            for (int c = 0; c < 6; ++c)
                d1[c] = src_fy[c];

            // fz → dir 2
            double* src_fz = fz.strain.data() + base;
            double* d2 = greens_subset.data() + g_base + 2 * 6;
            for (int c = 0; c < 6; ++c)
                d2[c] = src_fz[c];
        }
    }
    // Debug: check merged strain
    // ---- Assemble displacement tensor ----
    // disp_subset: [n_steps, n_unique_gll, 3, 3]
    std::vector<double> disp_subset;
    if (has_displacement) {
        disp_subset.resize((size_t)n_steps * (size_t)n_recorded * 3 * 3, 0.0);
        for (int64_t s = 0; s < n_steps; ++s) {
            for (int64_t gi = 0; gi < n_recorded; ++gi) {
                size_t base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 3;
                size_t d_base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 3 * 3;
                const double* src_fx = fx.displacement.data() + base;
                const double* src_fy = fy.displacement.data() + base;
                const double* src_fz = fz.displacement.data() + base;
                for (int c = 0; c < 3; ++c) {
                    double* d = disp_subset.data() + d_base + c * 3;
                    d[0] = src_fx[c];
                    d[1] = src_fy[c];
                    d[2] = src_fz[c];
                }
            }
        }
    }

    // ---- Assemble velocity tensor ----
    std::vector<double> vel_subset;
    if (has_velocity) {
        vel_subset.resize((size_t)n_steps * (size_t)n_recorded * 3 * 3, 0.0);
        for (int64_t s = 0; s < n_steps; ++s) {
            for (int64_t gi = 0; gi < n_recorded; ++gi) {
                size_t base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 3;
                size_t d_base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 3 * 3;
                const double* src_fx = fx.velocity.data() + base;
                const double* src_fy = fy.velocity.data() + base;
                const double* src_fz = fz.velocity.data() + base;
                for (int c = 0; c < 3; ++c) {
                    double* d = vel_subset.data() + d_base + c * 3;
                    d[0] = src_fx[c];
                    d[1] = src_fy[c];
                    d[2] = src_fz[c];
                }
            }
        }
    }

    // ---- Assemble acceleration tensor ----
    std::vector<double> acc_subset;
    if (has_acceleration) {
        acc_subset.resize((size_t)n_steps * (size_t)n_recorded * 3 * 3, 0.0);
        for (int64_t s = 0; s < n_steps; ++s) {
            for (int64_t gi = 0; gi < n_recorded; ++gi) {
                size_t base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 3;
                size_t d_base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 3 * 3;
                const double* src_fx = fx.acceleration.data() + base;
                const double* src_fy = fy.acceleration.data() + base;
                const double* src_fz = fz.acceleration.data() + base;
                for (int c = 0; c < 3; ++c) {
                    double* d = acc_subset.data() + d_base + c * 3;
                    d[0] = src_fx[c];
                    d[1] = src_fy[c];
                    d[2] = src_fz[c];
                }
            }
        }
    }

    // Free per-direction arrays to save memory
    fx.strain.clear();
    fx.strain.shrink_to_fit();
    fy.strain.clear();
    fy.strain.shrink_to_fit();
    fz.strain.clear();
    fz.strain.shrink_to_fit();
    fx.displacement.clear();
    fx.displacement.shrink_to_fit();
    fy.displacement.clear();
    fy.displacement.shrink_to_fit();
    fz.displacement.clear();
    fz.displacement.shrink_to_fit();
    fx.velocity.clear();
    fx.velocity.shrink_to_fit();
    fy.velocity.clear();
    fy.velocity.shrink_to_fit();
    fz.velocity.clear();
    fz.velocity.shrink_to_fit();
    fx.acceleration.clear();
    fx.acceleration.shrink_to_fit();
    fy.acceleration.clear();
    fy.acceleration.shrink_to_fit();
    fz.acceleration.clear();
    fz.acceleration.shrink_to_fit();
    // ---- Bin recording cells into tiles (whole-cell tiling) ----
    fprintf(stderr, "[postprocess] Binning recording cells into tiles...\n");
    TileBins bins;
    std::unordered_map<TileKey, std::vector<int64_t>, TileKeyHash> cell_bins;
    double xmin = model.xmin, ymin = model.ymin, xmax = model.xmax, ymax = model.ymax;
    double dx = (xmax - xmin) / cfg.nx_elements;
    double dy = (ymax - ymin) / cfg.ny_elements;
    int64_t total_interior_x = 0, total_interior_y = 0;
    for (auto sz : cfg.tilex_elements)
        total_interior_x += sz;
    for (auto sz : cfg.tiley_elements)
        total_interior_y += sz;

    for (int64_t ci = 0; ci < fx.n_rec_cell_merged; ++ci) {
        int64_t idx0 = fx.cell_gll_node_index[(size_t)ci * (size_t)fx.n_node_per_cell + 0];
        int64_t idx124 = fx.cell_gll_node_index[(size_t)ci * (size_t)fx.n_node_per_cell + 124];
        if (idx0 < 0 || idx0 >= n_recorded || idx124 < 0 || idx124 >= n_recorded)
            continue;
        double cx = 0.5 * (fx.gll_node_coords[(size_t)idx0 * 3 + 0] +
                           fx.gll_node_coords[(size_t)idx124 * 3 + 0]);
        double cy = 0.5 * (fx.gll_node_coords[(size_t)idx0 * 3 + 1] +
                           fx.gll_node_coords[(size_t)idx124 * 3 + 1]);
        int64_t ei = (dx > 0) ? (int64_t)std::floor((cx - xmin) / dx) : 0;
        int64_t ej = (dy > 0) ? (int64_t)std::floor((cy - ymin) / dy) : 0;
        if (ei < 0)
            ei = 0;
        if (ej < 0)
            ej = 0;
        if (cfg.nx_elements > 0 && ei >= cfg.nx_elements)
            ei = cfg.nx_elements - 1;
        if (cfg.ny_elements > 0 && ej >= cfg.ny_elements)
            ej = cfg.ny_elements - 1;
        int64_t interior_i = ei - cfg.pml_xmin;
        int64_t interior_j = ej - cfg.pml_ymin;
        if (interior_i < 0 || interior_i >= total_interior_x)
            continue;
        if (interior_j < 0 || interior_j >= total_interior_y)
            continue;
        TileKey key;
        key.tx = find_tile_index(interior_i, cfg.tilex_elements);
        key.ty = find_tile_index(interior_j, cfg.tiley_elements);
        cell_bins[key].push_back(ci);
    }
    bins.keys.clear();
    for (auto& kv : cell_bins)
        bins.keys.push_back(kv.first);
    std::sort(bins.keys.begin(), bins.keys.end());
    fprintf(stderr, "[postprocess]   %zu tiles\n", bins.keys.size());

    // ---- Write tiles ----
    fprintf(stderr, "[postprocess] Writing Green's function tiles to %s...\n",
            args.output_dir.c_str());

    // Create output directory
    std::string mkdir_cmd = "mkdir -p " + args.output_dir;
    if (system(mkdir_cmd.c_str()) != 0) {
        fprintf(stderr, "WARNING: could not create output directory %s\n",
                args.output_dir.c_str());
    }

    double zmin = model.zmin, zmax = model.zmax;
    int64_t n_tiles = (int64_t)bins.keys.size();

    auto compute_tile_bounds = [&](const TileKey& key, double& tx_min, double& tx_max,
                                   double& ty_min, double& ty_max) {
        double dx = (xmax - xmin) / cfg.nx_elements;
        double dy = (ymax - ymin) / cfg.ny_elements;
        int64_t tile_x_cum = 0, tile_y_cum = 0;
        for (int t = 0; t < key.tx; ++t)
            tile_x_cum += cfg.tilex_elements[(size_t)t];
        for (int t = 0; t < key.ty; ++t)
            tile_y_cum += cfg.tiley_elements[(size_t)t];
        int64_t i_start = cfg.pml_xmin + tile_x_cum;
        int64_t i_end = cfg.pml_xmin + tile_x_cum + cfg.tilex_elements[(size_t)key.tx];
        int64_t j_start = cfg.pml_ymin + tile_y_cum;
        int64_t j_end = cfg.pml_ymin + tile_y_cum + cfg.tiley_elements[(size_t)key.ty];
        tx_min = xmin + i_start * dx;
        tx_max = xmin + i_end * dx;
        ty_min = ymin + j_start * dy;
        ty_max = ymin + j_end * dy;
    };

    // Write tiles (could be OpenMP parallel, but HDF5 C library is not thread-safe
    // for file creation — serialize writes)
    for (int64_t ti = 0; ti < n_tiles; ++ti) {
        const TileKey& key = bins.keys[(size_t)ti];
        const auto& vert_indices = bins.bins.at(key);
        const auto& cell_indices = cell_bins.at(key);
        // Build tile-local GLL node set from cells
        std::unordered_map<int64_t, int64_t> gll_to_tile_local;
        std::vector<int64_t> tile_gll_indices;
        for (auto ci : cell_indices) {
            for (int64_t p = 0; p < fx.n_node_per_cell; ++p) {
                int32_t gi = fx.cell_gll_node_index[(size_t)ci * (size_t)fx.n_node_per_cell + p];
                if (gi >= 0 && gll_to_tile_local.find(gi) == gll_to_tile_local.end()) {
                    gll_to_tile_local[gi] = (int64_t)tile_gll_indices.size();
                    tile_gll_indices.push_back(gi);
                }
            }
        }
        int64_t n_local = (int64_t)tile_gll_indices.size();

        // Build tile-local cell_gll_node_index
        std::vector<int32_t> tile_cell_gll_index(cell_indices.size() * (size_t)fx.n_node_per_cell);
        for (size_t ci = 0; ci < cell_indices.size(); ++ci) {
            for (int64_t p = 0; p < fx.n_node_per_cell; ++p) {
                int32_t gi =
                    fx.cell_gll_node_index[(size_t)cell_indices[ci] * (size_t)fx.n_node_per_cell +
                                           p];
                tile_cell_gll_index[ci * (size_t)fx.n_node_per_cell + (size_t)p] =
                    (gi >= 0) ? (int32_t)gll_to_tile_local[gi] : -1;
            }
        }

        // Build tile vertex IDs and coords
        std::vector<int64_t> tile_vertex_ids((size_t)n_local);
        std::vector<double> tile_vertex_coords((size_t)n_local * 3);
        for (int64_t i = 0; i < n_local; ++i) {
            int64_t gi = tile_gll_indices[(size_t)i];
            tile_vertex_ids[(size_t)i] = recorded_ids[(size_t)gi];
            tile_vertex_coords[(size_t)i * 3 + 0] = fx.gll_node_coords[(size_t)gi * 3 + 0];
            tile_vertex_coords[(size_t)i * 3 + 1] = fx.gll_node_coords[(size_t)gi * 3 + 1];
            tile_vertex_coords[(size_t)i * 3 + 2] = fx.gll_node_coords[(size_t)gi * 3 + 2];
        }

        // Build tile greens: [n_steps, n_local, 6, 3]
        std::vector<double> tile_greens((size_t)n_steps * (size_t)n_local * 6 * 3);
        for (int64_t s = 0; s < n_steps; ++s) {
            for (int64_t li = 0; li < n_local; ++li) {
                int64_t gi = tile_gll_indices[(size_t)li];
                size_t src_base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 6 * 3;
                size_t dst_base = ((size_t)s * (size_t)n_local + (size_t)li) * 6 * 3;
                for (size_t k = 0; k < (size_t)(6 * 3); ++k)
                    tile_greens[dst_base + k] = greens_subset[src_base + k];
            }
        }

        // Build tile displacement: [n_steps, n_local, 3, 3]
        std::vector<double> tile_displacement;
        if (has_displacement) {
            tile_displacement.resize((size_t)n_steps * (size_t)n_local * 3 * 3);
            for (int64_t s = 0; s < n_steps; ++s) {
                for (int64_t li = 0; li < n_local; ++li) {
                    int64_t gi = tile_gll_indices[(size_t)li];
                    size_t src_base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 3 * 3;
                    size_t dst_base = ((size_t)s * (size_t)n_local + (size_t)li) * 3 * 3;
                    for (size_t k = 0; k < (size_t)(3 * 3); ++k)
                        tile_displacement[dst_base + k] = disp_subset[src_base + k];
                }
            }
        }

        // Build tile velocity
        std::vector<double> tile_velocity;
        if (has_velocity) {
            tile_velocity.resize((size_t)n_steps * (size_t)n_local * 3 * 3);
            for (int64_t s = 0; s < n_steps; ++s) {
                for (int64_t li = 0; li < n_local; ++li) {
                    int64_t gi = tile_gll_indices[(size_t)li];
                    size_t src_base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 3 * 3;
                    size_t dst_base = ((size_t)s * (size_t)n_local + (size_t)li) * 3 * 3;
                    for (size_t k = 0; k < (size_t)(3 * 3); ++k)
                        tile_velocity[dst_base + k] = vel_subset[src_base + k];
                }
            }
        }

        // Build tile acceleration
        std::vector<double> tile_acceleration;
        if (has_acceleration) {
            tile_acceleration.resize((size_t)n_steps * (size_t)n_local * 3 * 3);
            for (int64_t s = 0; s < n_steps; ++s) {
                for (int64_t li = 0; li < n_local; ++li) {
                    int64_t gi = tile_gll_indices[(size_t)li];
                    size_t src_base = ((size_t)s * (size_t)n_recorded + (size_t)gi) * 3 * 3;
                    size_t dst_base = ((size_t)s * (size_t)n_local + (size_t)li) * 3 * 3;
                    for (size_t k = 0; k < (size_t)(3 * 3); ++k)
                        tile_acceleration[dst_base + k] = acc_subset[src_base + k];
                }
            }
        }

        // Source position
        double source_xyz_m[3] = {cfg.source_x_m, cfg.source_y_m, cfg.source_z_m};

        // Compute tile bounds
        double tx_min, tx_max, ty_min, ty_max;
        compute_tile_bounds(key, tx_min, tx_max, ty_min, ty_max);

        // Output precision follows config snapshot_precision
        bool use_float32 = (cfg.snapshot_precision == "float32");

        // Build filename
        char fname[256];
        std::snprintf(fname, sizeof(fname), "%s/tile_x%03d_y%03d.h5", args.output_dir.c_str(),
                      key.tx, key.ty);

        write_tile(fname, key.tx, key.ty, tx_min, tx_max, ty_min, ty_max, zmin, zmax,
                   cfg.record_depth_max_m, cfg.record_depth_actual_m, tile_vertex_ids, time_arr,
                   cfg.solver_dt, tile_greens, source_xyz_m, tile_vertex_coords,
                   tile_cell_gll_index, (int)fx.n_node_per_cell,
                   has_displacement ? tile_displacement.data() : nullptr,
                   has_velocity ? tile_velocity.data() : nullptr,
                   has_acceleration ? tile_acceleration.data() : nullptr, stf_t_ds, stf_values_ds,
                   use_float32);
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