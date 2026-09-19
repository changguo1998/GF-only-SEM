/* postprocess/cpp/main_mpi.cpp - MPI tile-parallel Green's function postprocessor
 *
 * CLI:
 *   gf_postprocess_mpi <model.h5> <config.h5> \
 *       --fx <dir> --fy <dir> --fz <dir> -o <output_dir>
 *
 * Tile-parallel variant of main.cpp: tiles are round-robin distributed across
 * ranks, each tile written exactly once regardless of n_ranks (a rank handles
 * tiles tile_pos = rank, rank+nranks, rank+2*nranks, ...).
 *
 * Memory design (prevents OOM on multi-rank runs):
 *   Phase 1 - merge_metadata(): every rank builds GLL-node union + cell
 *             metadata only (~2 MB). No per-step field arrays.
 *   Phase 2 - binning: ranks without any assigned tile exit before any field
 *             allocation (mpi_rank >= n_tiles).
 *   Phase 3 - extract_tile_fields(): each rank reads records but accumulates
 *             field data only for its tiles' nodes (~1 GB/rank per tile, freed
 *             between tiles). All sharing cells are still iterated so averaging
 *             stays correct.
 */

#include <mpi.h>
#include <unistd.h>

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

#include "common.hpp"
#include "reader.hpp"
#include "writer.hpp"

// -----------------------------------------------------------------------
// CLI argument parsing
// -----------------------------------------------------------------------

/// Print command-line usage and exit.
static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s <model.h5> <config.h5> --fx <dir> --fy <dir> --fz <dir> [-o <dir>]\n"
            "\n"
            "Extract strain Green's functions from SEM record files (MPI tile-parallel).\n"
            "Reads per-step record_{r}_{step}.h5 files from three force-direction\n"
            "forward runs, merges per-rank records, assembles the full 3x6 Green's\n"
            "tensor at every recorded mesh vertex, and writes tiled HDF5 output.\n"
            "Tiles round-robin across ranks; ranks without a tile exit early.\n"
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

// -----------------------------------------------------------------------
// Per-file mapping (local GLL node id -> global merged index)
// -----------------------------------------------------------------------

struct FileMapping {
    RecordFileInfo info;
    std::vector<int32_t> local_to_global;
    hsize_t n_rec_cell = 0;
    hsize_t nnodes = 0;
    std::vector<int32_t> cell_gll_idx;        // [n_rec_cell * n_node]
    std::vector<int64_t> rec_cell_model_idx;  // [n_rec_cell]
};

// -----------------------------------------------------------------------
// Merged metadata for one force direction (NO per-step field arrays)
// -----------------------------------------------------------------------

struct MergedMetadata {
    std::vector<double> gll_node_coords;  // [n_unique_gll, 3]
    std::vector<int64_t> gll_node_ids;    // [n_unique_gll] 1-based global DOF
    // Cell-level data (for whole-cell tiling + mass-weighted L2 projection)
    std::vector<int32_t> cell_gll_node_index;         // [n_rec_cell_merged * n_node]
    std::vector<int64_t> recording_cell_model_index;  // [n_rec_cell_merged]
    std::vector<FileMapping> file_maps;
    std::vector<StepGroup> groups;  // per-step file groups
    int64_t n_unique_gll = 0;
    int64_t n_steps = 0;
    int64_t n_node_per_cell = 0;
    int64_t n_rec_cell_merged = 0;
    bool has_displacement = false;
    bool has_velocity = false;
    bool has_acceleration = false;
};

// -----------------------------------------------------------------------
// Phase 1: merge metadata (first pass + cell accumulation, no per-step fields).
// -----------------------------------------------------------------------

static MergedMetadata merge_metadata(const char* dir_path) {
    MergedMetadata result;
    fprintf(stderr, "[postprocess] Scanning metadata from %s...\n", dir_path);

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
            hsize_t idx_total = idxdims[0] * idxdims[1];
            cell_gll_idx_local.resize((size_t)idx_total);
            H5Dread(idxds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    cell_gll_idx_local.data());
            H5Sclose(ispace);
            H5Dclose(idxds);
        } else {
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

    result.groups = group_by_step(files);
    result.n_steps = (int64_t)result.groups.size();
    result.file_maps = std::move(file_maps);
    fprintf(stderr, "[postprocess]   %lld steps\n", (long long)result.n_steps);

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

    return result;
}

// -----------------------------------------------------------------------
// Per-direction tile-local field arrays (second pass, tile-local only)
// -----------------------------------------------------------------------

struct DirFields {
    std::vector<double> strain;        // [n_steps, n_local, 6]
    std::vector<double> displacement;  // [n_steps, n_local, 3]
    std::vector<double> velocity;      // [n_steps, n_local, 3]
    std::vector<double> acceleration;  // [n_steps, n_local, 3]
};

// Phase 3: extract per-step fields for ONE direction, tile-local nodes only.
// All recording cells are iterated (shared-node averaging correctness), but
// only tile-local nodes are accumulated. Memory: n_steps * n_local, not n_unique_gll.
// tile_local_index: [n_unique_gll] -> [0, n_local) or -1
static DirFields extract_tile_fields(const MergedMetadata& meta,
                                     const std::vector<int32_t>& tile_local_index, int64_t n_local,
                                     const std::vector<double>& cell_mass, int64_t n_model_cell,
                                     int ngll) {
    DirFields result;
    int64_t n_steps = meta.n_steps;
    int64_t ng = meta.n_unique_gll;
    int64_t n_node_per_cell = meta.n_node_per_cell;

    result.strain.resize((size_t)n_steps * (size_t)n_local * 6, 0.0);
    if (meta.has_displacement)
        result.displacement.resize((size_t)n_steps * (size_t)n_local * 3, 0.0);
    if (meta.has_velocity)
        result.velocity.resize((size_t)n_steps * (size_t)n_local * 3, 0.0);
    if (meta.has_acceleration)
        result.acceleration.resize((size_t)n_steps * (size_t)n_local * 3, 0.0);

    bool use_mass_weighted = !cell_mass.empty() && ngll > 0 && n_model_cell > 0;
    int ngll2 = ngll * ngll;
    int n_node_mass = ngll * ngll2;

    // --- Per-step GLL-node averaging (tile-local only) ---
    for (int64_t snap_idx = 0; snap_idx < n_steps; ++snap_idx) {
        const auto& group = meta.groups[(size_t)snap_idx];
        double* step_data = result.strain.data() + snap_idx * n_local * 6;
        double* step_disp =
            meta.has_displacement ? result.displacement.data() + snap_idx * n_local * 3 : nullptr;
        double* step_vel =
            meta.has_velocity ? result.velocity.data() + snap_idx * n_local * 3 : nullptr;
        double* step_acc =
            meta.has_acceleration ? result.acceleration.data() + snap_idx * n_local * 3 : nullptr;

        // Accumulate GLL mass for mass-weighted strain averaging (tile-local).
        std::vector<double> node_weight_sum((size_t)n_local, 0.0);
        gf_postprocess_common::VectorFieldAverager displacement_average(step_disp,
                                                                        (size_t)n_local);
        gf_postprocess_common::VectorFieldAverager velocity_average(step_vel, (size_t)n_local);
        gf_postprocess_common::VectorFieldAverager acceleration_average(step_acc, (size_t)n_local);

        for (const auto& fm : meta.file_maps) {
            const RecordFileInfo* gfi = nullptr;
            for (const auto& fi : group.files) {
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
                    int32_t li = tile_local_index[(size_t)global_idx];
                    if (li < 0)
                        continue;  // not tile-local

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
                        }
                    }
                    double* src = strain_buf.data() + (c * nnp + p) * 6;
                    double* dst = step_data + (size_t)li * 6;
                    for (int comp = 0; comp < 6; ++comp)
                        dst[comp] += src[comp] * weight;
                    node_weight_sum[(size_t)li] += weight;
                }
            }

            // Read and accumulate displacement/velocity/acceleration (identical
            // logic - only the dataset name and step buffer differ).
            auto accumulate_vector_field =
                [&](const char* ds_name, gf_postprocess_common::VectorFieldAverager& average) {
                    hsize_t frc = 0, fnp = 0;
                    std::vector<double> fbuf;
                    read_field_4d(fid, ds_name, frc, fnp, fbuf);
                    for (hsize_t c = 0; c < frc && c < nrc; ++c) {
                        for (hsize_t p = 0; p < fnp && p < n_node_per_cell; ++p) {
                            int32_t local_gll_idx = cell_gll_idx[c * (hsize_t)n_node_per_cell + p];
                            if (local_gll_idx < 0 ||
                                local_gll_idx >= (int32_t)fm.local_to_global.size())
                                continue;
                            int32_t global_idx = fm.local_to_global[(size_t)local_gll_idx];
                            if (global_idx < 0 || global_idx >= (int32_t)ng)
                                continue;
                            int32_t li = tile_local_index[(size_t)global_idx];
                            if (li < 0)
                                continue;
                            const double* fsrc = fbuf.data() + (c * fnp + p) * 3;
                            average.add((size_t)li, fsrc);
                        }
                    }
                };
            if (meta.has_displacement)
                accumulate_vector_field("displacement", displacement_average);
            if (meta.has_velocity)
                accumulate_vector_field("velocity", velocity_average);
            if (meta.has_acceleration)
                accumulate_vector_field("acceleration", acceleration_average);

            H5Fclose(fid);
        }

        // Normalize: mass-weighted average for strain, count-based for disp/vel/acc.
        for (int64_t li = 0; li < n_local; ++li) {
            if (node_weight_sum[(size_t)li] > 0.0) {
                double* dst = step_data + li * 6;
                double inv_mass = 1.0 / node_weight_sum[(size_t)li];
                for (int c = 0; c < 6; ++c)
                    dst[c] *= inv_mass;
            }
        }
        if (meta.has_displacement)
            displacement_average.normalize();
        if (meta.has_velocity)
            velocity_average.normalize();
        if (meta.has_acceleration)
            acceleration_average.normalize();
    }  // snap_idx loop

    return result;
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------

int main(int argc, char** argv) {
    // ---- MPI init ----
    MPI_Init(&argc, &argv);
    int mpi_rank = 0, mpi_nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_nranks);
    fprintf(stderr, "[postprocess] MPI rank %d/%d\n", mpi_rank, mpi_nranks);
    // Stagger file access to avoid HDF5 metadata contention
    MPI_Barrier(MPI_COMM_WORLD);
    usleep((unsigned int)(mpi_rank * 200000));  // 200ms stagger
    MPI_Barrier(MPI_COMM_WORLD);

    double start = 0.0;
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        start = ts.tv_sec + ts.tv_nsec * 1e-9;
    }

    fprintf(stderr, "[postprocess] Starting...\n");

    auto args = gf_postprocess_common::parse_args(argc, argv, print_usage);

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
    gf_postprocess_common::read_cell_mass(args.model_path.c_str(), cell_mass, n_model_cell,
                                          ngll_model);

    // ---- Phase 1: merge metadata for each direction (cheap, replicated) ----
    MergedMetadata mfx = merge_metadata(args.fx_dir.c_str());
    MergedMetadata mfy = merge_metadata(args.fy_dir.c_str());
    MergedMetadata mfz = merge_metadata(args.fz_dir.c_str());

    if (mfx.n_steps != mfy.n_steps || mfx.n_steps != mfz.n_steps) {
        fprintf(stderr,
                "ERROR: mismatched number of steps across directions "
                "(%lld, %lld, %lld)\n",
                (long long)mfx.n_steps, (long long)mfy.n_steps, (long long)mfz.n_steps);
        exit(1);
    }
    int64_t n_steps = mfx.n_steps;

    if (mfx.gll_node_ids != mfy.gll_node_ids || mfx.gll_node_ids != mfz.gll_node_ids) {
        fprintf(stderr, "[postprocess] WARNING: GLL node sets differ across directions\n");
    }

    // GLL node IDs (1-based, shared across directions)
    const auto& recorded_ids = mfx.gll_node_ids;
    int64_t n_recorded = mfx.n_unique_gll;
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
    // config STF is at solver_dt; tile time axis is at output_dt_s.
    std::vector<double> stf_t_ds, stf_values_ds;
    gf_postprocess_common::downsample_stf(cfg, n_steps, stf_t_ds, stf_values_ds);

    bool has_displacement = mfx.has_displacement && mfy.has_displacement && mfz.has_displacement;
    bool has_velocity = mfx.has_velocity && mfy.has_velocity && mfz.has_velocity;
    bool has_acceleration = mfx.has_acceleration && mfy.has_acceleration && mfz.has_acceleration;
    fprintf(stderr, "[postprocess]   displacement=%s velocity=%s acceleration=%s\n",
            has_displacement ? "yes" : "no", has_velocity ? "yes" : "no",
            has_acceleration ? "yes" : "no");

    // ---- Phase 2: bin recording cells into tiles (whole-cell tiling) ----
    fprintf(stderr, "[postprocess] Binning recording cells into tiles...\n");
    std::unordered_map<TileKey, std::vector<int64_t>, TileKeyHash> cell_bins;
    double xmin = model.xmin, ymin = model.ymin, xmax = model.xmax, ymax = model.ymax;
    double dx = (xmax - xmin) / cfg.nx_elements;
    double dy = (ymax - ymin) / cfg.ny_elements;
    int64_t total_interior_x = 0, total_interior_y = 0;
    for (auto sz : cfg.tilex_elements)
        total_interior_x += sz;
    for (auto sz : cfg.tiley_elements)
        total_interior_y += sz;

    for (int64_t ci = 0; ci < mfx.n_rec_cell_merged; ++ci) {
        int64_t idx0 = mfx.cell_gll_node_index[(size_t)ci * (size_t)mfx.n_node_per_cell + 0];
        int64_t idx124 = mfx.cell_gll_node_index[(size_t)ci * (size_t)mfx.n_node_per_cell + 124];
        if (idx0 < 0 || idx0 >= n_recorded || idx124 < 0 || idx124 >= n_recorded)
            continue;
        double cx = 0.5 * (mfx.gll_node_coords[(size_t)idx0 * 3 + 0] +
                           mfx.gll_node_coords[(size_t)idx124 * 3 + 0]);
        double cy = 0.5 * (mfx.gll_node_coords[(size_t)idx0 * 3 + 1] +
                           mfx.gll_node_coords[(size_t)idx124 * 3 + 1]);
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
    std::vector<TileKey> tile_keys;
    for (auto& kv : cell_bins)
        tile_keys.push_back(kv.first);
    std::sort(tile_keys.begin(), tile_keys.end());
    fprintf(stderr, "[postprocess]   %zu tiles\n", tile_keys.size());

    int64_t n_tiles = (int64_t)tile_keys.size();

    // ---- Rank exit check: ranks beyond n_tiles exit BEFORE any field alloc ----
    if (mpi_rank >= (int)n_tiles) {
        fprintf(stderr, "[postprocess] Rank %d: no tile assigned (n_tiles=%lld), exiting\n",
                mpi_rank, (long long)n_tiles);
        MPI_Finalize();
        return 0;
    }

    std::string mkdir_cmd = "mkdir -p " + args.output_dir;
    if (system(mkdir_cmd.c_str()) != 0) {
        fprintf(stderr, "WARNING: could not create output directory %s\n",
                args.output_dir.c_str());
    }

    double zmin = model.zmin, zmax = model.zmax;

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

    // ---- Write assigned tiles (round-robin: tile_pos = rank, rank+nranks, ...) ----
    for (int64_t tile_pos = (int64_t)mpi_rank; tile_pos < n_tiles; tile_pos += mpi_nranks) {
        const TileKey& key = tile_keys[(size_t)tile_pos];
        const auto& cell_indices = cell_bins.at(key);

        // Build tile-local GLL node set from cells
        std::unordered_map<int64_t, int64_t> gll_to_tile_local;
        std::vector<int64_t> tile_gll_indices;
        for (auto ci : cell_indices) {
            for (int64_t p = 0; p < mfx.n_node_per_cell; ++p) {
                int32_t gi = mfx.cell_gll_node_index[(size_t)ci * (size_t)mfx.n_node_per_cell + p];
                if (gi >= 0 && gll_to_tile_local.find(gi) == gll_to_tile_local.end()) {
                    gll_to_tile_local[gi] = (int64_t)tile_gll_indices.size();
                    tile_gll_indices.push_back(gi);
                }
            }
        }
        int64_t n_local = (int64_t)tile_gll_indices.size();

        // tile_local_index: [n_recorded] -> tile-local index or -1
        std::vector<int32_t> tile_local_index((size_t)n_recorded, -1);
        for (int64_t li = 0; li < n_local; ++li)
            tile_local_index[(size_t)tile_gll_indices[(size_t)li]] = (int32_t)li;

        // Build tile-local cell_gll_node_index
        std::vector<int32_t> tile_cell_gll_index(cell_indices.size() *
                                                 (size_t)mfx.n_node_per_cell);
        for (size_t ci = 0; ci < cell_indices.size(); ++ci) {
            for (int64_t p = 0; p < mfx.n_node_per_cell; ++p) {
                int32_t gi = mfx.cell_gll_node_index[(size_t)cell_indices[ci] *
                                                         (size_t)mfx.n_node_per_cell +
                                                     p];
                tile_cell_gll_index[ci * (size_t)mfx.n_node_per_cell + (size_t)p] =
                    (gi >= 0) ? (int32_t)gll_to_tile_local[gi] : -1;
            }
        }

        // Build tile vertex IDs and coords
        std::vector<int64_t> tile_vertex_ids((size_t)n_local);
        std::vector<double> tile_vertex_coords((size_t)n_local * 3);
        for (int64_t i = 0; i < n_local; ++i) {
            int64_t gi = tile_gll_indices[(size_t)i];
            tile_vertex_ids[(size_t)i] = recorded_ids[(size_t)gi];
            tile_vertex_coords[(size_t)i * 3 + 0] = mfx.gll_node_coords[(size_t)gi * 3 + 0];
            tile_vertex_coords[(size_t)i * 3 + 1] = mfx.gll_node_coords[(size_t)gi * 3 + 1];
            tile_vertex_coords[(size_t)i * 3 + 2] = mfx.gll_node_coords[(size_t)gi * 3 + 2];
        }

        // ---- Phase 3: assemble tile Green's tensor directly from per-direction
        //      tile-local fields. One direction at a time to bound peak memory.
        // tile_greens: [n_steps, n_local, comp(6), dir(3)] = 18 per (s,li)
        std::vector<double> tile_greens((size_t)n_steps * (size_t)n_local * 6 * 3, 0.0);
        std::vector<double> tile_displacement;
        std::vector<double> tile_velocity;
        std::vector<double> tile_acceleration;
        if (has_displacement)
            tile_displacement.resize((size_t)n_steps * (size_t)n_local * 3 * 3, 0.0);
        if (has_velocity)
            tile_velocity.resize((size_t)n_steps * (size_t)n_local * 3 * 3, 0.0);
        if (has_acceleration)
            tile_acceleration.resize((size_t)n_steps * (size_t)n_local * 3 * 3, 0.0);

        // Direction 0 = fx, 1 = fy, 2 = fz
        const MergedMetadata* metas[3] = {&mfx, &mfy, &mfz};
        for (int dir = 0; dir < 3; ++dir) {
            DirFields fields = extract_tile_fields(*metas[dir], tile_local_index, n_local,
                                                   cell_mass, n_model_cell, ngll_model);

            // Assemble strain -> tile_greens [s, li, comp(6), dir]
            for (int64_t s = 0; s < n_steps; ++s) {
                for (int64_t li = 0; li < n_local; ++li) {
                    size_t tg = ((size_t)s * (size_t)n_local + (size_t)li) * 18;
                    const double* src =
                        fields.strain.data() + ((size_t)s * (size_t)n_local + (size_t)li) * 6;
                    gf_postprocess_common::assign_strain_direction(src, tile_greens.data() + tg,
                                                                   dir);
                }
            }
            // Assemble disp/vel/acc -> tile_* [s, li, comp(3), dir(3)]
            auto assemble_vector_field = [&](const std::vector<double>& src,
                                             std::vector<double>& dst) {
                if (src.empty())
                    return;
                for (int64_t s = 0; s < n_steps; ++s) {
                    for (int64_t li = 0; li < n_local; ++li) {
                        size_t td = ((size_t)s * (size_t)n_local + (size_t)li) * 9;
                        const double* sp =
                            src.data() + ((size_t)s * (size_t)n_local + (size_t)li) * 3;
                        for (int c = 0; c < 3; ++c)
                            dst[td + (size_t)c * 3 + (size_t)dir] = sp[c];
                    }
                }
            };
            if (has_displacement)
                assemble_vector_field(fields.displacement, tile_displacement);
            if (has_velocity)
                assemble_vector_field(fields.velocity, tile_velocity);
            if (has_acceleration)
                assemble_vector_field(fields.acceleration, tile_acceleration);
            // fields freed at end of iteration (one direction at a time)
        }

        double source_xyz_m[3] = {cfg.source_x_m, cfg.source_y_m, cfg.source_z_m};

        double tx_min, tx_max, ty_min, ty_max;
        compute_tile_bounds(key, tx_min, tx_max, ty_min, ty_max);

        // Output precision follows config snapshot_precision
        bool use_float32 = (cfg.snapshot_precision == "float32");

        char fname[256];
        std::snprintf(fname, sizeof(fname), "%s/tile_x%03d_y%03d.h5", args.output_dir.c_str(),
                      key.tx, key.ty);

        write_tile(fname, key.tx, key.ty, tx_min, tx_max, ty_min, ty_max, zmin, zmax,
                   cfg.record_depth_max_m, cfg.record_depth_actual_m, tile_vertex_ids, time_arr,
                   cfg.solver_dt, tile_greens, source_xyz_m, tile_vertex_coords,
                   tile_cell_gll_index, (int)mfx.n_node_per_cell,
                   has_displacement ? tile_displacement.data() : nullptr,
                   has_velocity ? tile_velocity.data() : nullptr,
                   has_acceleration ? tile_acceleration.data() : nullptr, stf_t_ds, stf_values_ds,
                   use_float32);
        fprintf(stderr, "[postprocess]   rank %d wrote tile x%03d y%03d\n", mpi_rank, key.tx,
                key.ty);
    }

    MPI_Finalize();

    // ---- Print machine-parseable stats ----
    gf_postprocess_common::print_stats(start, n_steps, n_vertex, n_recorded, n_tiles);

    return 0;
}
