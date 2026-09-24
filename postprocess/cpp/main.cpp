/* postprocess/cpp/main.cpp - tile-batched Green's function postprocessor
 *
 * CLI:
 *   gf_postprocess[_mpi] <model.h5> <config.h5> \
 *       --fx <dir> --fy <dir> --fz <dir> -o <output_dir>
 *
 * Both executables use the same worker-local pipeline. The serial executable
 * processes every tile with worker 0/1. The MPI executable groups tiles by
 * shared record ranks while balancing work; each tile is written exactly once.
 *
 * Memory design:
 *   Phase 1 - rank 0 rebuilds layouts from partitions and prepares shared indexes;
 *             MPI broadcasts them without rebuilding on other workers.
 *   Phase 2 - each worker prepares indexes for its assigned tiles before field I/O.
 *   Phase 3 - each worker reads a record once and accumulates it into every
 *             assigned tile that depends on that record rank. All sharing cells
 *             are still iterated so averaging stays correct.
 */

#ifdef GF_POST_MPI
#include <mpi.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"
#include "reader.hpp"
#include "writer.hpp"

#ifdef DEBUG
#define GF_POST_DEBUG_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
#define GF_POST_DEBUG_LOG(...) ((void)0)
#endif

// -----------------------------------------------------------------------
// CLI argument parsing
// -----------------------------------------------------------------------

/// Print command-line usage and exit.
static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s <model.h5> <config.h5> --fx <dir> --fy <dir> --fz <dir> [-o <dir>]\n"
            "\n"
            "Extract strain Green's functions from SEM record files (tile-batched).\n"
            "Reads per-step record_{r}_{step}.h5 files from three force-direction\n"
            "forward runs, merges per-rank records, assembles the full 3x6 Green's\n"
            "tensor at every recorded mesh vertex, and writes tiled HDF5 output.\n"
            "The MPI build groups tiles by record-rank overlap; the serial build processes\n"
            "the same tiles sequentially.\n"
            "\n"
            "Arguments:\n"
            "  model.h5   Mesh file with /topology/vertex_to_coord + /domain/ bounds\n"
            "  config.h5  Simulation config with /simulation/ attrs and tile arrays\n"
            "  --fx dir   Directory with x-direction record files\n"
            "  --fy dir   Directory with y-direction record files\n"
            "  --fz dir   Directory with z-direction record files\n"
            "  -o dir     Output directory (default: greenfun/)\n"
            "\n"
            "Environment:\n"
            "  GF_POST_MEMORY_GB  Aggregate field-array budget in GiB (default: 32)\n",
            prog);
}

// -----------------------------------------------------------------------
// Per-rank mapping (local GLL node id -> global merged index)
// -----------------------------------------------------------------------

struct RankMapping {
    int rank = -1;
    std::vector<int32_t> local_to_global;
    hsize_t n_rec_cell = 0;
    hsize_t nnodes = 0;
    std::vector<int32_t> cell_gll_idx;         // [n_rec_cell * n_node]
    std::vector<int64_t> rec_cell_model_idx;   // [n_rec_cell]
    std::vector<int64_t> record_cell_indices;  // [n_rec_cell], index in full-domain record
    std::vector<int64_t> cell_point_mass_idx;  // [n_rec_cell * n_node]
};

// -----------------------------------------------------------------------
// Shared merged layout (NO per-step field arrays)
// -----------------------------------------------------------------------

struct LayoutMetadata {
    std::vector<double> gll_node_coords;  // [n_unique_gll, 3]
    std::vector<int64_t> gll_node_ids;    // [n_unique_gll] 1-based global DOF
    // Cell-level data (for whole-cell tiling + mass-weighted L2 projection)
    std::vector<int32_t> cell_gll_node_index;  // [n_rec_cell_merged * n_node]
    std::vector<RankMapping> rank_maps;
    int64_t n_unique_gll = 0;
    int64_t n_node_per_cell = 0;
    int64_t n_rec_cell_merged = 0;
};

struct DirectionRecords {
    std::vector<StepGroup> groups;  // discarded after record_paths is built
    std::vector<int> steps;
    std::vector<std::vector<std::string>> record_paths;  // [step][rank-map index]
#ifdef DEBUG
    bool has_displacement = false;
    bool has_velocity = false;
    bool has_acceleration = false;
#endif
};

// -----------------------------------------------------------------------
// Phase 1: scan direction records without rebuilding layout indexes.
// -----------------------------------------------------------------------

static DirectionRecords scan_direction_records(const char* dir_path) {
    DirectionRecords result;
    auto files = discover_records(dir_path);
    if (files.empty()) {
        fprintf(stderr, "ERROR: no record files found in %s\n", dir_path);
        exit(1);
    }
    result.groups = group_by_step(files);

#ifdef DEBUG
    hid_t probe = H5Fopen(files[0].path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (probe >= 0) {
        result.has_displacement = H5Lexists(probe, "displacement", H5P_DEFAULT) > 0;
        result.has_velocity = H5Lexists(probe, "velocity", H5P_DEFAULT) > 0;
        result.has_acceleration = H5Lexists(probe, "acceleration", H5P_DEFAULT) > 0;
        H5Fclose(probe);
    }
#endif
    return result;
}

static bool read_int_attribute(hid_t location, const char* name, int& value) {
    if (H5Aexists(location, name) <= 0)
        return false;
    hid_t attribute = H5Aopen(location, name, H5P_DEFAULT);
    if (attribute < 0)
        return false;
    const bool success = H5Aread(attribute, H5T_NATIVE_INT, &value) >= 0;
    H5Aclose(attribute);
    return success;
}

template <typename T>
static std::vector<T> read_flat_dataset(hid_t location, const char* name, hid_t memory_type,
                                        hsize_t& value_count) {
    hid_t dataset = H5Dopen2(location, name, H5P_DEFAULT);
    if (dataset < 0) {
        value_count = 0;
        return {};
    }
    hid_t space = H5Dget_space(dataset);
    const int dimension_count = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dimensions((size_t)dimension_count);
    H5Sget_simple_extent_dims(space, dimensions.data(), nullptr);
    value_count = 1;
    for (hsize_t dimension : dimensions)
        value_count *= dimension;
    std::vector<T> values((size_t)value_count);
    if (value_count > 0)
        H5Dread(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    H5Sclose(space);
    H5Dclose(dataset);
    return values;
}

static std::pair<int, int> record_partition_range(const DirectionRecords& records, int rank,
                                                  int partition_count,
                                                  const std::set<int>& record_ranks) {
    bool checked_record = false;
    for (const auto& group : records.groups) {
        for (const auto& file : group.files) {
            if (file.rank != rank)
                continue;
            checked_record = true;
            hid_t record = H5Fopen(file.path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            int start = 0;
            int count = 0;
            const bool found = record >= 0 &&
                               read_int_attribute(record, "source_partition_start", start) &&
                               read_int_attribute(record, "source_partition_count", count);
            if (record >= 0)
                H5Fclose(record);
            if (found)
                return {start, count};
            break;
        }
        if (checked_record)
            break;
    }

    bool contiguous_from_zero = true;
    int expected_rank = 0;
    for (int record_rank : record_ranks) {
        if (record_rank != expected_rank++) {
            contiguous_from_zero = false;
            break;
        }
    }
    const int effective_ranks = static_cast<int>(record_ranks.size());
    if (contiguous_from_zero && effective_ranks > 0 && effective_ranks < partition_count) {
        const int base = partition_count / effective_ranks;
        const int remainder = partition_count % effective_ranks;
        const int start = rank * base + std::min(rank, remainder);
        const int count = base + (rank < remainder ? 1 : 0);
        return {start, count};
    }
    return {rank, 1};
}

static LayoutMetadata merge_partition_metadata(const std::filesystem::path& partition_dir,
                                               const DirectionRecords& records) {
    LayoutMetadata result;
    GF_POST_DEBUG_LOG("[postprocess] Building recording layout from %s...\n",
                      partition_dir.c_str());

    std::set<int> ranks;
    for (const auto& group : records.groups)
        for (const auto& file : group.files)
            ranks.insert(file.rank);

    int partition_count = 0;
    while (std::filesystem::exists(partition_dir /
                                   ("partition_" + std::to_string(partition_count) + ".h5")))
        ++partition_count;
    if (partition_count == 0) {
        fprintf(stderr, "ERROR: no partition files found in %s\n", partition_dir.c_str());
        exit(1);
    }

    // Merge the original partition recording maps exactly as the solver merges
    // partitions for each output rank.
    std::unordered_map<int64_t, int32_t> global_to_merged_idx;
    std::vector<int64_t> merged_gll_node_ids;
    std::vector<double> merged_gll_node_coords;
    int64_t n_node_per_cell = 0;

    std::vector<RankMapping> rank_maps;
    for (int rank : ranks) {
        RankMapping mapping;
        mapping.rank = rank;
        std::unordered_map<int64_t, int32_t> rank_node_index;
        const auto [partition_start, assigned_partition_count] =
            record_partition_range(records, rank, partition_count, ranks);
        if (partition_start < 0 || assigned_partition_count <= 0 ||
            partition_start + assigned_partition_count > partition_count) {
            fprintf(stderr, "ERROR: invalid partition range [%d, %d) for record rank %d\n",
                    partition_start, partition_start + assigned_partition_count, rank);
            exit(1);
        }

        int64_t output_cell_offset = 0;
        for (int partition = partition_start;
             partition < partition_start + assigned_partition_count; ++partition) {
            const auto path = partition_dir / ("partition_" + std::to_string(partition) + ".h5");
            hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file < 0) {
                fprintf(stderr, "ERROR: cannot open partition file %s\n", path.c_str());
                exit(1);
            }
            hsize_t partition_local_cell_count = 0;
            auto local_cell_ids =
                read_int64_1d(file, "/partition/local_cell_ids", partition_local_cell_count);
            if (local_cell_ids.size() != (size_t)partition_local_cell_count) {
                fprintf(stderr, "ERROR: invalid local_cell_ids in %s\n", path.c_str());
                exit(1);
            }
            hid_t recording = -1;
            H5E_BEGIN_TRY {
                recording = H5Gopen2(file, "/recording", H5P_DEFAULT);
            }
            H5E_END_TRY;
            if (recording < 0) {
                output_cell_offset += static_cast<int64_t>(partition_local_cell_count);
                H5Fclose(file);
                continue;
            }

            hsize_t node_count = 0;
            auto partition_node_ids = read_int64_1d(recording, "gll_node_ids", node_count);
            hsize_t coordinate_count = 0;
            auto partition_node_coords = read_flat_dataset<double>(
                recording, "gll_node_coords", H5T_NATIVE_DOUBLE, coordinate_count);
            hsize_t recording_cell_count = 0;
            auto recording_cells =
                read_int64_1d(recording, "rec_cell_global_ids", recording_cell_count);
            hsize_t recording_local_cell_count = 0;
            auto recording_local_cells = read_flat_dataset<int32_t>(
                recording, "rec_cell_local", H5T_NATIVE_INT32, recording_local_cell_count);
            hsize_t cell_point_count = 0;
            auto partition_cell_nodes = read_flat_dataset<int32_t>(
                recording, "cell_gll_node_index", H5T_NATIVE_INT32, cell_point_count);
            H5Gclose(recording);
            H5Fclose(file);

            if (recording_cell_count == 0) {
                output_cell_offset += static_cast<int64_t>(partition_local_cell_count);
                continue;
            }
            if (recording_local_cell_count != recording_cell_count) {
                fprintf(stderr, "ERROR: inconsistent rec_cell_local in %s\n", path.c_str());
                exit(1);
            }
            for (int32_t local_cell : recording_local_cells) {
                if (local_cell < 0 ||
                    static_cast<hsize_t>(local_cell) >= partition_local_cell_count) {
                    fprintf(stderr, "ERROR: rec_cell_local out of range in %s\n", path.c_str());
                    exit(1);
                }
                mapping.record_cell_indices.push_back(output_cell_offset + local_cell);
            }
            const int64_t partition_nodes_per_cell =
                static_cast<int64_t>(cell_point_count / recording_cell_count);
            if (partition_nodes_per_cell * static_cast<int64_t>(recording_cell_count) !=
                    static_cast<int64_t>(cell_point_count) ||
                (n_node_per_cell != 0 && n_node_per_cell != partition_nodes_per_cell)) {
                fprintf(stderr, "ERROR: inconsistent recording map in %s\n", path.c_str());
                exit(1);
            }
            n_node_per_cell = partition_nodes_per_cell;

            std::vector<int32_t> partition_to_rank(node_count, -1);
            for (hsize_t node = 0; node < node_count; ++node) {
                const int64_t global_node_id = partition_node_ids[(size_t)node];
                auto rank_node = rank_node_index.find(global_node_id);
                if (rank_node == rank_node_index.end()) {
                    const int32_t output_node =
                        static_cast<int32_t>(mapping.local_to_global.size());
                    rank_node_index[global_node_id] = output_node;
                    partition_to_rank[(size_t)node] = output_node;

                    auto global_node = global_to_merged_idx.find(global_node_id);
                    int32_t merged_node = -1;
                    if (global_node == global_to_merged_idx.end()) {
                        merged_node = static_cast<int32_t>(merged_gll_node_ids.size());
                        global_to_merged_idx[global_node_id] = merged_node;
                        merged_gll_node_ids.push_back(global_node_id);
                        for (int component = 0; component < 3; ++component)
                            merged_gll_node_coords.push_back(
                                partition_node_coords[(size_t)node * 3 + (size_t)component]);
                    } else {
                        merged_node = global_node->second;
                    }
                    mapping.local_to_global.push_back(merged_node);
                } else {
                    partition_to_rank[(size_t)node] = rank_node->second;
                }
            }

            for (int32_t partition_node : partition_cell_nodes)
                mapping.cell_gll_idx.push_back(partition_to_rank[(size_t)partition_node]);
            mapping.rec_cell_model_idx.insert(mapping.rec_cell_model_idx.end(),
                                              recording_cells.begin(), recording_cells.end());
            output_cell_offset += static_cast<int64_t>(partition_local_cell_count);
        }

        mapping.n_rec_cell = mapping.rec_cell_model_idx.size();
        mapping.nnodes = mapping.local_to_global.size();
        rank_maps.push_back(std::move(mapping));
    }

    result.n_unique_gll = (int64_t)merged_gll_node_ids.size();
    result.n_node_per_cell = n_node_per_cell;
    result.gll_node_ids = std::move(merged_gll_node_ids);
    result.gll_node_coords = std::move(merged_gll_node_coords);

    // Accumulate merged cell-level data for whole-cell tiling
    for (auto& mapping : rank_maps) {
        for (hsize_t c = 0; c < mapping.n_rec_cell; ++c) {
            for (hsize_t p = 0; p < (hsize_t)n_node_per_cell; ++p) {
                int32_t local_idx = mapping.cell_gll_idx[c * (hsize_t)n_node_per_cell + p];
                int32_t global_idx =
                    (local_idx >= 0 && (size_t)local_idx < mapping.local_to_global.size())
                        ? mapping.local_to_global[(size_t)local_idx]
                        : -1;
                result.cell_gll_node_index.push_back(global_idx);
            }
            result.n_rec_cell_merged++;
        }
    }
    GF_POST_DEBUG_LOG("[postprocess]   %lld merged recording cells\n",
                      (long long)result.n_rec_cell_merged);
    GF_POST_DEBUG_LOG("[postprocess]   %lld unique GLL nodes from %zu output rank(s)\n",
                      (long long)result.n_unique_gll, rank_maps.size());

    result.rank_maps = std::move(rank_maps);
    return result;
}

static void build_direction_record_index(DirectionRecords& records, const LayoutMetadata& layout) {
    std::unordered_map<int, size_t> rank_to_mapping;
    for (size_t mapping_index = 0; mapping_index < layout.rank_maps.size(); ++mapping_index)
        rank_to_mapping[layout.rank_maps[mapping_index].rank] = mapping_index;

    records.steps.resize(records.groups.size());
    records.record_paths.assign(records.groups.size(),
                                std::vector<std::string>(layout.rank_maps.size()));
    for (size_t step_index = 0; step_index < records.groups.size(); ++step_index) {
        const auto& group = records.groups[step_index];
        records.steps[step_index] = group.step;
        for (const auto& file : group.files) {
            auto mapping = rank_to_mapping.find(file.rank);
            if (mapping != rank_to_mapping.end())
                records.record_paths[step_index][mapping->second] = file.path;
        }
    }
    records.groups.clear();
    records.groups.shrink_to_fit();
}

static void build_mass_indexes(LayoutMetadata& layout, int64_t n_model_cell, int ngll) {
    const int64_t nodes_per_cell = static_cast<int64_t>(ngll) * ngll * ngll;
    for (auto& mapping : layout.rank_maps) {
        mapping.cell_point_mass_idx.assign(mapping.n_rec_cell * (hsize_t)layout.n_node_per_cell,
                                           -1);
        if (nodes_per_cell != layout.n_node_per_cell)
            continue;
        for (hsize_t cell = 0; cell < mapping.n_rec_cell; ++cell) {
            if (cell >= mapping.rec_cell_model_idx.size())
                continue;
            int64_t model_cell = mapping.rec_cell_model_idx[(size_t)cell];
            if (model_cell < 0 || model_cell >= n_model_cell)
                continue;
            for (int64_t point = 0; point < layout.n_node_per_cell; ++point) {
                mapping.cell_point_mass_idx[(size_t)cell * (size_t)layout.n_node_per_cell +
                                            (size_t)point] = model_cell * nodes_per_cell + point;
            }
        }
        mapping.rec_cell_model_idx.clear();
        mapping.rec_cell_model_idx.shrink_to_fit();
    }
}

struct TileBin {
    TileKey key;
    std::vector<int64_t> cell_indices;
};

struct TilePlan {
    TileKey key;
    int64_t n_node_per_cell = 0;
    std::vector<int64_t> gll_indices;
    std::vector<int32_t> rank_ids;
    std::vector<int64_t> rank_entry_offsets;
    std::vector<int64_t> record_cell_point_index;
    std::vector<int32_t> tile_local_node_index;
    std::vector<int64_t> cell_mass_index;
    std::vector<int32_t> cell_gll_index;
    std::vector<int64_t> node_ids;
    std::vector<double> node_coords;
};

#ifdef GF_POST_MPI
template <typename T>
static void broadcast_vector(std::vector<T>& values, MPI_Datatype datatype, int worker_rank) {
    uint64_t size = static_cast<uint64_t>(values.size());
    MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (worker_rank != 0)
        values.resize(static_cast<size_t>(size));
    if (size > 0)
        MPI_Bcast(values.data(), static_cast<int>(size), datatype, 0, MPI_COMM_WORLD);
}

static void broadcast_string(std::string& value, int worker_rank) {
    uint64_t size = static_cast<uint64_t>(value.size());
    MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (worker_rank != 0)
        value.resize(static_cast<size_t>(size));
    if (size > 0)
        MPI_Bcast(value.data(), static_cast<int>(size), MPI_CHAR, 0, MPI_COMM_WORLD);
}

static void broadcast_layout(LayoutMetadata& layout, int worker_rank) {
    int64_t scalars[4] = {layout.n_unique_gll, layout.n_node_per_cell, layout.n_rec_cell_merged,
                          static_cast<int64_t>(layout.rank_maps.size())};
    MPI_Bcast(scalars, 4, MPI_INT64_T, 0, MPI_COMM_WORLD);
    if (worker_rank != 0) {
        layout.n_unique_gll = scalars[0];
        layout.n_node_per_cell = scalars[1];
        layout.n_rec_cell_merged = scalars[2];
        layout.rank_maps.resize(static_cast<size_t>(scalars[3]));
    }
    broadcast_vector(layout.gll_node_coords, MPI_DOUBLE, worker_rank);
    broadcast_vector(layout.gll_node_ids, MPI_INT64_T, worker_rank);
    broadcast_vector(layout.cell_gll_node_index, MPI_INT32_T, worker_rank);

    for (auto& mapping : layout.rank_maps) {
        int64_t mapping_scalars[3] = {mapping.rank, static_cast<int64_t>(mapping.n_rec_cell),
                                      static_cast<int64_t>(mapping.nnodes)};
        MPI_Bcast(mapping_scalars, 3, MPI_INT64_T, 0, MPI_COMM_WORLD);
        if (worker_rank != 0) {
            mapping.rank = static_cast<int>(mapping_scalars[0]);
            mapping.n_rec_cell = static_cast<hsize_t>(mapping_scalars[1]);
            mapping.nnodes = static_cast<hsize_t>(mapping_scalars[2]);
        }
        broadcast_vector(mapping.local_to_global, MPI_INT32_T, worker_rank);
        broadcast_vector(mapping.cell_gll_idx, MPI_INT32_T, worker_rank);
        broadcast_vector(mapping.record_cell_indices, MPI_INT64_T, worker_rank);
        broadcast_vector(mapping.cell_point_mass_idx, MPI_INT64_T, worker_rank);
    }
}

static void broadcast_direction_records(DirectionRecords& records, const LayoutMetadata& layout,
                                        int worker_rank) {
#ifdef DEBUG
    int flags[3] = {records.has_displacement ? 1 : 0, records.has_velocity ? 1 : 0,
                    records.has_acceleration ? 1 : 0};
    MPI_Bcast(flags, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (worker_rank != 0) {
        records.has_displacement = flags[0] != 0;
        records.has_velocity = flags[1] != 0;
        records.has_acceleration = flags[2] != 0;
    }
#endif
    broadcast_vector(records.steps, MPI_INT, worker_rank);
    if (worker_rank != 0) {
        records.record_paths.assign(records.steps.size(),
                                    std::vector<std::string>(layout.rank_maps.size()));
    }
    for (auto& step_paths : records.record_paths)
        for (auto& path : step_paths)
            broadcast_string(path, worker_rank);
}

static void broadcast_tile_bins(std::vector<TileBin>& tile_bins, int worker_rank) {
    uint64_t count = static_cast<uint64_t>(tile_bins.size());
    MPI_Bcast(&count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (worker_rank != 0)
        tile_bins.resize(static_cast<size_t>(count));
    for (auto& tile_bin : tile_bins) {
        int key[2] = {tile_bin.key.tx, tile_bin.key.ty};
        MPI_Bcast(key, 2, MPI_INT, 0, MPI_COMM_WORLD);
        if (worker_rank != 0) {
            tile_bin.key.tx = key[0];
            tile_bin.key.ty = key[1];
        }
        broadcast_vector(tile_bin.cell_indices, MPI_INT64_T, worker_rank);
    }
}
#endif

static std::vector<TileBin> build_tile_bins(const LayoutMetadata& layout, const ConfigParams& cfg,
                                            const ModelData& model) {
    std::unordered_map<TileKey, std::vector<int64_t>, TileKeyHash> bins;
    const double dx = (model.xmax - model.xmin) / cfg.nx_elements;
    const double dy = (model.ymax - model.ymin) / cfg.ny_elements;
    int64_t total_interior_x = 0;
    int64_t total_interior_y = 0;
    for (auto size : cfg.tilex_elements)
        total_interior_x += size;
    for (auto size : cfg.tiley_elements)
        total_interior_y += size;

    const int64_t last_node = layout.n_node_per_cell - 1;
    for (int64_t cell = 0; cell < layout.n_rec_cell_merged; ++cell) {
        int64_t first_index =
            layout.cell_gll_node_index[(size_t)cell * (size_t)layout.n_node_per_cell];
        int64_t last_index =
            layout.cell_gll_node_index[(size_t)cell * (size_t)layout.n_node_per_cell +
                                       (size_t)last_node];
        if (first_index < 0 || first_index >= layout.n_unique_gll || last_index < 0 ||
            last_index >= layout.n_unique_gll)
            continue;
        double center_x = 0.5 * (layout.gll_node_coords[(size_t)first_index * 3] +
                                 layout.gll_node_coords[(size_t)last_index * 3]);
        double center_y = 0.5 * (layout.gll_node_coords[(size_t)first_index * 3 + 1] +
                                 layout.gll_node_coords[(size_t)last_index * 3 + 1]);
        int64_t element_x = dx > 0 ? (int64_t)std::floor((center_x - model.xmin) / dx) : 0;
        int64_t element_y = dy > 0 ? (int64_t)std::floor((center_y - model.ymin) / dy) : 0;
        element_x = std::max<int64_t>(0, std::min<int64_t>(element_x, cfg.nx_elements - 1));
        element_y = std::max<int64_t>(0, std::min<int64_t>(element_y, cfg.ny_elements - 1));
        int64_t interior_x = element_x - cfg.pml_xmin;
        int64_t interior_y = element_y - cfg.pml_ymin;
        if (interior_x < 0 || interior_x >= total_interior_x || interior_y < 0 ||
            interior_y >= total_interior_y)
            continue;
        TileKey key{find_tile_index(interior_x, cfg.tilex_elements),
                    find_tile_index(interior_y, cfg.tiley_elements)};
        bins[key].push_back(cell);
    }

    std::vector<TileBin> result;
    result.reserve(bins.size());
    for (auto& entry : bins)
        result.push_back({entry.first, std::move(entry.second)});
    std::sort(result.begin(), result.end(),
              [](const TileBin& left, const TileBin& right) { return left.key < right.key; });
    return result;
}

static TilePlan build_tile_plan(const TileBin& tile_bin, const LayoutMetadata& layout) {
    TilePlan plan;
    plan.key = tile_bin.key;
    plan.n_node_per_cell = layout.n_node_per_cell;

    std::unordered_map<int64_t, int64_t> global_to_tile;
    for (int64_t cell : tile_bin.cell_indices) {
        for (int64_t point = 0; point < layout.n_node_per_cell; ++point) {
            int32_t global_index =
                layout.cell_gll_node_index[(size_t)cell * (size_t)layout.n_node_per_cell +
                                           (size_t)point];
            if (global_index >= 0 && global_to_tile.find(global_index) == global_to_tile.end()) {
                global_to_tile[global_index] = static_cast<int64_t>(plan.gll_indices.size());
                plan.gll_indices.push_back(global_index);
            }
        }
    }

    std::vector<int32_t> global_to_tile_local(static_cast<size_t>(layout.n_unique_gll), -1);
    for (size_t tile_local = 0; tile_local < plan.gll_indices.size(); ++tile_local) {
        global_to_tile_local[(size_t)plan.gll_indices[tile_local]] =
            static_cast<int32_t>(tile_local);
    }

    for (size_t mapping_index = 0; mapping_index < layout.rank_maps.size(); ++mapping_index) {
        const auto& mapping = layout.rank_maps[mapping_index];
        const int64_t entry_begin = static_cast<int64_t>(plan.record_cell_point_index.size());
        for (size_t point = 0; point < mapping.cell_gll_idx.size(); ++point) {
            int32_t rank_node = mapping.cell_gll_idx[point];
            if (rank_node < 0 || rank_node >= (int32_t)mapping.local_to_global.size())
                continue;
            int32_t global_node = mapping.local_to_global[(size_t)rank_node];
            if (global_node < 0 || global_node >= layout.n_unique_gll)
                continue;
            const int32_t tile_node = global_to_tile_local[(size_t)global_node];
            if (tile_node < 0)
                continue;
            plan.record_cell_point_index.push_back(static_cast<int64_t>(point));
            plan.tile_local_node_index.push_back(tile_node);
            plan.cell_mass_index.push_back(mapping.cell_point_mass_idx[point]);
        }
        const int64_t entry_end = static_cast<int64_t>(plan.record_cell_point_index.size());
        if (entry_end > entry_begin) {
            plan.rank_ids.push_back(mapping.rank);
            if (plan.rank_entry_offsets.empty())
                plan.rank_entry_offsets.push_back(entry_begin);
            plan.rank_entry_offsets.push_back(entry_end);
        }
    }

    plan.cell_gll_index.resize(tile_bin.cell_indices.size() * (size_t)layout.n_node_per_cell);
    for (size_t cell_index = 0; cell_index < tile_bin.cell_indices.size(); ++cell_index) {
        for (int64_t point = 0; point < layout.n_node_per_cell; ++point) {
            int32_t global_index =
                layout.cell_gll_node_index[(size_t)tile_bin.cell_indices[cell_index] *
                                               (size_t)layout.n_node_per_cell +
                                           (size_t)point];
            plan.cell_gll_index[cell_index * (size_t)layout.n_node_per_cell + (size_t)point] =
                global_index >= 0 ? static_cast<int32_t>(global_to_tile.at(global_index)) : -1;
        }
    }

    plan.node_ids.resize(plan.gll_indices.size());
    plan.node_coords.resize(plan.gll_indices.size() * 3);
    for (size_t local_index = 0; local_index < plan.gll_indices.size(); ++local_index) {
        int64_t global_index = plan.gll_indices[local_index];
        plan.node_ids[local_index] = layout.gll_node_ids[(size_t)global_index];
        for (int component = 0; component < 3; ++component) {
            plan.node_coords[local_index * 3 + (size_t)component] =
                layout.gll_node_coords[(size_t)global_index * 3 + (size_t)component];
        }
    }
    return plan;
}

template <typename T>
static void write_index_dataset(hid_t file, const char* name, hid_t type,
                                const std::vector<T>& values,
                                const std::vector<hsize_t>& dimensions) {
    hid_t space =
        H5Screate_simple(static_cast<int>(dimensions.size()), dimensions.data(), nullptr);
    hid_t dataset = H5Dcreate2(file, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0) {
        H5Sclose(space);
        throw std::runtime_error(std::string("cannot create tile index dataset: ") + name);
    }
    if (!values.empty())
        H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    H5Dclose(dataset);
    H5Sclose(space);
}

static void write_index_attribute(hid_t file, const char* name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attribute = H5Acreate2(file, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, H5T_NATIVE_INT, &value);
    H5Aclose(attribute);
    H5Sclose(space);
}

static std::filesystem::path tile_index_path(const std::filesystem::path& directory,
                                             const TileKey& key) {
    char filename[64];
    std::snprintf(filename, sizeof(filename), "tile_index_x%03d_y%03d.h5", key.tx, key.ty);
    return directory / filename;
}

static void write_tile_index(const std::filesystem::path& path, const TilePlan& plan) {
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
        throw std::runtime_error("cannot create tile index: " + path.string());

    write_index_attribute(file, "schema_version", 2);
    write_index_attribute(file, "tile_x_index", plan.key.tx);
    write_index_attribute(file, "tile_y_index", plan.key.ty);
    write_index_attribute(file, "n_node_per_cell", static_cast<int>(plan.n_node_per_cell));

    write_index_dataset(file, "rank_ids", H5T_NATIVE_INT32, plan.rank_ids, {plan.rank_ids.size()});
    write_index_dataset(file, "rank_entry_offsets", H5T_NATIVE_INT64, plan.rank_entry_offsets,
                        {plan.rank_entry_offsets.size()});
    write_index_dataset(file, "record_cell_point_index", H5T_NATIVE_INT64,
                        plan.record_cell_point_index, {plan.record_cell_point_index.size()});
    write_index_dataset(file, "tile_local_node_index", H5T_NATIVE_INT32,
                        plan.tile_local_node_index, {plan.tile_local_node_index.size()});
    write_index_dataset(file, "cell_mass_index", H5T_NATIVE_INT64, plan.cell_mass_index,
                        {plan.cell_mass_index.size()});
    write_index_dataset(file, "gll_node_ids", H5T_NATIVE_INT64, plan.node_ids,
                        {plan.node_ids.size()});
    write_index_dataset(file, "gll_node_coords", H5T_NATIVE_DOUBLE, plan.node_coords,
                        {plan.node_ids.size(), 3});
    write_index_dataset(file, "cell_gll_node_index", H5T_NATIVE_INT32, plan.cell_gll_index,
                        {plan.cell_gll_index.size() / (size_t)plan.n_node_per_cell,
                         (hsize_t)plan.n_node_per_cell});
    H5Fclose(file);
}

static TilePlan read_tile_index(const std::filesystem::path& path) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
        throw std::runtime_error("cannot open tile index: " + path.string());

    TilePlan plan;
    int schema_version = 0;
    int n_node_per_cell = 0;
    if (!read_int_attribute(file, "schema_version", schema_version) || schema_version != 2 ||
        !read_int_attribute(file, "tile_x_index", plan.key.tx) ||
        !read_int_attribute(file, "tile_y_index", plan.key.ty) ||
        !read_int_attribute(file, "n_node_per_cell", n_node_per_cell)) {
        H5Fclose(file);
        throw std::runtime_error("invalid tile index schema: " + path.string());
    }
    plan.n_node_per_cell = n_node_per_cell;

    hsize_t value_count = 0;
    plan.rank_ids = read_flat_dataset<int32_t>(file, "rank_ids", H5T_NATIVE_INT32, value_count);
    plan.rank_entry_offsets =
        read_flat_dataset<int64_t>(file, "rank_entry_offsets", H5T_NATIVE_INT64, value_count);
    plan.record_cell_point_index =
        read_flat_dataset<int64_t>(file, "record_cell_point_index", H5T_NATIVE_INT64, value_count);
    plan.tile_local_node_index =
        read_flat_dataset<int32_t>(file, "tile_local_node_index", H5T_NATIVE_INT32, value_count);
    plan.cell_mass_index =
        read_flat_dataset<int64_t>(file, "cell_mass_index", H5T_NATIVE_INT64, value_count);
    plan.node_ids =
        read_flat_dataset<int64_t>(file, "gll_node_ids", H5T_NATIVE_INT64, value_count);
    plan.node_coords =
        read_flat_dataset<double>(file, "gll_node_coords", H5T_NATIVE_DOUBLE, value_count);
    plan.cell_gll_index =
        read_flat_dataset<int32_t>(file, "cell_gll_node_index", H5T_NATIVE_INT32, value_count);
    H5Fclose(file);

    if (plan.rank_entry_offsets.size() != plan.rank_ids.size() + 1 ||
        plan.record_cell_point_index.size() != plan.tile_local_node_index.size() ||
        plan.record_cell_point_index.size() != plan.cell_mass_index.size() ||
        plan.rank_entry_offsets.empty() || plan.rank_entry_offsets.front() != 0 ||
        plan.rank_entry_offsets.back() !=
            static_cast<int64_t>(plan.record_cell_point_index.size()))
        throw std::runtime_error("inconsistent tile index offsets: " + path.string());
    return plan;
}

// -----------------------------------------------------------------------
// Per-direction worker-tile field arrays (second pass)
// -----------------------------------------------------------------------

struct DirFields {
    std::vector<double> strain;  // [n_steps, n_local, 6]
#ifdef DEBUG
    std::vector<double> displacement;  // [n_steps, n_local, 3]
    std::vector<double> velocity;      // [n_steps, n_local, 3]
    std::vector<double> acceleration;  // [n_steps, n_local, 3]
#endif
};

#ifdef DEBUG
struct PostprocessProfile {
    bool enabled = std::getenv("GF_POST_PROFILE") != nullptr;
    double argument_parse_s = 0.0;
    double config_read_s = 0.0;
    double model_read_s = 0.0;
    double mass_read_s = 0.0;
    double record_scan_s = 0.0;
    double layout_merge_s = 0.0;
    double mass_index_s = 0.0;
    double record_index_s = 0.0;
    double tile_bin_s = 0.0;
    double broadcast_s = 0.0;
    double time_stf_s = 0.0;
    double tile_plan_s = 0.0;
    double tile_allocation_s = 0.0;
    double field_allocation_s = 0.0;
    double step_buffer_init_s = 0.0;
    double file_open_s = 0.0;
    double file_close_s = 0.0;
    double hdf5_read_s = 0.0;
    double field_read_s[4] = {0.0, 0.0, 0.0, 0.0};
    double aggregation_s = 0.0;
    double field_aggregation_s[4] = {0.0, 0.0, 0.0, 0.0};
    double normalization_s = 0.0;
    double assembly_s = 0.0;
    double output_prepare_s = 0.0;
    double write_s = 0.0;
    double mpi_finalize_s = 0.0;
    uint64_t files_opened = 0;
    uint64_t dataset_reads = 0;
    uint64_t cell_values_visited = 0;
};

static double profile_now() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

#define GF_POST_PROFILE_ENABLED(profile) ((profile).enabled)
#define GF_POST_PROFILE_START(name, profile) \
    const double name = (profile).enabled ? profile_now() : 0.0
#define GF_POST_PROFILE_PTR_START(name, profile) \
    const double name = ((profile) && (profile)->enabled) ? profile_now() : 0.0
#define GF_POST_PROFILE(profile, ...) \
    do {                              \
        if ((profile).enabled) {      \
            __VA_ARGS__;              \
        }                             \
    } while (0)
#define GF_POST_PROFILE_PTR(profile, ...)      \
    do {                                       \
        if ((profile) && (profile)->enabled) { \
            __VA_ARGS__;                       \
        }                                      \
    } while (0)
#else
struct PostprocessProfile {};
#define GF_POST_PROFILE_ENABLED(profile) false
#define GF_POST_PROFILE_START(name, profile)
#define GF_POST_PROFILE_PTR_START(name, profile)
#define GF_POST_PROFILE(profile, ...)
#define GF_POST_PROFILE_PTR(profile, ...)
#endif

struct TileRankRoute {
    size_t tile_index = 0;
    int64_t entry_begin = 0;
    int64_t entry_end = 0;
};

static std::vector<std::vector<TileRankRoute>> build_worker_record_routes(
    const std::vector<TilePlan>& plans, const LayoutMetadata& layout) {
    std::unordered_map<int, size_t> rank_to_mapping;
    for (size_t mapping_index = 0; mapping_index < layout.rank_maps.size(); ++mapping_index)
        rank_to_mapping[layout.rank_maps[mapping_index].rank] = mapping_index;

    std::vector<std::vector<TileRankRoute>> routes(layout.rank_maps.size());
    for (size_t tile_index = 0; tile_index < plans.size(); ++tile_index) {
        const auto& plan = plans[tile_index];
        for (size_t rank_position = 0; rank_position < plan.rank_ids.size(); ++rank_position) {
            const auto mapping = rank_to_mapping.find(plan.rank_ids[rank_position]);
            if (mapping == rank_to_mapping.end())
                continue;
            routes[mapping->second].push_back({tile_index, plan.rank_entry_offsets[rank_position],
                                               plan.rank_entry_offsets[rank_position + 1]});
        }
    }
    return routes;
}

struct TileStepCounts {
    std::vector<double> strain_weights;
#ifdef DEBUG
    std::vector<int32_t> displacement;
    std::vector<int32_t> velocity;
    std::vector<int32_t> acceleration;
#endif
};

// Phase 3: read every record once per worker and scatter it to all assigned
// tiles that depend on the record rank.
static std::vector<DirFields> extract_worker_tile_fields(
    const DirectionRecords& records, const LayoutMetadata& layout,
    const std::vector<TilePlan>& plans, const std::vector<std::vector<TileRankRoute>>& routes,
    const std::vector<double>& cell_mass, PostprocessProfile* profile) {
    const int64_t n_steps = static_cast<int64_t>(records.steps.size());
    std::vector<DirFields> results(plans.size());

    GF_POST_PROFILE_PTR_START(field_allocation_start, profile);
    for (size_t tile_index = 0; tile_index < plans.size(); ++tile_index) {
        const size_t node_count = plans[tile_index].node_ids.size();
        results[tile_index].strain.resize((size_t)n_steps * node_count * 6, 0.0);
#ifdef DEBUG
        if (records.has_displacement)
            results[tile_index].displacement.resize((size_t)n_steps * node_count * 3, 0.0);
        if (records.has_velocity)
            results[tile_index].velocity.resize((size_t)n_steps * node_count * 3, 0.0);
        if (records.has_acceleration)
            results[tile_index].acceleration.resize((size_t)n_steps * node_count * 3, 0.0);
#endif
    }
    GF_POST_PROFILE_PTR(profile,
                        profile->field_allocation_s += profile_now() - field_allocation_start);

    for (int64_t snap_idx = 0; snap_idx < n_steps; ++snap_idx) {
        GF_POST_PROFILE_PTR_START(step_buffer_init_start, profile);
        std::vector<TileStepCounts> counts(plans.size());
        for (size_t tile_index = 0; tile_index < plans.size(); ++tile_index) {
            const size_t node_count = plans[tile_index].node_ids.size();
            counts[tile_index].strain_weights.resize(node_count, 0.0);
#ifdef DEBUG
            if (records.has_displacement)
                counts[tile_index].displacement.resize(node_count, 0);
            if (records.has_velocity)
                counts[tile_index].velocity.resize(node_count, 0);
            if (records.has_acceleration)
                counts[tile_index].acceleration.resize(node_count, 0);
#endif
        }
        GF_POST_PROFILE_PTR(profile,
                            profile->step_buffer_init_s += profile_now() - step_buffer_init_start);

        for (size_t mapping_index = 0; mapping_index < routes.size(); ++mapping_index) {
            if (routes[mapping_index].empty())
                continue;
            const auto& record_path = records.record_paths[(size_t)snap_idx][mapping_index];
            if (record_path.empty())
                continue;

            GF_POST_PROFILE_PTR_START(file_open_start, profile);
            hid_t file = H5Fopen(record_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            GF_POST_PROFILE_PTR(profile, profile->file_open_s += profile_now() - file_open_start);
            if (file < 0)
                continue;
            GF_POST_PROFILE_PTR(profile, ++profile->files_opened);

            hsize_t recording_cell_count = 0;
            hsize_t nodes_per_cell = 0;
            std::vector<double> field_buffer;
            const auto& selected_cells = layout.rank_maps[mapping_index].record_cell_indices;
            GF_POST_PROFILE_PTR_START(strain_io_start, profile);
            read_field_cells_4d(file, "strain", 6, selected_cells, recording_cell_count,
                                nodes_per_cell, field_buffer);
            GF_POST_PROFILE_PTR(profile, ++profile->dataset_reads;
                                const double elapsed_s = profile_now() - strain_io_start;
                                profile->hdf5_read_s += elapsed_s;
                                profile->field_read_s[0] += elapsed_s;);

            GF_POST_PROFILE_PTR_START(aggregation_start, profile);
            for (const auto& route : routes[mapping_index]) {
                const auto& plan = plans[route.tile_index];
                const size_t node_count = plan.node_ids.size();
                double* destination_step =
                    results[route.tile_index].strain.data() + (size_t)snap_idx * node_count * 6;
                for (int64_t entry = route.entry_begin; entry < route.entry_end; ++entry) {
                    GF_POST_PROFILE_PTR(profile, ++profile->cell_values_visited);
                    const int64_t point_index = plan.record_cell_point_index[(size_t)entry];
                    const hsize_t cell = point_index / plan.n_node_per_cell;
                    const hsize_t point = point_index % plan.n_node_per_cell;
                    if (cell >= recording_cell_count || point >= nodes_per_cell)
                        continue;
                    const int32_t tile_node = plan.tile_local_node_index[(size_t)entry];
                    double weight = 1.0;
                    const int64_t mass_index = plan.cell_mass_index[(size_t)entry];
                    if (mass_index >= 0 && (size_t)mass_index < cell_mass.size())
                        weight = cell_mass[(size_t)mass_index];
                    const double* source =
                        field_buffer.data() + (cell * nodes_per_cell + point) * 6;
                    double* destination = destination_step + (size_t)tile_node * 6;
                    for (int component = 0; component < 6; ++component)
                        destination[component] += source[component] * weight;
                    counts[route.tile_index].strain_weights[(size_t)tile_node] += weight;
                }
            }
            GF_POST_PROFILE_PTR(profile,
                                const double elapsed_s = profile_now() - aggregation_start;
                                profile->aggregation_s += elapsed_s;
                                profile->field_aggregation_s[0] += elapsed_s;);

#ifdef DEBUG
            auto read_and_accumulate_vector = [&](const char* dataset_name, int field_index,
                                                  std::vector<DirFields>& tile_results) {
                hsize_t field_cell_count = 0;
                hsize_t field_nodes_per_cell = 0;
                const double field_io_start = profile && profile->enabled ? profile_now() : 0.0;
                read_field_cells_4d(file, dataset_name, 3, selected_cells, field_cell_count,
                                    field_nodes_per_cell, field_buffer);
                if (profile && profile->enabled) {
                    ++profile->dataset_reads;
                    const double elapsed_s = profile_now() - field_io_start;
                    profile->hdf5_read_s += elapsed_s;
                    profile->field_read_s[field_index] += elapsed_s;
                }

                const double field_aggregation_start =
                    profile && profile->enabled ? profile_now() : 0.0;
                for (const auto& route : routes[mapping_index]) {
                    const auto& plan = plans[route.tile_index];
                    const size_t node_count = plan.node_ids.size();
                    std::vector<double>* values = nullptr;
                    std::vector<int32_t>* sample_counts = nullptr;
                    if (field_index == 1) {
                        values = &tile_results[route.tile_index].displacement;
                        sample_counts = &counts[route.tile_index].displacement;
                    } else if (field_index == 2) {
                        values = &tile_results[route.tile_index].velocity;
                        sample_counts = &counts[route.tile_index].velocity;
                    } else {
                        values = &tile_results[route.tile_index].acceleration;
                        sample_counts = &counts[route.tile_index].acceleration;
                    }
                    double* destination_step = values->data() + (size_t)snap_idx * node_count * 3;
                    for (int64_t entry = route.entry_begin; entry < route.entry_end; ++entry) {
                        const int64_t point_index = plan.record_cell_point_index[(size_t)entry];
                        const hsize_t cell = point_index / plan.n_node_per_cell;
                        const hsize_t point = point_index % plan.n_node_per_cell;
                        if (cell >= field_cell_count || point >= field_nodes_per_cell)
                            continue;
                        const int32_t tile_node = plan.tile_local_node_index[(size_t)entry];
                        const double* source =
                            field_buffer.data() + (cell * field_nodes_per_cell + point) * 3;
                        double* destination = destination_step + (size_t)tile_node * 3;
                        for (int component = 0; component < 3; ++component)
                            destination[component] += source[component];
                        ++(*sample_counts)[(size_t)tile_node];
                    }
                }
                if (profile && profile->enabled) {
                    const double elapsed_s = profile_now() - field_aggregation_start;
                    profile->aggregation_s += elapsed_s;
                    profile->field_aggregation_s[field_index] += elapsed_s;
                }
            };
            if (records.has_displacement)
                read_and_accumulate_vector("displacement", 1, results);
            if (records.has_velocity)
                read_and_accumulate_vector("velocity", 2, results);
            if (records.has_acceleration)
                read_and_accumulate_vector("acceleration", 3, results);
#endif

            GF_POST_PROFILE_PTR_START(file_close_start, profile);
            H5Fclose(file);
            GF_POST_PROFILE_PTR(profile,
                                profile->file_close_s += profile_now() - file_close_start);
        }

        GF_POST_PROFILE_PTR_START(normalization_start, profile);
        for (size_t tile_index = 0; tile_index < plans.size(); ++tile_index) {
            const size_t node_count = plans[tile_index].node_ids.size();
            double* strain_step =
                results[tile_index].strain.data() + (size_t)snap_idx * node_count * 6;
            for (size_t node = 0; node < node_count; ++node) {
                const double weight = counts[tile_index].strain_weights[node];
                if (weight > 0.0) {
                    for (int component = 0; component < 6; ++component)
                        strain_step[node * 6 + (size_t)component] /= weight;
                }
            }
#ifdef DEBUG
            auto normalize_vector = [&](std::vector<double>& values,
                                        const std::vector<int32_t>& sample_counts) {
                if (values.empty())
                    return;
                double* step = values.data() + (size_t)snap_idx * node_count * 3;
                for (size_t node = 0; node < node_count; ++node) {
                    if (sample_counts[node] == 0)
                        continue;
                    const double inverse_count = 1.0 / sample_counts[node];
                    for (int component = 0; component < 3; ++component)
                        step[node * 3 + (size_t)component] *= inverse_count;
                }
            };
            normalize_vector(results[tile_index].displacement, counts[tile_index].displacement);
            normalize_vector(results[tile_index].velocity, counts[tile_index].velocity);
            normalize_vector(results[tile_index].acceleration, counts[tile_index].acceleration);
#endif
        }
        GF_POST_PROFILE_PTR(profile,
                            profile->normalization_s += profile_now() - normalization_start);
    }
    return results;
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------

int main(int argc, char** argv) {
    uint64_t total_field_memory_budget = 0;
    try {
        total_field_memory_budget = gf_postprocess_common::postprocess_memory_budget_bytes(
            std::getenv("GF_POST_MEMORY_GB"));
    } catch (const std::invalid_argument& error) {
        fprintf(stderr, "[postprocess] Error: %s\n", error.what());
        return 1;
    }

    int worker_rank = 0;
    int worker_count = 1;
#ifdef GF_POST_MPI
    // ---- Optional MPI init ----
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &worker_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worker_count);
    GF_POST_DEBUG_LOG("[postprocess] MPI rank %d/%d\n", worker_rank, worker_count);
    // Stagger file access to avoid HDF5 metadata contention
    MPI_Barrier(MPI_COMM_WORLD);
    usleep((unsigned int)(worker_rank * 200000));  // 200ms stagger
    MPI_Barrier(MPI_COMM_WORLD);
#else
    GF_POST_DEBUG_LOG("[postprocess] Serial tile worker 0/1\n");
#endif

    double start = 0.0;
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        start = ts.tv_sec + ts.tv_nsec * 1e-9;
    }

    PostprocessProfile profile;
    if (GF_POST_PROFILE_ENABLED(profile))
        fprintf(stderr, "[postprocess] Profiling enabled (GF_POST_PROFILE=1)\n");

    if (worker_rank == 0)
        fprintf(stderr, "[postprocess] Starting...\n");

    GF_POST_PROFILE_START(argument_parse_start, profile);
    auto args = gf_postprocess_common::parse_args(argc, argv, print_usage);
    GF_POST_PROFILE(profile, profile.argument_parse_s += profile_now() - argument_parse_start);

    // ---- Read config ----
    GF_POST_DEBUG_LOG("[postprocess] Reading config from %s\n", args.config_path.c_str());
    GF_POST_PROFILE_START(config_read_start, profile);
    ConfigParams cfg = read_config(args.config_path.c_str());
    GF_POST_PROFILE(profile, profile.config_read_s += profile_now() - config_read_start);

    // ---- Read mesh ----
    GF_POST_DEBUG_LOG("[postprocess] Reading mesh geometry from %s\n", args.model_path.c_str());
    GF_POST_PROFILE_START(model_read_start, profile);
    ModelData model = read_model(args.model_path.c_str());
    GF_POST_PROFILE(profile, profile.model_read_s += profile_now() - model_read_start);
    int64_t n_vertex = model.n_vertex;  // kept for domain bounds
    GF_POST_DEBUG_LOG("[postprocess]   domain vertex count = %lld\n", (long long)n_vertex);

    // ---- Read cell mass from model.h5 for L2 projection ----
    std::vector<double> cell_mass;
    int64_t n_model_cell = 0;
    int ngll_model = 0;
    GF_POST_PROFILE_START(mass_read_start, profile);
    gf_postprocess_common::read_cell_mass(args.model_path.c_str(), cell_mass, n_model_cell,
                                          ngll_model);
    GF_POST_PROFILE(profile, profile.mass_read_s += profile_now() - mass_read_start);

    // ---- Phase 1: rank 0 builds all shared indexes before field processing ----
    LayoutMetadata layout;
    DirectionRecords directions[3];
    std::vector<TileBin> tile_bins;
    std::vector<int> tile_assignments;
    auto case_directory = std::filesystem::path(args.model_path).parent_path();
    if (case_directory.empty())
        case_directory = ".";
    const auto partition_directory = case_directory / "partitions";
    auto x_record_directory = std::filesystem::path(args.fx_dir).lexically_normal();
    if (x_record_directory.filename().empty())
        x_record_directory = x_record_directory.parent_path();
    const auto tile_index_directory = x_record_directory.parent_path() / "tile_indexes";
    if (worker_rank == 0) {
        GF_POST_PROFILE_START(record_scan_start, profile);
        directions[0] = scan_direction_records(args.fx_dir.c_str());
        directions[1] = scan_direction_records(args.fy_dir.c_str());
        directions[2] = scan_direction_records(args.fz_dir.c_str());
        GF_POST_PROFILE(profile, profile.record_scan_s += profile_now() - record_scan_start);

        GF_POST_PROFILE_START(layout_merge_start, profile);
        layout = merge_partition_metadata(partition_directory, directions[0]);
        GF_POST_PROFILE(profile, profile.layout_merge_s += profile_now() - layout_merge_start);

        GF_POST_PROFILE_START(mass_index_start, profile);
        build_mass_indexes(layout, n_model_cell, ngll_model);
        GF_POST_PROFILE(profile, profile.mass_index_s += profile_now() - mass_index_start);

        GF_POST_PROFILE_START(record_index_start, profile);
        for (auto& direction : directions)
            build_direction_record_index(direction, layout);
        GF_POST_PROFILE(profile, profile.record_index_s += profile_now() - record_index_start);

        if (directions[0].steps != directions[1].steps ||
            directions[0].steps != directions[2].steps) {
            fprintf(stderr, "ERROR: mismatched output steps across force directions\n");
#ifdef GF_POST_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            return 1;
#endif
        }
        GF_POST_PROFILE_START(tile_bin_start, profile);
        tile_bins = build_tile_bins(layout, cfg, model);
        GF_POST_PROFILE(profile, profile.tile_bin_s += profile_now() - tile_bin_start);

        GF_POST_PROFILE_START(tile_plan_start, profile);
        std::filesystem::create_directories(tile_index_directory);
        for (const auto& entry : std::filesystem::directory_iterator(tile_index_directory)) {
            const std::string filename = entry.path().filename().string();
            if (entry.is_regular_file() && filename.rfind("tile_index_", 0) == 0 &&
                entry.path().extension() == ".h5")
                std::filesystem::remove(entry.path());
        }
        std::unordered_map<int, uint64_t> rank_read_weights;
        for (const auto& mapping : layout.rank_maps)
            rank_read_weights[mapping.rank] = static_cast<uint64_t>(mapping.cell_gll_idx.size());
        std::vector<gf_postprocess_common::TileAssignmentInput> assignment_inputs;
        assignment_inputs.reserve(tile_bins.size());
        for (const auto& tile_bin : tile_bins) {
            const TilePlan plan = build_tile_plan(tile_bin, layout);
            write_tile_index(tile_index_path(tile_index_directory, plan.key), plan);
            gf_postprocess_common::TileAssignmentInput input;
            input.work_weight = static_cast<uint64_t>(plan.record_cell_point_index.size()) * 15 +
                                static_cast<uint64_t>(plan.node_ids.size()) * 45;
            for (int rank : plan.rank_ids)
                input.rank_read_weights.emplace_back(rank, rank_read_weights.at(rank));
            assignment_inputs.push_back(std::move(input));
        }
        tile_assignments =
            gf_postprocess_common::assign_tiles_by_rank_overlap(assignment_inputs, worker_count);
        GF_POST_PROFILE(profile, profile.tile_plan_s += profile_now() - tile_plan_start);
    }

#ifdef GF_POST_MPI
    GF_POST_PROFILE_START(broadcast_start, profile);
    broadcast_layout(layout, worker_rank);
    for (auto& direction : directions)
        broadcast_direction_records(direction, layout, worker_rank);
    broadcast_tile_bins(tile_bins, worker_rank);
    broadcast_vector(tile_assignments, MPI_INT, worker_rank);
    GF_POST_PROFILE(profile, profile.broadcast_s += profile_now() - broadcast_start);
#endif

    int64_t n_steps = static_cast<int64_t>(directions[0].steps.size());

    // GLL node IDs (1-based, shared across directions)
    int64_t n_recorded = layout.n_unique_gll;
    if (worker_rank == 0)
        fprintf(stderr, "[postprocess] %lld unique GLL nodes recorded\n", (long long)n_recorded);

    if (n_recorded == 0) {
        fprintf(stderr, "ERROR: no GLL nodes recorded\n");
        return 1;
    }

    // ---- Build time array ----
    GF_POST_PROFILE_START(time_stf_start, profile);
    std::vector<double> time_arr((size_t)n_steps);
    for (int64_t s = 0; s < n_steps; ++s) {
        time_arr[(size_t)s] = (double)s * cfg.output_dt_s;
    }

    // ---- Downsample STF to tile time axis ----
    // config STF is at solver_dt; tile time axis is at output_dt_s.
    std::vector<double> stf_t_ds, stf_values_ds;
    gf_postprocess_common::downsample_stf(cfg, n_steps, stf_t_ds, stf_values_ds);
    GF_POST_PROFILE(profile, profile.time_stf_s += profile_now() - time_stf_start);

#ifdef DEBUG
    bool has_displacement = directions[0].has_displacement && directions[1].has_displacement &&
                            directions[2].has_displacement;
    bool has_velocity =
        directions[0].has_velocity && directions[1].has_velocity && directions[2].has_velocity;
    bool has_acceleration = directions[0].has_acceleration && directions[1].has_acceleration &&
                            directions[2].has_acceleration;
    fprintf(stderr, "[postprocess]   displacement=%s velocity=%s acceleration=%s\n",
            has_displacement ? "yes" : "no", has_velocity ? "yes" : "no",
            has_acceleration ? "yes" : "no");
#endif

    // ---- Phase 2: prepare assigned tile indexes before any field I/O ----
    double xmin = model.xmin, ymin = model.ymin, xmax = model.xmax, ymax = model.ymax;
    int64_t n_tiles = static_cast<int64_t>(tile_bins.size());
    if (worker_rank == 0)
        fprintf(stderr, "[postprocess]   %lld tiles\n", (long long)n_tiles);

    const size_t assigned_tile_count = static_cast<size_t>(
        std::count(tile_assignments.begin(), tile_assignments.end(), worker_rank));
    // ---- Worker exit check: idle workers exit BEFORE any field allocation ----
    if (assigned_tile_count == 0) {
        GF_POST_DEBUG_LOG("[postprocess] Worker %d: no tile assigned (n_tiles=%lld), exiting\n",
                          worker_rank, (long long)n_tiles);
#ifdef GF_POST_MPI
        MPI_Finalize();
#endif
        return 0;
    }

    GF_POST_PROFILE_START(tile_plan_load_start, profile);
    std::vector<TilePlan> assigned_tile_plans;
    assigned_tile_plans.reserve(assigned_tile_count);
    for (int64_t tile_position = 0; tile_position < n_tiles; ++tile_position) {
        if (tile_assignments[(size_t)tile_position] != worker_rank)
            continue;
        const TileKey key = tile_bins[(size_t)tile_position].key;
        assigned_tile_plans.push_back(read_tile_index(tile_index_path(tile_index_directory, key)));
    }
    GF_POST_PROFILE(profile, profile.tile_plan_s += profile_now() - tile_plan_load_start);
    GF_POST_DEBUG_LOG("[postprocess]   worker %d prepared %zu tile index plan(s)\n", worker_rank,
                      assigned_tile_plans.size());

    // Divide the aggregate output + one-direction field budget across active
    // workers. Each batch still maximizes record reuse within that budget.
    const int active_worker_count = std::min<int>(worker_count, static_cast<int>(n_tiles));
    const uint64_t worker_memory_budget =
        total_field_memory_budget / static_cast<uint64_t>(active_worker_count);
    if (worker_rank == 0) {
        constexpr double bytes_per_gib = 1024.0 * 1024.0 * 1024.0;
        fprintf(stderr, "[postprocess] Field memory budget: %llu GiB total, %.2f GiB/worker\n",
                static_cast<unsigned long long>(total_field_memory_budget / bytes_per_gib),
                static_cast<double>(worker_memory_budget) / bytes_per_gib);
    }
    std::vector<uint64_t> tile_memory_bytes;
    tile_memory_bytes.reserve(assigned_tile_plans.size());
#ifdef DEBUG
    constexpr uint64_t field_components_per_node = 60;
#else
    constexpr uint64_t field_components_per_node = 24;
#endif
    for (const auto& plan : assigned_tile_plans) {
        tile_memory_bytes.push_back(static_cast<uint64_t>(n_steps) * plan.node_ids.size() *
                                    field_components_per_node *
                                    static_cast<uint64_t>(sizeof(double)));
    }
    std::vector<std::vector<TilePlan>> tile_plan_batches;
    for (const auto& [batch_begin, batch_end] :
         gf_postprocess_common::plan_memory_batches(tile_memory_bytes, worker_memory_budget)) {
        auto& batch = tile_plan_batches.emplace_back();
        batch.reserve(batch_end - batch_begin);
        for (size_t tile_index = batch_begin; tile_index < batch_end; ++tile_index)
            batch.push_back(std::move(assigned_tile_plans[tile_index]));
    }
    assigned_tile_plans.clear();
    assigned_tile_plans.shrink_to_fit();
    GF_POST_DEBUG_LOG("[postprocess]   worker %d grouped tiles into %zu memory batch(es)\n",
                      worker_rank, tile_plan_batches.size());

    GF_POST_PROFILE_START(output_prepare_start, profile);
    std::string mkdir_cmd = "mkdir -p " + args.output_dir;
    if (system(mkdir_cmd.c_str()) != 0) {
        fprintf(stderr, "WARNING: could not create output directory %s\n",
                args.output_dir.c_str());
    }
    GF_POST_PROFILE(profile, profile.output_prepare_s += profile_now() - output_prepare_start);

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

    for (const auto& tile_plans : tile_plan_batches) {
        // ---- Allocate final arrays for as many assigned tiles as fit the
        //      worker memory budget. Each record feeds every tile in the batch. ----
        GF_POST_PROFILE_START(tile_allocation_start, profile);
        std::vector<std::vector<double>> tile_greens(tile_plans.size());
#ifdef DEBUG
        std::vector<std::vector<double>> tile_displacements(tile_plans.size());
        std::vector<std::vector<double>> tile_velocities(tile_plans.size());
        std::vector<std::vector<double>> tile_accelerations(tile_plans.size());
#endif
        for (size_t tile_index = 0; tile_index < tile_plans.size(); ++tile_index) {
            const size_t node_count = tile_plans[tile_index].node_ids.size();
            tile_greens[tile_index].resize((size_t)n_steps * node_count * 18, 0.0);
#ifdef DEBUG
            if (has_displacement)
                tile_displacements[tile_index].resize((size_t)n_steps * node_count * 9, 0.0);
            if (has_velocity)
                tile_velocities[tile_index].resize((size_t)n_steps * node_count * 9, 0.0);
            if (has_acceleration)
                tile_accelerations[tile_index].resize((size_t)n_steps * node_count * 9, 0.0);
#endif
        }
        GF_POST_PROFILE(profile,
                        profile.tile_allocation_s += profile_now() - tile_allocation_start);

        const auto record_routes = build_worker_record_routes(tile_plans, layout);
        // Direction 0 = fx, 1 = fy, 2 = fz.
        for (int direction = 0; direction < 3; ++direction) {
            auto fields = extract_worker_tile_fields(directions[direction], layout, tile_plans,
                                                     record_routes, cell_mass, &profile);
            GF_POST_PROFILE_START(assembly_start, profile);
            for (size_t tile_index = 0; tile_index < tile_plans.size(); ++tile_index) {
                const size_t node_count = tile_plans[tile_index].node_ids.size();
                for (int64_t step = 0; step < n_steps; ++step) {
                    for (size_t node = 0; node < node_count; ++node) {
                        const size_t tensor_offset = ((size_t)step * node_count + node) * 18;
                        const double* strain = fields[tile_index].strain.data() +
                                               ((size_t)step * node_count + node) * 6;
                        gf_postprocess_common::assign_strain_direction(
                            strain, tile_greens[tile_index].data() + tensor_offset, direction);
                    }
                }
#ifdef DEBUG
                auto assemble_vector_field = [&](const std::vector<double>& source,
                                                 std::vector<double>& destination) {
                    if (source.empty())
                        return;
                    for (int64_t step = 0; step < n_steps; ++step) {
                        for (size_t node = 0; node < node_count; ++node) {
                            const size_t tensor_offset = ((size_t)step * node_count + node) * 9;
                            const double* components =
                                source.data() + ((size_t)step * node_count + node) * 3;
                            for (int component = 0; component < 3; ++component) {
                                destination[tensor_offset + (size_t)component * 3 +
                                            (size_t)direction] = components[component];
                            }
                        }
                    }
                };
                assemble_vector_field(fields[tile_index].displacement,
                                      tile_displacements[tile_index]);
                assemble_vector_field(fields[tile_index].velocity, tile_velocities[tile_index]);
                assemble_vector_field(fields[tile_index].acceleration,
                                      tile_accelerations[tile_index]);
#endif
            }
            GF_POST_PROFILE(profile, profile.assembly_s += profile_now() - assembly_start);
        }

        // ---- Each tile remains exclusively owned and is written once. ----
        for (size_t tile_index = 0; tile_index < tile_plans.size(); ++tile_index) {
            const auto& plan = tile_plans[tile_index];
            const TileKey& key = plan.key;
            double source_xyz_m[3] = {cfg.source_x_m, cfg.source_y_m, cfg.source_z_m};

            double tx_min, tx_max, ty_min, ty_max;
            compute_tile_bounds(key, tx_min, tx_max, ty_min, ty_max);

            // Output precision follows config snapshot_precision
            bool use_float32 = (cfg.snapshot_precision == "float32");

            char fname[256];
            std::snprintf(fname, sizeof(fname), "%s/tile_x%03d_y%03d.h5", args.output_dir.c_str(),
                          key.tx, key.ty);

            GF_POST_PROFILE_START(write_start, profile);
#ifdef DEBUG
            write_tile(fname, key.tx, key.ty, tx_min, tx_max, ty_min, ty_max, zmin, zmax,
                       cfg.record_depth_max_m, cfg.record_depth_actual_m, plan.node_ids, time_arr,
                       cfg.solver_dt, tile_greens[tile_index], source_xyz_m, plan.node_coords,
                       plan.cell_gll_index, (int)plan.n_node_per_cell,
                       has_displacement ? tile_displacements[tile_index].data() : nullptr,
                       has_velocity ? tile_velocities[tile_index].data() : nullptr,
                       has_acceleration ? tile_accelerations[tile_index].data() : nullptr,
                       stf_t_ds, stf_values_ds, use_float32);
#else
            write_tile(fname, key.tx, key.ty, tx_min, tx_max, ty_min, ty_max, zmin, zmax,
                       cfg.record_depth_max_m, cfg.record_depth_actual_m, plan.node_ids, time_arr,
                       cfg.solver_dt, tile_greens[tile_index], source_xyz_m, plan.node_coords,
                       plan.cell_gll_index, (int)plan.n_node_per_cell, stf_t_ds, stf_values_ds,
                       use_float32);
#endif
            GF_POST_PROFILE(profile, profile.write_s += profile_now() - write_start);
            GF_POST_DEBUG_LOG("[postprocess]   worker %d wrote tile x%03d y%03d\n", worker_rank,
                              key.tx, key.ty);
        }
    }

#ifdef GF_POST_MPI
    GF_POST_PROFILE_START(mpi_finalize_start, profile);
    MPI_Finalize();
    GF_POST_PROFILE(profile, profile.mpi_finalize_s += profile_now() - mpi_finalize_start);
#endif

    // ---- Print machine-parseable stats ----
    gf_postprocess_common::print_stats(start, n_steps, n_vertex, n_recorded, n_tiles);
#ifdef DEBUG
    if (profile.enabled) {
        const double total_s = profile_now() - start;
        const double measured_s =
            profile.argument_parse_s + profile.config_read_s + profile.model_read_s +
            profile.mass_read_s + profile.record_scan_s + profile.layout_merge_s +
            profile.mass_index_s + profile.record_index_s + profile.tile_bin_s +
            profile.broadcast_s + profile.time_stf_s + profile.tile_plan_s +
            profile.output_prepare_s + profile.tile_allocation_s + profile.field_allocation_s +
            profile.step_buffer_init_s + profile.file_open_s + profile.hdf5_read_s +
            profile.aggregation_s + profile.normalization_s + profile.assembly_s +
            profile.file_close_s + profile.write_s + profile.mpi_finalize_s;
        const double other_s = std::max(0.0, total_s - measured_s);
        const auto percentage = [total_s](double elapsed_s) {
            return total_s > 0.0 ? 100.0 * elapsed_s / total_s : 0.0;
        };
        fprintf(stderr,
                "[profile] rank=%d hdf5_read=%.3fs aggregation=%.3fs normalization=%.3fs "
                "assembly=%.3fs write=%.3fs files=%llu dataset_reads=%llu cell_values=%llu\n",
                worker_rank, profile.hdf5_read_s, profile.aggregation_s, profile.normalization_s,
                profile.assembly_s, profile.write_s,
                static_cast<unsigned long long>(profile.files_opened),
                static_cast<unsigned long long>(profile.dataset_reads),
                static_cast<unsigned long long>(profile.cell_values_visited));
        fprintf(stderr,
                "[profile] rank=%d percentages: hdf5_read=%.1f%% aggregation=%.1f%% "
                "normalization=%.1f%% assembly=%.1f%% write=%.1f%% other=%.1f%% "
                "total=%.3fs\n",
                worker_rank, percentage(profile.hdf5_read_s), percentage(profile.aggregation_s),
                percentage(profile.normalization_s), percentage(profile.assembly_s),
                percentage(profile.write_s), percentage(other_s), total_s);
        const auto print_stage = [&](const char* name, double elapsed_s) {
            fprintf(stderr, "[profile] rank=%d stage=%s seconds=%.6f percent=%.3f%%\n",
                    worker_rank, name, elapsed_s, percentage(elapsed_s));
        };
        print_stage("argument_parse", profile.argument_parse_s);
        print_stage("config_read", profile.config_read_s);
        print_stage("model_read", profile.model_read_s);
        print_stage("mass_read", profile.mass_read_s);
        print_stage("record_scan", profile.record_scan_s);
        print_stage("layout_merge", profile.layout_merge_s);
        print_stage("mass_index", profile.mass_index_s);
        print_stage("record_index", profile.record_index_s);
        print_stage("tile_bin", profile.tile_bin_s);
        print_stage("broadcast", profile.broadcast_s);
        print_stage("time_stf", profile.time_stf_s);
        print_stage("tile_plan", profile.tile_plan_s);
        print_stage("output_prepare", profile.output_prepare_s);
        print_stage("tile_allocation", profile.tile_allocation_s);
        print_stage("field_allocation", profile.field_allocation_s);
        print_stage("step_buffer_init", profile.step_buffer_init_s);
        print_stage("file_open", profile.file_open_s);
        print_stage("strain_read", profile.field_read_s[0]);
        print_stage("displacement_read", profile.field_read_s[1]);
        print_stage("velocity_read", profile.field_read_s[2]);
        print_stage("acceleration_read", profile.field_read_s[3]);
        print_stage("strain_aggregation", profile.field_aggregation_s[0]);
        print_stage("displacement_aggregation", profile.field_aggregation_s[1]);
        print_stage("velocity_aggregation", profile.field_aggregation_s[2]);
        print_stage("acceleration_aggregation", profile.field_aggregation_s[3]);
        print_stage("normalization", profile.normalization_s);
        print_stage("assembly", profile.assembly_s);
        print_stage("file_close", profile.file_close_s);
        print_stage("write", profile.write_s);
        print_stage("mpi_finalize", profile.mpi_finalize_s);
        print_stage("unclassified", other_s);
    }
#endif

    return 0;
}
