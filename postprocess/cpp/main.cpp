/* postprocess/cpp/main.cpp - tile-batched Green's function postprocessor
 *
 * CLI:
 *   gf_postprocess[_mpi] <model.h5> <config.h5> \
 *       --fx <dir> --fy <dir> --fz <dir> -o <output_dir>
 *
 * Both executables use the same tile-local pipeline. The serial executable
 * processes every tile with worker 0/1. The MPI executable distributes tiles
 * round-robin so each tile is written exactly once.
 *
 * Memory design:
 *   Phase 1 - rank 0 rebuilds layouts from partitions and prepares shared indexes;
 *             MPI broadcasts them without rebuilding on other workers.
 *   Phase 2 - each worker prepares indexes for its assigned tiles before field I/O.
 *   Phase 3 - extract_tile_fields(): each worker reads records but accumulates
 *             field data only for its tiles' nodes (~1 GB/worker per tile, freed
 *             between tiles). All sharing cells are still iterated so averaging
 *             stays correct.
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
            "The MPI build distributes tiles round-robin; the serial build processes\n"
            "the same tiles sequentially.\n"
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
// Per-rank mapping (local GLL node id -> global merged index)
// -----------------------------------------------------------------------

struct RankMapping {
    int rank = -1;
    std::vector<int32_t> local_to_global;
    hsize_t n_rec_cell = 0;
    hsize_t nnodes = 0;
    std::vector<int32_t> cell_gll_idx;         // [n_rec_cell * n_node]
    std::vector<int64_t> rec_cell_model_idx;   // [n_rec_cell]
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
    bool has_displacement = false;
    bool has_velocity = false;
    bool has_acceleration = false;
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

    hid_t probe = H5Fopen(files[0].path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (probe >= 0) {
        result.has_displacement = H5Lexists(probe, "displacement", H5P_DEFAULT) > 0;
        result.has_velocity = H5Lexists(probe, "velocity", H5P_DEFAULT) > 0;
        result.has_acceleration = H5Lexists(probe, "acceleration", H5P_DEFAULT) > 0;
        H5Fclose(probe);
    }
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
    fprintf(stderr, "[postprocess] Building recording layout from %s...\n", partition_dir.c_str());

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

        for (int partition = partition_start;
             partition < partition_start + assigned_partition_count; ++partition) {
            const auto path = partition_dir / ("partition_" + std::to_string(partition) + ".h5");
            hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            hid_t recording = -1;
            if (file >= 0) {
                H5E_BEGIN_TRY {
                    recording = H5Gopen2(file, "/recording", H5P_DEFAULT);
                }
                H5E_END_TRY;
            }
            if (recording < 0) {
                if (file >= 0)
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
            hsize_t cell_point_count = 0;
            auto partition_cell_nodes = read_flat_dataset<int32_t>(
                recording, "cell_gll_node_index", H5T_NATIVE_INT32, cell_point_count);
            H5Gclose(recording);
            H5Fclose(file);

            if (recording_cell_count == 0)
                continue;
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
    fprintf(stderr, "[postprocess]   %lld merged recording cells\n",
            (long long)result.n_rec_cell_merged);
    fprintf(stderr, "[postprocess]   %lld unique GLL nodes from %zu output rank(s)\n",
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
        broadcast_vector(mapping.cell_point_mass_idx, MPI_INT64_T, worker_rank);
    }
}

static void broadcast_direction_records(DirectionRecords& records, const LayoutMetadata& layout,
                                        int worker_rank) {
    int flags[3] = {records.has_displacement ? 1 : 0, records.has_velocity ? 1 : 0,
                    records.has_acceleration ? 1 : 0};
    MPI_Bcast(flags, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (worker_rank != 0) {
        records.has_displacement = flags[0] != 0;
        records.has_velocity = flags[1] != 0;
        records.has_acceleration = flags[2] != 0;
    }
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

    plan.rank_entry_offsets.push_back(0);
    for (size_t mapping_index = 0; mapping_index < layout.rank_maps.size(); ++mapping_index) {
        const auto& mapping = layout.rank_maps[mapping_index];
        plan.rank_ids.push_back(mapping.rank);
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
        plan.rank_entry_offsets.push_back(
            static_cast<int64_t>(plan.record_cell_point_index.size()));
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
// Per-direction tile-local field arrays (second pass, tile-local only)
// -----------------------------------------------------------------------

struct DirFields {
    std::vector<double> strain;        // [n_steps, n_local, 6]
    std::vector<double> displacement;  // [n_steps, n_local, 3]
    std::vector<double> velocity;      // [n_steps, n_local, 3]
    std::vector<double> acceleration;  // [n_steps, n_local, 3]
};

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

// Phase 3: extract per-step fields for ONE direction, tile-local nodes only.
// All recording cells are iterated (shared-node averaging correctness), but
// only tile-local nodes are accumulated. Memory: n_steps * n_local, not n_unique_gll.
static DirFields extract_tile_fields(const DirectionRecords& records, const TilePlan& plan,
                                     int64_t n_local, const std::vector<double>& cell_mass,
                                     PostprocessProfile* profile) {
    DirFields result;
    int64_t n_steps = static_cast<int64_t>(records.steps.size());
    int64_t n_node_per_cell = plan.n_node_per_cell;

    const double field_allocation_start = profile && profile->enabled ? profile_now() : 0.0;
    result.strain.resize((size_t)n_steps * (size_t)n_local * 6, 0.0);
    if (records.has_displacement)
        result.displacement.resize((size_t)n_steps * (size_t)n_local * 3, 0.0);
    if (records.has_velocity)
        result.velocity.resize((size_t)n_steps * (size_t)n_local * 3, 0.0);
    if (records.has_acceleration)
        result.acceleration.resize((size_t)n_steps * (size_t)n_local * 3, 0.0);
    if (profile && profile->enabled)
        profile->field_allocation_s += profile_now() - field_allocation_start;

    // --- Per-step GLL-node averaging (tile-local only) ---
    for (int64_t snap_idx = 0; snap_idx < n_steps; ++snap_idx) {
        double* step_data = result.strain.data() + snap_idx * n_local * 6;
        double* step_disp = records.has_displacement
                                ? result.displacement.data() + snap_idx * n_local * 3
                                : nullptr;
        double* step_vel =
            records.has_velocity ? result.velocity.data() + snap_idx * n_local * 3 : nullptr;
        double* step_acc = records.has_acceleration
                               ? result.acceleration.data() + snap_idx * n_local * 3
                               : nullptr;

        // Accumulate GLL mass for mass-weighted strain averaging (tile-local).
        const double step_buffer_init_start = profile && profile->enabled ? profile_now() : 0.0;
        std::vector<double> node_weight_sum((size_t)n_local, 0.0);
        gf_postprocess_common::VectorFieldAverager displacement_average(step_disp,
                                                                        (size_t)n_local);
        gf_postprocess_common::VectorFieldAverager velocity_average(step_vel, (size_t)n_local);
        gf_postprocess_common::VectorFieldAverager acceleration_average(step_acc, (size_t)n_local);
        if (profile && profile->enabled)
            profile->step_buffer_init_s += profile_now() - step_buffer_init_start;

        for (size_t mapping_index = 0; mapping_index < plan.rank_ids.size(); ++mapping_index) {
            const auto& record_path = records.record_paths[(size_t)snap_idx][mapping_index];
            if (record_path.empty())
                continue;

            const double file_open_start = profile && profile->enabled ? profile_now() : 0.0;
            hid_t fid = H5Fopen(record_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (profile && profile->enabled)
                profile->file_open_s += profile_now() - file_open_start;
            if (fid < 0)
                continue;
            if (profile && profile->enabled)
                ++profile->files_opened;

            // Read 4D strain [1, n_rec_cell, n_node_per_cell, 6]
            hsize_t nrc = 0, nnp = 0;
            std::vector<double> strain_buf;
            const double strain_io_start = profile && profile->enabled ? profile_now() : 0.0;
            read_strain_4d(fid, "strain", nrc, nnp, strain_buf);
            if (profile && profile->enabled) {
                ++profile->dataset_reads;
                const double elapsed_s = profile_now() - strain_io_start;
                profile->hdf5_read_s += elapsed_s;
                profile->field_read_s[0] += elapsed_s;
            }

            const double aggregation_start = profile && profile->enabled ? profile_now() : 0.0;
            const int64_t entry_begin = plan.rank_entry_offsets[mapping_index];
            const int64_t entry_end = plan.rank_entry_offsets[mapping_index + 1];
            for (int64_t entry = entry_begin; entry < entry_end; ++entry) {
                if (profile && profile->enabled)
                    ++profile->cell_values_visited;
                const int64_t point_index = plan.record_cell_point_index[(size_t)entry];
                const hsize_t cell = static_cast<hsize_t>(point_index / n_node_per_cell);
                const hsize_t point = static_cast<hsize_t>(point_index % n_node_per_cell);
                if (cell >= nrc || point >= nnp)
                    continue;
                const int32_t tile_node = plan.tile_local_node_index[(size_t)entry];
                double weight = 1.0;
                const int64_t mass_index = plan.cell_mass_index[(size_t)entry];
                if (mass_index >= 0 && (size_t)mass_index < cell_mass.size())
                    weight = cell_mass[(size_t)mass_index];
                const double* source = strain_buf.data() + (cell * nnp + point) * 6;
                double* destination = step_data + (size_t)tile_node * 6;
                for (int component = 0; component < 6; ++component)
                    destination[component] += source[component] * weight;
                node_weight_sum[(size_t)tile_node] += weight;
            }
            if (profile && profile->enabled) {
                const double elapsed_s = profile_now() - aggregation_start;
                profile->aggregation_s += elapsed_s;
                profile->field_aggregation_s[0] += elapsed_s;
            }

            // Read and accumulate displacement/velocity/acceleration (identical
            // logic - only the dataset name and step buffer differ).
            auto accumulate_vector_field =
                [&](const char* ds_name, int field_index,
                    gf_postprocess_common::VectorFieldAverager& average) {
                    hsize_t frc = 0, fnp = 0;
                    std::vector<double> fbuf;
                    const double field_io_start =
                        profile && profile->enabled ? profile_now() : 0.0;
                    read_field_4d(fid, ds_name, frc, fnp, fbuf);
                    if (profile && profile->enabled) {
                        ++profile->dataset_reads;
                        const double elapsed_s = profile_now() - field_io_start;
                        profile->hdf5_read_s += elapsed_s;
                        profile->field_read_s[field_index] += elapsed_s;
                    }
                    const double field_aggregation_start =
                        profile && profile->enabled ? profile_now() : 0.0;
                    for (int64_t entry = entry_begin; entry < entry_end; ++entry) {
                        const int64_t point_index = plan.record_cell_point_index[(size_t)entry];
                        const hsize_t cell = static_cast<hsize_t>(point_index / n_node_per_cell);
                        const hsize_t point = static_cast<hsize_t>(point_index % n_node_per_cell);
                        if (cell >= frc || point >= fnp)
                            continue;
                        const int32_t tile_node = plan.tile_local_node_index[(size_t)entry];
                        const double* source = fbuf.data() + (cell * fnp + point) * 3;
                        average.add((size_t)tile_node, source);
                    }
                    if (profile && profile->enabled) {
                        const double elapsed_s = profile_now() - field_aggregation_start;
                        profile->aggregation_s += elapsed_s;
                        profile->field_aggregation_s[field_index] += elapsed_s;
                    }
                };
            if (records.has_displacement)
                accumulate_vector_field("displacement", 1, displacement_average);
            if (records.has_velocity)
                accumulate_vector_field("velocity", 2, velocity_average);
            if (records.has_acceleration)
                accumulate_vector_field("acceleration", 3, acceleration_average);

            const double file_close_start = profile && profile->enabled ? profile_now() : 0.0;
            H5Fclose(fid);
            if (profile && profile->enabled)
                profile->file_close_s += profile_now() - file_close_start;
        }

        // Normalize: mass-weighted average for strain, count-based for disp/vel/acc.
        const double normalization_start = profile && profile->enabled ? profile_now() : 0.0;
        for (int64_t li = 0; li < n_local; ++li) {
            if (node_weight_sum[(size_t)li] > 0.0) {
                double* dst = step_data + li * 6;
                double inv_mass = 1.0 / node_weight_sum[(size_t)li];
                for (int c = 0; c < 6; ++c)
                    dst[c] *= inv_mass;
            }
        }
        if (records.has_displacement)
            displacement_average.normalize();
        if (records.has_velocity)
            velocity_average.normalize();
        if (records.has_acceleration)
            acceleration_average.normalize();
        if (profile && profile->enabled)
            profile->normalization_s += profile_now() - normalization_start;
    }  // snap_idx loop

    return result;
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------

int main(int argc, char** argv) {
    int worker_rank = 0;
    int worker_count = 1;
#ifdef GF_POST_MPI
    // ---- Optional MPI init ----
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &worker_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worker_count);
    fprintf(stderr, "[postprocess] MPI rank %d/%d\n", worker_rank, worker_count);
    // Stagger file access to avoid HDF5 metadata contention
    MPI_Barrier(MPI_COMM_WORLD);
    usleep((unsigned int)(worker_rank * 200000));  // 200ms stagger
    MPI_Barrier(MPI_COMM_WORLD);
#else
    fprintf(stderr, "[postprocess] Serial tile worker 0/1\n");
#endif

    double start = 0.0;
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        start = ts.tv_sec + ts.tv_nsec * 1e-9;
    }

    PostprocessProfile profile;
    if (profile.enabled)
        fprintf(stderr, "[postprocess] Profiling enabled (GF_POST_PROFILE=1)\n");

    fprintf(stderr, "[postprocess] Starting...\n");

    const double argument_parse_start = profile.enabled ? profile_now() : 0.0;
    auto args = gf_postprocess_common::parse_args(argc, argv, print_usage);
    if (profile.enabled)
        profile.argument_parse_s += profile_now() - argument_parse_start;

    // ---- Read config ----
    fprintf(stderr, "[postprocess] Reading config from %s\n", args.config_path.c_str());
    const double config_read_start = profile.enabled ? profile_now() : 0.0;
    ConfigParams cfg = read_config(args.config_path.c_str());
    if (profile.enabled)
        profile.config_read_s += profile_now() - config_read_start;

    // ---- Read mesh ----
    fprintf(stderr, "[postprocess] Reading mesh geometry from %s\n", args.model_path.c_str());
    const double model_read_start = profile.enabled ? profile_now() : 0.0;
    ModelData model = read_model(args.model_path.c_str());
    if (profile.enabled)
        profile.model_read_s += profile_now() - model_read_start;
    int64_t n_vertex = model.n_vertex;  // kept for domain bounds
    fprintf(stderr, "[postprocess]   domain vertex count = %lld\n", (long long)n_vertex);

    // ---- Read cell mass from model.h5 for L2 projection ----
    std::vector<double> cell_mass;
    int64_t n_model_cell = 0;
    int ngll_model = 0;
    const double mass_read_start = profile.enabled ? profile_now() : 0.0;
    gf_postprocess_common::read_cell_mass(args.model_path.c_str(), cell_mass, n_model_cell,
                                          ngll_model);
    if (profile.enabled)
        profile.mass_read_s += profile_now() - mass_read_start;

    // ---- Phase 1: rank 0 builds all shared indexes before field processing ----
    LayoutMetadata layout;
    DirectionRecords directions[3];
    std::vector<TileBin> tile_bins;
    auto case_directory = std::filesystem::path(args.model_path).parent_path();
    if (case_directory.empty())
        case_directory = ".";
    const auto partition_directory = case_directory / "partitions";
    auto x_record_directory = std::filesystem::path(args.fx_dir).lexically_normal();
    if (x_record_directory.filename().empty())
        x_record_directory = x_record_directory.parent_path();
    const auto tile_index_directory = x_record_directory.parent_path() / "tile_indexes";
    if (worker_rank == 0) {
        const double record_scan_start = profile.enabled ? profile_now() : 0.0;
        directions[0] = scan_direction_records(args.fx_dir.c_str());
        directions[1] = scan_direction_records(args.fy_dir.c_str());
        directions[2] = scan_direction_records(args.fz_dir.c_str());
        if (profile.enabled)
            profile.record_scan_s += profile_now() - record_scan_start;

        const double layout_merge_start = profile.enabled ? profile_now() : 0.0;
        layout = merge_partition_metadata(partition_directory, directions[0]);
        if (profile.enabled)
            profile.layout_merge_s += profile_now() - layout_merge_start;

        const double mass_index_start = profile.enabled ? profile_now() : 0.0;
        build_mass_indexes(layout, n_model_cell, ngll_model);
        if (profile.enabled)
            profile.mass_index_s += profile_now() - mass_index_start;

        const double record_index_start = profile.enabled ? profile_now() : 0.0;
        for (auto& direction : directions)
            build_direction_record_index(direction, layout);
        if (profile.enabled)
            profile.record_index_s += profile_now() - record_index_start;

        if (directions[0].steps != directions[1].steps ||
            directions[0].steps != directions[2].steps) {
            fprintf(stderr, "ERROR: mismatched output steps across force directions\n");
#ifdef GF_POST_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            return 1;
#endif
        }
        const double tile_bin_start = profile.enabled ? profile_now() : 0.0;
        tile_bins = build_tile_bins(layout, cfg, model);
        if (profile.enabled)
            profile.tile_bin_s += profile_now() - tile_bin_start;

        const double tile_plan_start = profile.enabled ? profile_now() : 0.0;
        std::filesystem::create_directories(tile_index_directory);
        for (const auto& entry : std::filesystem::directory_iterator(tile_index_directory)) {
            const std::string filename = entry.path().filename().string();
            if (entry.is_regular_file() && filename.rfind("tile_index_", 0) == 0 &&
                entry.path().extension() == ".h5")
                std::filesystem::remove(entry.path());
        }
        for (const auto& tile_bin : tile_bins) {
            const TilePlan plan = build_tile_plan(tile_bin, layout);
            write_tile_index(tile_index_path(tile_index_directory, plan.key), plan);
        }
        if (profile.enabled)
            profile.tile_plan_s += profile_now() - tile_plan_start;
    }

#ifdef GF_POST_MPI
    const double broadcast_start = profile.enabled ? profile_now() : 0.0;
    broadcast_layout(layout, worker_rank);
    for (auto& direction : directions)
        broadcast_direction_records(direction, layout, worker_rank);
    broadcast_tile_bins(tile_bins, worker_rank);
    if (profile.enabled)
        profile.broadcast_s += profile_now() - broadcast_start;
#endif

    int64_t n_steps = static_cast<int64_t>(directions[0].steps.size());

    // GLL node IDs (1-based, shared across directions)
    int64_t n_recorded = layout.n_unique_gll;
    fprintf(stderr, "[postprocess] %lld unique GLL nodes recorded\n", (long long)n_recorded);

    if (n_recorded == 0) {
        fprintf(stderr, "ERROR: no GLL nodes recorded\n");
        return 1;
    }

    // ---- Build time array ----
    const double time_stf_start = profile.enabled ? profile_now() : 0.0;
    std::vector<double> time_arr((size_t)n_steps);
    for (int64_t s = 0; s < n_steps; ++s) {
        time_arr[(size_t)s] = (double)s * cfg.output_dt_s;
    }

    // ---- Downsample STF to tile time axis ----
    // config STF is at solver_dt; tile time axis is at output_dt_s.
    std::vector<double> stf_t_ds, stf_values_ds;
    gf_postprocess_common::downsample_stf(cfg, n_steps, stf_t_ds, stf_values_ds);
    if (profile.enabled)
        profile.time_stf_s += profile_now() - time_stf_start;

    bool has_displacement = directions[0].has_displacement && directions[1].has_displacement &&
                            directions[2].has_displacement;
    bool has_velocity =
        directions[0].has_velocity && directions[1].has_velocity && directions[2].has_velocity;
    bool has_acceleration = directions[0].has_acceleration && directions[1].has_acceleration &&
                            directions[2].has_acceleration;
    fprintf(stderr, "[postprocess]   displacement=%s velocity=%s acceleration=%s\n",
            has_displacement ? "yes" : "no", has_velocity ? "yes" : "no",
            has_acceleration ? "yes" : "no");

    // ---- Phase 2: prepare assigned tile indexes before any field I/O ----
    double xmin = model.xmin, ymin = model.ymin, xmax = model.xmax, ymax = model.ymax;
    int64_t n_tiles = static_cast<int64_t>(tile_bins.size());
    fprintf(stderr, "[postprocess]   %lld tiles\n", (long long)n_tiles);

    // ---- Worker exit check: workers beyond n_tiles exit BEFORE any field alloc ----
    if (worker_rank >= (int)n_tiles) {
        fprintf(stderr, "[postprocess] Worker %d: no tile assigned (n_tiles=%lld), exiting\n",
                worker_rank, (long long)n_tiles);
#ifdef GF_POST_MPI
        MPI_Finalize();
#endif
        return 0;
    }

    const double tile_plan_load_start = profile.enabled ? profile_now() : 0.0;
    std::vector<TilePlan> tile_plans;
    for (int64_t tile_position = worker_rank; tile_position < n_tiles;
         tile_position += worker_count) {
        const TileKey key = tile_bins[(size_t)tile_position].key;
        tile_plans.push_back(read_tile_index(tile_index_path(tile_index_directory, key)));
    }
    if (profile.enabled)
        profile.tile_plan_s += profile_now() - tile_plan_load_start;
    fprintf(stderr, "[postprocess]   worker %d prepared %zu tile index plan(s)\n", worker_rank,
            tile_plans.size());

    const double output_prepare_start = profile.enabled ? profile_now() : 0.0;
    std::string mkdir_cmd = "mkdir -p " + args.output_dir;
    if (system(mkdir_cmd.c_str()) != 0) {
        fprintf(stderr, "WARNING: could not create output directory %s\n",
                args.output_dir.c_str());
    }
    if (profile.enabled)
        profile.output_prepare_s += profile_now() - output_prepare_start;

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

    // ---- Process assigned tiles using the prebuilt plans ----
    for (const auto& plan : tile_plans) {
        const TileKey& key = plan.key;
        int64_t n_local = static_cast<int64_t>(plan.node_ids.size());
        // ---- Phase 3: assemble tile Green's tensor directly from per-direction
        //      tile-local fields. One direction at a time to bound peak memory.
        // tile_greens: [n_steps, n_local, comp(6), dir(3)] = 18 per (s,li)
        const double tile_allocation_start = profile.enabled ? profile_now() : 0.0;
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
        if (profile.enabled)
            profile.tile_allocation_s += profile_now() - tile_allocation_start;

        // Direction 0 = fx, 1 = fy, 2 = fz
        for (int dir = 0; dir < 3; ++dir) {
            DirFields fields =
                extract_tile_fields(directions[dir], plan, n_local, cell_mass, &profile);

            // Assemble strain -> tile_greens [s, li, comp(6), dir]
            const double assembly_start = profile.enabled ? profile_now() : 0.0;
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
            if (profile.enabled)
                profile.assembly_s += profile_now() - assembly_start;
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

        const double write_start = profile.enabled ? profile_now() : 0.0;
        write_tile(fname, key.tx, key.ty, tx_min, tx_max, ty_min, ty_max, zmin, zmax,
                   cfg.record_depth_max_m, cfg.record_depth_actual_m, plan.node_ids, time_arr,
                   cfg.solver_dt, tile_greens, source_xyz_m, plan.node_coords, plan.cell_gll_index,
                   (int)plan.n_node_per_cell,
                   has_displacement ? tile_displacement.data() : nullptr,
                   has_velocity ? tile_velocity.data() : nullptr,
                   has_acceleration ? tile_acceleration.data() : nullptr, stf_t_ds, stf_values_ds,
                   use_float32);
        if (profile.enabled)
            profile.write_s += profile_now() - write_start;
        fprintf(stderr, "[postprocess]   worker %d wrote tile x%03d y%03d\n", worker_rank, key.tx,
                key.ty);
    }

#ifdef GF_POST_MPI
    const double mpi_finalize_start = profile.enabled ? profile_now() : 0.0;
    MPI_Finalize();
    if (profile.enabled)
        profile.mpi_finalize_s += profile_now() - mpi_finalize_start;
#endif

    // ---- Print machine-parseable stats ----
    gf_postprocess_common::print_stats(start, n_steps, n_vertex, n_recorded, n_tiles);
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

    return 0;
}
