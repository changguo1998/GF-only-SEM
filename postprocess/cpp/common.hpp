/* postprocess/cpp/common.hpp — helpers for the shared serial/MPI tile-batched
 * postprocessor pipeline in main.cpp.
 *
 * Nothing here depends on MPI. Everything is inline/header-only so a single
 * definition is shared by both build variants.
 */

#ifndef GF_POSTPROCESS_COMMON_HPP
#define GF_POSTPROCESS_COMMON_HPP

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "reader.hpp"  // ConfigParams (includes <hdf5.h>)

namespace gf_postprocess_common {

struct TileAssignmentInput {
    uint64_t work_weight = 0;
    std::vector<std::pair<int, uint64_t>> rank_read_weights;
};

/// Parse the aggregate field-array budget in GiB. A missing environment value
/// preserves the default; present values must be positive whole numbers.
inline uint64_t postprocess_memory_budget_bytes(const char* environment_value) {
    constexpr uint64_t bytes_per_gib = 1024ULL * 1024 * 1024;
    constexpr uint64_t default_budget_gib = 32;
    if (environment_value == nullptr)
        return default_budget_gib * bytes_per_gib;

    const uint64_t maximum_budget_gib = std::numeric_limits<uint64_t>::max() / bytes_per_gib;
    uint64_t budget_gib = 0;
    if (environment_value[0] == '\0')
        throw std::invalid_argument("GF_POST_MEMORY_GB must be a positive integer");
    for (const char* character = environment_value; *character != '\0'; ++character) {
        if (*character < '0' || *character > '9')
            throw std::invalid_argument("GF_POST_MEMORY_GB must be a positive integer");
        const uint64_t digit = static_cast<uint64_t>(*character - '0');
        if (budget_gib > (maximum_budget_gib - digit) / 10)
            throw std::invalid_argument("GF_POST_MEMORY_GB is too large");
        budget_gib = budget_gib * 10 + digit;
    }
    if (budget_gib == 0)
        throw std::invalid_argument("GF_POST_MEMORY_GB must be greater than zero");
    return budget_gib * bytes_per_gib;
}

/// Assign heavier tiles first while preferring workers that already need the
/// same record ranks. The balance window prevents I/O reuse from collapsing
/// all tiles onto one worker.
inline std::vector<int> assign_tiles_by_rank_overlap(const std::vector<TileAssignmentInput>& tiles,
                                                     int worker_count) {
    std::vector<int> assignments(tiles.size(), 0);
    if (tiles.empty() || worker_count <= 1)
        return assignments;

    const int active_workers = std::min<int>(worker_count, static_cast<int>(tiles.size()));
    std::vector<size_t> order(tiles.size());
    for (size_t tile = 0; tile < tiles.size(); ++tile)
        order[tile] = tile;
    std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        return tiles[left].work_weight > tiles[right].work_weight;
    });

    std::vector<uint64_t> worker_loads(static_cast<size_t>(active_workers), 0);
    std::vector<std::unordered_set<int>> worker_ranks(static_cast<size_t>(active_workers));
    for (size_t tile_index : order) {
        const auto& tile = tiles[tile_index];
        uint64_t minimum_projected_load = std::numeric_limits<uint64_t>::max();
        for (int worker = 0; worker < active_workers; ++worker) {
            minimum_projected_load =
                std::min(minimum_projected_load,
                         worker_loads[static_cast<size_t>(worker)] + tile.work_weight);
        }
        const uint64_t balance_slack = std::max<uint64_t>(1, tile.work_weight / 10);

        int best_worker = 0;
        uint64_t best_reused_read_weight = 0;
        uint64_t best_projected_load = std::numeric_limits<uint64_t>::max();
        bool found = false;
        for (int worker = 0; worker < active_workers; ++worker) {
            const uint64_t projected_load =
                worker_loads[static_cast<size_t>(worker)] + tile.work_weight;
            if (projected_load > minimum_projected_load + balance_slack)
                continue;
            uint64_t reused_read_weight = 0;
            for (const auto& [rank, read_weight] : tile.rank_read_weights) {
                if (worker_ranks[static_cast<size_t>(worker)].count(rank) != 0)
                    reused_read_weight += read_weight;
            }
            if (!found || reused_read_weight > best_reused_read_weight ||
                (reused_read_weight == best_reused_read_weight &&
                 projected_load < best_projected_load)) {
                best_worker = worker;
                best_reused_read_weight = reused_read_weight;
                best_projected_load = projected_load;
                found = true;
            }
        }

        assignments[tile_index] = best_worker;
        worker_loads[static_cast<size_t>(best_worker)] += tile.work_weight;
        for (const auto& [rank, read_weight] : tile.rank_read_weights) {
            (void)read_weight;
            worker_ranks[static_cast<size_t>(best_worker)].insert(rank);
        }
    }
    return assignments;
}

/// Pack consecutive assigned tiles into the fewest batches that fit the
/// per-worker memory budget. A single oversized tile remains a valid batch.
inline std::vector<std::pair<size_t, size_t>> plan_memory_batches(
    const std::vector<uint64_t>& item_bytes, uint64_t memory_budget) {
    std::vector<std::pair<size_t, size_t>> ranges;
    size_t batch_begin = 0;
    uint64_t batch_bytes = 0;
    for (size_t item = 0; item < item_bytes.size(); ++item) {
        if (item > batch_begin && batch_bytes + item_bytes[item] > memory_budget) {
            ranges.emplace_back(batch_begin, item);
            batch_begin = item;
            batch_bytes = 0;
        }
        batch_bytes += item_bytes[item];
    }
    if (batch_begin < item_bytes.size())
        ranges.emplace_back(batch_begin, item_bytes.size());
    return ranges;
}

/// Store one force-direction strain vector in a [component(6), direction(3)] tensor.
inline void assign_strain_direction(const double* strain_components, double* greens_tensor,
                                    int force_direction) {
    for (int component = 0; component < 6; ++component)
        greens_tensor[component * 3 + force_direction] = strain_components[component];
}

/// Accumulate and count one three-component nodal vector field.
///
/// Displacement, velocity, and acceleration must each own an instance. Sharing
/// their sample counts would divide every field by the number of enabled
/// quantities instead of by the number of contributing cells.
class VectorFieldAverager {
public:
    VectorFieldAverager(double* values, size_t n_nodes)
        : values_(values), sample_counts_(n_nodes, 0) {}

    void add(size_t node_index, const double* sample) {
        double* destination = values_ + node_index * 3;
        for (int component = 0; component < 3; ++component)
            destination[component] += sample[component];
        sample_counts_[node_index]++;
    }

    void normalize() {
        for (size_t node_index = 0; node_index < sample_counts_.size(); ++node_index) {
            int count = sample_counts_[node_index];
            if (count == 0)
                continue;
            double* destination = values_ + node_index * 3;
            double inverse_count = 1.0 / static_cast<double>(count);
            for (int component = 0; component < 3; ++component)
                destination[component] *= inverse_count;
        }
    }

private:
    double* values_;
    std::vector<int> sample_counts_;
};

struct Args {
    std::string model_path;
    std::string config_path;
    std::string fx_dir;
    std::string fy_dir;
    std::string fz_dir;
    std::string output_dir = "greenfun";
};

/// Parse CLI into `Args`. On error prints `usage(prog)` (per-binary help text)
/// and exits(1).
inline Args parse_args(int argc, char** argv, void (*usage)(const char*)) {
    if (argc < 7) {
        usage(argv[0]);
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
        usage(argv[0]);
        exit(1);
    }
    return args;
}

/// Read /field/cell/mass from model.h5 for L2 projection. Best-effort: on any
/// read failure the arrays are left empty and counts zeroed.
inline void read_cell_mass(const char* model_path, std::vector<double>& cell_mass,
                           int64_t& n_model_cell, int& ngll_model) {
    cell_mass.clear();
    n_model_cell = 0;
    ngll_model = 0;

    hid_t model_fid = H5Fopen(model_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (model_fid < 0)
        return;
    hid_t mass_ds = H5Dopen2(model_fid, "/field/cell/mass", H5P_DEFAULT);
    if (mass_ds >= 0) {
        hid_t mass_space = H5Dget_space(mass_ds);
        hsize_t mass_dims[4] = {0, 0, 0, 0};
        H5Sget_simple_extent_dims(mass_space, mass_dims, nullptr);
        n_model_cell = (int64_t)mass_dims[0];
        ngll_model = (int)mass_dims[1];
        hsize_t total = mass_dims[0] * mass_dims[1] * mass_dims[2] * mass_dims[3];
        cell_mass.resize((size_t)total);
        H5Dread(mass_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cell_mass.data());
        fprintf(stderr, "[postprocess]   read cell mass [%lld, %d, %d, %d]\n",
                (long long)n_model_cell, ngll_model, ngll_model, ngll_model);
        H5Sclose(mass_space);
        H5Dclose(mass_ds);
    }
    H5Fclose(model_fid);
}

/// Resample the config STF (sampled at solver_dt) onto the tile time axis
/// [n_steps] (sampled at output_dt_s) so users can deconvolve with an STF
/// sampled at the same rate as the Green's function tensors. Leaves output
/// arrays empty when the config carries no STF.
inline void downsample_stf(const ConfigParams& cfg, int64_t n_steps, std::vector<double>& stf_t_ds,
                           std::vector<double>& stf_values_ds) {
    stf_t_ds.clear();
    stf_values_ds.clear();
    if (cfg.stf_t.empty() || n_steps <= 0)
        return;

    int64_t nstf = (int64_t)cfg.stf_t.size();
    int64_t stride = nstf / n_steps;
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

/// Print the machine-parseable summary block (STAT_* lines + human stderr).
inline void print_stats(double start, int64_t n_steps, int64_t n_vertex, int64_t n_recorded,
                        int64_t n_tiles) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double elapsed = (ts.tv_sec + ts.tv_nsec * 1e-9) - start;

    printf("STAT_NSTEPS=%lld\n", (long long)n_steps);
    printf("STAT_NVERTEX=%lld\n", (long long)n_vertex);
    printf("STAT_NRECORDED=%lld\n", (long long)n_recorded);
    printf("STAT_NTILES=%lld\n", (long long)n_tiles);
    printf("STAT_ELAPSED_S=%.1f\n", elapsed);
    fflush(stdout);

    fprintf(stderr, "[postprocess] Done in %.1fs - %lld tile(s), %lld recorded vertex(ices)\n",
            elapsed, (long long)n_tiles, (long long)n_recorded);
}

}  // namespace gf_postprocess_common

#endif  // GF_POSTPROCESS_COMMON_HPP
