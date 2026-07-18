// forward/share/src/assembly.cpp
#include "gf/assembly.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace gf {

void scatter_to_rank(const std::vector<double>& local_cell_residual,
                     const std::vector<int32_t>& local_cell2rank_node, int n_local_cell,
                     int n_node, std::vector<double>& rank_node_residual) {
    // Zero global residual before accumulation
    std::fill(rank_node_residual.begin(), rank_node_residual.end(), 0.0);

    for (int e = 0; e < n_local_cell; ++e) {
        for (int n = 0; n < n_node; ++n) {
            int node_id = local_cell2rank_node[e * n_node + n];
            int elem_base = (e * n_node + n) * 3;
            int glob_base = node_id * 3;
            for (int d = 0; d < 3; ++d) {
                rank_node_residual[glob_base + d] += local_cell_residual[elem_base + d];
            }
        }
    }
}

void gather_from_rank(const std::vector<double>& rank_node_field,
                      const std::vector<int32_t>& local_cell2rank_node, int n_local_cell,
                      int n_node, std::vector<double>& local_cell_field) {
    for (int e = 0; e < n_local_cell; ++e) {
        for (int n = 0; n < n_node; ++n) {
            int node_id = local_cell2rank_node[e * n_node + n];
            int elem_base = (e * n_node + n) * 3;
            int glob_base = node_id * 3;
            for (int d = 0; d < 3; ++d) {
                local_cell_field[elem_base + d] = rank_node_field[glob_base + d];
            }
        }
    }
}

// ---- Legacy API (for existing tests) ----

void assemble_residual(const std::vector<double>& local_cell_residual, const RankData& rank_data,
                       std::vector<double>& rank_node_residual) {
    const int ngll = rank_data.ngll;
    const int n_node = ngll * ngll * ngll;
    const int n_dof_per_elem = n_node * 3;

    for (int e = 0; e < rank_data.n_local_cell; ++e) {
        const int global_offset = e * n_dof_per_elem;
        const int local_offset = e * n_dof_per_elem;

        for (int d = 0; d < n_dof_per_elem; ++d) {
            rank_node_residual[global_offset + d] = local_cell_residual[local_offset + d];
        }
    }
}

void add_source_to_rhs(int elem_idx, int gll_i, int gll_j, int gll_k, double fx, double fy,
                       double fz, const RankData& rank_data, std::vector<double>& rhs) {
    const int ngll = rank_data.ngll;
    const int n_node = ngll * ngll * ngll;

    const int node_idx = (gll_i * ngll + gll_j) * ngll + gll_k;
    const int elem_base = elem_idx * n_node * 3;
    const int dof_base = elem_base + node_idx * 3;

    rhs[dof_base + 0] += fx;
    rhs[dof_base + 1] += fy;
    rhs[dof_base + 2] += fz;
}

}  // namespace gf