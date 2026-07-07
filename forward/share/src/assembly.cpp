// forward/share/src/assembly.cpp
#include "gf/assembly.hpp"

#include <cassert>

namespace gf {

void assemble_residual(const std::vector<double>& elem_residual, const RankData& rank_data,
                       std::vector<double>& global_residual) {
    const int ngll = rank_data.ngll;
    const int n_node = ngll * ngll * ngll;
    const int n_dof_per_elem = n_node * 3;

    // For each local element, copy its residual block to the corresponding
    // global offset. Local elements 0..n_local_elem-1 map to the same
    // positions in the global array.
    for (int e = 0; e < rank_data.n_local_elem; ++e) {
        const int global_offset = e * n_dof_per_elem;
        const int local_offset = e * n_dof_per_elem;

        for (int d = 0; d < n_dof_per_elem; ++d) {
            global_residual[global_offset + d] = elem_residual[local_offset + d];
        }
    }
}

void add_source_to_rhs(int elem_idx, int gll_i, int gll_j, int gll_k, double fx, double fy,
                       double fz, const RankData& rank_data, std::vector<double>& rhs) {
    const int ngll = rank_data.ngll;
    const int n_node = ngll * ngll * ngll;

    // Compute 1D flat index for (gll_i, gll_j, gll_k) within the element
    const int node_idx = (gll_i * ngll + gll_j) * ngll + gll_k;

    // Global DOF base index for this element
    const int elem_base = elem_idx * n_node * 3;
    const int dof_base = elem_base + node_idx * 3;

    rhs[dof_base + 0] += fx;
    rhs[dof_base + 1] += fy;
    rhs[dof_base + 2] += fz;
}

}  // namespace gf