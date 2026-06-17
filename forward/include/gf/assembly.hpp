#pragma once

#include <vector>
#include <cstddef>
#include "gf/types.hpp"

namespace gf {

/// Scatter element-local GLL node contributions to global array.
///
/// Copies element_residual[3 * node] to global_residual[global_offset + 3 * node]
/// where global_offset = elem_idx * ngll^3 * 3. The mapping is direct because
/// each element occupies a contiguous block in the global array.
///
/// @param[in] elem_residual    [n_local_elem * ngll^3 * 3] element contributions, elem-major
/// @param[in] rank_data         per-rank metadata (ngll, element counts)
/// @param[in,out] global_residual [n_total_elem * ngll^3 * 3] global array, accumulated into
void assemble_residual(
    const std::vector<double>& elem_residual,
    const RankData& rank_data,
    std::vector<double>& global_residual
);

/// Add a point source force at a specific GLL node within an element to the RHS.
///
/// Finds the global DOF index for element elem_idx (0-based local index),
/// GLL indices (gll_i, gll_j, gll_k), and adds the force components.
///
/// @param[in] elem_idx  0-based local element index
/// @param[in] gll_i     GLL index in xi direction
/// @param[in] gll_j     GLL index in eta direction
/// @param[in] gll_k     GLL index in zeta direction
/// @param[in] fx,fy,fz  force components
/// @param[in] rank_data per-rank metadata
/// @param[in,out] rhs   global RHS vector, accumulated into
void add_source_to_rhs(
    int elem_idx, int gll_i, int gll_j, int gll_k,
    double fx, double fy, double fz,
    const RankData& rank_data,
    std::vector<double>& rhs
);

} // namespace gf