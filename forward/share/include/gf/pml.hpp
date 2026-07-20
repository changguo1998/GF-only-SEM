#pragma once

#include <cstddef>
#include <vector>

#include "gf/types.hpp"

namespace gf {

/// Apply legacy PML damping to the velocity field (backward compatibility).
///
/// For each DOF, the velocity is damped by a precomputed damping profile:
///   v[i] -= damping_profile[node] * v[i]
///
/// Used when C-PML data is not available (old partition files).
///
/// @param[in] damping_profile  per-GLL-node damping values [n_elem * NGLL^3]
/// @param[in] u                displacement field [n_dof]
/// @param[in,out] v            velocity field [n_dof], modified in place
/// @param[in] n_dof            total DOF count (= n_elem * NGLL^3 * 3)
void apply_pml_damping(const std::vector<double>& damping_profile, const std::vector<double>& u,
                       std::vector<double>& v, int n_dof);

// ---------------------------------------------------------------------------
// C-PML functions (recursive convolution, Wang et al. 2006)
// ---------------------------------------------------------------------------

/// Initialize C-PML memory state arrays.
///
/// Allocates and zero-initializes pml_displ_old, pml_displ_new,
/// rmemory_displ, and rmemory_strain based on the number of PML elements.
///
/// @param[in,out] part     RankData with C-PML coefficients loaded
/// @param[in] n_node       NGLL^3 (nodes per element)
void cpml_initialize(RankData& part, int n_node);

/// Update PML displacement fields for second-order convolution.
///
/// PML_displ_new = u + (1-2θ)/2 * dt * v                    (current step)
/// PML_displ_old = u_prev + (1-2θ)/2 * dt * v_prev + (1-θ)/2 * dt² * a_prev
///               (copied from previous PML_displ_new before update)
///
/// @param[in,out] part     RankData with PML displacement fields
/// @param[in] displacement Global displacement array [n_rank_dof]
/// @param[in] velocity     Global velocity array [n_rank_dof]
/// @param[in] acceleration Global acceleration array [n_rank_dof]
/// @param[in] dt           Solver timestep
/// @param[in] n_node       NGLL^3
void cpml_update_displ_fields(RankData& part, const std::vector<double>& displacement,
                              const std::vector<double>& velocity,
                              const std::vector<double>& acceleration, double dt, int n_node);

/// Update C-PML displacement memory variables.
///
/// For each PML element and GLL node:
///   rmemory[d] = coef0_α[d] * rmemory[d]
///              + coef1_α[d] * PML_displ_new
///              + coef2_α[d] * PML_displ_old
///
/// @param[in,out] part     RankData with memory variables and coefficients
/// @param[in] n_node       NGLL^3
void cpml_update_displ_memory(RankData& part, int n_node);

/// Compute C-PML acceleration contribution and add to element-local residual.
///
/// For each PML element and GLL node:
///   residual += w * (1/ρ) * J * (Ā₁*v + Ā₂*u + Ā₃*mem_x + Ā₄*mem_y + Ā₅*mem_z)
///
/// @param[in] part         RankData with C-PML coefficients and memory
/// @param[in] displacement Global displacement array [n_rank_dof]
/// @param[in] velocity     Global velocity array [n_rank_dof]
/// @param[in] local_cell2rank_node  Element-to-rank-node mapping
/// @param[in] gll_weights  GLL quadrature weights [NGLL]
/// @param[in,out] residual Element-local residual [n_local_cell * n_node * 3]
/// @param[in] n_local_cell Number of local elements
/// @param[in] n_node       NGLL^3
void cpml_accel_contribution(const RankData& part, const std::vector<double>& displacement,
                             const std::vector<double>& velocity,
                             const std::vector<int32_t>& local_cell2rank_node,
                             const std::vector<double>& gll_weights, std::vector<double>& residual,
                             int n_local_cell, int n_node);

}  // namespace gf
