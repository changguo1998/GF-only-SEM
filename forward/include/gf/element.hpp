#pragma once

#include <cstddef>
#include <vector>

#include "gf/types.hpp"

namespace gf {

/// Compute the element internal force (stiffness residual): r += K_e * u
///
/// Matrix-free: no global stiffness matrix. For each GLL quadrature node (i,j,k):
///   1. Compute displacement gradient ∂u_l/∂x_m via chain rule with precomputed dξ/dx
///   2. Form symmetric strain ε_lm = ½(∂u_l/∂x_m + ∂u_m/∂x_l)
///   3. Compute stress σ_lm = λ·δ_lm·ε_kk + 2μ·ε_lm (isotropic)
///   4. Accumulate f = ∇N · σ · detJ · w_i · w_j · w_k (contributing to each node's residual)
///
/// Precomputed arrays (per element, length NGLL^3):
///   dxi_dx[9 * n_node]  — per GLL node: [dξ/dx, dη/dx, dζ/dx, dξ/dy, dη/dy, dζ/dy, dξ/dz, dη/dz,
///   dζ/dz] jacobian[n_node]    — det(J) per GLL node vp, vs, density     — material properties
///   per GLL node
///
/// GLL quadrature:
///   D[NGLL * NGLL]  — derivative matrix (row-major)
///   weights[NGLL]   — 1D GLL quadrature weights
///   NGLL            — N+1
///
/// @param[in] dxi_dx   [9 * n_node]  — per GLL node: d(xi_i)/dx_j in row-major order (9 values per
/// node)
/// @param[in] jacobian [n_node]      — det(J) per GLL node
/// @param[in] vp       [n_node]      — P-wave velocity per GLL node
/// @param[in] vs       [n_node]      — S-wave velocity per GLL node
/// @param[in] density  [n_node]      — density per GLL node
/// @param[in] D        [NGLL*NGLL]   — 1D GLL derivative matrix (row-major)
/// @param[in] weights  [NGLL]        — 1D GLL quadrature weights
/// @param[in] NGLL     — N+1 (number of GLL points per axis)
/// @param[in] u        [3 * n_node]  — displacement at all GLL nodes of element (flat x/y/z
/// interleaved)
/// @param[out] r       [3 * n_node]  — residual accumulation (internal force with minus sign)
///
void compute_element_residual(const double* dxi_dx, const double* jacobian, const double* vp,
                              const double* vs, const double* density, const double* D,
                              const double* weights, int NGLL, const double* u, double* r);

}  // namespace gf