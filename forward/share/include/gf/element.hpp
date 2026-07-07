#pragma once

#include <cstddef>
#include <vector>

#include "gf/backend.hpp"
#include "gf/types.hpp"

namespace gf {

// -----------------------------------------------------------------------
// compute_element_residual — backend-dispatched kernel
//
// Compute the internal force (stiffness residual) for a batch of elements:
//   r += K_e * u   for e = 0..n_elem-1
//
// Matrix-free: no global stiffness matrix. For each GLL quadrature node (i,j,k):
//   1. Compute displacement gradient ∂u_l/∂x_m via chain rule with precomputed dξ/dx
//   2. Form symmetric strain ε_lm = ½(∂u_l/∂x_m + ∂u_m/∂x_l)
//   3. Compute stress σ_lm = λ·δ_lm·ε_kk + 2μ·ε_lm (isotropic)
//   4. Accumulate f = ∇N · σ · detJ · w_i · w_j · w_k
//
// All arrays are element-major contiguous (n_elem blocks of NGLL^3 nodes).
//
// @tparam Backend  Tag type selecting the device backend.
// @param[in]  n_elem      Number of elements in this batch
// @param[in]  dxi_dx      [n_elem * NGLL^3 * 9]  d(xi_i)/dx_j per GLL node
// @param[in]  jacobian    [n_elem * NGLL^3]       det(J) per GLL node
// @param[in]  lambda_     [n_elem * NGLL^3]       Lamé parameter λ per GLL node (precomputed)
// @param[in]  mu_         [n_elem * NGLL^3]       Shear modulus μ per GLL node (precomputed)
// @param[in]  D           [NGLL * NGLL]            1D GLL derivative matrix (row-major)
// @param[in]  weights     [NGLL]                   1D GLL quadrature weights
// @param[in]  NGLL        N+1 (number of GLL points per axis)
// @param[in]  u           [n_elem * NGLL^3 * 3]   displacement (x/y/z interleaved)
// @param[out] r           [n_elem * NGLL^3 * 3]   residual (internal force, accumulated)
//
// -----------------------------------------------------------------------

template <typename Backend>
void compute_element_residual(int n_elem, const double* dxi_dx, const double* jacobian,
                              const double* lambda_, const double* mu_, const double* D,
                              const double* weights, int NGLL, const double* u, double* r);

// --- Explicit instantiation declarations ---
// Guarded: specialization source files define GF_ELEMENT_{CPU,CUDA}_SOURCE
// to suppress the extern template declaration and avoid
// "specialization after instantiation" errors.

#ifndef GF_ELEMENT_CPU_SOURCE
extern template void compute_element_residual<BackendCPU>(int, const double*, const double*,
                                                          const double*, const double*,
                                                          const double*, const double*, int,
                                                          const double*, double*);
#endif

#ifdef GF_WITH_CUDA
#ifndef GF_ELEMENT_CUDA_SOURCE
extern template void compute_element_residual<BackendCUDA>(int, const double*, const double*,
                                                           const double*, const double*,
                                                           const double*, const double*, int,
                                                           const double*, double*);
#endif
#endif

}  // namespace gf