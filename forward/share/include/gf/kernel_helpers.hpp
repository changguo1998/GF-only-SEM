#pragma once
/// \file kernel_helpers.hpp
/// \brief Shared element kernel helpers — geometry/mechanics primitives
///        independent of the constitutive stress model.
///
/// Five inline functions extracted from the elastic element kernel:
///   1. compute_reference_gradient   — GLL derivative → dudξ/dη/dζ
///   2. transform_to_physical        — reference → physical gradient
///   3. apply_cpml_strain_correction — C-PML gradient modification
///   4. compute_strain_tensor        — symmetric strain ε = ½(∇u + ∇uᵀ)
///   5. scatter_residual             — stress → residual via GLL quadrature
///
/// These are pure geometric/mechanical primitives.  Different constitutive
/// models (elastic, viscoelastic, poroelastic) reuse all five and only
/// replace the stress computation that sits between #4 and #5.

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "gf/pml.hpp"

namespace gf {

// ---------------------------------------------------------------------------
// 1. Reference-space displacement gradient (GLL derivative)
// ---------------------------------------------------------------------------
/// Compute dudξ, dudη, dudζ at GLL node (i,j,k) inside one element.
///
/// \param[in]  i, j, k   GLL node index within the element
/// \param[in]  NGLL      number of GLL points per dimension
/// \param[in]  D         GLL derivative matrix  [NGLL × NGLL], row-major
/// \param[in]  elem_u    element displacement    [NGLL³ × 3], flat
/// \param[out] dudxi     3-component gradient along ξ
/// \param[out] dudeta    3-component gradient along η
/// \param[out] dudzeta   3-component gradient along ζ
inline void compute_reference_gradient(int i, int j, int k, int NGLL,
                                       const double* __restrict D,
                                       const double* __restrict elem_u,
                                       double* __restrict dudxi,
                                       double* __restrict dudeta,
                                       double* __restrict dudzeta) {
    dudxi[0] = 0.0;   dudxi[1] = 0.0;   dudxi[2] = 0.0;
    dudeta[0] = 0.0;  dudeta[1] = 0.0;  dudeta[2] = 0.0;
    dudzeta[0] = 0.0; dudzeta[1] = 0.0; dudzeta[2] = 0.0;

    for (int s = 0; s < NGLL; ++s) {
        const double Di_s = D[i * NGLL + s];
        const double Dj_s = D[j * NGLL + s];
        const double Dk_s = D[k * NGLL + s];

        const int n_sjk = (s * NGLL + j) * NGLL + k;
        const int n_isk = (i * NGLL + s) * NGLL + k;
        const int n_ijs = (i * NGLL + j) * NGLL + s;

        for (int dir = 0; dir < 3; ++dir) {
            dudxi[dir]   += Di_s * elem_u[3 * n_sjk + dir];
            dudeta[dir]  += Dj_s * elem_u[3 * n_isk + dir];
            dudzeta[dir] += Dk_s * elem_u[3 * n_ijs + dir];
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Transform reference gradient → physical gradient
// ---------------------------------------------------------------------------
/// Apply inverse Jacobian to convert reference-space gradients to
/// physical-space gradients.
///
/// \param[in]  dudxi, dudeta, dudzeta   reference gradient (3-vectors)
/// \param[in]  dd  inverse Jacobian [9]:
///                 dd[0..2] = (dξ/dx, dη/dx, dζ/dx)
///                 dd[3..5] = (dξ/dy, dη/dy, dζ/dy)
///                 dd[6..8] = (dξ/dz, dη/dz, dζ/dz)
/// \param[out] du_dx  physical gradient [3][3]:
///                    du_dx[comp][0]=∂/∂x, [1]=∂/∂y, [2]=∂/∂z
inline void transform_to_physical(const double dudxi[3],
                                   const double dudeta[3],
                                   const double dudzeta[3],
                                   const double dd[9],
                                   double du_dx[3][3]) {
    for (int comp = 0; comp < 3; ++comp) {
        du_dx[comp][0] = dudxi[comp] * dd[0] + dudeta[comp] * dd[1] + dudzeta[comp] * dd[2];
        du_dx[comp][1] = dudxi[comp] * dd[3] + dudeta[comp] * dd[4] + dudzeta[comp] * dd[5];
        du_dx[comp][2] = dudxi[comp] * dd[6] + dudeta[comp] * dd[7] + dudzeta[comp] * dd[8];
    }
}

// ---------------------------------------------------------------------------
// 3. C-PML strain correction
// ---------------------------------------------------------------------------
/// Modify physical gradients for PML elements using the precomputed
/// A₆…A₂₃ coefficients and strain memory variables.
///
/// No-op when pml_region is nullptr or pml_region[elem] == 0.
///
/// \param[in]     elem              element index
/// \param[in]     node              GLL node flat index within element (0..NGLL³-1)
/// \param[in]     pml_region        PML region flags per element (nullptr → skip)
/// \param[in]     pml_coef_strain   A₆…A₂₃ coefficients [n_elem·NGLL³·18]
/// \param[in]     rmemory_strain    strain memory [n_elem·NGLL³·27]
/// \param[in,out] du_dx             physical gradients, modified in-place
inline void apply_cpml_strain_correction(int elem, int node, int n_node,
                                          const int32_t* pml_region,
                                          const double* pml_coef_strain,
                                          const double* rmemory_strain,
                                          double du_dx[3][3]) {
    if (!pml_region || pml_region[elem] == 0) return;

    // --- Bring CpmlStrain symbols into scope ---
    using namespace CpmlStrain;
    const int global_node = elem * n_node + node;
    StrainCoefficients coef = load_strain_coefficients(pml_coef_strain, global_node);


    // Off-diagonal: duy/dx and duz/dx — corrected by grad_wrt_x (lijk z,y,x)
    for (int comp : {DUY, DUZ}) {
        double grad = du_dx[comp][DX];
        size_t m_base = strain_memory_offset(global_node, gradient_of(comp, DX), 0);
        double mem_z = rmemory_strain[m_base + CONV_Z];
        double mem_y = rmemory_strain[m_base + CONV_Y];
        double mem_x = rmemory_strain[m_base + CONV_X];
        du_dx[comp][DX] = coef.grad_wrt_x.gradient_prefactor * grad
                        + coef.grad_wrt_x.memory_coef_conv_dir0 * mem_z
                        + coef.grad_wrt_x.memory_coef_conv_dir1 * mem_y
                        + coef.grad_wrt_x.memory_coef_conv_dir2 * mem_x;
    }

    // Off-diagonal: dux/dy and duz/dy — corrected by grad_wrt_y (lijk x,z,y)
    for (int comp : {DUX, DUZ}) {
        double grad = du_dx[comp][DY];
        size_t m_base = strain_memory_offset(global_node, gradient_of(comp, DY), 0);
        double mem_x = rmemory_strain[m_base + CONV_X];
        double mem_z = rmemory_strain[m_base + CONV_Z];
        double mem_y = rmemory_strain[m_base + CONV_Y];
        du_dx[comp][DY] = coef.grad_wrt_y.gradient_prefactor * grad
                        + coef.grad_wrt_y.memory_coef_conv_dir0 * mem_x
                        + coef.grad_wrt_y.memory_coef_conv_dir1 * mem_z
                        + coef.grad_wrt_y.memory_coef_conv_dir2 * mem_y;
    }

    // Off-diagonal: dux/dz and duy/dz — corrected by grad_wrt_z (lijk x,y,z)
    for (int comp : {DUX, DUY}) {
        double grad = du_dx[comp][DZ];
        size_t m_base = strain_memory_offset(global_node, gradient_of(comp, DZ), 0);
        double mem_x = rmemory_strain[m_base + CONV_X];
        double mem_y = rmemory_strain[m_base + CONV_Y];
        double mem_z = rmemory_strain[m_base + CONV_Z];
        du_dx[comp][DZ] = coef.grad_wrt_z.gradient_prefactor * grad
                        + coef.grad_wrt_z.memory_coef_conv_dir0 * mem_x
                        + coef.grad_wrt_z.memory_coef_conv_dir1 * mem_y
                        + coef.grad_wrt_z.memory_coef_conv_dir2 * mem_z;
    }

    // Diagonal: dux/dx — corrected by dux_dx (lx)
    {
        double grad = du_dx[DUX][DX];
        size_t m_off = strain_memory_offset(global_node, DUX_DX, CONV_X);
        double mem_x = rmemory_strain[m_off];
        du_dx[DUX][DX] = coef.dux_dx.gradient_prefactor * grad
                       + coef.dux_dx.memory_coef_local_dir * mem_x;
    }

    // Diagonal: duy/dy — corrected by duy_dy (ly)
    {
        double grad = du_dx[DUY][DY];
        size_t m_off = strain_memory_offset(global_node, DUY_DY, CONV_Y);
        double mem_y = rmemory_strain[m_off];
        du_dx[DUY][DY] = coef.duy_dy.gradient_prefactor * grad
                       + coef.duy_dy.memory_coef_local_dir * mem_y;
    }

    // Diagonal: duz/dz — corrected by duz_dz (lz)
    {
        double grad = du_dx[DUZ][DZ];
        size_t m_off = strain_memory_offset(global_node, DUZ_DZ, CONV_Z);
        double mem_z = rmemory_strain[m_off];
        du_dx[DUZ][DZ] = coef.duz_dz.gradient_prefactor * grad
                       + coef.duz_dz.memory_coef_local_dir * mem_z;
    }
}

// ---------------------------------------------------------------------------
// 4. Symmetric strain tensor
// ---------------------------------------------------------------------------
/// Compute symmetric infinitesimal strain from the displacement gradient.
///
/// \param[in]  du_dx  physical displacement gradient [3][3]
/// \param[out] eps    symmetric strain tensor [3][3]
inline void compute_strain_tensor(const double du_dx[3][3], double eps[3][3]) {
    for (int l = 0; l < 3; ++l) {
        for (int m = 0; m < 3; ++m) {
            eps[l][m] = 0.5 * (du_dx[l][m] + du_dx[m][l]);
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Residual scatter (GLL quadrature assembly)
// ---------------------------------------------------------------------------
/// Scatter the stress contribution at GLL node (i,j,k) to the element
/// residual vector via the weak-form gradient of the Lagrange basis.
///
/// Computes:  r -= σ : ∇N · J · w_i w_j w_k   for all NGLL nodes
/// in the ξ, η, ζ directions.
///
/// \param[in]  i, j, k    GLL node index within the element
/// \param[in]  NGLL       number of GLL points per dimension
/// \param[in]  sigma      stress tensor [3][3] at this node
/// \param[in]  dd         inverse Jacobian [9] (same layout as transform_to_physical)
/// \param[in]  D          GLL derivative matrix  [NGLL × NGLL]
/// \param[in]  weights    GLL quadrature weights [NGLL]
/// \param[in]  jacobian_det   |J| at this node
/// \param[in,out] elem_r  element residual [NGLL³ × 3], accumulated
inline void scatter_residual(int i, int j, int k, int NGLL,
                              const double sigma[3][3],
                              const double dd[9],
                              const double* __restrict D,
                              const double* __restrict weights,
                              double jacobian_det,
                              double* __restrict elem_r) {
    const double factor = jacobian_det * weights[i] * weights[j] * weights[k];

    // --- ξ-direction: contributions to nodes (s, j, k) ---
    for (int s = 0; s < NGLL; ++s) {
        const double Dis = D[i * NGLL + s];
        const double gradN[3] = {Dis * dd[0], Dis * dd[3], Dis * dd[6]};
        const int n_s = (s * NGLL + j) * NGLL + k;
        elem_r[3 * n_s + 0] -= (sigma[0][0] * gradN[0] + sigma[0][1] * gradN[1] +
                                sigma[0][2] * gradN[2]) * factor;
        elem_r[3 * n_s + 1] -= (sigma[1][0] * gradN[0] + sigma[1][1] * gradN[1] +
                                sigma[1][2] * gradN[2]) * factor;
        elem_r[3 * n_s + 2] -= (sigma[2][0] * gradN[0] + sigma[2][1] * gradN[1] +
                                sigma[2][2] * gradN[2]) * factor;
    }

    // --- η-direction: contributions to nodes (i, s, k) ---
    for (int s = 0; s < NGLL; ++s) {
        const double Djs = D[j * NGLL + s];
        const double gradN[3] = {Djs * dd[1], Djs * dd[4], Djs * dd[7]};
        const int n_s = (i * NGLL + s) * NGLL + k;
        elem_r[3 * n_s + 0] -= (sigma[0][0] * gradN[0] + sigma[0][1] * gradN[1] +
                                sigma[0][2] * gradN[2]) * factor;
        elem_r[3 * n_s + 1] -= (sigma[1][0] * gradN[0] + sigma[1][1] * gradN[1] +
                                sigma[1][2] * gradN[2]) * factor;
        elem_r[3 * n_s + 2] -= (sigma[2][0] * gradN[0] + sigma[2][1] * gradN[1] +
                                sigma[2][2] * gradN[2]) * factor;
    }

    // --- ζ-direction: contributions to nodes (i, j, s) ---
    for (int s = 0; s < NGLL; ++s) {
        const double Dks = D[k * NGLL + s];
        const double gradN[3] = {Dks * dd[2], Dks * dd[5], Dks * dd[8]};
        const int n_s = (i * NGLL + j) * NGLL + s;
        elem_r[3 * n_s + 0] -= (sigma[0][0] * gradN[0] + sigma[0][1] * gradN[1] +
                                sigma[0][2] * gradN[2]) * factor;
        elem_r[3 * n_s + 1] -= (sigma[1][0] * gradN[0] + sigma[1][1] * gradN[1] +
                                sigma[1][2] * gradN[2]) * factor;
        elem_r[3 * n_s + 2] -= (sigma[2][0] * gradN[0] + sigma[2][1] * gradN[1] +
                                sigma[2][2] * gradN[2]) * factor;
    }
}

}  // namespace gf