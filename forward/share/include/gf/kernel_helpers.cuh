#pragma once
/// \file kernel_helpers.cuh
/// \brief CUDA __device__ versions of the five shared element kernel helpers.
///
/// Counterpart to kernel_helpers.hpp — identical logic with CUDA-specific
/// atomicAdd scatter.  Included from element_cuda.cu.

#include <cstddef>
#include <cstdint>

#include "gf/pml.hpp"

namespace gf {

// ---------------------------------------------------------------------------
// 1. Reference-space displacement gradient (GLL derivative)
// ---------------------------------------------------------------------------
__device__ inline void compute_reference_gradient(int i, int j, int k, int NGLL,
                                                   const double* __restrict__ D,
                                                   const double* __restrict__ elem_u,
                                                   double* __restrict__ dudxi,
                                                   double* __restrict__ dudeta,
                                                   double* __restrict__ dudzeta) {
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
__device__ inline void transform_to_physical(const double dudxi[3],
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
__device__ inline void apply_cpml_strain_correction(int global_node,
                                                     const int32_t* pml_region,
                                                     int elem_index,
                                                     const double* pml_coef_strain,
                                                     const double* rmemory_strain,
                                                     double du_dx[3][3]) {
    if (!pml_region || pml_region[elem_index] == 0) return;

    using namespace CpmlStrain;
    StrainCoefficients coef = load_strain_coefficients(pml_coef_strain, global_node);

    // Off-diagonal: duy/dx — corrected by grad_wrt_x (lijk z,y,x)
    {
        constexpr int comp = DUY;
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
    // Off-diagonal: duz/dx — corrected by grad_wrt_x (lijk z,y,x)
    {
        constexpr int comp = DUZ;
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

    // Off-diagonal: dux/dy — corrected by grad_wrt_y (lijk x,z,y)
    {
        constexpr int comp = DUX;
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
    // Off-diagonal: duz/dy — corrected by grad_wrt_y (lijk x,z,y)
    {
        constexpr int comp = DUZ;
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

    // Off-diagonal: dux/dz — corrected by grad_wrt_z (lijk x,y,z)
    {
        constexpr int comp = DUX;
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
    // Off-diagonal: duy/dz — corrected by grad_wrt_z (lijk x,y,z)
    {
        constexpr int comp = DUY;
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
__device__ inline void compute_strain_tensor(const double du_dx[3][3], double eps[3][3]) {
    for (int l = 0; l < 3; ++l) {
        for (int m = 0; m < 3; ++m) {
            eps[l][m] = 0.5 * (du_dx[l][m] + du_dx[m][l]);
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Residual scatter with atomicAdd (CUDA)
// ---------------------------------------------------------------------------
__device__ inline void scatter_residual(int i, int j, int k, int NGLL,
                                         const double sigma[3][3],
                                         const double dd[9],
                                         const double* __restrict__ D,
                                         const double* __restrict__ weights,
                                         double jacobian_det,
                                         int elem_offset,
                                         double* __restrict__ elem_r) {
    const double factor = jacobian_det * weights[i] * weights[j] * weights[k];

    // --- ξ-direction ---
    for (int s = 0; s < NGLL; ++s) {
        const double Dis = D[i * NGLL + s];
        const double gradN[3] = {Dis * dd[0], Dis * dd[3], Dis * dd[6]};
        const int base = 3 * (elem_offset + (s * NGLL + j) * NGLL + k);

        double r0 = -(sigma[0][0] * gradN[0] + sigma[0][1] * gradN[1] + sigma[0][2] * gradN[2]) * factor;
        double r1 = -(sigma[1][0] * gradN[0] + sigma[1][1] * gradN[1] + sigma[1][2] * gradN[2]) * factor;
        double r2 = -(sigma[2][0] * gradN[0] + sigma[2][1] * gradN[1] + sigma[2][2] * gradN[2]) * factor;

        atomicAdd(&elem_r[base + 0], r0);
        atomicAdd(&elem_r[base + 1], r1);
        atomicAdd(&elem_r[base + 2], r2);
    }

    // --- η-direction ---
    for (int s = 0; s < NGLL; ++s) {
        const double Djs = D[j * NGLL + s];
        const double gradN[3] = {Djs * dd[1], Djs * dd[4], Djs * dd[7]};
        const int base = 3 * (elem_offset + (i * NGLL + s) * NGLL + k);

        double r0 = -(sigma[0][0] * gradN[0] + sigma[0][1] * gradN[1] + sigma[0][2] * gradN[2]) * factor;
        double r1 = -(sigma[1][0] * gradN[0] + sigma[1][1] * gradN[1] + sigma[1][2] * gradN[2]) * factor;
        double r2 = -(sigma[2][0] * gradN[0] + sigma[2][1] * gradN[1] + sigma[2][2] * gradN[2]) * factor;

        atomicAdd(&elem_r[base + 0], r0);
        atomicAdd(&elem_r[base + 1], r1);
        atomicAdd(&elem_r[base + 2], r2);
    }

    // --- ζ-direction ---
    for (int s = 0; s < NGLL; ++s) {
        const double Dks = D[k * NGLL + s];
        const double gradN[3] = {Dks * dd[2], Dks * dd[5], Dks * dd[8]};
        const int base = 3 * (elem_offset + (i * NGLL + j) * NGLL + s);

        double r0 = -(sigma[0][0] * gradN[0] + sigma[0][1] * gradN[1] + sigma[0][2] * gradN[2]) * factor;
        double r1 = -(sigma[1][0] * gradN[0] + sigma[1][1] * gradN[1] + sigma[1][2] * gradN[2]) * factor;
        double r2 = -(sigma[2][0] * gradN[0] + sigma[2][1] * gradN[1] + sigma[2][2] * gradN[2]) * factor;

        atomicAdd(&elem_r[base + 0], r0);
        atomicAdd(&elem_r[base + 1], r1);
        atomicAdd(&elem_r[base + 2], r2);
    }
}

}  // namespace gf