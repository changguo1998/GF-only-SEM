/**
 * @file element_cuda.cu
 * @brief CUDA specialization of compute_element_residual<BackendCUDA>.
 *
 * Batched kernel: grid.x = n_elem (one block per element), one thread per
 * GLL node (i,j,k). Each thread computes the quadrature-point contribution
 * and scatters to element residual with atomicAdd.
 */

#define GF_ELEMENT_CUDA_SOURCE
#include "gf/element.hpp"

#include <cstdio>
#include <cstdlib>

#include "gf/cuda_check.h"
#include "gf/cuda_device_manager.hpp"

namespace gf {

// -----------------------------------------------------------------------
// Device helpers (used only inside kernel)
// -----------------------------------------------------------------------

/// 1D flat index from (i, j, k) within an element.
__device__ static inline int idx(int i, int j, int k, int NGLL) {
    return (i * NGLL + j) * NGLL + k;
}

// -----------------------------------------------------------------------
// Element residual kernel (batched over all local elements)
// -----------------------------------------------------------------------

/**
 * CUDA global kernel: compute element residual for all local elements.
 *
 * Grid:    dim3 grid(n_elem, 1, 1)
 * Block:   dim3 block(NGLL, NGLL, NGLL)
 *
 * Each thread (i,j,k) within block e accumulates contributions from
 * quadrature node (i,j,k) to all 3*NGLL^3 DOFs of element e via
 * atomicAdd on the residual.
 */
__global__ void element_residual_kernel(const double* __restrict__ dxi_dx,
                                        const double* __restrict__ jacobian,
                                        const double* __restrict__ lambda_,
                                        const double* __restrict__ mu_,
                                        const double* __restrict__ D,
                                        const double* __restrict__ weights, int NGLL,
                                        const double* __restrict__ u, double* r) {
    // Element index from block
    int e = blockIdx.x;

    // GLL node (i, j, k) from thread
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.z;
    if (i >= NGLL || j >= NGLL || k >= NGLL)
        return;

    int n_node = NGLL * NGLL * NGLL;
    int elem_offset = e * n_node;
    int n = idx(i, j, k, NGLL);  // local node index within element

    // --- Precomputed elastic coefficients at this GLL node ---
    double lambda = lambda_[elem_offset + n];
    double mu = mu_[elem_offset + n];
    if (mu <= 0.0)
        return;

    // --- Inverse Jacobian at this node ---
    const double* dd = &dxi_dx[9 * (elem_offset + n)];
    // dd[0]=dξ/dx, dd[1]=dη/dx, dd[2]=dζ/dx
    // dd[3]=dξ/dy, dd[4]=dη/dy, dd[5]=dζ/dy
    // dd[6]=dξ/dz, dd[7]=dη/dz, dd[8]=dζ/dz

    // --- Displacement gradient in reference space ---
    double dudxi[3] = {0.0, 0.0, 0.0};
    double dudeta[3] = {0.0, 0.0, 0.0};
    double dudzeta[3] = {0.0, 0.0, 0.0};

    for (int s = 0; s < NGLL; ++s) {
        double Di_s = D[i * NGLL + s];
        double Dj_s = D[j * NGLL + s];
        double Dk_s = D[k * NGLL + s];

        int n_sjk = idx(s, j, k, NGLL);
        int n_isk = idx(i, s, k, NGLL);
        int n_ijs = idx(i, j, s, NGLL);

        for (int dir = 0; dir < 3; ++dir) {
            dudxi[dir] += Di_s * u[3 * (elem_offset + n_sjk) + dir];
            dudeta[dir] += Dj_s * u[3 * (elem_offset + n_isk) + dir];
            dudzeta[dir] += Dk_s * u[3 * (elem_offset + n_ijs) + dir];
        }
    }

    // --- Transform to physical gradient ---
    double du_dx[3][3];
    for (int comp = 0; comp < 3; ++comp) {
        du_dx[comp][0] =
            dudxi[comp] * dd[0] + dudeta[comp] * dd[1] + dudzeta[comp] * dd[2];
        du_dx[comp][1] =
            dudxi[comp] * dd[3] + dudeta[comp] * dd[4] + dudzeta[comp] * dd[5];
        du_dx[comp][2] =
            dudxi[comp] * dd[6] + dudeta[comp] * dd[7] + dudzeta[comp] * dd[8];
    }

    // --- Symmetric strain tensor ---
    double eps[3][3];
    for (int l = 0; l < 3; ++l) {
        for (int m = 0; m < 3; ++m) {
            eps[l][m] = 0.5 * (du_dx[l][m] + du_dx[m][l]);
            if (fabs(eps[l][m]) < 1.0e-14) {
                eps[l][m] = 0.0;
            }
        }
    }

    // --- Isotropic stress ---
    double eps_kk = eps[0][0] + eps[1][1] + eps[2][2];
    double sigma[3][3];
    for (int l = 0; l < 3; ++l) {
        for (int m = 0; m < 3; ++m) {
            sigma[l][m] = 2.0 * mu * eps[l][m];
        }
        sigma[l][l] += lambda * eps_kk;
    }

    // --- Quadrature weight factor ---
    double factor = jacobian[elem_offset + n] * weights[i] * weights[j] * weights[k];

    // --- Scatter residual contributions with atomicAdd ---
    // ξ-direction contributions to nodes (s, j, k)
    for (int s = 0; s < NGLL; ++s) {
        double Dis = D[i * NGLL + s];
        double gradN[3] = {Dis * dd[0], Dis * dd[3], Dis * dd[6]};
        int base = 3 * (elem_offset + idx(s, j, k, NGLL));

        double r0 = -(sigma[0][0] * gradN[0] + sigma[0][1] * gradN[1] + sigma[0][2] * gradN[2]) * factor;
        double r1 = -(sigma[1][0] * gradN[0] + sigma[1][1] * gradN[1] + sigma[1][2] * gradN[2]) * factor;
        double r2 = -(sigma[2][0] * gradN[0] + sigma[2][1] * gradN[1] + sigma[2][2] * gradN[2]) * factor;

        atomicAdd(&r[base + 0], r0);
        atomicAdd(&r[base + 1], r1);
        atomicAdd(&r[base + 2], r2);
    }

    // η-direction contributions to nodes (i, s, k)
    for (int s = 0; s < NGLL; ++s) {
        double Djs = D[j * NGLL + s];
        double gradN[3] = {Djs * dd[1], Djs * dd[4], Djs * dd[7]};
        int base = 3 * (elem_offset + idx(i, s, k, NGLL));

        double r0 = -(sigma[0][0] * gradN[0] + sigma[0][1] * gradN[1] + sigma[0][2] * gradN[2]) * factor;
        double r1 = -(sigma[1][0] * gradN[0] + sigma[1][1] * gradN[1] + sigma[1][2] * gradN[2]) * factor;
        double r2 = -(sigma[2][0] * gradN[0] + sigma[2][1] * gradN[1] + sigma[2][2] * gradN[2]) * factor;

        atomicAdd(&r[base + 0], r0);
        atomicAdd(&r[base + 1], r1);
        atomicAdd(&r[base + 2], r2);
    }

    // ζ-direction contributions to nodes (i, j, s)
    for (int s = 0; s < NGLL; ++s) {
        double Dks = D[k * NGLL + s];
        double gradN[3] = {Dks * dd[2], Dks * dd[5], Dks * dd[8]};
        int base = 3 * (elem_offset + idx(i, j, s, NGLL));

        double r0 = -(sigma[0][0] * gradN[0] + sigma[0][1] * gradN[1] + sigma[0][2] * gradN[2]) * factor;
        double r1 = -(sigma[1][0] * gradN[0] + sigma[1][1] * gradN[1] + sigma[1][2] * gradN[2]) * factor;
        double r2 = -(sigma[2][0] * gradN[0] + sigma[2][1] * gradN[1] + sigma[2][2] * gradN[2]) * factor;

        atomicAdd(&r[base + 0], r0);
        atomicAdd(&r[base + 1], r1);
        atomicAdd(&r[base + 2], r2);
    }
}

// -----------------------------------------------------------------------
// CUDA specialization: batched element residual
// -----------------------------------------------------------------------

// File-scope persistent device buffer cache.
namespace {
CudaDeviceBuffers g_cuda_buffers;
}  // anonymous namespace

template <>
void compute_element_residual<BackendCUDA>(int n_elem, const double* dxi_dx,
                                           const double* jacobian, const double* lambda_,
                                           const double* mu_,
                                           const double* D, const double* weights, int NGLL,
                                           const double* u, double* r) {
#ifdef GF_WITH_CUDA
    const int n_node = NGLL * NGLL * NGLL;

    // --- Allocate / reuse device buffers ---
    if (!g_cuda_buffers.allocated || g_cuda_buffers.n_elem != n_elem ||
        g_cuda_buffers.n_node_per_elem != n_node) {
        if (g_cuda_buffers.allocated) {
            free_device_buffers(g_cuda_buffers);
        }
        g_cuda_buffers = allocate_device_buffers(n_elem, NGLL, dxi_dx, jacobian, lambda_, mu_,
                                                  D, weights);
    }

    // --- Copy displacement to device ---
    copy_u_to_device(g_cuda_buffers, u);

    // --- Zero device residual ---
    GF_CUDA_CHECK(cudaMemset(g_cuda_buffers.d_r, 0, n_elem * n_node * 3 * sizeof(double)));

    // --- Launch kernel (one block per element) ---
    dim3 block(NGLL, NGLL, NGLL);
    dim3 grid(n_elem, 1, 1);
    element_residual_kernel<<<grid, block>>>(g_cuda_buffers.d_dxi_dx, g_cuda_buffers.d_jacobian,
                                              g_cuda_buffers.d_lambda, g_cuda_buffers.d_mu,
                                              g_cuda_buffers.d_D,
                                              g_cuda_buffers.d_weights, NGLL, g_cuda_buffers.d_u,
                                              g_cuda_buffers.d_r);

    // --- Check for launch errors ---
    GF_CUDA_CHECK(cudaGetLastError());

    // --- Copy residual back to host ---
    copy_r_to_host(g_cuda_buffers, r);

    // --- Synchronize ---
    GF_CUDA_CHECK(cudaDeviceSynchronize());
#else
    (void)n_elem;
    (void)dxi_dx;
    (void)jacobian;
    (void)lambda_;
    (void)mu_;
    (void)D;
    (void)weights;
    (void)NGLL;
    (void)u;
    (void)r;
    fprintf(stderr,
            "compute_element_residual<BackendCUDA> called without GF_WITH_CUDA. "
            "Recompile with -DGF_WITH_CUDA and CUDA toolkit.\n");
    std::abort();
#endif
}

}  // namespace gf