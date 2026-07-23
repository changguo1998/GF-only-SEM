/**
 * @file element_cuda.cu
 * @brief CUDA compute_element_residual — elastic.
 *
 * Batched kernel: grid.x = n_elem (one block per element), one thread per
 * GLL node (i,j,k).  Calls the five shared geometry/mechanics helpers from
 * kernel_helpers.cuh and inserts elastic isotropic stress.
 */

#include <cstdio>
#include <cstdlib>

#include "gf/cuda_check.hpp"
#include "gf/cuda_device_manager.hpp"
#include "gf/cuda_step.hpp"
#include "gf/element.hpp"
#include "gf/kernel_helpers.cuh"
#include "gf/pml.hpp"

namespace gf {

// -----------------------------------------------------------------------
// Device helpers
// -----------------------------------------------------------------------

/// 1D flat index from (i, j, k) within an element.
__device__ static inline int idx(int i, int j, int k, int NGLL) {
    return (i * NGLL + j) * NGLL + k;
}

// -----------------------------------------------------------------------
// Element residual kernel (batched over all local elements)
// -----------------------------------------------------------------------

__global__ void element_residual_kernel(
    const double* __restrict__ dxi_dx, const double* __restrict__ jacobian,
    const double* __restrict__ lambda_, const double* __restrict__ mu_,
    const double* __restrict__ D, const double* __restrict__ weights, int NGLL,
    const double* __restrict__ u, double* r, const int32_t* __restrict__ pml_region,
    const double* __restrict__ pml_coef_strain, const double* __restrict__ rmemory_strain,
    double* __restrict__ rmemory_sls, double* __restrict__ sigma_old,
    const double* __restrict__ sls_coef_a, const double* __restrict__ sls_coef_b,
    bool has_attenuation) {
    int e = blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.y;
    int k = threadIdx.z;
    if (i >= NGLL || j >= NGLL || k >= NGLL)
        return;

    int n_node = NGLL * NGLL * NGLL;
    int elem_offset = e * n_node;
    int n = idx(i, j, k, NGLL);
    int global_node = elem_offset + n;

    // --- Material coefficients ---
    double lambda = lambda_[global_node];
    double mu = mu_[global_node];
    if (mu <= 0.0)
        return;

    const double* dd = &dxi_dx[9 * global_node];

    // Point into this element's displacement sub-array
    const double* elem_u = u + 3 * elem_offset;

    // --- [1] Reference-space gradient ---
    double dudxi[3], dudeta[3], dudzeta[3];
    compute_reference_gradient(i, j, k, NGLL, D, elem_u, dudxi, dudeta, dudzeta);

    // --- [2] Physical gradient ---
    double du_dx[3][3];
    transform_to_physical(dudxi, dudeta, dudzeta, dd, du_dx);

    // --- [3-5] Stress and scatter ---
    double sigma[3][3];
    if (pml_region && pml_region[e] != 0) {
        compute_pml_non_symmetric_stress(global_node, du_dx, lambda, mu, pml_coef_strain,
                                         rmemory_strain, sigma);
    } else {
        double eps[3][3];
        compute_strain_tensor(du_dx, eps);
        double eps_kk = eps[0][0] + eps[1][1] + eps[2][2];
        for (int l = 0; l < 3; ++l) {
            for (int m = 0; m < 3; ++m) {
                sigma[l][m] = 2.0 * mu * eps[l][m];
            }
            sigma[l][l] += lambda * eps_kk;
        }
    }
    (void)rmemory_sls;
    (void)sigma_old;
    (void)sls_coef_a;
    (void)sls_coef_b;
    (void)has_attenuation;

    // --- [5] Residual scatter (atomicAdd) ---
    scatter_residual(i, j, k, NGLL, sigma, dd, D, weights, jacobian[global_node], elem_offset, r);
}

// -----------------------------------------------------------------------
// Host-side CUDA dispatch: batched element residual
// -----------------------------------------------------------------------

// File-scope persistent device buffer cache.
namespace {
CudaDeviceBuffers g_cuda_buffers;
}  // anonymous namespace

void compute_element_residual(int n_elem, const double* dxi_dx, const double* jacobian,
                              const double* lambda_, const double* mu_, const double* D,
                              const double* weights, int NGLL, const double* u, double* r,
                              const int32_t* /*pml_region*/, const double* /*pml_coef_strain*/,
                              const double* /*rmemory_strain*/, double* /*rmemory_sls*/,
                              double* /*sigma_old*/, const double* /*sls_coef_a*/,
                              const double* /*sls_coef_b*/, bool /*has_attenuation*/) {
    const int n_node = NGLL * NGLL * NGLL;

    // --- Allocate / reuse device buffers ---
    if (!g_cuda_buffers.allocated || g_cuda_buffers.n_elem != n_elem ||
        g_cuda_buffers.n_node_per_elem != n_node) {
        if (g_cuda_buffers.allocated) {
            free_device_buffers(g_cuda_buffers);
        }
        g_cuda_buffers =
            allocate_device_buffers(n_elem, NGLL, dxi_dx, jacobian, lambda_, mu_, D, weights);
    }

    // --- Copy displacement to device ---
    copy_u_to_device(g_cuda_buffers, u);

    // --- Zero device residual ---
    GF_CUDA_CHECK(cudaMemset(g_cuda_buffers.d_r, 0, n_elem * n_node * 3 * sizeof(double)));

    // --- Launch kernel (one block per element) ---
    dim3 block(NGLL, NGLL, NGLL);
    dim3 grid(n_elem, 1, 1);
    element_residual_kernel<<<grid, block>>>(
        g_cuda_buffers.d_dxi_dx, g_cuda_buffers.d_jacobian, g_cuda_buffers.d_lambda,
        g_cuda_buffers.d_mu, g_cuda_buffers.d_D, g_cuda_buffers.d_weights, NGLL,
        g_cuda_buffers.d_u, g_cuda_buffers.d_r, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, false);

    // --- Check for launch errors ---
    GF_CUDA_CHECK(cudaGetLastError());

    // --- Copy residual back to host ---
    copy_r_to_host(g_cuda_buffers, r);

    // --- Synchronize ---
    GF_CUDA_CHECK(cudaDeviceSynchronize());
}

// Element residual kernel launch using pre-existing device pointers (GPU-native mode).
void cuda_launch_element_residual(const CudaDeviceState& state, int ngll, int n_elem) {
    const int n_node = ngll * ngll * ngll;
    dim3 block(ngll, ngll, ngll);
    dim3 grid(n_elem, 1, 1);

    const double* d_input =
        state.use_global_dof ? state.d_local_cell_displacement : state.d_displacement_tilde;
    double* d_output = state.use_global_dof ? state.d_local_cell_residual : state.d_residual;

    GF_CUDA_CHECK(cudaMemset(d_output, 0, n_elem * n_node * 3 * sizeof(double)));
    element_residual_kernel<<<grid, block>>>(
        state.d_dxi_dx, state.d_jacobian, state.d_lambda_, state.d_mu_, state.d_D, state.d_weights,
        ngll, d_input, d_output, state.d_pml_region, state.d_pml_coef_strain,
        state.d_rmemory_strain, nullptr, nullptr, nullptr, nullptr, false);
    GF_CUDA_CHECK(cudaGetLastError());
    GF_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace gf