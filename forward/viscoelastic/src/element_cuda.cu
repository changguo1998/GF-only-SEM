/**
 * @file element_cuda.cu
 * @brief Viscoelastic CUDA element kernel with SLS attenuation.
 *
 * Calls the five shared __device__ helpers from kernel_helpers.cuh.
 * Replaces elastic isotropic stress with SLS viscoelastic stress:
 *   sigma = sigma_elastic - sum(R_l)
 * and updates SLS memory variables inline.
 */

#include "gf/attenuation.hpp"
#include "gf/cuda_check.hpp"
#include "gf/cuda_device_manager.hpp"
#include "gf/cuda_step.hpp"
#include "gf/element.hpp"
#include "gf/kernel_helpers.cuh"
#include "gf/pml.hpp"

namespace gf {

namespace {
// 1D flat index within an element
__device__ static inline int idx(int i, int j, int k, int NGLL) {
    return (i * NGLL + j) * NGLL + k;
}
}  // anonymous namespace

// -----------------------------------------------------------------------
// Element residual kernel
// -----------------------------------------------------------------------
__global__ void element_residual_kernel(
    const double* __restrict__ dxi_dx, const double* __restrict__ jacobian,
    const double* __restrict__ lambda_, const double* __restrict__ mu_,
    const double* __restrict__ D, const double* __restrict__ weights, int NGLL,
    const double* __restrict__ u, double* r, const int32_t* __restrict__ pml_region,
    const double* __restrict__ pml_coef_strain, const double* __restrict__ rmemory_strain,
    double* __restrict__ rmemory_sls, double* __restrict__ strain_old,
    const double* __restrict__ sls_decay, const double* __restrict__ sls_forcing_mu,
    const double* __restrict__ sls_forcing_kappa, bool has_attenuation) {
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

    double lambda = lambda_[global_node];
    double mu = mu_[global_node];
    if (mu <= 0.0)
        return;

    const double* dd = &dxi_dx[9 * global_node];
    const double* elem_u = u + 3 * elem_offset;

    // [1] Reference-space gradient
    double dudxi[3], dudeta[3], dudzeta[3];
    compute_reference_gradient(i, j, k, NGLL, D, elem_u, dudxi, dudeta, dudzeta);

    // [2] Physical gradient
    double du_dx[3][3];
    transform_to_physical(dudxi, dudeta, dudzeta, dd, du_dx);

    // [3-5] PML non-symmetric stress or symmetric SLS path
    double sigma[3][3];
    if (pml_region && pml_region[e] != 0) {
        compute_pml_non_symmetric_stress(global_node, du_dx, lambda, mu, pml_coef_strain,
                                         rmemory_strain, sigma);
    } else {
        // [4] Strain tensor
        double eps[3][3];
        compute_strain_tensor(du_dx, eps);

        // === Step A: elastic trial stress ===
        const double eps_kk = eps[0][0] + eps[1][1] + eps[2][2];
        for (int l = 0; l < 3; ++l) {
            for (int m = 0; m < 3; ++m) {
                sigma[l][m] = 2.0 * mu * eps[l][m];
            }
            sigma[l][l] += lambda * eps_kk;
        }

        // === Step B: subtract old deviatoric and bulk memory ===
        if (has_attenuation && rmemory_sls != nullptr) {
            const double dev_xx = eps[0][0] - eps_kk / 3.0;
            const double dev_yy = eps[1][1] - eps_kk / 3.0;
            const double current_strain[SLS::VOIGT_COMPONENTS] = {dev_xx,    dev_yy,    eps_kk,
                                                                  eps[0][1], eps[0][2], eps[1][2]};

            double memory_sum[SLS::VOIGT_COMPONENTS] = {};
            for (int mechanism = 0; mechanism < SLS::N_SLS; ++mechanism) {
                for (int component = 0; component < SLS::VOIGT_COMPONENTS; ++component) {
                    memory_sum[component] +=
                        rmemory_sls[SLS::sls_memory_offset(global_node, mechanism, component)];
                }
            }
            sigma[0][0] -= memory_sum[SLS::DEVIATORIC_XX] + memory_sum[SLS::TRACE];
            sigma[1][1] -= memory_sum[SLS::DEVIATORIC_YY] + memory_sum[SLS::TRACE];
            sigma[2][2] += memory_sum[SLS::DEVIATORIC_XX] + memory_sum[SLS::DEVIATORIC_YY] -
                           memory_sum[SLS::TRACE];
            sigma[0][1] -= memory_sum[SLS::XY];
            sigma[0][2] -= memory_sum[SLS::XZ];
            sigma[1][2] -= memory_sum[SLS::YZ];

            // === Step C: update memory from previous/current strain ===
            const double kappa = lambda + 2.0 * mu / 3.0;
            for (int mechanism = 0; mechanism < SLS::N_SLS; ++mechanism) {
                const double decay = sls_decay[SLS::coef_offset(global_node, mechanism)];
                const double mu_previous =
                    sls_forcing_mu[SLS::forcing_offset(global_node, mechanism, 0)];
                const double mu_current =
                    sls_forcing_mu[SLS::forcing_offset(global_node, mechanism, 1)];
                const double kappa_previous =
                    sls_forcing_kappa[SLS::forcing_offset(global_node, mechanism, 0)];
                const double kappa_current =
                    sls_forcing_kappa[SLS::forcing_offset(global_node, mechanism, 1)];

                for (int component = SLS::DEVIATORIC_XX; component <= SLS::DEVIATORIC_YY;
                     ++component) {
                    const size_t memory_offset =
                        SLS::sls_memory_offset(global_node, mechanism, component);
                    const double previous = strain_old[SLS::strain_offset(global_node, component)];
                    rmemory_sls[memory_offset] =
                        decay * rmemory_sls[memory_offset] +
                        2.0 * mu *
                            (mu_previous * previous + mu_current * current_strain[component]);
                }
                for (int component = SLS::XY; component <= SLS::YZ; ++component) {
                    const size_t memory_offset =
                        SLS::sls_memory_offset(global_node, mechanism, component);
                    const double previous = strain_old[SLS::strain_offset(global_node, component)];
                    rmemory_sls[memory_offset] =
                        decay * rmemory_sls[memory_offset] +
                        2.0 * mu *
                            (mu_previous * previous + mu_current * current_strain[component]);
                }
                const size_t bulk_offset =
                    SLS::sls_memory_offset(global_node, mechanism, SLS::TRACE);
                const double previous_trace =
                    strain_old[SLS::strain_offset(global_node, SLS::TRACE)];
                rmemory_sls[bulk_offset] =
                    decay * rmemory_sls[bulk_offset] +
                    kappa * (kappa_previous * previous_trace + kappa_current * eps_kk);
            }

            for (int component = 0; component < SLS::VOIGT_COMPONENTS; ++component) {
                strain_old[SLS::strain_offset(global_node, component)] = current_strain[component];
            }
        }

        // Symmetrize
        sigma[1][0] = sigma[0][1];
        sigma[2][0] = sigma[0][2];
        sigma[2][1] = sigma[1][2];
    }  // end SLS (non-PML) path

    // [5] Residual scatter (atomicAdd)
    scatter_residual(i, j, k, NGLL, sigma, dd, D, weights, jacobian[global_node], elem_offset, r);
}

// -----------------------------------------------------------------------
// CUDA specialization: batched element residual
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
                              double* /*strain_old*/, const double* /*sls_decay*/,
                              const double* /*sls_forcing_mu*/,
                              const double* /*sls_forcing_kappa*/, bool /*has_attenuation*/) {
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
        nullptr, nullptr, nullptr, false);

    // --- Check for launch errors ---
    GF_CUDA_CHECK(cudaGetLastError());

    // --- Copy residual back to host ---
    copy_r_to_host(g_cuda_buffers, r);

    // --- Synchronize ---
    GF_CUDA_CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------
// Element residual kernel launch (GPU-native mode)
// -----------------------------------------------------------------------
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
        state.d_rmemory_strain, state.d_rmemory_sls, state.d_strain_old, state.d_sls_decay,
        state.d_sls_forcing_mu, state.d_sls_forcing_kappa, state.has_attenuation);
    GF_CUDA_CHECK(cudaGetLastError());
    GF_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace gf
