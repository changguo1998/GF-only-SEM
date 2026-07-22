// forward/share/src/cuda_step.cu
// GPU kernels for single-GPU mode: Newmark, PML, source injection, strain,
// and CG-SEM global scatter/gather.
// Keeps all state on device, no per-step H2D/D2H copies.

#include "gf/cuda_check.h"
#include "gf/cuda_step.hpp"
#include "gf/pml.hpp"

namespace gf {
using namespace CpmlStrain;

// -----------------------------------------------------------------------
// Helper: device flat index
// -----------------------------------------------------------------------
__device__ static inline int node_idx(int i, int j, int k, int ngll) {
    return (i * ngll + j) * ngll + k;
}

// =======================================================================
// Element-local kernels (legacy / backward compat)
// =======================================================================

__global__ void newmark_predict_kernel(double* d_disp_tilde, const double* d_disp,
                                       const double* d_vel, const double* d_acc, double dt,
                                       double beta_factor, int n_dof) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_dof) {
        d_disp_tilde[i] = d_disp[i] + dt * d_vel[i] + beta_factor * d_acc[i];
    }
}

__global__ void pml_damping_kernel(double* d_vel, const double* d_pml, int n_dof, int n_nodes) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_dof) {
        int node = i / 3;
        if (node < n_nodes) {
            double d = d_pml[node];
            if (d > 0.0) {
                d_vel[i] -= d * d_vel[i];
            }
        }
    }
}

__global__ void source_injection_kernel(double* d_residual, const double* d_src_weights,
                                        double stf_val, int dir, int n_src, int n_node,
                                        const int* d_src_elem_offsets) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n_src * n_node) {
        int si = idx / n_node;
        int n = idx % n_node;
        double w = d_src_weights[si * n_node + n];
        if (w != 0.0) {
            int elem = d_src_elem_offsets[si];
            if (elem >= 0) {
                int dof_base = (elem * n_node + n) * 3;
                atomicAdd(&d_residual[dof_base + dir], stf_val * w);
            }
        }
    }
}

__global__ void newmark_correct_kernel(double* d_disp, double* d_vel, double* d_acc,
                                       const double* d_residual, const double* d_mass, double dt,
                                       double beta, double gamma, int n_dof, int n_nodes) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_dof) {
        int node = i / 3;
        if (node < n_nodes) {
            double m = d_mass[node];
            if (m <= 0.0) {
                d_acc[i] = 0.0;
                return;
            }
            double a_old = d_acc[i];
            double a_new = d_residual[i] / m;
            d_disp[i] += dt * d_vel[i] + dt * dt * ((0.5 - beta) * a_old + beta * a_new);
            d_vel[i] += dt * ((1.0 - gamma) * a_old + gamma * a_new);
            d_acc[i] = a_new;
        }
    }
}

// =======================================================================
// Global-DOF kernels (CG-SEM assembly)
// =======================================================================

/// Scatter element-local residual → global residual.
/// Uses atomicAdd because multiple elements sharing a physical node (same
/// node_id) write concurrently to the same destination DOF.
__global__ void scatter_to_rank_kernel(const double* d_local_cell_residual,
                                       const int* d_local_cell2rank_node,
                                       double* d_rank_node_residual, int n_local_cell,
                                       int n_node) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = n_local_cell * n_node * 3;
    if (idx < total) {
        int direction = idx % 3;
        int node = (idx / 3) % n_node;
        int elem = idx / (n_node * 3);
        int node_id = d_local_cell2rank_node[elem * n_node + node];
        atomicAdd(&d_rank_node_residual[node_id * 3 + direction], d_local_cell_residual[idx]);
    }
}

/// Gather global field → element-local.
/// One-to-one mapping, no atomics needed.
__global__ void gather_from_rank_kernel(const double* d_rank_node_field,
                                        const int* d_local_cell2rank_node, double* d_elem_field,
                                        int n_local_cell, int n_node) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = n_local_cell * n_node * 3;
    if (idx < total) {
        int direction = idx % 3;
        int node = (idx / 3) % n_node;
        int elem = idx / (n_node * 3);
        int node_id = d_local_cell2rank_node[elem * n_node + node];
        d_elem_field[idx] = d_rank_node_field[node_id * 3 + direction];
    }
}

/// PML damping on global velocity (direct — each physical node appears once).
__global__ void pml_damping_rank_kernel(double* d_rank_node_velocity,
                                        const double* d_rank_node_damping, int n_rank_node) {
    int node_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_id < n_rank_node) {
        double d = d_rank_node_damping[node_id];
        if (d > 0.0) {
            int base = node_id * 3;
            d_rank_node_velocity[base + 0] -= d * d_rank_node_velocity[base + 0];
            d_rank_node_velocity[base + 1] -= d * d_rank_node_velocity[base + 1];
            d_rank_node_velocity[base + 2] -= d * d_rank_node_velocity[base + 2];
        }
    }
}

/// Newmark predictor on global arrays.
__global__ void newmark_predict_rank_kernel(double* d_disp_tilde, const double* d_disp,
                                            const double* d_vel, const double* d_acc, double dt,
                                            double beta_factor, int n_rank_dof) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_rank_dof) {
        d_disp_tilde[i] = d_disp[i] + dt * d_vel[i] + beta_factor * d_acc[i];
    }
}

/// Newmark corrector on global arrays.  Uses per-node global mass.
/// u += dt*v + dt²*((0.5-β)*a_old + β*a_new), v += dt*((1-γ)*a_old + γ*a_new)
__global__ void newmark_correct_rank_kernel(double* d_disp, double* d_vel, double* d_acc,
                                            const double* d_residual,
                                            const double* d_rank_node_mass, double dt, double beta,
                                            double gamma, int n_rank_dof) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_rank_dof) {
        int node_id = i / 3;
        double m = d_rank_node_mass[node_id];
        if (m <= 0.0) {
            d_acc[i] = 0.0;
            return;
        }
        double a_old = d_acc[i];
        double a_new = d_residual[i] / m;
        d_disp[i] += dt * d_vel[i] + dt * dt * ((0.5 - beta) * a_old + beta * a_new);
        d_vel[i] += dt * ((1.0 - gamma) * a_old + gamma * a_new);
        d_acc[i] = a_new;
    }
}

// =======================================================================
// Strain computation kernels
// =======================================================================
// Host-callable wrappers
// =======================================================================

static int grid_blocks(int n, int threads_per_block = 256) {
    return (n + threads_per_block - 1) / threads_per_block;
}

void cuda_newmark_predict(CudaDeviceState& state, double dt, double beta) {
    double beta_factor = 0.5 * dt * dt * (1.0 - 2.0 * beta);
    if (state.use_global_dof) {
        int n = state.n_rank_node * 3;
        newmark_predict_rank_kernel<<<grid_blocks(n), 256>>>(
            state.d_rank_node_displacement_tilde, state.d_rank_node_displacement,
            state.d_rank_node_velocity, state.d_rank_node_acceleration, dt, beta_factor, n);
    } else {
        newmark_predict_kernel<<<grid_blocks(state.n_dof), 256>>>(
            state.d_displacement_tilde, state.d_displacement, state.d_velocity,
            state.d_acceleration, dt, beta_factor, state.n_dof);
    }
    GF_CUDA_CHECK(cudaGetLastError());
    GF_CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_zero_residual(CudaDeviceState& state) {
    if (state.use_global_dof) {
        GF_CUDA_CHECK(
            cudaMemset(state.d_local_cell_residual, 0, state.n_local_cell_dof * sizeof(double)));
    } else {
        GF_CUDA_CHECK(cudaMemset(state.d_residual, 0, state.n_dof * sizeof(double)));
    }
}

void cuda_pml_damping(CudaDeviceState& state) {
    if (state.use_global_dof) {
        pml_damping_rank_kernel<<<grid_blocks(state.n_rank_node), 256>>>(
            state.d_rank_node_velocity, state.d_rank_node_damping, state.n_rank_node);
    } else {
        pml_damping_kernel<<<grid_blocks(state.n_dof), 256>>>(state.d_velocity, state.d_pml,
                                                              state.n_dof, state.n_total_nodes);
    }
    GF_CUDA_CHECK(cudaGetLastError());
}

void cuda_source_injection(CudaDeviceState& state, int direction, double stf_val,
                           const double* h_src_weights, int n_src_cell) {
    if (stf_val == 0.0 || n_src_cell == 0)
        return;

    // Upload source weights to device (cached)
    static double* d_src_weights = nullptr;
    static int d_src_weights_size = 0;
    int total_weights = n_src_cell * state.n_node;
    if (d_src_weights == nullptr || d_src_weights_size < total_weights) {
        if (d_src_weights)
            cudaFree(d_src_weights);
        GF_CUDA_CHECK(cudaMalloc(&d_src_weights, total_weights * sizeof(double)));
        d_src_weights_size = total_weights;
    }
    GF_CUDA_CHECK(cudaMemcpy(d_src_weights, h_src_weights, total_weights * sizeof(double),
                             cudaMemcpyHostToDevice));

    int block = 256;
    int total_threads = n_src_cell * state.n_node;
    int grid = (total_threads + block - 1) / block;

    // Source always injects into element-local residual
    // (with global DOF: local_cell_residual → scatter_to_rank later)
    double* d_target = state.use_global_dof ? state.d_local_cell_residual : state.d_residual;
    source_injection_kernel<<<grid, block>>>(d_target, d_src_weights, stf_val, direction,
                                             n_src_cell, state.n_node, state.d_src_elem_offsets);
    GF_CUDA_CHECK(cudaGetLastError());
}

void cuda_newmark_correct(CudaDeviceState& state, double dt, double beta, double gamma) {
    if (state.use_global_dof) {
        int n = state.n_rank_node * 3;
        newmark_correct_rank_kernel<<<grid_blocks(n), 256>>>(
            state.d_rank_node_displacement, state.d_rank_node_velocity,
            state.d_rank_node_acceleration, state.d_rank_node_residual, state.d_rank_node_mass, dt,
            beta, gamma, n);
    } else {
        newmark_correct_kernel<<<grid_blocks(state.n_dof), 256>>>(
            state.d_displacement, state.d_velocity, state.d_acceleration, state.d_residual,
            state.d_mass, dt, beta, gamma, state.n_dof, state.n_total_nodes);
    }
    GF_CUDA_CHECK(cudaGetLastError());
    GF_CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_copy_state_to_host(const CudaDeviceState& state, std::vector<double>& h_displacement,
                             std::vector<double>& h_velocity,
                             std::vector<double>& h_acceleration) {
    if (state.use_global_dof) {
        int n = state.n_rank_node * 3;
        GF_CUDA_CHECK(cudaMemcpy(h_displacement.data(), state.d_rank_node_displacement,
                                 n * sizeof(double), cudaMemcpyDeviceToHost));
        GF_CUDA_CHECK(cudaMemcpy(h_velocity.data(), state.d_rank_node_velocity, n * sizeof(double),
                                 cudaMemcpyDeviceToHost));
        GF_CUDA_CHECK(cudaMemcpy(h_acceleration.data(), state.d_rank_node_acceleration,
                                 n * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        GF_CUDA_CHECK(cudaMemcpy(h_displacement.data(), state.d_displacement,
                                 state.n_dof * sizeof(double), cudaMemcpyDeviceToHost));
        GF_CUDA_CHECK(cudaMemcpy(h_velocity.data(), state.d_velocity, state.n_dof * sizeof(double),
                                 cudaMemcpyDeviceToHost));
        GF_CUDA_CHECK(cudaMemcpy(h_acceleration.data(), state.d_acceleration,
                                 state.n_dof * sizeof(double), cudaMemcpyDeviceToHost));
    }
}

// =======================================================================
// CG-SEM global scatter/gather host wrappers
// =======================================================================

void cuda_scatter_to_rank(CudaDeviceState& state) {
    // Zero global residual before accumulation (mirrors CPU scatter_to_rank
    // which std::fill(rank_node_residual, 0) before += ). Without this, the
    // atomicAdd accumulates across timesteps -> unbounded growth -> explosion.
    GF_CUDA_CHECK(
        cudaMemset(state.d_rank_node_residual, 0, state.n_rank_node * 3 * sizeof(double)));
    scatter_to_rank_kernel<<<grid_blocks(state.n_local_cell_dof), 256>>>(
        state.d_local_cell_residual, state.d_local_cell2rank_node, state.d_rank_node_residual,
        state.n_local_cell, state.n_node);
    GF_CUDA_CHECK(cudaGetLastError());
}

void cuda_gather_from_rank(CudaDeviceState& state) {
    gather_from_rank_kernel<<<grid_blocks(state.n_local_cell_dof), 256>>>(
        state.d_rank_node_displacement, state.d_local_cell2rank_node,
        state.d_local_cell_displacement, state.n_local_cell, state.n_node);
    GF_CUDA_CHECK(cudaGetLastError());
}

/// Gather predicted displacement (displacement_tilde) for the element kernel.
void cuda_gather_predicted(CudaDeviceState& state) {
    gather_from_rank_kernel<<<grid_blocks(state.n_local_cell_dof), 256>>>(
        state.d_rank_node_displacement_tilde, state.d_local_cell2rank_node,
        state.d_local_cell_displacement, state.n_local_cell, state.n_node);
    GF_CUDA_CHECK(cudaGetLastError());
}

// =======================================================================
// MPI exchange staging (host-staged D2H/H2D for multi-rank CG-SEM)
// =======================================================================
// These copy state vectors device↔host so solver.cpp can run exchange_halo
// (MPI) on host buffers. For single-GPU (GF_NO_MPI) exchange_halo is a no-op,
// and exchange_patterns is empty, so these are never called.

void cuda_copy_utilde_to_host(const CudaDeviceState& state, double* host_buf) {
    GF_CUDA_CHECK(cudaMemcpy(host_buf, state.d_rank_node_displacement_tilde,
                             state.n_rank_node * 3 * sizeof(double), cudaMemcpyDeviceToHost));
}

void cuda_copy_utilde_from_host(CudaDeviceState& state, const double* host_buf) {
    GF_CUDA_CHECK(cudaMemcpy(state.d_rank_node_displacement_tilde, host_buf,
                             state.n_rank_node * 3 * sizeof(double), cudaMemcpyHostToDevice));
}

void cuda_copy_residual_to_host(const CudaDeviceState& state, double* host_buf) {
    if (state.use_global_dof) {
        GF_CUDA_CHECK(cudaMemcpy(host_buf, state.d_rank_node_residual,
                                 state.n_rank_node * 3 * sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        GF_CUDA_CHECK(cudaMemcpy(host_buf, state.d_residual, state.n_dof * sizeof(double),
                                 cudaMemcpyDeviceToHost));
    }
}

void cuda_copy_residual_from_host(CudaDeviceState& state, const double* host_buf) {
    if (state.use_global_dof) {
        GF_CUDA_CHECK(cudaMemcpy(state.d_rank_node_residual, host_buf,
                                 state.n_rank_node * 3 * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        GF_CUDA_CHECK(cudaMemcpy(state.d_residual, host_buf, state.n_dof * sizeof(double),
                                 cudaMemcpyHostToDevice));
    }
}

// =======================================================================
// C-PML kernels and wrappers
// =======================================================================

// Per-node update of PML displacement fields.
// Each thread processes one GLL node in one PML element.
__global__ void cpml_displ_fields_kernel(double* d_displ_new, double* d_displ_old,
                                         const double* d_rank_node_displacement,
                                         const double* d_rank_node_velocity,
                                         const double* d_rank_node_acceleration,
                                         const int* d_local_cell2rank_node,
                                         const int32_t* d_pml_region, int n_local_cell, int n_node,
                                         double c1, double c2) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_local_cell * n_node)
        return;

    int e = idx / n_node;
    if (d_pml_region[e] == 0)
        return;

    int elem_off = idx * 3;  // 3 DOFs per node
    int rank_node = d_local_cell2rank_node[idx];
    int rank_dof = rank_node * 3;

    // PML_displ_new = u + c1 * v
    for (int d = 0; d < 3; ++d) {
        d_displ_new[elem_off + d] =
            d_rank_node_displacement[rank_dof + d] + c1 * d_rank_node_velocity[rank_dof + d];
    }

    // PML_displ_old += c2 * a  (old was swapped to previous new in host)
    for (int d = 0; d < 3; ++d) {
        d_displ_old[elem_off + d] += c2 * d_rank_node_acceleration[rank_dof + d];
    }
}

// Per-node update of displacement memory (α-convolution).
__global__ void cpml_displ_memory_kernel(double* d_rmemory_displ, const double* d_pml_displ_new,
                                         const double* d_pml_displ_old,
                                         const double* d_pml_coef_alpha,
                                         const int32_t* d_pml_region, int n_local_cell,
                                         int n_node) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_local_cell * n_node)
        return;

    int e = idx / n_node;
    if (d_pml_region[e] == 0)
        return;

    int elem_off = idx;
    int node_mem_off = elem_off * 9;
    int coef_off = elem_off * 9;
    int displ_off = elem_off * 3;

    for (int comp = 0; comp < 3; ++comp) {
        double new_val = d_pml_displ_new[displ_off + comp];
        double old_val = d_pml_displ_old[displ_off + comp];
        for (int d = 0; d < 3; ++d) {
            int mem_idx = node_mem_off + comp * 3 + d;
            int coef_idx = coef_off + d * 3;
            d_rmemory_displ[mem_idx] = d_pml_coef_alpha[coef_idx + 0] * d_rmemory_displ[mem_idx] +
                                       d_pml_coef_alpha[coef_idx + 1] * new_val +
                                       d_pml_coef_alpha[coef_idx + 2] * old_val;
        }
    }
}

// Per-node update of strain memory (β-convolution).
// Computes reference-space gradients via diff ops on PML displ fields,
// transforms to physical gradients, then updates rmemory_strain.
__global__ void cpml_strain_memory_kernel(double* d_rmemory_strain, const double* d_pml_displ_new,
                                          const double* d_pml_displ_old,
                                          const double* d_pml_coef_beta,
                                          const int32_t* d_pml_region, const double* d_D,
                                          const double* dxi_dx, int ngll, int n_local_cell,
                                          int n_node) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_local_cell * n_node)
        return;

    int e = idx / n_node;
    if (d_pml_region[e] == 0)
        return;

    int n = idx % n_node;
    int elem_off = e * n_node;
    int i = n / (ngll * ngll);
    int j = (n / ngll) % ngll;
    int k = n % ngll;

    // --- 1. Compute reference-space gradients ---
    double new_dudxi[3] = {0, 0, 0};
    double new_dudeta[3] = {0, 0, 0};
    double new_dudzeta[3] = {0, 0, 0};
    double old_dudxi[3] = {0, 0, 0};
    double old_dudeta[3] = {0, 0, 0};
    double old_dudzeta[3] = {0, 0, 0};

    for (int s = 0; s < ngll; ++s) {
        double Di_s = d_D[i * ngll + s];
        double Dj_s = d_D[j * ngll + s];
        double Dk_s = d_D[k * ngll + s];

        int n_sjk = (s * ngll + j) * ngll + k;
        int n_isk = (i * ngll + s) * ngll + k;
        int n_ijs = (i * ngll + j) * ngll + s;

        int new_sjk = (elem_off + n_sjk) * 3;
        int new_isk = (elem_off + n_isk) * 3;
        int new_ijs = (elem_off + n_ijs) * 3;
        int old_sjk = (elem_off + n_sjk) * 3;
        int old_isk = (elem_off + n_isk) * 3;
        int old_ijs = (elem_off + n_ijs) * 3;

        for (int dir = 0; dir < 3; ++dir) {
            new_dudxi[dir] += Di_s * d_pml_displ_new[new_sjk + dir];
            new_dudeta[dir] += Dj_s * d_pml_displ_new[new_isk + dir];
            new_dudzeta[dir] += Dk_s * d_pml_displ_new[new_ijs + dir];
            old_dudxi[dir] += Di_s * d_pml_displ_old[old_sjk + dir];
            old_dudeta[dir] += Dj_s * d_pml_displ_old[old_isk + dir];
            old_dudzeta[dir] += Dk_s * d_pml_displ_old[old_ijs + dir];
        }
    }

    // --- 2. Transform to physical gradients ---
    const double* dd = &dxi_dx[(elem_off + n) * 9];
    double new_phys_grad[NUM_GRADIENT_COMPS];
    double old_phys_grad[NUM_GRADIENT_COMPS];

    for (int comp = 0; comp < 3; ++comp) {
        int c3 = comp * 3;
        new_phys_grad[c3 + 0] =
            new_dudxi[comp] * dd[0] + new_dudeta[comp] * dd[1] + new_dudzeta[comp] * dd[2];
        new_phys_grad[c3 + 1] =
            new_dudxi[comp] * dd[3] + new_dudeta[comp] * dd[4] + new_dudzeta[comp] * dd[5];
        new_phys_grad[c3 + 2] =
            new_dudxi[comp] * dd[6] + new_dudeta[comp] * dd[7] + new_dudzeta[comp] * dd[8];
        old_phys_grad[c3 + 0] =
            old_dudxi[comp] * dd[0] + old_dudeta[comp] * dd[1] + old_dudzeta[comp] * dd[2];
        old_phys_grad[c3 + 1] =
            old_dudxi[comp] * dd[3] + old_dudeta[comp] * dd[4] + old_dudzeta[comp] * dd[5];
        old_phys_grad[c3 + 2] =
            old_dudxi[comp] * dd[6] + old_dudeta[comp] * dd[7] + old_dudzeta[comp] * dd[8];
    }

    // --- 3. Update strain memory with β convolution ---
    for (int grad = 0; grad < NUM_GRADIENT_COMPS; ++grad) {
        double new_g = new_phys_grad[grad];
        double old_g = old_phys_grad[grad];
        for (int conv_dir = 0; conv_dir < NUM_CONV_DIRECTIONS; ++conv_dir) {
            size_t mem_off = strain_memory_offset(elem_off + n, grad, conv_dir);
            int beta_off = (elem_off + n) * BETA_COEFS_PER_NODE + conv_dir * BETA_COEFS_PER_DIR;
            d_rmemory_strain[mem_off] =
                d_pml_coef_beta[beta_off + BETA_COEF0] * d_rmemory_strain[mem_off] +
                d_pml_coef_beta[beta_off + BETA_COEF1] * new_g +
                d_pml_coef_beta[beta_off + BETA_COEF2] * old_g;
        }
    }
}

// Per-node acceleration correction (Ā₁…Ā₅).
// Adds PML contribution to element-local residual.
__global__ void cpml_accel_kernel(double* d_residual, const double* d_rank_node_displacement,
                                  const double* d_rank_node_velocity,
                                  const int* d_local_cell2rank_node, const int32_t* d_pml_region,
                                  const double* d_pml_coef_abar, const double* d_rmemory_displ,
                                  const double* d_density, const double* d_jacobian,
                                  const double* d_weights, int ngll, int n_local_cell,
                                  int n_node) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_local_cell * n_node)
        return;

    int e = idx / n_node;
    if (d_pml_region[e] == 0)
        return;

    int n = idx % n_node;
    int elem_off = e * n_node;
    int rank_node = d_local_cell2rank_node[idx];
    int rank_dof = rank_node * 3;

    // GLL weight product
    int nk = n % ngll;
    int nj = (n / ngll) % ngll;
    int ni = n / (ngll * ngll);
    double wgll = d_weights[ni] * d_weights[nj] * d_weights[nk];

    // Coefficients Ā₁…Ā₅
    int node_coef_off = (elem_off + n) * 5;
    double A1 = d_pml_coef_abar[node_coef_off + 0];
    double A2 = d_pml_coef_abar[node_coef_off + 1];
    double A3 = d_pml_coef_abar[node_coef_off + 2];
    double A4 = d_pml_coef_abar[node_coef_off + 3];
    double A5 = d_pml_coef_abar[node_coef_off + 4];

    // Scale factor: w * ρ * J (matches SPECFEM3D pml_compute_accel_contribution)
    double rho = d_density[elem_off + n];
    double jac = d_jacobian[elem_off + n];
    double rho_val = (rho > 0.0) ? rho : 0.0;
    double scale = wgll * rho_val * jac;

    int node_mem_off = (elem_off + n) * 9;
    int elem_resid_off = idx * 3;

    for (int comp = 0; comp < 3; ++comp) {
        double u_val = d_rank_node_displacement[rank_dof + comp];
        double v_val = d_rank_node_velocity[rank_dof + comp];
        double mem_x = d_rmemory_displ[node_mem_off + comp * 3 + 0];
        double mem_y = d_rmemory_displ[node_mem_off + comp * 3 + 1];
        double mem_z = d_rmemory_displ[node_mem_off + comp * 3 + 2];

        double accel_pml =
            scale * (A1 * v_val + A2 * u_val + A3 * mem_x + A4 * mem_y + A5 * mem_z);
        d_residual[elem_resid_off + comp] += accel_pml;
    }
}

// -----------------------------------------------------------------------
// Host wrappers
// -----------------------------------------------------------------------

void cuda_cpml_update_displ_fields(CudaDeviceState& state, double solver_dt, int n_node) {
    if (!state.has_cpml)
        return;

    constexpr double THETA_CPML = 1.0 / 8.0;
    double c1 = (1.0 - 2.0 * THETA_CPML) * 0.5 * solver_dt;
    double c2 = (1.0 - THETA_CPML) * 0.5 * solver_dt * solver_dt;

    int n_total = state.n_local_cell * n_node;
    cpml_displ_fields_kernel<<<grid_blocks(n_total), 256>>>(
        state.d_pml_displ_new, state.d_pml_displ_old, state.d_rank_node_displacement,
        state.d_rank_node_velocity, state.d_rank_node_acceleration, state.d_local_cell2rank_node,
        state.d_pml_region, state.n_local_cell, n_node, c1, c2);
    GF_CUDA_CHECK(cudaGetLastError());
}

void cuda_cpml_update_displ_memory(CudaDeviceState& state, int n_node) {
    if (!state.has_cpml)
        return;

    int n_total = state.n_local_cell * n_node;
    cpml_displ_memory_kernel<<<grid_blocks(n_total), 256>>>(
        state.d_rmemory_displ, state.d_pml_displ_new, state.d_pml_displ_old,
        state.d_pml_coef_alpha, state.d_pml_region, state.n_local_cell, n_node);
    GF_CUDA_CHECK(cudaGetLastError());
}

void cuda_cpml_update_strain_memory(CudaDeviceState& state, int ngll, int n_node) {
    if (!state.has_cpml)
        return;

    int n_total = state.n_local_cell * n_node;
    cpml_strain_memory_kernel<<<grid_blocks(n_total), 256>>>(
        state.d_rmemory_strain, state.d_pml_displ_new, state.d_pml_displ_old,
        state.d_pml_coef_beta, state.d_pml_region, state.d_D, state.d_dxi_dx, ngll,
        state.n_local_cell, n_node);
    GF_CUDA_CHECK(cudaGetLastError());
}

void cuda_cpml_accel_contribution(CudaDeviceState& state, int ngll, int n_node) {
    if (!state.has_cpml)
        return;

    int n_total = state.n_local_cell * n_node;
    cpml_accel_kernel<<<grid_blocks(n_total), 256>>>(
        state.d_local_cell_residual, state.d_rank_node_displacement, state.d_rank_node_velocity,
        state.d_local_cell2rank_node, state.d_pml_region, state.d_pml_coef_abar,
        state.d_rmemory_displ, state.d_density, state.d_jacobian, state.d_weights, ngll,
        state.n_local_cell, n_node);
    GF_CUDA_CHECK(cudaGetLastError());
}
// =======================================================================
// Allocation / free
// =======================================================================

CudaDeviceState cuda_allocate_state(
    int n_local_cell, int ngll, const std::vector<double>& mass,
    const std::vector<double>& pml_damping, const std::vector<double>& dxi_dx,
    const std::vector<double>& jacobian, const std::vector<double>& lambda_,
    const std::vector<double>& mu_, const double* h_D, const double* h_weights,
    const ConfigData& cfg, int n_local_cell_dof, const std::vector<int32_t>& local_cell2rank_node,
    int n_rank_node, const std::vector<double>& rank_node_mass,
    const std::vector<double>& rank_node_damping) {
    CudaDeviceState state;
    state.n_dof = n_local_cell_dof;
    state.n_local_cell_dof = n_local_cell_dof;
    state.n_total_nodes = n_local_cell * ngll * ngll * ngll;
    state.n_node = ngll * ngll * ngll;
    state.n_src_cell = cfg.n_src_cell;
    state.n_local_cell = n_local_cell;

    // Detect global DOF mode
    state.use_global_dof = (n_rank_node > 0 && !local_cell2rank_node.empty());
    state.n_rank_node = n_rank_node;

    // Allocate read-only data
    auto alloc_d = [](auto*& ptr, size_t bytes) { GF_CUDA_CHECK(cudaMalloc(&ptr, bytes)); };
    auto upload = [](auto* d_ptr, const auto* h_ptr, size_t bytes) {
        GF_CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
    };

    alloc_d(state.d_mass, state.n_total_nodes * sizeof(double));
    alloc_d(state.d_pml, state.n_total_nodes * sizeof(double));
    alloc_d(state.d_dxi_dx, state.n_total_nodes * 9 * sizeof(double));
    alloc_d(state.d_jacobian, state.n_total_nodes * sizeof(double));
    alloc_d(state.d_lambda_, state.n_total_nodes * sizeof(double));
    alloc_d(state.d_mu_, state.n_total_nodes * sizeof(double));
    int D_bytes = ngll * ngll * sizeof(double);
    alloc_d(state.d_D, D_bytes);
    alloc_d(state.d_weights, ngll * sizeof(double));
    alloc_d(state.d_src_elem_offsets, state.n_src_cell * sizeof(int));

    upload(state.d_mass, mass.data(), state.n_total_nodes * sizeof(double));
    upload(state.d_pml, pml_damping.data(), state.n_total_nodes * sizeof(double));
    upload(state.d_dxi_dx, dxi_dx.data(), state.n_total_nodes * 9 * sizeof(double));
    upload(state.d_jacobian, jacobian.data(), state.n_total_nodes * sizeof(double));
    upload(state.d_lambda_, lambda_.data(), state.n_total_nodes * sizeof(double));
    upload(state.d_mu_, mu_.data(), state.n_total_nodes * sizeof(double));
    upload(state.d_D, h_D, D_bytes);
    upload(state.d_weights, h_weights, ngll * sizeof(double));

    GF_CUDA_CHECK(cudaMemset(state.d_src_elem_offsets, 0, state.n_src_cell * sizeof(int)));

    if (state.use_global_dof) {
        // --- Global DOF arrays ---
        alloc_d(state.d_local_cell2rank_node, state.n_total_nodes * sizeof(int));
        upload(state.d_local_cell2rank_node, local_cell2rank_node.data(),
               state.n_total_nodes * sizeof(int32_t));

        alloc_d(state.d_rank_node_mass, n_rank_node * sizeof(double));
        upload(state.d_rank_node_mass, rank_node_mass.data(), n_rank_node * sizeof(double));

        alloc_d(state.d_rank_node_damping, n_rank_node * sizeof(double));
        upload(state.d_rank_node_damping, rank_node_damping.data(), n_rank_node * sizeof(double));

        // Global state vectors
        int n_glob_dof = n_rank_node * 3;
        alloc_d(state.d_rank_node_displacement, n_glob_dof * sizeof(double));
        alloc_d(state.d_rank_node_velocity, n_glob_dof * sizeof(double));
        alloc_d(state.d_rank_node_acceleration, n_glob_dof * sizeof(double));
        alloc_d(state.d_rank_node_residual, n_glob_dof * sizeof(double));
        alloc_d(state.d_rank_node_displacement_tilde, n_glob_dof * sizeof(double));

        GF_CUDA_CHECK(cudaMemset(state.d_rank_node_displacement, 0, n_glob_dof * sizeof(double)));
        GF_CUDA_CHECK(cudaMemset(state.d_rank_node_velocity, 0, n_glob_dof * sizeof(double)));
        GF_CUDA_CHECK(cudaMemset(state.d_rank_node_acceleration, 0, n_glob_dof * sizeof(double)));
        GF_CUDA_CHECK(cudaMemset(state.d_rank_node_residual, 0, n_glob_dof * sizeof(double)));
        GF_CUDA_CHECK(
            cudaMemset(state.d_rank_node_displacement_tilde, 0, n_glob_dof * sizeof(double)));

        // Element-local temp arrays for kernel
        alloc_d(state.d_local_cell_displacement, n_local_cell_dof * sizeof(double));
        alloc_d(state.d_local_cell_residual, n_local_cell_dof * sizeof(double));
        GF_CUDA_CHECK(
            cudaMemset(state.d_local_cell_displacement, 0, n_local_cell_dof * sizeof(double)));
        GF_CUDA_CHECK(
            cudaMemset(state.d_local_cell_residual, 0, n_local_cell_dof * sizeof(double)));
    } else {
        // --- Legacy element-local state ---
        alloc_d(state.d_displacement, n_local_cell_dof * sizeof(double));
        alloc_d(state.d_velocity, n_local_cell_dof * sizeof(double));
        alloc_d(state.d_acceleration, n_local_cell_dof * sizeof(double));
        alloc_d(state.d_residual, n_local_cell_dof * sizeof(double));
        alloc_d(state.d_displacement_tilde, n_local_cell_dof * sizeof(double));

        GF_CUDA_CHECK(cudaMemset(state.d_displacement, 0, n_local_cell_dof * sizeof(double)));
        GF_CUDA_CHECK(cudaMemset(state.d_velocity, 0, n_local_cell_dof * sizeof(double)));
        GF_CUDA_CHECK(cudaMemset(state.d_acceleration, 0, n_local_cell_dof * sizeof(double)));
        GF_CUDA_CHECK(cudaMemset(state.d_residual, 0, n_local_cell_dof * sizeof(double)));
        GF_CUDA_CHECK(
            cudaMemset(state.d_displacement_tilde, 0, n_local_cell_dof * sizeof(double)));
    }

    state.allocated = true;
    return state;
}

void cuda_upload_cpml_data(CudaDeviceState& state, const RankData& part, int n_node) {
    if (!part.has_cpml || !state.allocated)
        return;

    auto upload = [](auto*& d_ptr, const auto& host_vec, size_t elem_size) {
        if (!host_vec.empty()) {
            size_t bytes = host_vec.size() * elem_size;
            GF_CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
            GF_CUDA_CHECK(cudaMemcpy(d_ptr, host_vec.data(), bytes, cudaMemcpyHostToDevice));
        }
    };

    int n_pml_nodes = part.n_local_cell * n_node;

    upload(state.d_pml_region, part.pml_region, sizeof(int32_t));
    upload(state.d_pml_coef_alpha, part.pml_coef_alpha, sizeof(double));
    upload(state.d_pml_coef_beta, part.pml_coef_beta, sizeof(double));
    upload(state.d_pml_coef_abar, part.pml_coef_abar, sizeof(double));
    upload(state.d_pml_coef_strain, part.pml_coef_strain, sizeof(double));
    upload(state.d_pml_displ_old, part.pml_displ_old, sizeof(double));
    upload(state.d_pml_displ_new, part.pml_displ_new, sizeof(double));
    upload(state.d_rmemory_displ, part.rmemory_displ, sizeof(double));
    upload(state.d_rmemory_strain, part.rmemory_strain, sizeof(double));
    upload(state.d_density, part.density, sizeof(double));

    state.n_pml_nodes_used = n_pml_nodes;
    state.has_cpml = true;
}

void cuda_free_cpml_data(CudaDeviceState& state) {
    if (!state.has_cpml)
        return;
    auto f = [](auto*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    f(state.d_pml_region);
    f(state.d_pml_coef_alpha);
    f(state.d_pml_coef_beta);
    f(state.d_pml_coef_abar);
    f(state.d_pml_coef_strain);
    f(state.d_pml_displ_old);
    f(state.d_pml_displ_new);
    f(state.d_rmemory_displ);
    f(state.d_rmemory_strain);
    f(state.d_density);
    state.has_cpml = false;
    state.n_pml_nodes_used = 0;
}

// ---------------------------------------------------------------------------
// SLS attenuation — device upload / free
// ---------------------------------------------------------------------------

void cuda_upload_sls_data(CudaDeviceState& state, const RankData& part, int n_node) {
    if (!part.has_attenuation || !state.allocated)
        return;

    auto upload = [](auto*& d_ptr, const auto& host_vec, size_t elem_size) {
        if (!host_vec.empty()) {
            size_t bytes = host_vec.size() * elem_size;
            GF_CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
            GF_CUDA_CHECK(cudaMemcpy(d_ptr, host_vec.data(), bytes, cudaMemcpyHostToDevice));
        }
    };

    upload(state.d_rmemory_sls, part.rmemory_sls, sizeof(double));
    upload(state.d_sigma_old, part.sigma_old, sizeof(double));
    upload(state.d_sls_coef_a, part.sls_coef_a, sizeof(double));
    upload(state.d_sls_coef_b, part.sls_coef_b, sizeof(double));

    state.has_attenuation = true;
}

void cuda_free_sls_data(CudaDeviceState& state) {
    if (!state.has_attenuation)
        return;
    auto f = [](auto*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    f(state.d_rmemory_sls);
    f(state.d_sigma_old);
    f(state.d_sls_coef_a);
    f(state.d_sls_coef_b);
    state.has_attenuation = false;
}

void cuda_free_state(CudaDeviceState& state) {
    if (!state.allocated)
        return;
    cuda_free_cpml_data(state);
    cuda_free_sls_data(state);
    auto f = [](auto*& ptr) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };
    f(state.d_mass);
    f(state.d_pml);
    f(state.d_dxi_dx);
    f(state.d_jacobian);
    f(state.d_lambda_);
    f(state.d_mu_);
    f(state.d_weights);
    f(state.d_src_elem_offsets);
    // Element-local state
    f(state.d_displacement);
    f(state.d_velocity);
    f(state.d_acceleration);
    f(state.d_residual);
    f(state.d_displacement_tilde);
    // Global state
    f(state.d_rank_node_displacement);
    f(state.d_rank_node_displacement_tilde);
    f(state.d_rank_node_velocity);
    f(state.d_rank_node_acceleration);
    f(state.d_rank_node_residual);
    f(state.d_rank_node_mass);
    f(state.d_rank_node_damping);
    f(state.d_local_cell2rank_node);
    f(state.d_local_cell_displacement);
    f(state.d_local_cell_residual);
    state.allocated = false;
}

}  // namespace gf