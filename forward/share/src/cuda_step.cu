// forward/share/src/cuda_step.cu
// GPU kernels for single-GPU mode: Newmark, PML, source injection, strain,
// and CG-SEM global scatter/gather.
// Keeps all state on device, no per-step H2D/D2H copies.

#include "gf/cuda_check.h"
#include "gf/cuda_step.hpp"

namespace gf {

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
                                       double gamma, int n_dof, int n_nodes) {
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
            d_disp[i] += dt * d_vel[i] + 0.5 * dt * dt * a_old;
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
__global__ void scatter_to_rank_kernel(const double* d_local_element_residual,
                                       const int* d_local_element2rank_node,
                                       double* d_rank_node_residual, int n_local_element,
                                       int n_node) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = n_local_element * n_node * 3;
    if (idx < total) {
        int direction = idx % 3;
        int node = (idx / 3) % n_node;
        int elem = idx / (n_node * 3);
        int node_id = d_local_element2rank_node[elem * n_node + node];
        atomicAdd(&d_rank_node_residual[node_id * 3 + direction], d_local_element_residual[idx]);
    }
}

/// Gather global field → element-local.
/// One-to-one mapping, no atomics needed.
__global__ void gather_from_rank_kernel(const double* d_rank_node_field,
                                        const int* d_local_element2rank_node, double* d_elem_field,
                                        int n_local_element, int n_node) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = n_local_element * n_node * 3;
    if (idx < total) {
        int direction = idx % 3;
        int node = (idx / 3) % n_node;
        int elem = idx / (n_node * 3);
        int node_id = d_local_element2rank_node[elem * n_node + node];
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
__global__ void newmark_correct_rank_kernel(double* d_disp, double* d_vel, double* d_acc,
                                            const double* d_residual,
                                            const double* d_rank_node_mass, double dt,
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
        d_disp[i] += dt * d_vel[i] + 0.5 * dt * dt * a_old;
        d_vel[i] += dt * ((1.0 - gamma) * a_old + gamma * a_new);
        d_acc[i] = a_new;
    }
}

// =======================================================================
// Strain computation kernels
// =======================================================================

/// Element-local strain (used with both element-local and global-gathered displacement).
/// d_disp is the element-local displacement array (gathered if needed beforehand).
__global__ void recorded_strain_kernel(const double* d_disp, const double* d_dxi_dx,
                                       const double* d_D, int ngll, const int* d_rec_elem,
                                       const int* d_rec_corner, double* d_strain, int n_vertices,
                                       int n_node) {
    int v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= n_vertices)
        return;

    int elem = d_rec_elem[v];
    int corner = d_rec_corner[v];
    int corner_i = (corner & 1) ? (ngll - 1) : 0;
    int corner_j = (corner & 2) ? (ngll - 1) : 0;
    int corner_k = (corner & 4) ? (ngll - 1) : 0;

    int corner_node = node_idx(corner_i, corner_j, corner_k, ngll);
    const double* dd = &d_dxi_dx[(elem * n_node + corner_node) * 9];
    const double* disp_ptr = &d_disp[(elem * n_node + corner_node) * 3];

    // Reference gradient
    double dudxi[3] = {0.0, 0.0, 0.0};
    double dudeta[3] = {0.0, 0.0, 0.0};
    double dudzeta[3] = {0.0, 0.0, 0.0};

    for (int s = 0; s < ngll; ++s) {
        double Di_s = d_D[corner_i * ngll + s];
        double Dj_s = d_D[corner_j * ngll + s];
        double Dk_s = d_D[corner_k * ngll + s];

        int n_sjk = node_idx(s, corner_j, corner_k, ngll);
        int n_isk = node_idx(corner_i, s, corner_k, ngll);
        int n_ijs = node_idx(corner_i, corner_j, s, ngll);

        for (int d = 0; d < 3; ++d) {
            dudxi[d] += Di_s * disp_ptr[3 * n_sjk + d];
            dudeta[d] += Dj_s * disp_ptr[3 * n_isk + d];
            dudzeta[d] += Dk_s * disp_ptr[3 * n_ijs + d];
        }
    }

    // Physical gradient
    double du_dx[3][3];
    for (int comp = 0; comp < 3; ++comp) {
        du_dx[comp][0] = dudxi[comp] * dd[0] + dudeta[comp] * dd[1] + dudzeta[comp] * dd[2];
        du_dx[comp][1] = dudxi[comp] * dd[3] + dudeta[comp] * dd[4] + dudzeta[comp] * dd[5];
        du_dx[comp][2] = dudxi[comp] * dd[6] + dudeta[comp] * dd[7] + dudzeta[comp] * dd[8];
    }

    // Symmetric strain (Voigt order)
    double* out = &d_strain[v * 6];
    out[0] = du_dx[0][0];                        // eps_xx
    out[1] = du_dx[1][1];                        // eps_yy
    out[2] = du_dx[2][2];                        // eps_zz
    out[3] = 0.5 * (du_dx[0][1] + du_dx[1][0]);  // eps_xy
    out[4] = 0.5 * (du_dx[0][2] + du_dx[2][0]);  // eps_xz
    out[5] = 0.5 * (du_dx[1][2] + du_dx[2][1]);  // eps_yz
}

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
        GF_CUDA_CHECK(cudaMemset(state.d_local_element_residual, 0,
                                 state.n_local_element_dof * sizeof(double)));
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
                           const double* h_src_weights, int n_src_elements) {
    if (stf_val == 0.0 || n_src_elements == 0)
        return;

    // Upload source weights to device (cached)
    static double* d_src_weights = nullptr;
    static int d_src_weights_size = 0;
    int total_weights = n_src_elements * state.n_node;
    if (d_src_weights == nullptr || d_src_weights_size < total_weights) {
        if (d_src_weights)
            cudaFree(d_src_weights);
        GF_CUDA_CHECK(cudaMalloc(&d_src_weights, total_weights * sizeof(double)));
        d_src_weights_size = total_weights;
    }
    GF_CUDA_CHECK(cudaMemcpy(d_src_weights, h_src_weights, total_weights * sizeof(double),
                             cudaMemcpyHostToDevice));

    int block = 256;
    int total_threads = n_src_elements * state.n_node;
    int grid = (total_threads + block - 1) / block;

    // Source always injects into element-local residual
    // (with global DOF: local_element_residual → scatter_to_rank later)
    double* d_target = state.use_global_dof ? state.d_local_element_residual : state.d_residual;
    source_injection_kernel<<<grid, block>>>(d_target, d_src_weights, stf_val, direction,
                                             n_src_elements, state.n_node,
                                             state.d_src_elem_offsets);
    GF_CUDA_CHECK(cudaGetLastError());
}

void cuda_newmark_correct(CudaDeviceState& state, double dt, double gamma) {
    if (state.use_global_dof) {
        int n = state.n_rank_node * 3;
        newmark_correct_rank_kernel<<<grid_blocks(n), 256>>>(
            state.d_rank_node_displacement, state.d_rank_node_velocity,
            state.d_rank_node_acceleration, state.d_rank_node_residual, state.d_rank_node_mass, dt,
            gamma, n);
    } else {
        newmark_correct_kernel<<<grid_blocks(state.n_dof), 256>>>(
            state.d_displacement, state.d_velocity, state.d_acceleration, state.d_residual,
            state.d_mass, dt, gamma, state.n_dof, state.n_total_nodes);
    }
    GF_CUDA_CHECK(cudaGetLastError());
    GF_CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_compute_strain(CudaDeviceState& state, const double* h_D, int ngll,
                         const std::vector<double>& /*dxi_dx*/) {
    if (state.n_vertices == 0)
        return;
    int block = 256;
    int grid = (state.n_vertices + block - 1) / block;
    // Upload D matrix to device (cached across calls)
    static double* d_D_cache = nullptr;
    static int d_D_size = 0;
    int D_bytes = ngll * ngll * sizeof(double);
    if (d_D_cache == nullptr || d_D_size < D_bytes) {
        if (d_D_cache)
            cudaFree(d_D_cache);
        GF_CUDA_CHECK(cudaMalloc(&d_D_cache, D_bytes));
        d_D_size = D_bytes;
    }
    GF_CUDA_CHECK(cudaMemcpy(d_D_cache, h_D, D_bytes, cudaMemcpyHostToDevice));

    // When using global DOF, displacement has been gathered into local_element_displacement
    // before calling this function (done in solver.cpp).
    // The strain kernel always reads from the element-local displacement array.
    double* d_strain_disp =
        state.use_global_dof ? state.d_local_element_displacement : state.d_displacement;

    recorded_strain_kernel<<<grid, block>>>(d_strain_disp, state.d_dxi_dx, d_D_cache, ngll,
                                            state.d_rec_src_elem, state.d_rec_corner,
                                            state.d_strain_buffer, state.n_vertices, state.n_node);
    GF_CUDA_CHECK(cudaGetLastError());
}

void cuda_copy_strain_to_host(CudaDeviceState& state, double* h_strain) {
    if (state.n_vertices == 0)
        return;
    GF_CUDA_CHECK(cudaMemcpy(h_strain, state.d_strain_buffer,
                             state.n_vertices * 6 * sizeof(double), cudaMemcpyDeviceToHost));
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
    scatter_to_rank_kernel<<<grid_blocks(state.n_local_element_dof), 256>>>(
        state.d_local_element_residual, state.d_local_element2rank_node,
        state.d_rank_node_residual, state.n_local_element, state.n_node);
    GF_CUDA_CHECK(cudaGetLastError());
}

void cuda_gather_from_rank(CudaDeviceState& state) {
    gather_from_rank_kernel<<<grid_blocks(state.n_local_element_dof), 256>>>(
        state.d_rank_node_displacement, state.d_local_element2rank_node,
        state.d_local_element_displacement, state.n_local_element, state.n_node);
    GF_CUDA_CHECK(cudaGetLastError());
}

/// Gather predicted displacement (displacement_tilde) for the element kernel.
void cuda_gather_predicted(CudaDeviceState& state) {
    gather_from_rank_kernel<<<grid_blocks(state.n_local_element_dof), 256>>>(
        state.d_rank_node_displacement_tilde, state.d_local_element2rank_node,
        state.d_local_element_displacement, state.n_local_element, state.n_node);
    GF_CUDA_CHECK(cudaGetLastError());
}

// =======================================================================
// Allocation / free
// =======================================================================

CudaDeviceState cuda_allocate_state(
    int n_local_element, int ngll, const std::vector<double>& mass,
    const std::vector<double>& pml_damping, const std::vector<double>& dxi_dx,
    const std::vector<double>& jacobian, const std::vector<double>& lambda_,
    const std::vector<double>& mu_, const double* h_D, const double* h_weights,
    const ConfigData& cfg, const RankData::RecordingMap& rec_map, int n_local_element_dof,
    const std::vector<int32_t>& local_element2rank_node, int n_rank_node,
    const std::vector<double>& rank_node_mass, const std::vector<double>& rank_node_damping) {
    CudaDeviceState state;
    state.n_dof = n_local_element_dof;
    state.n_local_element_dof = n_local_element_dof;
    state.n_total_nodes = n_local_element * ngll * ngll * ngll;
    state.n_node = ngll * ngll * ngll;
    state.n_vertices = static_cast<int>(rec_map.vertex_ids.size());
    state.n_src_elements = cfg.n_src_elements;
    state.n_local_element = n_local_element;

    // Detect global DOF mode
    state.use_global_dof = (n_rank_node > 0 && !local_element2rank_node.empty());
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
    alloc_d(state.d_rec_src_elem, state.n_vertices * sizeof(int));
    alloc_d(state.d_rec_corner, state.n_vertices * sizeof(int));
    alloc_d(state.d_src_elem_offsets, state.n_src_elements * sizeof(int));

    upload(state.d_mass, mass.data(), state.n_total_nodes * sizeof(double));
    upload(state.d_pml, pml_damping.data(), state.n_total_nodes * sizeof(double));
    upload(state.d_dxi_dx, dxi_dx.data(), state.n_total_nodes * 9 * sizeof(double));
    upload(state.d_jacobian, jacobian.data(), state.n_total_nodes * sizeof(double));
    upload(state.d_lambda_, lambda_.data(), state.n_total_nodes * sizeof(double));
    upload(state.d_mu_, mu_.data(), state.n_total_nodes * sizeof(double));
    upload(state.d_D, h_D, D_bytes);
    upload(state.d_weights, h_weights, ngll * sizeof(double));

    if (state.n_vertices > 0) {
        upload(state.d_rec_src_elem, rec_map.src_elem_local.data(),
               state.n_vertices * sizeof(int32_t));
        upload(state.d_rec_corner, rec_map.src_corner.data(), state.n_vertices * sizeof(int32_t));
    }

    GF_CUDA_CHECK(cudaMemset(state.d_src_elem_offsets, 0, state.n_src_elements * sizeof(int)));

    if (state.use_global_dof) {
        // --- Global DOF arrays ---
        alloc_d(state.d_local_element2rank_node, state.n_total_nodes * sizeof(int));
        upload(state.d_local_element2rank_node, local_element2rank_node.data(),
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
        alloc_d(state.d_local_element_displacement, n_local_element_dof * sizeof(double));
        alloc_d(state.d_local_element_residual, n_local_element_dof * sizeof(double));
        GF_CUDA_CHECK(cudaMemset(state.d_local_element_displacement, 0,
                                 n_local_element_dof * sizeof(double)));
        GF_CUDA_CHECK(
            cudaMemset(state.d_local_element_residual, 0, n_local_element_dof * sizeof(double)));
    } else {
        // --- Legacy element-local state ---
        alloc_d(state.d_displacement, n_local_element_dof * sizeof(double));
        alloc_d(state.d_velocity, n_local_element_dof * sizeof(double));
        alloc_d(state.d_acceleration, n_local_element_dof * sizeof(double));
        alloc_d(state.d_residual, n_local_element_dof * sizeof(double));
        alloc_d(state.d_displacement_tilde, n_local_element_dof * sizeof(double));

        GF_CUDA_CHECK(cudaMemset(state.d_displacement, 0, n_local_element_dof * sizeof(double)));
        GF_CUDA_CHECK(cudaMemset(state.d_velocity, 0, n_local_element_dof * sizeof(double)));
        GF_CUDA_CHECK(cudaMemset(state.d_acceleration, 0, n_local_element_dof * sizeof(double)));
        GF_CUDA_CHECK(cudaMemset(state.d_residual, 0, n_local_element_dof * sizeof(double)));
        GF_CUDA_CHECK(
            cudaMemset(state.d_displacement_tilde, 0, n_local_element_dof * sizeof(double)));
    }

    // Strain buffer
    alloc_d(state.d_strain_buffer, state.n_vertices * 6 * sizeof(double));

    state.allocated = true;
    return state;
}

void cuda_free_state(CudaDeviceState& state) {
    if (!state.allocated)
        return;
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
    f(state.d_D);
    f(state.d_weights);
    f(state.d_rec_src_elem);
    f(state.d_rec_corner);
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
    f(state.d_local_element2rank_node);
    f(state.d_local_element_displacement);
    f(state.d_local_element_residual);
    f(state.d_strain_buffer);
    state.allocated = false;
}

}  // namespace gf