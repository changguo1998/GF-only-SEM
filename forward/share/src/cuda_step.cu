// forward/share/src/cuda_step.cu
// GPU kernels for single-GPU mode: Newmark, PML, source injection, strain.
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

// -----------------------------------------------------------------------
// Newmark predictor kernel: u_tilde = u + dt*v + 0.5*dt^2*(1-2*beta)*a
// -----------------------------------------------------------------------
__global__ void newmark_predict_kernel(double* d_disp_tilde, const double* d_disp,
                                       const double* d_vel, const double* d_acc, double dt,
                                       double beta_factor, int n_dof) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_dof) {
        d_disp_tilde[i] = d_disp[i] + dt * d_vel[i] + beta_factor * d_acc[i];
    }
}

// -----------------------------------------------------------------------
// PML damping kernel: v[i] -= damping_profile[node] * v[i]
// -----------------------------------------------------------------------
__global__ void pml_damping_kernel(double* d_vel, const double* d_pml, int n_dof, int n_nodes) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_dof) {
        int node = i / 3;  // same damping for all 3 DOFs at a node
        if (node < n_nodes) {
            double d = d_pml[node];
            if (d > 0.0) {
                d_vel[i] -= d * d_vel[i];
            }
        }
    }
}

// -----------------------------------------------------------------------
// Source injection kernel: residual[base + dir] += stf_val * weight
// -----------------------------------------------------------------------
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

// -----------------------------------------------------------------------
// Newmark corrector kernel:
//   a_new = r / mass[node]
//   u += dt * v + 0.5*dt^2 * a_old
//   v += dt * ((1-gamma) * a_old + gamma * a_new)
//   acc = a_new
// -----------------------------------------------------------------------
__global__ void newmark_correct_kernel(double* d_disp, double* d_vel, double* d_acc,
                                       const double* d_residual, const double* d_mass, double dt,
                                       double gamma, int n_dof, int n_nodes) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_dof) {
        int node = i / 3;
        if (node < n_nodes) {
            double a_old = d_acc[i];
            double a_new = d_residual[i] / d_mass[node];
            d_disp[i] += dt * d_vel[i] + 0.5 * dt * dt * a_old;
            d_vel[i] += dt * ((1.0 - gamma) * a_old + gamma * a_new);
            d_acc[i] = a_new;
        }
    }
}

// -----------------------------------------------------------------------
// Strain computation kernel (recorded mesh vertices only)
// -----------------------------------------------------------------------
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

void cuda_newmark_predict(CudaDeviceState& state, double dt, double beta) {
    double beta_factor = 0.5 * dt * dt * (1.0 - 2.0 * beta);
    int block = 256;
    int grid = (state.n_dof + block - 1) / block;
    newmark_predict_kernel<<<grid, block>>>(state.d_displacement_tilde, state.d_displacement,
                                            state.d_velocity, state.d_acceleration, dt,
                                            beta_factor, state.n_dof);
    GF_CUDA_CHECK(cudaGetLastError());
    GF_CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_zero_residual(CudaDeviceState& state) {
    GF_CUDA_CHECK(cudaMemset(state.d_residual, 0, state.n_dof * sizeof(double)));
}

void cuda_pml_damping(CudaDeviceState& state) {
    int block = 256;
    int grid = (state.n_dof + block - 1) / block;
    pml_damping_kernel<<<grid, block>>>(state.d_velocity, state.d_pml, state.n_dof,
                                        state.n_total_nodes);
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
    source_injection_kernel<<<grid, block>>>(state.d_residual, d_src_weights, stf_val, direction,
                                             n_src_elements, state.n_node,
                                             state.d_src_elem_offsets);
    GF_CUDA_CHECK(cudaGetLastError());
}

void cuda_newmark_correct(CudaDeviceState& state, double dt, double gamma) {
    int block = 256;
    int grid = (state.n_dof + block - 1) / block;
    newmark_correct_kernel<<<grid, block>>>(state.d_displacement, state.d_velocity,
                                            state.d_acceleration, state.d_residual, state.d_mass,
                                            dt, gamma, state.n_dof, state.n_total_nodes);
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

    recorded_strain_kernel<<<grid, block>>>(state.d_displacement, state.d_dxi_dx, d_D_cache, ngll,
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
    GF_CUDA_CHECK(cudaMemcpy(h_displacement.data(), state.d_displacement,
                             state.n_dof * sizeof(double), cudaMemcpyDeviceToHost));
    GF_CUDA_CHECK(cudaMemcpy(h_velocity.data(), state.d_velocity, state.n_dof * sizeof(double),
                             cudaMemcpyDeviceToHost));
    GF_CUDA_CHECK(cudaMemcpy(h_acceleration.data(), state.d_acceleration,
                             state.n_dof * sizeof(double), cudaMemcpyDeviceToHost));
}

// =======================================================================
// Allocation / free
// =======================================================================

CudaDeviceState cuda_allocate_state(int n_local_elem, int ngll, const std::vector<double>& mass,
                                    const std::vector<double>& pml_damping,
                                    const std::vector<double>& dxi_dx,
                                    const std::vector<double>& jacobian,
                                    const std::vector<double>& lambda_,
                                    const std::vector<double>& mu_, const double* h_D,
                                    const double* h_weights, const ConfigData& cfg,
                                    const RankData::RecordingMap& rec_map, int n_local_dof) {
    CudaDeviceState state;
    state.n_dof = n_local_dof;
    state.n_total_nodes = n_local_elem * ngll * ngll * ngll;
    state.n_node = ngll * ngll * ngll;
    state.n_vertices = static_cast<int>(rec_map.vertex_ids.size());
    state.n_src_elements = cfg.n_src_elements;

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

    // Source element offsets: map global element IDs to local element indices
    // We upload the already-computed mapping from solver.cpp
    // (filled by src_elem_to_local logic). For allocation, leave as zeros;
    // solver.cpp will upload the actual mapping.
    GF_CUDA_CHECK(cudaMemset(state.d_src_elem_offsets, 0, state.n_src_elements * sizeof(int)));

    // Allocate per-timestep state
    alloc_d(state.d_displacement, state.n_dof * sizeof(double));
    alloc_d(state.d_velocity, state.n_dof * sizeof(double));
    alloc_d(state.d_acceleration, state.n_dof * sizeof(double));
    alloc_d(state.d_residual, state.n_dof * sizeof(double));
    alloc_d(state.d_displacement_tilde, state.n_dof * sizeof(double));

    // Initialize to zero
    GF_CUDA_CHECK(cudaMemset(state.d_displacement, 0, state.n_dof * sizeof(double)));
    GF_CUDA_CHECK(cudaMemset(state.d_velocity, 0, state.n_dof * sizeof(double)));
    GF_CUDA_CHECK(cudaMemset(state.d_acceleration, 0, state.n_dof * sizeof(double)));
    GF_CUDA_CHECK(cudaMemset(state.d_residual, 0, state.n_dof * sizeof(double)));
    GF_CUDA_CHECK(cudaMemset(state.d_displacement_tilde, 0, state.n_dof * sizeof(double)));

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
    f(state.d_displacement);
    f(state.d_velocity);
    f(state.d_acceleration);
    f(state.d_residual);
    f(state.d_displacement_tilde);
    f(state.d_strain_buffer);
    state.allocated = false;
}

}  // namespace gf