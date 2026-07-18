#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gf/cuda_check.h"
#include "gf/types.hpp"

namespace gf {

// -----------------------------------------------------------------------
// CudaDeviceState — persistent GPU-side state for single-GPU mode
// -----------------------------------------------------------------------
/// Manages all device buffers needed for the GPU-native time loop.
/// Allocates once, holds state across timesteps, frees at end.
struct CudaDeviceState {
    // --- Read-only geometry (already in CudaDeviceBuffers, duplicates here for convenience) ---
    // --- Geometry buffers (uploaded once, persistent) ---
    double* d_mass = nullptr;       // [n_total_nodes] lumped mass (element-local)
    double* d_pml = nullptr;        // [n_total_nodes] PML damping profile (element-local)
    double* d_dxi_dx = nullptr;     // [n_total_nodes * 9]
    double* d_jacobian = nullptr;   // [n_total_nodes]
    double* d_lambda_ = nullptr;    // [n_total_nodes] λ at GLL nodes
    double* d_mu_ = nullptr;        // [n_total_nodes] μ at GLL nodes
    double* d_D = nullptr;          // [ngll * ngll] derivative matrix
    double* d_weights = nullptr;    // [ngll] quadrature weights
    int* d_rec_src_elem = nullptr;  // [n_vertices] local element index for each recorded vertex
    int* d_rec_corner = nullptr;    // [n_vertices] corner index 0-7
    int* d_src_elem_offsets = nullptr;  // [n_src_cell] local element index for source elems

    // --- Global DOF arrays (CG-SEM assembly) ---
    double* d_rank_node_mass = nullptr;     // [n_rank_node] — per-node mass
    double* d_rank_node_damping = nullptr;  // [n_rank_node] — per-node damping
    int* d_local_cell2rank_node = nullptr;  // [n_local_cell * n_node] — flat element→node id map

    // --- Per-node state vectors (CG-SEM, allocated when use_global_dof) ---
    double* d_rank_node_displacement = nullptr;        // [n_rank_node * 3]
    double* d_rank_node_displacement_tilde = nullptr;  // [n_rank_node * 3] — predictor output
    double* d_rank_node_velocity = nullptr;            // [n_rank_node * 3]
    double* d_rank_node_acceleration = nullptr;        // [n_rank_node * 3]
    double* d_rank_node_residual = nullptr;            // [n_rank_node * 3] — global residual

    // --- Element-local temp arrays for kernel (always element-local) ---
    double* d_local_cell_displacement = nullptr;  // [n_local_cell * n_node * 3]
    double* d_local_cell_residual = nullptr;      // [n_local_cell * n_node * 3]

    // --- Per-timestep state (persistent on device) ---
    // Element-local (backward compat / legacy)
    double* d_displacement = nullptr;        // [n_dof]
    double* d_velocity = nullptr;            // [n_dof]
    double* d_acceleration = nullptr;        // [n_dof]
    double* d_residual = nullptr;            // [n_dof]
    double* d_displacement_tilde = nullptr;  // [n_dof]
    double* d_strain_buffer = nullptr;       // [n_vertices * 6] for snapshot output

    // --- Sizes ---
    int n_dof = 0;
    int n_local_cell_dof = 0;  // n_local_cell * n_node * 3 (element-local DOF)
    int n_total_nodes = 0;
    int n_vertices = 0;
    int n_src_cell = 0;
    int n_node = 0;        // NGLL^3
    int n_rank_node = 0;   // unique rank-level nodes (0 = not using global DOF)
    int n_local_cell = 0;  // number of local elements
    bool use_global_dof = false;

    bool allocated = false;
};

/// Allocate GPU state and upload read-only data (mass, pml, source weights, recording map).
CudaDeviceState cuda_allocate_state(
    int n_local_cell, int ngll, const std::vector<double>& mass,
    const std::vector<double>& pml_damping, const std::vector<double>& dxi_dx,
    const std::vector<double>& jacobian, const std::vector<double>& lambda_,
    const std::vector<double>& mu_, const double* h_D, const double* h_weights,
    const ConfigData& cfg, const RankData::RecordingMap& rec_map, int n_local_cell_dof,
    const std::vector<int32_t>& local_cell2rank_node = {}, int n_rank_node = 0,
    const std::vector<double>& rank_node_mass = {},
    const std::vector<double>& rank_node_damping = {});

/// Free all device buffers.
void cuda_free_state(CudaDeviceState& state);

// -----------------------------------------------------------------------
// Per-step GPU kernels (host wrappers)
// -----------------------------------------------------------------------

/// Newmark predictor: d_displacement_tilde = d_displacement + dt * d_velocity + 0.5*dt^2 *
/// d_acceleration
void cuda_newmark_predict(CudaDeviceState& state, double dt, double beta);

/// Zero residual on device (cudaMemset).
void cuda_zero_residual(CudaDeviceState& state);

/// PML damping: d_velocity[i] -= d_pml[node] * d_velocity[i]
void cuda_pml_damping(CudaDeviceState& state);

/// Source injection: add STF * weights to residual at source element nodes.
void cuda_source_injection(CudaDeviceState& state, int direction, double stf_val,
                           const double* h_src_weights, int n_src_cell);

/// Newmark corrector: a_new = r/mass,
/// u += dt*v + dt²*((0.5-β)*a_old + β*a_new),
/// v += dt*((1-γ)*a_old + γ*a_new)
void cuda_newmark_correct(CudaDeviceState& state, double dt, double beta, double gamma);

/// Compute strain at recorded vertices, store in d_strain_buffer (then copy to host for I/O).
void cuda_compute_strain(CudaDeviceState& state, const double* h_D, int ngll,
                         const std::vector<double>& dxi_dx);

/// Copy strain buffer from device to host.
void cuda_copy_strain_to_host(CudaDeviceState& state, double* h_strain);

/// Copy state vectors to host (for restart).
void cuda_copy_state_to_host(const CudaDeviceState& state, std::vector<double>& h_displacement,
                             std::vector<double>& h_velocity, std::vector<double>& h_acceleration);

/// Launch element residual kernel using pre-existing device pointers (GPU-native mode).
/// Skips H2D/D2H copies — uses geometry buffers already resident on device.
void cuda_launch_element_residual(const CudaDeviceState& state, int ngll, int n_elem);

/// CG-SEM global scatter: local_cell_residual → rank_node_residual (with atomicAdd).
void cuda_scatter_to_rank(CudaDeviceState& state);

/// CG-SEM global gather: global_displacement → local_cell_displacement.
void cuda_gather_from_rank(CudaDeviceState& state);

/// CG-SEM global gather of predicted displacement for element kernel.
void cuda_gather_predicted(CudaDeviceState& state);

/// Copy predicted displacement (displacement_tilde) device → host for MPI exchange staging.
void cuda_copy_utilde_to_host(const CudaDeviceState& state, double* host_buf);

/// Copy predicted displacement host → device after MPI exchange + averaging.
void cuda_copy_utilde_from_host(CudaDeviceState& state, const double* host_buf);

/// Copy global residual device → host for MPI halo exchange staging.
void cuda_copy_residual_to_host(const CudaDeviceState& state, double* host_buf);

/// Copy global residual host → device after MPI halo exchange.
void cuda_copy_residual_from_host(CudaDeviceState& state, const double* host_buf);

}  // namespace gf