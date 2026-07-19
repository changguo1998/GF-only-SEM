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
/// Recording extraction (strain + displacement per GLL node) is done
/// on the host side in solver.cpp; GPU only handles element residual +
/// Newmark time stepping.
struct CudaDeviceState {
    // --- Geometry buffers (uploaded once, persistent) ---
    double* d_mass = nullptr;           // [n_total_nodes] lumped mass (element-local)
    double* d_pml = nullptr;            // [n_total_nodes] PML damping profile (element-local)
    double* d_dxi_dx = nullptr;         // [n_total_nodes * 9]
    double* d_jacobian = nullptr;       // [n_total_nodes]
    double* d_lambda_ = nullptr;        // [n_total_nodes] λ at GLL nodes
    double* d_mu_ = nullptr;            // [n_total_nodes] μ at GLL nodes
    double* d_D = nullptr;              // [ngll * ngll] derivative matrix
    double* d_weights = nullptr;        // [ngll] quadrature weights
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
    double* d_displacement = nullptr;        // [n_dof]
    double* d_velocity = nullptr;            // [n_dof]
    double* d_acceleration = nullptr;        // [n_dof]
    double* d_residual = nullptr;            // [n_dof]
    double* d_displacement_tilde = nullptr;  // [n_dof]

    // --- Sizes ---
    int n_dof = 0;
    int n_local_cell_dof = 0;  // n_local_cell * n_node * 3 (element-local DOF)
    int n_total_nodes = 0;
    int n_src_cell = 0;
    int n_node = 0;        // NGLL^3
    int n_rank_node = 0;   // unique rank-level nodes (0 = not using global DOF)
    int n_local_cell = 0;  // number of local elements
    bool use_global_dof = false;

    bool allocated = false;
};

/// Allocate GPU state and upload read-only data (mass, pml, source weights).
/// Recording map is no longer uploaded to GPU — strain extraction happens on host.
CudaDeviceState cuda_allocate_state(
    int n_local_cell, int ngll, const std::vector<double>& mass,
    const std::vector<double>& pml_damping, const std::vector<double>& dxi_dx,
    const std::vector<double>& jacobian, const std::vector<double>& lambda_,
    const std::vector<double>& mu_, const double* h_D, const double* h_weights,
    const ConfigData& cfg, int n_local_cell_dof,
    const std::vector<int32_t>& local_cell2rank_node = {}, int n_rank_node = 0,
    const std::vector<double>& rank_node_mass = {},
    const std::vector<double>& rank_node_damping = {});

/// Free all device buffers.
void cuda_free_state(CudaDeviceState& state);

// -----------------------------------------------------------------------
// Per-step GPU kernels (host wrappers)
// -----------------------------------------------------------------------

/// Newmark predictor.
void cuda_newmark_predict(CudaDeviceState& state, double dt, double beta);

/// Zero residual on device (cudaMemset).
void cuda_zero_residual(CudaDeviceState& state);

/// PML damping: d_velocity[i] -= d_pml[node] * d_velocity[i]
void cuda_pml_damping(CudaDeviceState& state);

/// Source injection: add STF * weights to residual at source element nodes.
void cuda_source_injection(CudaDeviceState& state, int direction, double stf_val,
                           const double* h_src_weights, int n_src_cell);

/// Newmark corrector.
void cuda_newmark_correct(CudaDeviceState& state, double dt, double beta, double gamma);

/// Copy state vectors to host (for restart / snapshot).
void cuda_copy_state_to_host(const CudaDeviceState& state, std::vector<double>& h_displacement,
                             std::vector<double>& h_velocity, std::vector<double>& h_acceleration);

/// Launch element residual kernel using pre-existing device pointers (GPU-native mode).
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