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
    double* d_mass = nullptr;       // [n_total_nodes] lumped mass
    double* d_pml = nullptr;        // [n_total_nodes] PML damping profile
    double* d_dxi_dx = nullptr;     // [n_total_nodes * 9]
    double* d_jacobian = nullptr;   // [n_total_nodes]
    double* d_lambda_ = nullptr;    // [n_total_nodes] λ at GLL nodes
    double* d_mu_ = nullptr;        // [n_total_nodes] μ at GLL nodes
    double* d_D = nullptr;          // [ngll * ngll] derivative matrix
    double* d_weights = nullptr;    // [ngll] quadrature weights
    int* d_rec_src_elem = nullptr;  // [n_vertices] local element index for each recorded vertex
    int* d_rec_corner = nullptr;    // [n_vertices] corner index 0-7
    int* d_src_elem_offsets = nullptr;  // [n_src_elements] local element index for source elems

    // --- Per-timestep state (persistent on device) ---
    double* d_displacement = nullptr;        // [n_dof]
    double* d_velocity = nullptr;            // [n_dof]
    double* d_acceleration = nullptr;        // [n_dof]
    double* d_residual = nullptr;            // [n_dof]
    double* d_displacement_tilde = nullptr;  // [n_dof]
    double* d_strain_buffer = nullptr;       // [n_vertices * 6] for snapshot output

    // --- Sizes ---
    int n_dof = 0;
    int n_total_nodes = 0;
    int n_vertices = 0;
    int n_src_elements = 0;
    int n_node = 0;  // NGLL^3

    bool allocated = false;
};

/// Allocate GPU state and upload read-only data (mass, pml, source weights, recording map).
CudaDeviceState cuda_allocate_state(int n_local_elem, int ngll, const std::vector<double>& mass,
                                    const std::vector<double>& pml_damping,
                                    const std::vector<double>& dxi_dx,
                                    const std::vector<double>& jacobian,
                                    const std::vector<double>& lambda_,
                                    const std::vector<double>& mu_, const double* h_D,
                                    const double* h_weights, const ConfigData& cfg,
                                    const RankData::RecordingMap& rec_map, int n_local_dof);

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
                           const double* h_src_weights, int n_src_elements);

/// Newmark corrector: a_new = r/mass, u += dt*v + 0.5*dt^2*a_old, v += 0.5*dt*(a_old + a_new)
void cuda_newmark_correct(CudaDeviceState& state, double dt, double gamma);

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

}  // namespace gf