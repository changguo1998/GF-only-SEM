#pragma once

/// @file
/// Persistent device memory manager for CUDA element residual kernel.
///
/// Manages allocation, deallocation, and H2D/D2H transfers for the
/// read-only mesh data (dxi_dx, jacobian, material, D, weights) and
/// per-timestep fields (displacement u, residual r).
///
/// Device arrays persist across timesteps to avoid repeated cudaMalloc/
/// cudaFree. The manager is a plain struct — no RAII wrappers, no
/// virtual dispatch. Caller owns lifecycle.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "gf/cuda_check.h"

namespace gf {

/// Persistent device-side pointers for element residual kernel.
///
/// Allocated once via allocate_device_buffers(), freed via free_device_buffers().
/// After each timestep, copy_u_to_device() and copy_r_to_host() transfer
/// the dynamic fields. Read-only geometry stays on device permanently.
struct CudaDeviceBuffers {
    // --- Read-only geometry (allocated once, never freed mid-loop) ---
    double* d_dxi_dx = nullptr;   // [n_elem * NGLL^3 * 9]
    double* d_jacobian = nullptr;  // [n_elem * NGLL^3]
    double* d_vp = nullptr;       // [n_elem * NGLL^3]
    double* d_vs = nullptr;       // [n_elem * NGLL^3]
    double* d_density = nullptr;  // [n_elem * NGLL^3]
    double* d_D = nullptr;        // [NGLL * NGLL]
    double* d_weights = nullptr;  // [NGLL]

    // --- Per-timestep fields (pinned / transfer each step) ---
    double* d_u = nullptr;  // [n_elem * NGLL^3 * 3]  displacement (predicted)
    double* d_r = nullptr;  // [n_elem * NGLL^3 * 3]  residual (output)

    // Sizes (in elements, not bytes)
    int n_elem = 0;
    int n_node_per_elem = 0;  // NGLL^3

    bool allocated = false;
};

/// Allocate device buffers for @p n_elem elements with @p ngll GLL points per axis.
/// Copies read-only mesh data to device. Returns initialized CudaDeviceBuffers.
inline CudaDeviceBuffers allocate_device_buffers(int n_elem, int ngll,
                                                 const double* h_dxi_dx,
                                                 const double* h_jacobian, const double* h_vp,
                                                 const double* h_vs, const double* h_density,
                                                 const double* h_D, const double* h_weights) {
#ifdef GF_WITH_CUDA
    CudaDeviceBuffers buf;
    buf.n_elem = n_elem;
    buf.n_node_per_elem = ngll * ngll * ngll;

    int n_node_total = n_elem * buf.n_node_per_elem;

    // --- Allocate read-only geometry ---
    GF_CUDA_CHECK(cudaMalloc(&buf.d_dxi_dx, n_node_total * 9 * sizeof(double)));
    GF_CUDA_CHECK(cudaMalloc(&buf.d_jacobian, n_node_total * sizeof(double)));
    GF_CUDA_CHECK(cudaMalloc(&buf.d_vp, n_node_total * sizeof(double)));
    GF_CUDA_CHECK(cudaMalloc(&buf.d_vs, n_node_total * sizeof(double)));
    GF_CUDA_CHECK(cudaMalloc(&buf.d_density, n_node_total * sizeof(double)));
    GF_CUDA_CHECK(cudaMalloc(&buf.d_D, ngll * ngll * sizeof(double)));
    GF_CUDA_CHECK(cudaMalloc(&buf.d_weights, ngll * sizeof(double)));

    // --- Allocate per-timestep fields ---
    GF_CUDA_CHECK(cudaMalloc(&buf.d_u, n_node_total * 3 * sizeof(double)));
    GF_CUDA_CHECK(cudaMalloc(&buf.d_r, n_node_total * 3 * sizeof(double)));

    // --- Copy read-only data to device ---
    GF_CUDA_CHECK(cudaMemcpy(buf.d_dxi_dx, h_dxi_dx, n_node_total * 9 * sizeof(double),
                             cudaMemcpyHostToDevice));
    GF_CUDA_CHECK(
        cudaMemcpy(buf.d_jacobian, h_jacobian, n_node_total * sizeof(double), cudaMemcpyHostToDevice));
    GF_CUDA_CHECK(
        cudaMemcpy(buf.d_vp, h_vp, n_node_total * sizeof(double), cudaMemcpyHostToDevice));
    GF_CUDA_CHECK(
        cudaMemcpy(buf.d_vs, h_vs, n_node_total * sizeof(double), cudaMemcpyHostToDevice));
    GF_CUDA_CHECK(
        cudaMemcpy(buf.d_density, h_density, n_node_total * sizeof(double), cudaMemcpyHostToDevice));
    GF_CUDA_CHECK(
        cudaMemcpy(buf.d_D, h_D, ngll * ngll * sizeof(double), cudaMemcpyHostToDevice));
    GF_CUDA_CHECK(
        cudaMemcpy(buf.d_weights, h_weights, ngll * sizeof(double), cudaMemcpyHostToDevice));

    buf.allocated = true;
    return buf;
#else
    (void)n_elem;
    (void)ngll;
    (void)h_dxi_dx;
    (void)h_jacobian;
    (void)h_vp;
    (void)h_vs;
    (void)h_density;
    (void)h_D;
    (void)h_weights;
    fprintf(stderr, "CudaDeviceBuffers: CUDA not enabled. Call allocate_device_buffers() only with GF_WITH_CUDA.\n");
    std::abort();
#endif
}

/// Free all device buffers.
inline void free_device_buffers(CudaDeviceBuffers& buf) {
#ifdef GF_WITH_CUDA
    if (!buf.allocated)
        return;
    GF_CUDA_CHECK(cudaFree(buf.d_dxi_dx));
    GF_CUDA_CHECK(cudaFree(buf.d_jacobian));
    GF_CUDA_CHECK(cudaFree(buf.d_vp));
    GF_CUDA_CHECK(cudaFree(buf.d_vs));
    GF_CUDA_CHECK(cudaFree(buf.d_density));
    GF_CUDA_CHECK(cudaFree(buf.d_D));
    GF_CUDA_CHECK(cudaFree(buf.d_weights));
    GF_CUDA_CHECK(cudaFree(buf.d_u));
    GF_CUDA_CHECK(cudaFree(buf.d_r));
    buf = CudaDeviceBuffers{};
#else
    (void)buf;
#endif
}

/// Copy predicted displacement u from host to device.
inline void copy_u_to_device(const CudaDeviceBuffers& buf, const double* h_u) {
#ifdef GF_WITH_CUDA
    int n_node_total = buf.n_elem * buf.n_node_per_elem;
    GF_CUDA_CHECK(cudaMemcpy(buf.d_u, h_u, n_node_total * 3 * sizeof(double),
                             cudaMemcpyHostToDevice));
#else
    (void)buf;
    (void)h_u;
#endif
}

/// Copy computed residual r from device to host.
inline void copy_r_to_host(const CudaDeviceBuffers& buf, double* h_r) {
#ifdef GF_WITH_CUDA
    int n_node_total = buf.n_elem * buf.n_node_per_elem;
    GF_CUDA_CHECK(cudaMemcpy(h_r, buf.d_r, n_node_total * 3 * sizeof(double),
                             cudaMemcpyDeviceToHost));
#else
    (void)buf;
    (void)h_r;
#endif
}

}  // namespace gf