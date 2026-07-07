#pragma once

/// @file
/// Device backend tags for compile-time dispatch.
///
/// Each backend is an empty tag type. The active backend is selected
/// via CMake define GF_DEVICE_BACKEND and aliased as ActiveBackend.
///
/// Usage:
///   compute_element_residual<gf::ActiveBackend>(...);
///
/// New backends add a tag struct and a specialization in their own
/// source file (e.g., element_cuda.cu, element_hip.hip.cpp).

namespace gf {

// -----------------------------------------------------------------------
// Backend tag types — empty structs for compile-time dispatch
// -----------------------------------------------------------------------

/// CPU backend (default, always available).
struct BackendCPU {};

/// NVIDIA CUDA backend (requires GF_WITH_CUDA).
struct BackendCUDA {};

// Future backends (reserved):
// struct BackendHIP {};
// struct BackendSYCL {};

// -----------------------------------------------------------------------
// Active backend alias — set by CMake compile definition
// -----------------------------------------------------------------------
//
// Default: BackendCPU. When GF_DEVICE_BACKEND=CUDA is configured, CMake
// adds -DGF_ACTIVE_BACKEND=1 and the alias below resolves to
// BackendCUDA. The same pattern applies for HIP, SYCL, etc.
//
// The define GF_ACTIVE_BACKEND is set by CMakeLists.txt.

#if defined(GF_ACTIVE_BACKEND) && GF_ACTIVE_BACKEND == 1  // CUDA
// Future: GF_ACTIVE_BACKEND==2 → BackendHIP, 3 → BackendSYCL
using ActiveBackend = BackendCUDA;
#else
using ActiveBackend = BackendCPU;
#endif

}  // namespace gf