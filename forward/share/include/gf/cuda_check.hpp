#pragma once

/// @file
/// CUDA error-checking utility macro.
///
/// Wraps CUDA runtime API calls and reports file/line on failure.
/// Only active when GF_WITH_CUDA is defined.

#include <cstdio>
#include <cstdlib>

#ifdef GF_WITH_CUDA
#include <cuda_runtime.h>

/// Check a CUDA runtime API call and abort on error with file/line info.
#define GF_CUDA_CHECK(call)                                                   \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

#else
// Stub — no-op when CUDA is disabled
#define GF_CUDA_CHECK(call) (void)(call)
#endif  // GF_WITH_CUDA