# GPU/DCU Device Abstraction — Implementation

> Parent: [../design-decisions.md](../design-decisions.md)
> Precedent: [design.md](design.md)
> Status: **CUDA backend implemented** (CPU+CUDA). HIP/SYCL deferred.

## Goal

Add a device-agnostic path for the element residual kernel — the throughput bottleneck.
Newmark, source injection, MPI exchange, and I/O stay on CPU.

Design rules:

1. Zero-cost dispatch for a single compiled backend.
1. No new dependency in the default CPU build.
1. Preserve existing CPU numerical behavior.
1. Each backend in its own source file.

## Architecture

### Backend Tags (`include/gf/backend.hpp`)

Empty tag types with compile-time dispatch via `ActiveBackend` alias:

```cpp
namespace gf {
struct BackendCPU {};    // always available
struct BackendCUDA {};   // requires GF_WITH_CUDA
// struct BackendHIP {};    // deferred
// struct BackendSYCL {};   // deferred

#if defined(GF_ACTIVE_BACKEND) && GF_ACTIVE_BACKEND == 1
using ActiveBackend = BackendCUDA;
#else
using ActiveBackend = BackendCPU;
#endif
}
```

### Kernel Entry (`include/gf/element.hpp`)

Backend-templated function with batched element interface:

```cpp
template <typename Backend>
void compute_element_residual(
    int n_elem,            // <-- batched: process all elements in one call
    const double* dxi_dx, const double* jacobian,
    const double* lambda_, const double* mu_,
    const double* D, const double* weights, int NGLL,
    const double* u, double* r);

extern template void compute_element_residual<BackendCPU>(...);
#ifdef GF_WITH_CUDA
extern template void compute_element_residual<BackendCUDA>(...);
#endif
```

**Why batched?** GPU throughput requires launching all elements in one kernel
call (grid.x = n_elem). Per-element dispatch would serialize kernel launches
and H2D/D2H transfers. The CPU backend loops internally — zero overhead
from batching.

### Support Headers

| Header | Purpose |
|--------|---------|
| `include/gf/backend.hpp` | Backend tag types + `ActiveBackend` alias |
| `include/gf/cuda_check.h` | `GF_CUDA_CHECK()` macro wrapping CUDA runtime API |
| `include/gf/cuda_device_manager.hpp` | `CudaDeviceBuffers` struct + allocate/free/copy helpers |

### Source Files

```
forward/src/
├── element_cpu.cpp        — CPU specialization (loops over n_elem internally)
├── element_cuda.cu        — CUDA kernel + specialization (grid.x = n_elem)
├── element_hip.hip.cpp    — deferred
└── element_sycl.cpp       — deferred
```

### Solver Loop (`src/solver.cpp`)

The per-element loop is removed. One batched call replaces it:

```cpp
// Before (CPU-only):
for (int elem = 0; elem < n_local; ++elem) {
    /* slice pointers */
    compute_element_residual(..., elem_u, elem_r);
}

// After (backend-agnostic):
compute_element_residual<gf::ActiveBackend>(
    n_local,
    part.dxi_dx.data(), part.jacobian.data(),
    part.lambda_.data(), part.mu_.data(),
    D_mat.data(), gll_wts.data(), ngll,
    displacement_tilde.data(), residual.data());
```

## CUDA Kernel

### Launch Configuration

```
grid:   dim3(n_elem, 1, 1)          — one block per element
block:  dim3(NGLL, NGLL, NGLL)      — one thread per GLL node (i,j,k)
```

### Per-Thread Work

Each thread (i,j,k) within element block `e`:

1. Read elastic coefficients (`lambda_`, `mu_`) and geometry (`dxi_dx`, `jacobian`) for node (i,j,k)
1. Compute displacement gradient in reference space via derivative matrix `D`
1. Transform to physical gradient via chain rule with `dxi_dx`
1. Form symmetric strain ε, isotropic stress σ
1. Scatter force contributions to all `3*NGLL^3` DOFs via `atomicAdd`

### Persistent Device Memory

`CudaDeviceBuffers` (file-scope singleton per MPI rank) caches device arrays:

| Data | Lifetime | Transfer |
|------|----------|----------|
| `dxi_dx`, `jacobian`, `lambda_`, `mu_` | Once (first call) | H2D at allocation |
| `D`, `weights` | Once (first call) | H2D at allocation |
| `u` (predicted displacement) | Each timestep | H2D before kernel |
| `r` (residual) | Each timestep | D2H after kernel |

Buffers are freed on shape change (reallocation). Cleanup before MPI_Finalize is not yet implemented — device memory is freed by OS on process exit.

> **Multi-GPU per node:** Each MPI rank must bind to a distinct GPU device
> (e.g., via `CUDA_VISIBLE_DEVICES` or `cudaSetDevice`). The code does not
> call `cudaSetDevice` — it relies on the MPI launcher or environment to assign
> devices. On multi-GPU nodes with multiple ranks, all ranks default to GPU 0
> without this binding, causing contention and memory exhaustion.

## CMake Configuration

### Root `CMakeLists.txt`

```cmake
set(GF_DEVICE_BACKEND "CPU" CACHE STRING "Device backend: CPU, CUDA")
```

### `forward/CMakeLists.txt`

```cmake
if(GF_DEVICE_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    list(APPEND BACKEND_SRCS src/element_cuda.cu)
    target_compile_definitions(libgf PRIVATE GF_WITH_CUDA)
    target_compile_definitions(libgf PRIVATE GF_ACTIVE_BACKEND=1)
    set_target_properties(libgf PROPERTIES CUDA_ARCHITECTURES "80;86;87;90")
endif()
```

### Building

```bash
# CPU (default)
cmake -B build -DGF_DEVICE_BACKEND=CPU

# CUDA
cmake -B build -DGF_DEVICE_BACKEND=CUDA
cmake --build build
```

## Tests

| File | Backend | Condition |
|------|---------|-----------|
| `tests/test_element.cpp` | CPU | Always built (updated to new batched API) |
| `tests/test_element_cuda.cu` | CUDA | Built only when `GF_DEVICE_BACKEND=CUDA` |

CUDA tests compare `compute_element_residual<BackendCPU>` vs
`compute_element_residual<BackendCUDA>` on random input, requiring
identical residual to `1e-12` tolerance.

## Limits & Future Work

1. **Device memory:** Large meshes may exceed GPU memory. Streaming (partition into tiles) is deferred.
1. **Atomics:** `atomicAdd` on double may contend. Future: shared-memory per-element reduction, then atomic per element (not per node).
1. **MPI:** MPI is always required and always used (CPU-side exchange). After each GPU kernel, residual is copied back to host for `exchange_halo`. CUDA-aware MPI is optional future work — would let exchanged `r` stay on device, eliminating D2H+H2D per timestep.
1. **Occupancy:** NGLL=4 → 64 threads/block. Low occupancy. Future: launch multiple elements per block or use 2D block with inner k-loop.
1. **r stays on device:** Residual is copied back to CPU after each step for MPI exchange. If MPI exchange stays on CPU, this is fine. If CUDA-aware MPI is used, `d_r` can persist — eliminating D2H sync.
1. **HIP/SYCL backends:** Follow the same pattern — add tag struct, add source file, add CMake branch.
1. **Device cleanup:** `g_cuda_buffers` is not explicitly freed before `MPI_Finalize`. Currently device memory is reclaimed by OS on process exit. For clean MPI shutdown, add explicit `free_device_buffers()` call before `MPI_Finalize`.

## File Summary

```
forward/
├── include/gf/
│   ├── backend.hpp              — backend tags + ActiveBackend
│   ├── cuda_check.h             — GF_CUDA_CHECK macro
│   ├── cuda_device_manager.hpp  — persistent device buffer manager
│   └── element.hpp              — backend-templated batched residual
├── src/
│   ├── element_cpu.cpp          — CPU specialization (batched)
│   ├── element_cuda.cu          — CUDA kernel + specialization
│   ├── element_hip.hip.cpp      — HIP (deferred)
│   └── element_sycl.cpp         — SYCL (deferred)
├── CMakeLists.txt               — GF_DEVICE_BACKEND option
└── solver.cpp                   — single batched call with ActiveBackend
tests/
├── test_element.cpp             — CPU tests (updated API)
├── test_element_cuda.cu         — CUDA-vs-CPU comparison tests
└── CMakeLists.txt               — conditional CUDA test build
```
