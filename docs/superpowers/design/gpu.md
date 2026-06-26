# GPU/DCU Device Abstraction — Technical Design

> Parent: [docs/design-decisions.md](../../design-decisions.md)
> Design precedent: [forward.md](forward.md) (existing pure-MPI architecture)

## Goal

Provide a device-agnostic abstraction for the element residual kernel
(`compute_element_residual`) that supports CPU, CUDA (NVIDIA), HIP (AMD/DCU),
and future backends — without modifying the solver loop or any other component.

## Design Principle

The **element residual kernel is the only throughput-critical compute kernel**
in the forward solver (~85%+ of runtime). All other operations (Newmark update,
source injection, MPI exchange, checkpoint I/O) are memory-bound or latency-bound
and remain on CPU. Therefore the abstraction targets this single kernel, not
the entire codebase.

The abstraction must:

1. **Zero cost** when compiled for a single backend (no virtual dispatch per element)
1. **Add no new dependencies** to the base build (GPU backends are opt-in via CMake)
1. **Preserve the existing CPU path** as the default — unchanged behavior, no new headers
1. **Support gradual adoption** — each backend is a separate `.cu` / `.hip.cpp` file

## Architecture

### Policy Tag

A type tag selects the backend at compile time, set by a CMake definition:

```cpp
// forward/include/gf/backend.hpp

namespace gf {

// Backend tags (empty types — compile-time dispatch only)
struct BackendCPU    {};
struct BackendCUDA   {};
struct BackendHIP    {};
struct BackendSYCL   {};

// The active backend is set by -DGF_DEVICE_BACKEND=CUDA etc.
// Default when no flag is set:
#ifndef GF_DEVICE_BACKEND
#  define GF_DEVICE_BACKEND CPU
#endif

using ActiveBackend = Backend##GF_DEVICE_BACKEND;

} // namespace gf
```

### Templatized Kernel Entry Point

The existing `compute_element_residual` becomes a **template on the backend**:

```cpp
// forward/include/gf/element.hpp

#include "gf/backend.hpp"

namespace gf {

template<typename Backend>
void compute_element_residual(
    const double* dxi_dx,
    const double* jacobian,
    const double* vp,
    const double* vs,
    const double* density,
    const double* D,
    const double* weights,
    int NGLL,
    const double* u,
    double* r
);

// Extern template declarations — each .cpp instantiates its own backend
extern template void compute_element_residual<BackendCPU>(...);

#ifdef GF_WITH_CUDA
extern template void compute_element_residual<BackendCUDA>(...);
#endif

#ifdef GF_WITH_HIP
extern template void compute_element_residual<BackendHIP>(...);
#endif

} // namespace gf
```

### File Layout

Each backend lives in a separate translation unit, compiled by the appropriate
compiler (host C++ for CPU, `nvcc` for CUDA, `hipcc` for HIP):

```
forward/src/
├── element_cpu.cpp        → BackendCPU     (existing element.cpp, renamed)
├── element_cuda.cu        → BackendCUDA    (new, nvcc)
├── element_hip.cpp        → BackendHIP     (new, hipcc)
└── element_sycl.cpp       → BackendSYCL    (future, DPC++)
```

Each file:

- Includes `gf/element.hpp`
- Provides a template specialization of `compute_element_residual` for its backend
- GPU backends manage device memory internally (alloc/copy if needed, but see
  Optimization Notes below for persistent device allocation)

### Solver Loop — No Changes

The solver in `solver.cpp` uses the active backend:

```cpp
// solver.cpp
#include "gf/element.hpp"

// In the time loop:
for (int e = 0; e < n_local; ++e) {
    /* ... same pointer math ... */
    compute_element_residual<gf::ActiveBackend>(
        elem_dxi_dx, elem_jac, elem_vp, elem_vs,
        elem_rho, D_mat.data(), gll_wts.data(),
        ngll, elem_u, elem_r
    );
}
```

The loop body is unchanged — only the function template gains a backend tag.

## CMake Integration

```cmake
# forward/CMakeLists.txt

# Default: CPU backend
set(GF_DEVICE_BACKEND "CPU" CACHE STRING "Device backend: CPU, CUDA, HIP, SYCL")

# --- Backend sources ---
set(BACKEND_SRCS src/element_cpu.cpp)

if(GF_DEVICE_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    list(APPEND BACKEND_SRCS src/element_cuda.cu)
    target_compile_definitions(libgf PRIVATE GF_WITH_CUDA)

elseif(GF_DEVICE_BACKEND STREQUAL "HIP")
    find_package(HIP REQUIRED)
    list(APPEND BACKEND_SRCS src/element_hip.hip.cpp)
    target_compile_definitions(libgf PRIVATE GF_WITH_HIP)
    # hipcc handles the .hip.cpp compilation
    set_source_files_properties(src/element_hip.hip.cpp
        PROPERTIES LANGUAGE HIP)

elseif(GF_DEVICE_BACKEND STREQUAL "SYCL")
    # DPC++ / oneAPI path
    list(APPEND BACKEND_SRCS src/element_sycl.cpp)
    target_compile_definitions(libgf PRIVATE GF_WITH_SYCL)

endif()

target_compile_definitions(libgf PRIVATE
    GF_DEVICE_BACKEND=${GF_DEVICE_BACKEND})

# Replace the old element.cpp with backend sources
target_sources(libgf PRIVATE ${BACKEND_SRCS})

# Remove element.cpp from the static source list
# (already listed in add_library, so we exclude it conditionally)
```

## The CUDA Kernel Structure

The computational core of each GPU implementation is the same as `element_cpu.cpp`
but expressed as a GPU kernel where **each GLL node** is one thread/thread-block:

```cuda
// forward/src/element_cuda.cu
#include "gf/element.hpp"

namespace gf {

__global__ void element_residual_kernel(
    const double* dxi_dx, const double* jacobian,
    const double* vp, const double* vs, const double* density,
    const double* D, const double* weights,
    int NGLL, int n_elem,
    const double* u, double* r
) {
    int e = blockIdx.x;                          // one block per element
    int i = threadIdx.z;                         // ξ index
    int j = threadIdx.y;                         // η index
    int k = threadIdx.x;                         // ζ index

    if (e >= n_elem) return;

    int n_node = NGLL * NGLL * NGLL;
    int n = (i * NGLL + j) * NGLL + k;           // flat GLL node index

    const double* dd   = &dxi_dx[9 * (e * n_node + n)];
    const double rho   = density[e * n_node + n];
    if (rho <= 0.0) return;

    // ... compute stress σ at this node (same math as element_cpu.cpp) ...

    // Accumulate into residual (atomic add since multiple threads target same node)
    for (int s = 0; s < NGLL; ++s) {
        int n_s = (s * NGLL + j) * NGLL + k;
        atomicAdd(&r[3 * (e * n_node + n_s) + 0], contribution_x);
        atomicAdd(&r[3 * (e * n_node + n_s) + 1], contribution_y);
        atomicAdd(&r[3 * (e * n_node + n_s) + 2], contribution_z);
    }
}
```

Launch configuration:

```cuda
dim3 block(NGLL, NGLL, NGLL);   // e.g., (4,4,4) for N=3, (5,5,5) for N=4
dim3 grid(n_elem, 1, 1);
element_residual_kernel<<<grid, block>>>(...);
cudaDeviceSynchronize();
```

## Optimization Notes

### Persistent Device Memory

For production use, the CPU-to-GPU copy of mesh data per timestep is prohibitive.
The `compute_element_residual<BackendCUDA>` specialization should:

1. **First call**: allocate device memory and copy all mesh data (dxi_dx, jacobian,
   vp, vs, density, D, weights) to the device — these are read-only throughout the run
1. **Each timestep**: copy `u` → device, launch kernel, copy `r` → host
1. **Optimized path**: if the solver can keep `r` on device and only copy when MPI
   exchange needs it, the device↔host transfer is reduced to field `u` and ghost values

This persistent allocation can be managed via:

```cpp
struct DeviceMesh {
    double *d_dxi_dx, *d_jac, *d_vp, *d_vs, *d_rho, *d_D, *d_wts, *d_u, *d_r;
    bool allocated = false;
};

// thread_local or static within the specialization
```

### HIP (DCU) Implementation

The HIP kernel is syntactically identical to the CUDA kernel — just replace `__global__`
with `__global__` (HIP uses the same syntax) and `hipLaunchKernelGGL` or `<<<>>>`:

```cpp
// forward/src/element_hip.cpp
#include "gf/element.hpp"

__global__ void element_residual_kernel_hip(...) {
    // same logic as CUDA kernel
}

template<>
void compute_element_residual<BackendHIP>(...) {
    // same pattern as CUDA specialization
}
```

The CMake integration uses `hipcc` as the compiler for HIP translation units.

### NGLL Tile Size

The block dimensions are `(NGLL, NGLL, NGLL)`. For NGLL=4 (N=3, test mode) this is 64
threads per block — under-utilizing modern GPUs. For NGLL=5 (N=4) it's 125 threads,
still modest. Options:

1. **Multiple elements per block**: launch fewer blocks, have each block process
   multiple elements sequentially (increases occupancy)
1. **Split NGLL loop**: one thread per (i,j) pair, inner k-loop (parallelism along ζ
   only, NGLL threads per block = NGLL², works for NGLL ≤ 8)
1. **Element batch processing**: combine multiple elements' data into larger tiles
   (requires strided memory layout)

Deferred to optimization phase — start with simple 1-thread-per-GLL-node for correctness.

## Maintainability

### Adding a New Backend

1. Create a new translation unit `element_<backend>.cpp`
1. Write `template<> void compute_element_residual<BackendX>(...)`
1. Add CMake detection for the new compiler/toolchain
1. Add compiler definition to `target_compile_definitions`

### Testing

The existing tests in `tests/test_element.cpp` test the CPU path.
GPU backends require separate integration tests:

```cpp
// tests/test_element_cuda.cpp (compiled with nvcc)
TEST_CASE("CUDA element residual matches CPU", "[element][cuda]") {
    // Generate random input
    // Run compute_element_residual<BackendCPU>
    // Run compute_element_residual<BackendCUDA>
    // Compare r with tolerance
}
```

These tests only build when the corresponding backend is enabled.

## Limitations

1. **Device memory is a finite resource**: for large meshes, the mesh data on GPU
   may exceed device memory. A streaming/partitioning scheme would be needed,
   processing a subset of elements per launch. This is deferred — for initial
   GPU support, mesh size is bounded by available GPU memory.

1. **Atomic contention**: the residual accumulation uses `atomicAdd` because
   multiple GLL nodes' stress contributions scatter to the same DOF indices.
   For high-order elements (NGLL ≥ 6, 216+ threads), contention on the residual
   vector may limit throughput. A shared-memory reduction within each block
   (element-local residual, then coalesced atomic adds to global) can mitigate this.

1. **MPI+GPU hybrid**: the MPI exchange currently operates on CPU memory.
   If `r` lives on the GPU, either:
   a. Copy to CPU for exchange (simple, O(n_interface) per timestep)
   b. Use CUDA-aware MPI (requires MPI implementation with GPU support → OpenMPI 4.0+)
   Initial implementation uses (a) — GPU is a per-rank accelerator, not a replacement
   for MPI.

## File Summary

```
forward/
├── include/gf/
│   ├── backend.hpp              ← New: BackendCPU/CUDA/HIP tags
│   └── element.hpp              ← Modified: template on Backend
├── src/
│   ├── element_cpu.cpp          ← Renamed from element.cpp, unchanged logic
│   ├── element_cuda.cu          ← New: CUDA kernel
│   ├── element_hip.hip.cpp      ← New: HIP kernel
│   └── element_sycl.cpp         ← Future: SYCL kernel
└── tests/
    ├── test_element.cpp         ← Existing CPU test (unchanged)
    ├── test_element_cuda.cu     ← New: CUDA correctness test
    └── test_element_hip.hip.cpp ← New: HIP correctness test
```

The existing `element.cpp` is renamed to `element_cpu.cpp` for consistency.
All other solver code (`solver.cpp`, `main.cpp`, `assembly.cpp`, etc.) is unchanged.
