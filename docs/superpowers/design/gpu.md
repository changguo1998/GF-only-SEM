# GPU/DCU Device Abstraction — Technical Design

> Parent: [docs/design-decisions.md](../../design-decisions.md)
> Precedent: [forward.md](forward.md)

## Goal

Add a device-agnostic path for the element residual kernel. Support CPU first, then CUDA, HIP/DCU, and future SYCL. Do not change the solver loop.

## Principle

`compute_element_residual` is the throughput kernel. It takes most runtime. Newmark, source injection, MPI exchange, and I/O stay on CPU.

Rules:

1. Zero-cost dispatch for a single compiled backend.
1. No new dependency in the default CPU build.
1. Preserve existing CPU behavior.
1. Add each backend in its own source file.

## Architecture

### Backend Tag

Use empty tag types and compile-time dispatch:

```cpp
namespace gf {
struct BackendCPU {};
struct BackendCUDA {};
struct BackendHIP {};
struct BackendSYCL {};
using ActiveBackend = BackendCPU;  // default, set by CMake in real build
}
```

### Kernel Entry

Make `compute_element_residual` a backend template:

```cpp
template <typename Backend>
void compute_element_residual(...);

extern template void compute_element_residual<BackendCPU>(...);
#ifdef GF_WITH_CUDA
extern template void compute_element_residual<BackendCUDA>(...);
#endif
#ifdef GF_WITH_HIP
extern template void compute_element_residual<BackendHIP>(...);
#endif
```

### Files

```
forward/src/
├── element_cpu.cpp        — CPU specialization, current logic
├── element_cuda.cu        — CUDA specialization
├── element_hip.hip.cpp    — HIP specialization
└── element_sycl.cpp       — future SYCL specialization
```

Each file includes `gf/element.hpp` and specializes `compute_element_residual`.
GPU files own their device-memory logic.

### Solver Loop

Only template argument changes:

```cpp
compute_element_residual<gf::ActiveBackend>(...);
```

Pointer math and loop structure stay the same.

## CMake

Default backend is CPU:

```cmake
set(GF_DEVICE_BACKEND "CPU" CACHE STRING "Device backend: CPU, CUDA, HIP, SYCL")
set(BACKEND_SRCS src/element_cpu.cpp)

if(GF_DEVICE_BACKEND STREQUAL "CUDA")
  enable_language(CUDA)
  list(APPEND BACKEND_SRCS src/element_cuda.cu)
  target_compile_definitions(libgf PRIVATE GF_WITH_CUDA)
elseif(GF_DEVICE_BACKEND STREQUAL "HIP")
  find_package(HIP REQUIRED)
  list(APPEND BACKEND_SRCS src/element_hip.hip.cpp)
  target_compile_definitions(libgf PRIVATE GF_WITH_HIP)
endif()

target_sources(libgf PRIVATE ${BACKEND_SRCS})
```

## CUDA/HIP Kernel Shape

Start simple: one block per element, one thread per GLL node.

```cuda
dim3 block(NGLL, NGLL, NGLL);
dim3 grid(n_elem, 1, 1);
element_residual_kernel<<<grid, block>>>(...);
```

Inside the kernel:

1. Map block to element and thread to `(i,j,k)`.
1. Read `dxi_dx`, `jacobian`, material, `D`, weights, and `u`.
1. Compute strain and stress.
1. Scatter residual with `atomicAdd`.

HIP uses the same kernel logic with HIP launch syntax or `<<<>>>`.

## Optimization Notes

### Persistent Device Memory

Do not copy mesh data each timestep. First call allocates device arrays and copies read-only mesh data. Each timestep copies `u`, runs the kernel, and copies `r` back. Later optimization can keep `r` on device. Then only MPI interface data moves.

### NGLL Occupancy

For NGLL=4 or 5, one element block has 64–125 threads. That is correct but may underuse GPUs. Later options:

1. Multiple elements per block.
1. One thread per `(i,j)` with an inner `k` loop.
1. Batched element tiles.

Correctness comes first.

## Adding a Backend

1. Add `element_<backend>.cpp`.
1. Specialize `compute_element_residual<BackendX>`.
1. Add compiler/toolchain detection in CMake.
1. Add compile definitions.

## Testing

CPU tests stay in `tests/test_element.cpp`.
GPU tests build only when that backend is enabled:

```cpp
TEST_CASE("CUDA element residual matches CPU", "[element][cuda]") {
  // generate input, run CPU and CUDA, compare residual
}
```

## Limits

1. **Device memory:** large meshes may exceed GPU memory. Streaming is deferred.
1. **Atomics:** residual scatter may bottleneck. Shared-memory reductions can help later.
1. **MPI:** exchange uses CPU memory. Initial GPU path copies interface data to CPU. CUDA-aware MPI is optional future work.

## File Summary

```
forward/
├── include/gf/
│   ├── backend.hpp              — backend tags
│   └── element.hpp              — backend-templated residual
├── src/
│   ├── element_cpu.cpp          — renamed current element.cpp
│   ├── element_cuda.cu          — CUDA kernel
│   ├── element_hip.hip.cpp      — HIP kernel
│   └── element_sycl.cpp         — future SYCL kernel
└── tests/
    ├── test_element.cpp         — CPU
    ├── test_element_cuda.cu     — CUDA optional
    └── test_element_hip.hip.cpp — HIP optional
```

Only `element.cpp` is renamed to `element_cpu.cpp`. Other solver files stay unchanged.
