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

Empty tag types (kept for future template-based code paths):

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

Plain (non-template) function with batched element interface. The CPU and CUDA
implementations share the same signature; the correct one is selected at link
time by which library (`libgf_elastic` or `libgf_elastic_cuda_*`) is linked.

```cpp
void compute_element_residual(
    int n_cell,            // <-- batched: process all elements in one call
    const double* dxi_dx, const double* jacobian,
    const double* lambda_, const double* mu_,
    const double* D, const double* weights, int NGLL,
    const double* u, double* r);
```

**Why batched?** GPU throughput requires launching all elements in one kernel
call (grid.x = n_cell). Per-element dispatch would serialize kernel launches
and H2D/D2H transfers. The CPU backend loops internally — zero overhead
from batching.

### Support Headers

| Header | Purpose |
|--------|---------|
| `include/gf/backend.hpp` | Backend tag types (kept for future template paths) |
| `include/gf/cuda_check.hpp` | `GF_CUDA_CHECK()` macro wrapping CUDA runtime API |
| `include/gf/cuda_device_manager.hpp` | `CudaDeviceBuffers` struct + allocate/free/copy helpers |

### Source Files

```
forward/share/src/
├── element_cpu.cpp        — CPU implementation (loops over n_cell internally)
├── element_cuda.cu        — CUDA kernel + implementation (grid.x = n_cell)
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
compute_element_residual(
    n_local,
    part.dxi_dx.data(), part.jacobian.data(),
    part.lambda_.data(), part.mu_.data(),
    D_mat.data(), gll_wts.data(), ngll,
    displacement_tilde.data(), residual.data());
```

## CUDA Kernel

### Launch Configuration

```
grid:   dim3(n_cell, 1, 1)          — one block per cell
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

> **Multi-GPU per node:** `main.cpp` auto-binds GPU via `cudaSetDevice(rank % n_devices)`.
> When MPI ranks on a shared-memory node exceed GPUs, the solver warns, reduces to
> 1 rank per GPU, and redistributes partitions via `read_partition_range()`.
> Excess ranks (shm_rank >= n_devices) exit early.

## CG-SEM Global DOF Path on GPU

When `use_global_dof=true` (partition file has `local_cell2rank_node`), the GPU solver
supports an alternative code path that mirrors the CPU CG-SEM assembly. Additional
CUDA kernels in `cuda_step.cu` implement the global DOF operations:

| Kernel | Purpose | Shared memory |
|--------|---------|---------------|
| `cuda_newmark_predict` | Newmark predictor on rank-level arrays | No |
| `cuda_gather_predicted` | Gather `d_rank_node_displacement_tilde` → element-local via ibool | No |
| `cuda_zero_residual` | Zero element-local residual array | No |
| `cuda_launch_element_residual` | Element kernel (same as legacy, reads/writes elem-local) | Depends on NGLL |
| `cuda_cpml_update_displ_fields` | Build old/new C-PML auxiliary displacement fields | No |
| `cuda_cpml_update_displ_memory` | Update Ā₁–Ā₅ displacement memory | No |
| `cuda_cpml_update_strain_memory` | Update A₆–A₂₃ strain memory | No |
| `cuda_cpml_accel_contribution` | Add C-PML acceleration correction | No |
| `cuda_pml_damping` | Legacy damping when C-PML data are absent | No |
| `cuda_source_injection` | Source injection into element-local residual | No |
| `cuda_scatter_to_rank` | Atomic `atomicAdd` from elem-local → rank-level residual via ibool | No |
| `cuda_newmark_correct` | Newmark corrector with mass-exchange handling (skip ghost-only nodes) | No |
| `cuda_gather_from_rank` | Gather rank-level displacement → element-local for strain recording | No |

The GPU scatter kernel uses `atomicAdd` on `d_rank_node_residual` to correctly
accumulate element contributions at shared nodes. No MPI exchange is performed
on GPU — the residual is left on device and the host-side `exchange_halo` handles
cross-rank assembly (GPU→CPU copy avoids device-side MPI complexity).

The CUDA C-PML sequence mirrors the CPU path: build the old/new auxiliary
displacement fields after the synchronized Newmark predictor, update displacement
and strain memory, evaluate the element residual, then add the acceleration
correction. Legacy velocity damping is mutually exclusive with C-PML.

**Device state allocation**: `cuda_allocate_state()` in `cuda_step.cu` detects
`use_global_dof` from partition data and allocates appropriate arrays:

- Global DOF: `d_rank_node_*` arrays of size `[n_rank_node × 3]` + element-local temps `[n_local_cell × n_node × 3]`
- Legacy: `d_*` arrays of size `[n_local_cell × n_node × 3]` (element-local only)

Both paths share the same element residual kernel. The global DOF path adds
gather/scatter wrapper kernels and rank-level arrays.

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
    target_compile_definitions(libgf_elastic PRIVATE GF_WITH_CUDA)
    target_compile_definitions(libgf_elastic PRIVATE GF_ACTIVE_BACKEND=1)
    set_target_properties(libgf_elastic PROPERTIES CUDA_ARCHITECTURES "80;86;87;90")
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

CUDA tests compare the CUDA result against an inline CPU reference
(`reference_element_residual` in the test file), requiring identical
residual to `1e-12` tolerance. The C-PML CUDA test additionally compares
the old/new auxiliary fields, displacement and strain memory, and acceleration
residual against the CPU implementation for one complete update.

## 限制与后续工作

1. **显存容量：** 大模型可能超过显存容量。按网格块流式计算仍处于延期状态。
1. **原子操作竞争：** double `atomicAdd` 可能形成瓶颈。可先在共享内存中进行单元内归约，
   再按单元执行原子累加。
1. **CUDA + MPI 数据搬运：** 多 rank CUDA 路径每步将残差复制到主机执行
   `exchange_halo`，再复制回设备。CUDA-aware MPI 可消除这组 D2H/H2D 搬运；单 GPU
   可执行文件使用空交换实现，不受 MPI 通信限制。
1. **占用率：** NGLL=4 时每 block 只有 64 个线程。可让一个 block 处理多个单元，或使用
   二维 block 并在内部循环 k 方向。
1. **HIP/SYCL 后端：** 可沿用 CUDA 模式增加 backend tag、源文件和 CMake 分支。
1. **设备清理：** 求解器状态通过 `cuda_free_state()` 显式释放；单元核的静态
   `g_cuda_buffers` 仍由进程退出时回收。若需要严格的 MPI 退出清理，可补充显式释放入口。

## File Summary

```
forward/
├── include/gf/
│   ├── backend.hpp              — backend tags (kept for future use)
│   ├── cuda_check.hpp             — GF_CUDA_CHECK macro
│   ├── cuda_device_manager.hpp  — persistent device buffer manager
│   └── element.hpp              — plain function, link-time dispatch
├── src/
│   ├── element_cpu.cpp          — CPU implementation (batched)
│   ├── element_cuda.cu          — CUDA kernel + implementation
│   ├── element_hip.hip.cpp      — HIP (deferred)
│   └── element_sycl.cpp         — SYCL (deferred)
├── CMakeLists.txt               — GF_DEVICE_BACKEND option
└── solver.cpp                   — single batched call with link-time dispatch
tests/
├── test_element.cpp             — CPU tests (updated API)
├── test_element_cuda.cu         — CUDA-vs-CPU comparison tests
└── CMakeLists.txt               — conditional CUDA test build
```
