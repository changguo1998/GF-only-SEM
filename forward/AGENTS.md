# forward/ — AGENTS.md

## Purpose

Elastic CG-SEM solver. Reads `config.h5` + `partition_{r}.h5`. Computes full volume. Writes shallow mesh-vertex strain snapshots and latest-only restart files.

## Architecture

### Library (`libgf`)

| Header | Implementation | Responsibility |
|--------|----------------|----------------|
| `types.hpp` | — | config and rank data structs |
| `gll.hpp` | — | GLL nodes, weights, derivative matrices |
| `backend.hpp` | — | device backend tags (`BackendCPU`, `BackendCUDA`, `ActiveBackend`) |
| `cuda_check.h` | — | `GF_CUDA_CHECK()` error macro |
| `cuda_device_manager.hpp` | — | persistent CUDA device buffer manager |
| `element.hpp` | `element_cpu.cpp`, `element_cuda.cu` | backend-templated element residual (K·u) |
| `assembly.hpp` | `assembly.cpp` | global residual assembly |
| `newmark.hpp` | `newmark.cpp` | explicit Newmark time step |
| `source.hpp` | `source.cpp` | point force injection |
| `pml.hpp` | `pml.cpp` | C-PML update |
| `exchange.hpp` | `exchange.cpp`, `exchange_noop.cpp` | MPI halo exchange (or no-op stub) |
| `io.hpp` | `io.cpp` | HDF5 input |
| `record.hpp` | `record.cpp` | shallow strain writer |
| `solver.hpp` | `solver.cpp` | time loop (backend-agnostic, dispatches via `ActiveBackend`) |
| — | `main.cpp` | CLI for all 3 binaries, `--direction` |

### Device Backend

`compute_element_residual` is backend-templated (`BackendCPU` / `BackendCUDA`).
Each solver binary uses a compile-time backend selection via `GF_ACTIVE_BACKEND`.

Elastic coefficients λ and μ are precomputed at GLL nodes during preprocessing
and read from partition files — the kernel receives λ, μ directly instead of
computing them from Vp, Vs, density every timestep.

MPI is optional at link time: `gf_solver_elastic_cuda` links a no-op exchange stub
and guards `MPI_Init` via `#ifndef GF_NO_MPI`. All other solvers use real MPI.
The GPU backend replaces *only* `compute_element_residual` — Newmark, PML,
source, exchange, I/O stay on CPU. After each GPU kernel, residual is copied
back to host for MPI exchange.

CMake `GF_DEVICE_BACKEND=CUDA` enables the CUDA path. All other solver components
(Newmark, PML, source, exchange, I/O) remain on CPU.

- **CPU**: `element_cpu.cpp` — batched loop over elements (default)
- **CUDA**: `element_cuda.cu` — one block/element, one thread/GLL node, `atomicAdd` scatter

See [`../docs/design/gpu.md`](../docs/design/gpu.md) for full design.

> **GPU auto-binding:** `gf_solver_elastic_cuda` and `gf_solver_elastic_mpi_cuda` automatically detect
> available GPUs via `cudaGetDeviceCount()` and assign `cudaSetDevice(rank % n_devices)`.
> If MPI ranks on a shared-memory node exceed GPUs, the solver warns and reduces to
> 1 rank per GPU: excess ranks exit early, remaining ranks redistribute partitions
> via `read_partition_range()` (block distribution, no cross-rank exchange).

### Executables

Three solver binaries are produced, selectable by name:

| Binary | Backend | MPI | Use case |
|--------|---------|-----|----------|
| `gf_solver_elastic_mpi` | CPU | yes | CPU cluster, workstation |
| `gf_solver_elastic_cuda` | CUDA | no | Single GPU, no MPI needed |
| `gf_solver_elastic_mpi_cuda` | CUDA | yes | Multi-GPU cluster |

All three share the same source code. MPI-dependent code is guarded by
`#ifndef GF_NO_MPI`. Backend dispatch uses `ActiveBackend` (set by CMake
define `GF_ACTIVE_BACKEND`).

```bash
# MPI + CPU (default)
mpirun -n N gf_solver_elastic_mpi --direction x

# Single GPU (no mpirun)
gf_solver_elastic_cuda --direction x

# Multi-GPU via MPI
mpirun -n N gf_solver_elastic_mpi_cuda --direction x
```

Frozen paths from CWD:

- input: `config.h5`, `partitions/partition_{r}.h5`
- strain: `wavefields/{direction}/record_{r}_{step}.h5` (one file per snapshot)
- restart: `restart/{direction}/restart_{r}.h5`

Caller creates directories.

## Time Loop

```
Newmark predict
→ residual K·u  [dispatched to active backend]
→ C-PML
→ source
→ MPI exchange
→ Newmark correct
→ strain + L2 smoothing
→ write shallow strain if step % snapshot_stride == 0
→ overwrite restart if step % restart_stride == 0
```

## Progress Output

On each snapshot write (rank 0 only), an in-place progress line is printed:

```
 500/5000  10%  elapsed= 123.4s  eta= 1110.6s  finish~2026-06-28 15:30:45
```

Fields: step/total, percentage, wall-clock elapsed, estimated remaining time,
estimated finish time (yyyy-mm-dd HH:MM:SS). Updated in-place via carriage return + ANSI clear-line.

- Writes directly to /dev/tty (controlling terminal) to bypass MPI I/O
  forwarding (orte/iof), which line-buffers rank output pipes.
- Log file gets full timestamped lines without escape sequences (no in-place).

Some ranks may have zero recorded vertices (no shallow elements). These ranks
write an empty record file with `vertex_ids (0,)` and `strain (0,0,6)` and skip
strain computation. The solver does not fall back to full-volume GLL strain when
recording mode is enabled (`record_depth_max_m > 0`).

## Config Fields

- `solver_dt`: Newmark timestep
- `snapshot_stride`: strain write cadence
- `restart_stride`: restart write cadence
- `record_depth_actual_m`: snapped bottom depth for records

## Record Schema

`wavefields/{direction}/record_{r}_{step}.h5` — one per snapshot.

Attrs: `rank`, `source_direction`, `basis="mesh_vertices"`, `record_depth_max_m`, `record_depth_actual_m`, `excludes_pml`.

Datasets:

- `vertex_ids`: `int64[n_record_vertices]`, global mesh vertex IDs, 1-based
- `strain`: `float32[1, n_record_vertices, 6]`
- `displacement`: `float32[1, n_record_vertices, 3]`
- `velocity`: `float32[1, n_record_vertices, 3]`
- `acceleration`: `float32[1, n_record_vertices, 3]`

## Restart Schema

`restart/{direction}/restart_{r}.h5` is overwritten in place. It stores exact-resume state:

- `displacement`, `velocity`, `acceleration`: full local GLL volume
- all active C-PML memory variables
- attrs: `step`, `time_s`, `source_direction`, `ngll`

## Build

### Prerequisites

On the development machine, dependencies are managed via Spack:

```bash
source $HOME/.spack/share/spack/setup-env.sh
spack load cuda        # CUDA 13.2 — needed for CUDA backend
spack load /zkrqzmds   # OpenMPI 5.0.10
```

### CMake

```bash
# CPU + MPI (default — always available)
cmake -B build
cmake --build build --target gf_solver_elastic_mpi

# CUDA single-GPU (requires CUDA toolkit)
cmake -B build
cmake --build build --target gf_solver_elastic_cuda

# CUDA + MPI multi-GPU
cmake -B build
cmake --build build --target gf_solver_elastic_mpi_cuda

# All available targets
cmake --build build
```

CMake auto-detects MPI and CUDA via `find_package`. Targets whose
dependencies are not found are silently skipped.

## Tests

Catch2 tests cover GLL, element (CPU + CUDA), assembly, Newmark, PML, source, exchange, IO, record, compress, integration.

## Design Docs

- [`../docs/design/forward.md`](../docs/design/forward.md) — solver architecture
- [`../docs/design/gpu.md`](../docs/design/gpu.md) — GPU backend design
