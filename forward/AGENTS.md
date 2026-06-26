# forward/ — C++ Elastic SEM Forward Solver

## Purpose

Elastic wave propagation solver using Continuous Galerkin Spectral Element Method (CG-SEM).
Reads partition files + config.h5, writes strain snapshots to record files.

## Architecture

### Library (libgf)

| Header | Implementation | Responsibility |
|--------|---------------|----------------|
| `types.hpp` | — | `SimulationConfig`, `MeshConfig`, `RankData`, `ExchangePattern` structs |
| `gll.hpp` | — | GLL quadrature nodes, weights, derivative matrices (inline, templated) |
| `element.hpp` | `element.cpp` | Element-level K·u matrix-free residual, strain computation |
| `assembly.hpp` | `assembly.cpp` | Global assembly: scatter-add element residuals, L2 strain projection |
| `newmark.hpp` | `newmark.cpp` | Newmark predictor-corrector (β=0, γ=½ explicit) |
| `source.hpp` | `source.cpp` | Source injection via precomputed Lagrange weights |
| `pml.hpp` | `pml.cpp` | C-PML memory variable update + acceleration correction |
| `exchange.hpp` | `exchange.cpp` | MPI halo exchange using precomputed face-pair patterns |
| `io.hpp` | `io.cpp` | HDF5 reader for partition/config files |
| `record.hpp` | `record.cpp` | Extendible HDF5 strain snapshot writer + restart state |
| `solver.hpp` | `solver.cpp` | `run_forward()` main time integration loop |
| — | `main.cpp` | MPI entry point, CLI parsing (`--direction` flag) |

### Solver Executable (`gf_solver`)

```
mpirun -n N gf_solver --direction x     # from CWD with frozen paths
```

All I/O paths are frozen relative to CWD:
- Input:  `config.h5`, `partitions/partition_{r}.h5`
- Output: `wavefields/{direction}/record_{r}.h5`

Directory creation is the caller's responsibility (shell script does `mkdir -p`).

### Time Loop (per step)

```
Newmark predict (ũ, ṽ)
→ element residual (matrix-free K·u)
→ C-PML memory update + acceleration correction
→ source injection (precomputed weights)
→ MPI halo exchange (face-pair patterns)
→ Newmark correct (u, v, a)
→ strain compute (element pass)
→ L2 strain smoothing (global projection)
→ snapshot write (if step % snapshot_stride == 0)
```

### Runtime Loop Driver

```mermaid
flowchart LR
    A[Newmark Predict] --> B[Element Residual]
    B --> C[C-PML]
    C --> D[Source]
    D --> E[MPI Exchange]
    E --> F[Newmark Correct]
    F --> G[Strain Compute + L2 Smooth]
    G --> H{step % stride == 0?}
    H -->|yes| I[Write Snapshot]
    H -->|no| J{step < nsteps?}
    I --> J
    J -->|yes| A
    J -->|no| K[Done]
```

## Config Source

`solver_dt` — Newmark timestep (from config.h5 `/simulation/solver_dt`).
`snapshot_stride` — write snapshot when `step % snapshot_stride == 0`.

## Record File Schema

`wavefields/{direction}/record_{r}.h5` attrs: `rank`, `source_direction`, `ngll`.
Strain dataset: `[n_snapshots, n_elem_local, NGLL, NGLL, NGLL, 6]` (extendible along dim 0).
Restart: `/restart/displacement`, `/restart/velocity`, `/restart/acceleration` (overwritten each snapshot).

## Tests

`tests/test_*.cpp` — 48 Catch2 tests across GLL, element, assembly, Newmark, PML, source, exchange, IO, record, compress, integration.

## Design Doc

`docs/superpowers/design/forward.md`