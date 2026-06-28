# forward/ — AGENTS.md

## Purpose

Elastic CG-SEM solver. Reads `config.h5` + `partition_{r}.h5`. Computes full volume. Writes shallow mesh-vertex strain snapshots and latest-only restart files.

## Architecture

### Library (`libgf`)

| Header | Implementation | Responsibility |
|--------|----------------|----------------|
| `types.hpp` | — | config and rank data structs |
| `gll.hpp` | — | GLL nodes, weights, derivative matrices |
| `element.hpp` | `element.cpp` | element residual and strain |
| `assembly.hpp` | `assembly.cpp` | global residual assembly |
| `newmark.hpp` | `newmark.cpp` | explicit Newmark time step |
| `source.hpp` | `source.cpp` | point force injection |
| `pml.hpp` | `pml.cpp` | C-PML update |
| `exchange.hpp` | `exchange.cpp` | MPI halo exchange |
| `io.hpp` | `io.cpp` | HDF5 input |
| `record.hpp` | `record.cpp` | shallow strain writer |
| `solver.hpp` | `solver.cpp` | time loop |
| — | `main.cpp` | MPI CLI, `--direction` |

### Executable

```bash
mpirun -n N gf_solver --direction x
```

Binary at `bin/gf_solver` (built by CMake). Add `bin/` to `$PATH` or run as `./bin/gf_solver`.

Frozen paths from CWD:

- input: `config.h5`, `partitions/partition_{r}.h5`
- strain: `wavefields/{direction}/record_{r}.h5`
- restart: `restart/{direction}/restart_{r}.h5`

Caller creates directories.

## Time Loop

```
Newmark predict
→ residual K·u
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
estimated finish time (yyyy-mm-dd HH:MM:SS). Updated in-place via carriage return

- ANSI clear-line (`\r\x1b[K`, no newline). Finalised with newline on completion.

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

`wavefields/{direction}/record_{r}.h5`

Attrs: `rank`, `source_direction`, `basis="mesh_vertices"`, `record_depth_max_m`, `record_depth_actual_m`, `excludes_pml`.

Datasets:

- `vertex_ids`: `int64[n_record_vertices]`, global mesh vertex IDs, 1-based
- `strain`: `[n_snapshots, n_record_vertices, 6]`, extendible in time

## Restart Schema

`restart/{direction}/restart_{r}.h5` is overwritten in place. It stores exact-resume state:

- `displacement`, `velocity`, `acceleration`: full local GLL volume
- all active C-PML memory variables
- attrs: `step`, `time_s`, `source_direction`, `ngll`

## Tests

Catch2 tests cover GLL, element, assembly, Newmark, PML, source, exchange, IO, record, compress, integration.

## Design Doc

[`docs/superpowers/design/forward.md`](../docs/superpowers/design/forward.md)
