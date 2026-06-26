# preprocess/ — Python Preprocessor Pipeline

## Purpose

Reads `mesh.h5` + `config.py` → produces extended `mesh.h5`, per-rank `partition_{r}.h5`, and `config.h5`.

## Files

| File | Responsibility |
|------|---------------|
| `topology_reader.py` | Reads mesh.h5 topology (vertices, edges, surfaces, cells) |
| `gll_geometry.py` | GLL quadrature nodes/weights, Jacobian, dξ/dx, lumped mass |
| `model_loader.py` | Interpolate Vp, Vs, density to GLL nodes (from callable or binary) |
| `boundary_detector.py` | Auto-detect free surface (z≈zmin) vs absorbing boundaries |
| `cfl_validator.py` | `compute_cfl_dt()` + `compute_solver_dt()`: derive `solver_dt`, `snapshot_stride` |
| `stf_evaluator.py` | Evaluate user STF callable at `solver_dt` intervals |
| `source_locator.py` | Find containing elements + Lagrange weights on free surface |
| `pml.py` | C-PML damping profiles (face/edge/corner type classification) |
| `preflight.py` | Comprehensive validation (mesh quality, material, CFL, boundary, source, STF, storage) |
| `partition.py` | METIS k-way partition + GLL global numbering + MPI exchange patterns |
| `config_loader.py` | `load_config()` — importable Python config validation |
| `config_writer.py` | Write `config.h5` with `/simulation/`, `/domain/`, `/source/` groups |
| `model_writer.py` | Write extended mesh.h5 fields + per-rank partition_{r}.h5 files |
| `cli.py` | CLI entry point: orchestrates full pipeline |

## Data Pipeline

```
config.py ─┐
mesh.h5 ───┤
            ↓
    topology_reader → gll_geometry → model_loader → boundary_detector
                                                          ↓
    cfl_validator → stf_evaluator → source_locator → preflight (validation)
                                                          ↓
    pml (damping) → partition (METIS) → model_writer → config_writer
                                                          ↓
    mesh.h5 (extended) + partitions/partition_{r}.h5 + config.h5
```

## Config Schema (config.h5 /simulation/)

| Attribute | Type | Source |
|-----------|------|--------|
| `solver_dt` | float64 | Auto-computed from CFL |
| `output_dt_s` | float64 | User config |
| `snapshot_stride` | int32 | Derived: `output_dt_s / solver_dt` |
| `nsteps` | int32 | Derived: `ceil(total_duration_s / solver_dt)` |
| `cfl_safety` | float64 | User config |
| `snapshot_precision` | string | User config ("float32" or "float64") |
| `storage_limit_gb` | int/float | User config |
| `polynomial_order` | int32 | User config |

Removed fields: `dt`, `nsteps` (user), `cfl_threshold`, `checkpoint_interval`, `checkpoint_precision`

## Tests

`tests/preprocess/test_*.py` — 74 tests covering each pipeline step.
`examples/halfspace/run.sh` — End-to-end pipeline (mesh generation → preprocess → forward solver in 3 directions).

## Design Doc

`docs/superpowers/design/preprocess.md`