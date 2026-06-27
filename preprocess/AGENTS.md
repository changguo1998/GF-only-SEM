# preprocess/ — Python Preprocessor Pipeline

## Purpose

Reads `mesh.h5` + `config.py` → produces extended `mesh.h5`, per-rank `partition_{r}.h5`, and `config.h5`.
Also derives the shallow recording map used by forward to write mesh-vertex Green's-function strain records.

## Files

| File | Responsibility |
|------|---------------|
| `topology_reader.py` | Reads mesh.h5 topology (vertices, edges, surfaces, cells) |
| `gll_geometry.py` | GLL quadrature nodes/weights, Jacobian, dξ/dx, lumped mass |
| `model_loader.py` | Interpolate Vp, Vs, density to GLL nodes (from callable or binary) |
| `boundary_detector.py` | Auto-detect free surface (z≈zmin) vs absorbing boundaries |
| `cfl_validator.py` | `compute_cfl_dt()` + `compute_solver_dt()`: derive `solver_dt`, `snapshot_stride`, `restart_stride` |
| `stf_evaluator.py` | Evaluate user STF callable at `solver_dt` intervals |
| `source_locator.py` | Find containing elements + Lagrange weights on free surface |
| `pml.py` | C-PML damping profiles (face/edge/corner type classification) |
| `preflight.py` | Comprehensive validation (mesh quality, material, CFL, boundary, source, STF, storage, recording map) |
| `partition.py` | METIS k-way partition + GLL global numbering + MPI exchange patterns |
| `config_loader.py` | `load_config()` — importable Python config validation |
| `config_writer.py` | Write `config.h5` with `/simulation/`, `/domain/`, `/source/` groups |
| `model_writer.py` | Write extended mesh.h5 fields + per-rank `partition_{r}.h5` files including `/recording/` maps |
| `cli.py` | CLI entry point: reads `mesh.h5` + `config.py` from CWD, orchestrates full pipeline |

## Data Pipeline

```
config.py ─┐
mesh.h5 ───┤
            ↓
    topology_reader → gll_geometry → model_loader → boundary_detector
                                                          ↓
    cfl_validator → recording map → stf_evaluator → source_locator
                                                          ↓
    pml (damping) → partition (METIS) → preflight → model_writer → config_writer
                                                          ↓
    mesh.h5 (extended) + partitions/partition_{r}.h5 + config.h5
```

## Recording Map

Preprocess snaps `record_depth_max_m` to a horizontal spectral-element face at or deeper than the requested depth. It marks non-PML elements/vertices on or above that face.

Per-rank partition files include:

```text
/recording/
  attrs: basis="mesh_vertices", record_depth_max_m, record_depth_actual_m,
         green_tile_size_m, excludes_pml=true
  save_element_mask          bool[n_local_elem]
  vertex_ids                 int64[n_record_vertices]
  source_element_local_index int32[n_record_vertices]
  source_corner_index        int8[n_record_vertices]
```

## Config Schema (config.h5 /simulation/)

| Attribute | Type | Source |
|-----------|------|--------|
| `solver_dt` | float64 | Auto-computed from CFL |
| `output_dt_s` | float64 | User config |
| `snapshot_stride` | int32 | Derived: `output_dt_s / solver_dt` |
| `restart_dt_s` | float64 | User config |
| `restart_stride` | int32 | Derived: `restart_dt_s / solver_dt` |
| `record_depth_max_m` | float64 | User config |
| `record_depth_actual_m` | float64 | Derived: snapped horizontal spectral-element face depth |
| `green_tile_size_m` | float64 | User config |
| `nsteps` | int32 | Derived: `ceil(total_duration_s / solver_dt)` |
| `cfl_safety` | float64 | User config |
| `snapshot_precision` | string | User config ("float32" or "float64") |
| `storage_limit_gb` | int/float | User config |
| `polynomial_order` | int32 | User config |

Removed fields: `dt`, `nsteps` (user), `cfl_threshold`, `checkpoint_interval`, `checkpoint_precision`

## Tests

`tests/preprocess/test_*.py` — tests covering each pipeline step.
`examples/halfspace/run.sh` — End-to-end pipeline (mesh generation → preprocess → forward solver in 3 directions).

## Design Doc

`docs/superpowers/design/preprocess.md`
