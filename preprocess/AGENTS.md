# preprocess/ — AGENTS.md

## Purpose

Read `mesh.h5` + `config.py`. Write extended `mesh.h5`, `config.h5`, and per-rank `partition_{r}.h5`. Also build shallow mesh-vertex recording maps.

## Files

| File | Responsibility |
|------|----------------|
| `gll_geometry.py` | GLL nodes, weights, Jacobian, `dxi_dx`, mass |
| `model_loader.py` | evaluate `vp`, `vs`, density at GLL nodes |
| `boundary_detector.py` | detect free surface and absorbing faces |
| `cfl_validator.py` | derive `solver_dt`, `snapshot_stride`, `restart_stride` |
| `stf_evaluator.py` | sample user STF at solver steps |
| `source_locator.py` | find source elements and weights |
| `pml.py` | C-PML profiles and element tags |
| `preflight.py` | validate mesh, material, CFL, source, storage, recording map |
| `partition.py` | METIS partition, GLL numbering, MPI exchange |
| `config_loader.py` | import and validate `config.py` |
| `config_writer.py` | write `config.h5` |
| `model_writer.py` | write mesh fields and partition files, including `/recording/`; precomputes λ, μ from Vp, Vs, density |
| `accelerator.py` | optional C++ subprocess for GLL geometry, CFL, PML damping |
| `cli.py` | run full pipeline from CWD |

## Pipeline

```
mesh.h5 + config.py
→ load config
→ [C++ accelerator: GLL geometry, CFL h_min, PML damping ramps]
→ material at GLL nodes
→ CFL + solver_dt + strides
→ source + STF
→ PML masking        (1-layer from boundary detection + layer expansion via i,j,k grid)
→ validation
→ METIS partition
→ recording map      (snaps depth, builds tile_index clamped to PML + recording depth)
→ write mesh.h5, config.h5, partitions/partition_{r}.h5
```

## Config Rules

- `config.py` is sole source of truth.
- Use SI suffixes: `_m`, `_s`, `_m_s`, `_kg_m3`.
- No YAML/TOML.
- No receivers.
- No force direction in config. Forward gets `--direction`.

## Recording Map

Preprocess selects non-PML mesh vertices in the shallow output volume:

- requested bottom: `record_depth_max_m`
- actual bottom: `record_depth_actual_m`, snapped to element face
- tile sizes (elements): `tilex_elements`, `tiley_elements`

Each rank writes:

```
/recording/
  attrs: basis="mesh_vertices", record_depth_max_m,
         record_depth_actual_m, excludes_pml=true
  save_element_mask
  vertex_ids
  source_element_local_index
  source_corner_index
```

## Tests

`tests/` has pytest coverage for geometry, material interpolation, CFL, source, PML, partitioning, config writing, and integration.

## Design Doc

[`../docs/design/preprocess.md`](../docs/design/preprocess.md)
## C++ Accelerator

Heavy numerical loops (GLL geometry, CFL h_min, PML damping ramps) can be
offloaded to a compiled C++ executable. See `cpp/main.cpp`.

- Binary: `bin/gf_preprocess_cpp` (built by CMake to `bin/`, or g++ — see docs)
- Threading: OpenMP multi-threading enabled by default (single-thread without OpenMP).
  Set `OMP_NUM_THREADS` to control; accelerator logs actual thread count at INFO.
- Fallback: pure Python if binary absent or CFL info cannot be parsed.
- Integration: `accelerator.py` → runs subprocess, reads results from HDF5
- Diagnostics: C++ stdout prints `H_MIN`, `CFL_DT`, `CFL_SAFETY`, `OMP_THREADS` for Python to capture and log.
