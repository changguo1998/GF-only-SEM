# preprocess/ — AGENTS.md

## Purpose

Read `model.h5` + `config.py`. Write extended `model.h5`, `config.h5`, and per-rank `partition_{r}.h5`. Also build shallow mesh-vertex recording maps.

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
| `stage2_runner.py` | wrap `gf_preprocess_stage2` for λ/μ, solver_dt, nsteps |
| `topology_reader.py` | read `/topology/` group from model.h5 |
| `recording_map.py` | build shallow mesh-vertex recording map |
| `accelerator.py` | optional C++ subprocess for GLL geometry, CFL, PML damping |
| `cli.py` | run full pipeline from CWD |

## Pipeline

```
model.h5 + config.py
→ load config
→ C++ stage1? → GLL geometry + CFL h_min + PML + boundary
→ else → Python gll_geometry + boundary_detector + PML
→ material at GLL nodes (Python model_loader.py)
→ C++ stage2? → λ/μ + CFL solver_dt + nsteps
→ else → Python numpy + cfl_validator
→ PML masking        (1-layer from boundary detection + layer expansion via i,j,k grid)
→ validation
→ METIS partition
→ recording map      (snaps depth, builds tile_index clamped to PML + recording depth)
→ write model.h5, config.h5, partitions/partition_{r}.h5
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

Two binaries:

- **`gf_preprocess_cpp`** (stage1): GLL geometry, CFL h_min, PML expansion + damping, boundary tagging.

  - Source: `cpp/main.cpp`, built to `bin/gf_preprocess_cpp`
  - CLI: `gf_preprocess_cpp <model.h5> <N> <cfl_safety> <nx> <ny> [pml_thickness...]`
  - OpenMP multi-threading; logs `H_MIN`, `CFL_DT`, `OMP_THREADS` to stdout

- **`gf_preprocess_stage2`** (stage2): λ/μ from Vp/Vs/density, CFL solver_dt, snapshot_stride, nsteps, pre-flight stats.

  - Source: `cpp/stage2_main.cpp`, built to `bin/gf_preprocess_stage2`
  - CLI: `gf_preprocess_stage2 <model.h5>`
  - Reads `/field/element/{vp,vs,density}` + `/config/` attrs; writes `/field/element/{lambda,mu}`
  - Prints `STAT_*` lines parsed by `stage2_runner.py`

Integration: `cli.py` discovers both binaries at startup, runs each step via subprocess when
the binary exists. Falls back to pure Python per step independently. `accelerator.py` provides
`_ensure_domain_attrs()` helper (legacy; `run_accelerator` superseded by `stage2_runner.py`).

Both binaries built from single `cpp/CMakeLists.txt`. CPU only (no MPI, no CUDA).
