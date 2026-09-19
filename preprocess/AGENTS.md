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
| `pml.py` | legacy damping profiles and element tags |
| `pml_cpml.py` | complete C-PML profiles and convolution coefficients (Python path) |
| `preflight.py` | validate mesh, material, CFL, source, storage, recording map |
| `partition.py` | METIS partition, GLL numbering, MPI exchange |
| `config_loader.py` | import and validate `config.py` |
| `config_writer.py` | write `config.h5` |
| `model_writer.py` | write mesh fields and partition files, including `/recording/`; precomputes λ, μ from Vp, Vs, density |
| `stage2_runner.py` | wrap `gf_preprocess stage2` for λ/μ, solver_dt, nsteps (fallback) |
| `topology_reader.py` | read `/topology/` group from model.h5 |
| `recording_map.py` | build shallow GLL recording map (Python path) |
| `accelerator.py` | optional C++ subprocess for GLL geometry, CFL, PML damping |
| `cli.py` | run full pipeline from CWD |

## Pipeline

```
model.h5 + config.py
→ load config
→ C++ stage1? → GLL geometry + boundary tags + PML damping
→ else → Python gll_geometry + boundary_detector + PML
→ material at GLL nodes (Python model_loader.py)
→ C++ stage2? → λ/μ + CFL solver_dt + nsteps
→ else → Python numpy + cfl_validator
→ Python C-PML + source + STF (from config.py)
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

Preprocess selects non-PML GLL nodes in the shallow output volume:

- requested bottom: `record_depth_max_m`
- actual bottom: `record_depth_actual_m`, snapped to element face
- tile sizes (elements): `tilex_elements`, `tiley_elements`

Each rank writes:

```
/recording/
  attrs: basis="gll", record_depth_max_m,
         record_depth_actual_m, excludes_pml=true
  gll_node_ids
  gll_node_coords
  rec_cell_local
  rec_cell_global_ids
  cell_gll_node_index
```

## Tests

`tests/` has pytest coverage for geometry, material interpolation, CFL, source, PML, partitioning, config writing, and integration.

## Design Doc

[`../docs/design/preprocess.md`](../docs/design/preprocess.md)

Single binary with subcommands:

- **`gf_preprocess stage1`**: GLL geometry, CFL h_min, PML expansion + damping, boundary tagging.

  - Source: `cpp/main.cpp`
  - CLI: `gf_preprocess stage1 <model.h5> --N N --cfl-safety VAL [--nx N] [--ny N] [--pml-* THICK]`
  - OpenMP multi-threading; logs `H_MIN`, `CFL_DT`, `OMP_THREADS` to stdout

- **`gf_preprocess stage2`**: λ/μ from Vp/Vs/density, CFL solver_dt, snapshot_stride, nsteps, pre-flight stats.

  - Source: `cpp/stage2_main.cpp`
  - CLI: `gf_preprocess stage2 <model.h5>`
  - Reads `/field/element/{vp,vs,density}` + `/config/` attrs; writes `/field/element/{lambda,mu}`
  - Prints `STAT_*` lines parsed by `stage2_runner.py`

Integration: `cli.py` discovers `gf_preprocess` at startup, dispatches via subprocess with
`stage1`/`stage2` subcommand. Falls back to pure Python per step independently. The Python
entry point does not call `gf_preprocess run`, because that command uses a compile-time C++
configuration and cannot represent arbitrary `config.py` material/STF callables.

`gf_preprocess run` is the Python-free preprocessing path after topology-only `model.h5` exists.
It writes `config.h5` and all `partitions/partition_{r}.h5` files, including rank node maps,
MPI exchange patterns, `/recording/`, and complete C-PML coefficients from the compiled
`GF_USER_CONFIG`.

Built from `preprocess/cpp/CMakeLists.txt`. CPU only (no MPI, no CUDA).
