# gf-calculation — Root AGENTS.md

## Project Purpose

3D viscoelastic SEM forward solver + post-hoc Green's function extraction.
Python pre/post + C++17 kernel + HDF5 I/O + METIS partitioning.

Full design decisions: [`docs/design-decisions.md`](docs/design-decisions.md)
Full math formulation: [`docs/math.md`](docs/math.md)

## Modules

| Module | Language | Purpose | AGENTS.md |
|--------|----------|---------|-----------|
| `preprocess/` | Python + C++17 | GLL geometry, material interpolation, PML, partition, config; C++ accelerator (default, OpenMP) | [`preprocess/AGENTS.md`](preprocess/AGENTS.md) |
| `forward/` | C++17 | Elastic SEM solver (libgf) + MPI executable | [`forward/AGENTS.md`](forward/AGENTS.md) |
| `compress/` | C++17 | HDF5 compression utilities (header-only) | [`compress/AGENTS.md`](compress/AGENTS.md) |
| `postprocess/` | Python | Strain Green's function extraction | [`postprocess/AGENTS.md`](postprocess/AGENTS.md) |
| `tools/` | Python | Mesh conversion (GMSH→HDF5) + VTK visualization tools | [`tools/AGENTS.md`](tools/AGENTS.md) |
| `tests/` | Python + C++ | Shared test infrastructure (pytest + Catch2) | [`tests/AGENTS.md`](tests/AGENTS.md) |

## Tech Stack

| Layer | Tool |
|-------|------|
| Core compute | C++17, MPI (OpenMPI/MPICH), CUDA (future), Eigen (small matrices) |
| Build | CMake |
| I/O | HDF5 |
| Mesh partitioning | METIS (called from preprocessor) |
| Pre/post | Python + C++17 (accelerator, OpenMP) |
| External reference | `external_reference_codes/` (read-only, untracked by git) |
| Design docs | `docs/design-decisions.md`, `docs/math.md`, `docs/superpowers/design/` |

## Project State

Implementation complete — 5 modules + tests (144 tests: 96 Python, 48 C++).
Elastic-only forward solver (SLS/attenuation deferred).

See each module's `AGENTS.md` for details.

## Cross-Cutting Conventions

- **Naming**: X2Y for topology relations, 1-based with signed direction
- **Config**: Python importable scripts (no YAML/TOML)
- **Data model**: `model.h5` = mesh-dependent precomputed data; `config.h5` = simulation params
- **SI-unit suffixes** on config fields (`_m`, `_s`, `_m_s`, `_kg_m3`)
- **Full names for scientific/physical variables**: Use descriptive full words, not single-letter or abbreviated names. Examples: `displacement`, `velocity`, `acceleration`, `strain`, `solver_dt`, `snapshot_stride`, `vertex_ids`, `green_tile_size_m`. No `u`, `v`, `a`, `dt`, `ss`, `vid`, `tile_sz` — even in local scope. Single letters allowed only for pure math indices (`i`, `j`, `k`) in tight loops
- **Timestep split**: `solver_dt` (auto from CFL) + `output_dt_s` (user snapshot interval), `snapshot_stride = output_dt_s / solver_dt`
- **No receivers**: Postprocess uses shallow mesh-vertex strain records, not receiver locations. No receivers.csv, receiver search, or interpolation. See `docs/design-decisions.md`.
- **config.py is source of truth**: All simulation parameters live in `config.py`. Scripts read it at runtime; no duplicated constants. See `examples/halfspace/config.py`.
- **Fixed filenames**: Examples use fixed output names (`mesh.h5`, `config.h5`, `partition_{r}.h5`, `record_{r}.h5`, `restart_{r}.h5`). No CLI overrides, except input paths.
- **Console scripts in root pyproject**: Tool entry points live in root `pyproject.toml` `[project.scripts]` and install via `gf-calculation`, not `gf-preprocess`.
- **Pipeline scripts read config**: `run.sh` derives `N_RANKS` and runtime params from `config.py`, not hardcoded values.
- **Run formatter before stage/commit**: Run `bash format.sh` before `git add` or `git commit`. It formats Python, Markdown, C/C++, CUDA, and CMake files.

## External Reference Codes

`external_reference_codes/` has SPECFEM3D Cartesian and Globe implementations
(read-only, untracked by git) — study SEM patterns only.
