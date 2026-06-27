# gf-calculation — Root AGENTS.md

## Project Purpose

3D viscoelastic SEM forward solver + post-hoc Green's function extraction.
Python pre/post + C++17 kernel + HDF5 I/O + METIS partitioning.

Full design decisions: [`docs/design-decisions.md`](docs/design-decisions.md)
Full math formulation: [`docs/math.md`](docs/math.md)

## Modules

| Module | Language | Purpose | AGENTS.md |
|--------|----------|---------|-----------|
| `preprocess/` | Python | GLL geometry, material interpolation, PML, partition, config | [`preprocess/AGENTS.md`](preprocess/AGENTS.md) |
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
| Pre/post | Python |
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
- **Timestep split**: `solver_dt` (auto from CFL) + `output_dt_s` (user snapshot interval), `snapshot_stride = output_dt_s / solver_dt`
- **No receivers**: Postprocess does NOT use receiver locations. Green's functions are extracted at all GLL nodes. No receivers.csv, no receiver positions, no receiver search in postprocess. This is a design constraint — see `docs/design-decisions.md` line 253.
- **config.py is the single source of truth**: All simulation parameters (mesh dimensions, material, source, boundary, parallelism) defined in `config.py` only. No script or shell script duplicates or hardcodes these values — they must be read from `config.py` at runtime. See `examples/halfspace/config.py` for the canonical schema.
- **Fixed filenames per design docs**: Example scripts (mesh generators, converters) use fixed output filenames per the design docs (`mesh.h5`, `config.h5`, `partition_{r}.h5`, `record_{r}.h5`). No CLI args to override them. Only I/O paths that vary per run (e.g., input GMSH file path) use CLI arguments.
- **Console scripts in root pyproject**: All tool entry points (`mesh2vtk`, `partition2vtk`, `wavefield2vtk`, `wavefield2vtk_detail`) are defined in the root `pyproject.toml` under `[project.scripts]` and installed as console_scripts via `gf-calculation`, not `gf-preprocess`.
- **Pipeline scripts read config**: `run.sh` and similar pipeline scripts derive `N_RANKS` and other runtime params from `config.py` via Python extraction, not hardcoded values.
- **Run formatter before stage/commit**: Execute `bash format.sh` before `git add` or `git commit` to ensure all `.py`, `.md`, `.c/.cpp/.h/.hpp/.cu`, `.cmake`/`CMakeLists.txt` files are formatted consistently. Silent on success, fails fast on error.

## External Reference Codes

`external_reference_codes/` has SPECFEM3D Cartesian and Globe implementations
(read-only, untracked by git) — study SEM patterns only.
