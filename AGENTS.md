# gf-calculation — Root AGENTS.md

## Project Purpose

3D viscoelastic SEM forward solver + post-hoc Green's function extraction.
Python pre/post + C++17 kernel + HDF5 I/O + METIS partitioning.

Full design decisions: [`docs/design-decisions.md`](docs/design-decisions.md)

## Modules

| Module | Language | Purpose | AGENTS.md |
|--------|----------|---------|-----------|
| `preprocess/` | Python | GLL geometry, material interpolation, PML, partition, config | [`preprocess/AGENTS.md`](preprocess/AGENTS.md) |
| `forward/` | C++17 | Elastic SEM solver (libgf) + MPI executable | [`forward/AGENTS.md`](forward/AGENTS.md) |
| `compress/` | C++17 | HDF5 compression utilities (header-only) | [`compress/AGENTS.md`](compress/AGENTS.md) |
| `postprocess/` | Python | Strain Green's function extraction | [`postprocess/AGENTS.md`](postprocess/AGENTS.md) |
| `tools/` | Python | GMSH mesh → mesh.h5 converter | [`tools/AGENTS.md`](tools/AGENTS.md) |
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
| Design docs | `docs/design-decisions.md`, `docs/superpowers/design/` |

## Project State

Implementation complete — 5 modules + tests (145 tests: 97 Python, 48 C++).
Elastic-only forward solver (SLS/attenuation deferred).

See each module's `AGENTS.md` for details.

## Cross-Cutting Conventions

- **Naming**: X2Y for topology relations, 1-based with signed direction
- **Config**: Python importable scripts (no YAML/TOML)
- **Data model**: `model.h5` = mesh-dependent precomputed data; `config.h5` = simulation params
- **SI-unit suffixes** on config fields (`_m`, `_s`, `_m_s`, `_kg_m3`)
- **Timestep split**: `solver_dt` (auto from CFL) + `output_dt_s` (user snapshot interval), `snapshot_stride = output_dt_s / solver_dt`

## External Reference Codes

`external_reference_codes/` has SPECFEM3D Cartesian and Globe implementations
(read-only, untracked by git) — study SEM patterns only.