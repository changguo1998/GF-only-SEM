# AGENTS.md - Green's Function Calculation

## Project Purpose

3D viscoelastic SEM forward solver + post-hoc Green's function extraction.
Python pre/post + C++17 kernel + HDF5 I/O + METIS partitioning.

Full design decisions: [`docs/design-decisions.md`](docs/design-decisions.md)
Module-level designs: [`docs/superpowers/design/`](docs/superpowers/design/)

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

## External Reference Codes

`external_reference_codes/` contains upstream SPECFEM implementations
(SPECFEM3D Cartesian and SPECFEM3D Globe) as read-only reference.
They are untracked by git (`*.gitignore`), never built or edited —
used only to study SEM implementation patterns.

## Project State

Implementation complete — all 5 modules have source code and passing tests (126 tests across Python + C++). Elastic-only forward solver (SLS/attenuation deferred).

Key design docs:
- `docs/design-decisions.md` — system-level decisions (CG-SEM, hexahedra, Newmark, etc.)
- `docs/superpowers/design/mesh.md` — mesh.h5 topology format, model.h5 schema
- `docs/superpowers/design/preprocess.md` — GLL geometry, 3D model interpolation, SLS, partition
- `docs/superpowers/design/forward.md` — libgf physics + MPI solver
- `docs/superpowers/design/compress.md` — checkpoint compression utilities
- `docs/superpowers/design/postprocess.md` — strain Green's function extraction

When adding code to this repo:
- Match SPECFEM3D conventions for computational code
- 1-based indexing with signed direction for topology (X2Y naming)
- Python config scripts (importable `.py`) instead of YAML/TOML
- model.h5 = Green's function database (all mesh-dependent data precomputed)
- config.h5 = simulation configuration separate from model data