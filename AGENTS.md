# gf-calculation — Root AGENTS.md

## Project Purpose

3D viscoelastic SEM forward solver + post-hoc Green's function extraction.
Python pre + C++17 kernel + HDF5 I/O + METIS partitioning.

Full design decisions: [`docs/design-decisions.md`](docs/design-decisions.md)
Full math formulation: [`docs/math.md`](docs/math.md)

## Modules

| Module | Language | Purpose | AGENTS.md |
|--------|----------|---------|-----------|
| `preprocess/` | Python + C++17 | GLL geometry, material interpolation, PML, partition, config; C++ accelerator (default, OpenMP) | [`preprocess/AGENTS.md`](preprocess/AGENTS.md) |
| `forward/` | C++17 | Elastic SEM solver (libgf) + MPI executable | [`forward/AGENTS.md`](forward/AGENTS.md) |
| `forward/viscoelastic/` | C++17 | Viscoelastic SEM solver (SLS) — skeleton, implementation deferred | [`forward/viscoelastic/AGENTS.md`](forward/viscoelastic/AGENTS.md) |
| `compress/` | C++17 | HDF5 compression utilities (header-only) | [`compress/AGENTS.md`](compress/AGENTS.md) |
| `postprocess/` | C++17 | Strain Green's function extraction (Python archived in `_archive/`) | [`postprocess/AGENTS.md`](postprocess/AGENTS.md) |
| `tools/` | C++17 + Python | VTK visualization tools (C++ primary, Python archived); GMSH→HDF5 conversion (Python) | [`tools/AGENTS.md`](tools/AGENTS.md) |
| `tests/` | Python + C++ | Shared test infrastructure (pytest + Catch2) | [`tests/AGENTS.md`](tests/AGENTS.md) |
| `greenfun/` | Python | Green's function reader with reciprocity query | [`greenfun/AGENTS.md`](greenfun/AGENTS.md) |

## Tech Stack

| Layer | Tool |
|-------|------|
| Core compute | C++17, MPI (OpenMPI/MPICH), CUDA (implemented), Eigen (small matrices) |
| Build | CMake |
| I/O | HDF5 |
| Mesh partitioning | METIS (called from preprocessor) |
| Pre/post + VTK | Python + C++17 (OpenMP for preprocessor and VTK tools) |
| External reference | `external_reference_codes/` (read-only, untracked by git) |
| Design docs | `docs/design-decisions.md`, `docs/math.md`, `docs/design/` — per-module design docs in `docs/design/` |

## Build Environment

### Spack (development machine)

Dependencies managed via Spack. Activate before building:

```bash
source $HOME/.spack/share/spack/setup-env.sh
spack load cuda        # CUDA 13.2 — required for CUDA backend
spack load /zkrqzmds   # OpenMPI 5.0.10 (use hash to disambiguate)
```

Available packages: `openmpi@5.0.10`, `cuda@13.2.1`, `eigen@3.4.0`.
System HDF5 at `/usr/include/hdf5/serial/`.

### Building Forward Solver

```bash
cd forward
# CPU (default)
cmake -B build -DGF_DEVICE_BACKEND=CPU
cmake --build build

# CUDA (requires cuda loaded via spack)
cmake -B build -DGF_DEVICE_BACKEND=CUDA
cmake --build build
```

### Formatting

Run `bash format.sh` before staging/committing. Requires `.venv` (ruff, mdformat)
and spack-installed `llvm` for clang-format.

## Project State

CG-SEM global-DOF assembly fix complete — waves now correctly propagate across element interfaces (both within-rank and cross-rank). All 221 tests pass (202 Python + 19 C++ Catch2).

Elastic-only forward solver (SLS/attenuation deferred).

Buried source support implemented (`source_z_m = None`→free surface, `float`→buried). Preprocessor auto-detects surface vs buried mode and excludes PML elements for buried sources.

**Example validation pipelines** (`examples/halfspace`, `examples/layer`) run end-to-end: SEM → reference → comparison. Diagonal displacement components match analytic/PyFK references within 3%. P-SV coupling components show ~0.5-2× bias due to Cartesian mesh anisotropy (documented in [`docs/deferred.md`](docs/deferred.md) §6 and [`docs/superpowers/plans/2026-07-16-pysv-coupling-debug.md`](docs/superpowers/plans/2026-07-16-pysv-coupling-debug.md)).

| Solver variant | Multi-rank | DOF numbering | Status |
|---------------|------------|---------------|--------|
| CPU + MPI | ✅ (16 ranks) | Global (ibool) | ✅ Verified — diagonals 1.01-1.03× ref |
| CUDA single | N/A | Element-local (legacy) | ⚠ Completes but wrong — legacy element-local path (read_partition_all clears ibool → can't use CG-SEM). rel_l2≈1.0 (uncorrelated) |
| CUDA + MPI | ✅ (4 ranks) | Global (ibool) | ✅ Verified — rel_l2=0.644 matches CPU 16-rank. MPI exchange + scatter fix (570c58b) |

CUDA single-GPU falls back to legacy element-local path because `read_partition_all` clears `local_element2rank_node` (can't merge per-rank ibool into a single global numbering). The CG-SEM path requires per-rank partitions via `read_partition`. This is a known architectural limitation, not a correctness regression.

## Cross-Cutting Conventions

- **Naming**: X2Y for topology relations, 1-based with signed direction
- **Config**: Python importable scripts (no YAML/TOML)
- **Data model**: `model.h5` = mesh-dependent precomputed data; `config.h5` = simulation params
- **SI-unit suffixes** on config fields (`_m`, `_s`, `_m_s`, `_kg_m3`)
- **Full names for scientific/physical variables**: Use descriptive full words, not single-letter or abbreviated names. Examples: `displacement`, `velocity`, `acceleration`, `strain`, `solver_dt`, `snapshot_stride`, `vertex_ids`, `green_tile_size_m`, `tilex_elements`. No `u`, `v`, `a`, `dt`, `ss`, `vid`, `tile_sz` — even in local scope. Single letters allowed only for pure math indices (`i`, `j`, `k`) in tight loops
- **Timestep split**: `solver_dt` (auto from CFL) + `output_dt_s` (user snapshot interval), `snapshot_stride = output_dt_s / solver_dt`
- **No receivers**: Postprocess uses shallow mesh-vertex strain records, not receiver locations. No receivers.csv, receiver search, or interpolation. See `docs/design-decisions.md`.
- **config.py is source of truth**: All simulation parameters live in `config.py`. Scripts read it at runtime; no duplicated constants. See `examples/halfspace/config.py`.
- **Fixed filenames**: Examples use fixed output names (`model.h5`, `config.h5`, `partition_{r}.h5`, `record_{r}.h5`, `restart_{r}.h5`). No CLI overrides, except input paths.
- **Console scripts in root pyproject**: Tool entry points live in root `pyproject.toml` `[project.scripts]` and install via `gf-calculation`, not `gf-preprocess`.
- **Pipeline scripts read config**: `run.sh` derives `N_RANKS` and runtime params from `config.py`, not hardcoded values.
- **Run formatter before stage/commit**: Run `bash format.sh` before `git add` or `git commit`. It formats Python, Markdown, C/C++, CUDA, and CMake files.

## External Reference Codes

`external_reference_codes/` has SPECFEM3D Cartesian and Globe implementations
(read-only, untracked by git) — study SEM patterns only.
