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
| `forward/` | C++17 | Elastic SEM solver (libgf_elastic) + MPI executable | [`forward/AGENTS.md`](forward/AGENTS.md) |
| `forward/viscoelastic/` | C++17 | Viscoelastic SEM solver (SLS) — SLS attenuation — complete | [`forward/viscoelastic/AGENTS.md`](forward/viscoelastic/AGENTS.md) |
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

### Build Environment

First-time setup:

```bash
uv sync                                # create Python venv + dev tools
source scripts/env.sh                  # load Spack packages + add bin/ to PATH
```

Build all solvers and tools:

```bash
scripts/build.sh                       # auto-detect CPU / CUDA
scripts/build.sh cpu                   # CPU only
scripts/build.sh cuda                  # CPU + CUDA
scripts/build.sh -t gf_postprocess     # single target
```

Run a solver:

```bash
scripts/solver.sh                      # interactive menu
scripts/solver.sh elastic cpu mpi -- --direction x
scripts/solver.sh elastic cuda -- --direction x
```

Install to system (requires Spack + venv):

```bash
scripts/install.sh                      # → /usr/local/bin, auto-backend
scripts/install.sh /opt/gf-calculation  # → custom prefix
scripts/install.sh ~/.local cpu         # → user install, CPU only
```

To uninstall:

```bash
# List installed files
cmake --install build-install --component gf_solver --prefix /tmp/list 2>/dev/null; \
  grep -r ... ;# alternatively, check scripts/install.sh output
rm -f /usr/local/bin/gf_solver_*
rm -f /usr/local/bin/gf_preprocess
rm -f /usr/local/bin/gf_postprocess
rm -f /usr/local/bin/gf_model2vtk
rm -f /usr/local/bin/gf_partition2vtk
rm -f /usr/local/bin/gf_wavefield2vtk*
rm -f /usr/local/bin/solver.sh build.sh env.sh
```

Spack packages required: `openmpi@5.0.10`, `cuda@13.2.1` (optional), `eigen@3.4.0`.
System HDF5 at `/usr/include/hdf5/serial/`.

### Formatting

Run `bash format.sh` before staging/committing. Requires `.venv` (ruff, mdformat)
and spack-installed `llvm` for clang-format.

## Project State

CG-SEM global-DOF assembly fix complete — waves now correctly propagate across element interfaces (both within-rank and cross-rank). All 207 Python tests pass. C++ Catch2 tests (17) require MPI-enabled build configuration.

Elastic + viscoelastic (SLS) forward solvers complete. See `scripts/solver.sh` to select.

Buried source support implemented (`source_z_m = None`→free surface, `float`→buried). Preprocessor auto-detects surface vs buried mode and excludes PML elements for buried sources.

**Example validation pipelines** (`examples/halfspace`, `examples/layer`) run end-to-end: SEM → reference → comparison. After fixing a postprocess mass-weighting bug (commit `6f90c12`), scaled waveform correlation improved from 0.945 to 0.991 (halfspace) and 0.745 to 0.745 (layer, unchanged — amplitude-only fix). A residual systematic factor of ~3× (2.95 halfspace, 2.60 layer) remains, attributed to SEM GLL discretization vs continuous Green's function — see [`docs/bugs.md`](docs/bugs.md) §Issue 3. The earlier "P-SV coupling bias" was a Green tensor index convention mismatch (transpose bug) in the postprocess, fixed 2026-07-19 — see [`docs/deferred.md`](docs/deferred.md) §6. Interpolated query points degrade accuracy due to trilinear interpolation of off-diagonal components.

| Solver variant | Multi-rank | DOF numbering | Status |
|---------------|------------|---------------|--------|
| CPU + MPI | ✅ (16 ranks) | Global (ibool) | ✅ Verified — diagonals 1.01-1.03× ref |
| CUDA single | N/A | Global (ibool) | ✅ Verified — rel_l2=0.644 matches CPU 16-rank. Global node ID merge in read_partition_all (93e8c32) |

## Cross-Cutting Conventions

- **Naming**: X2Y for topology relations, 1-based with signed direction
- **Header extension**: `.hpp` for C++ headers (no `.hh` or `.h`). CUDA-specific headers use `.cuh`.
- **Config**: Python importable scripts (no YAML/TOML)
- **Data model**: `model.h5` = mesh-dependent precomputed data; `config.h5` = simulation params
- **SI-unit suffixes** on config fields (`_m`, `_s`, `_m_s`, `_kg_m3`)
- **Full names for scientific/physical variables**: Use descriptive full words, not single-letter or abbreviated names. Examples: `displacement`, `velocity`, `acceleration`, `strain`, `solver_dt`, `snapshot_stride`, `vertex_ids`, `green_tile_size_m`, `tilex_elements`. No `u`, `v`, `a`, `dt`, `ss`, `vid`, `tile_sz` — even in local scope. Single letters allowed only for pure math indices (`i`, `j`, `k`) in tight loops
- **Timestep split**: `solver_dt` (auto from CFL) + `output_dt_s` (user snapshot interval), `snapshot_stride = output_dt_s / solver_dt`
- **No receivers**: Postprocess uses shallow mesh-vertex strain records, not receiver locations. No receivers.csv, receiver search, or interpolation. See `docs/design-decisions.md`.
- **config.py is source of truth**: All simulation parameters live in `config.py`. Scripts read it at runtime; no duplicated constants. See `examples/halfspace/config.py`.
- **Fixed filenames**: Examples use fixed output names (`model.h5`, `config.h5`, `partition_{r}.h5`, `record_{r}.h5`, `restart_{r}.h5`). No CLI overrides, except input paths.
- **Console scripts in root pyproject**: Tool entry points live in root `pyproject.toml` `[project.scripts]` and install via `gf-calculation`, not `gf-preprocess`.
- **Pipeline scripts read config**: `compare.sh` derives `N_RANKS` and runtime params from `config.py`, not hardcoded values.
- **Run formatter before stage/commit**: Run `bash format.sh` before `git add` or `git commit`. It formats Python, Markdown, C/C++, CUDA, and CMake files.

## External Reference Codes

`external_reference_codes/` has SPECFEM3D Cartesian and Globe implementations
(read-only, untracked by git) — study SEM patterns only.
