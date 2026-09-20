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
scripts/build.sh --backend cpu         # CPU only
scripts/build.sh --backend cuda        # CPU + CUDA
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

All previously tracked bugs (C-PML divergence, postprocess velocity/acceleration zeros,
postprocess mass-weighting, shared vector-field averaging count) are fixed and verified. See
[`docs/bugs.md`](docs/bugs.md) (archived) and
[`docs/design/known-limitations.md`](docs/design/known-limitations.md) (remaining shape error).

Elastic + viscoelastic (SLS) forward solvers complete and verified. SLS elastic-limit
regression (Q→∞) confirmed: max_rel_l2=0.0 (bit-identical to elastic). Examples are
unified on viscoelastic (SLS) config — the solver is chosen by the user via commented-out
entries in `examples/*/forward.sh` (or interactively via `scripts/solver.sh`); no more
test-case naming for GPU/CPU or solver.

Finite-Q SLS was reverified 2026-09-20 against the SPECFEM constitutive formulation:
Qμ=20 shear, Qκ=10 volumetric, and timestep-refinement tests pass. A 16×10×10,
2-rank CPU versus single-GPU CUDA propagation check found the finite-Q/elastic response-norm
ratio agrees within 6.18e-4. Late-time direct CPU–CUDA field differences are not specific to
attenuation: the elastic control differs by 9.76%, versus 8.16% for finite Q.

CG-SEM global-DOF assembly fix complete — waves correctly propagate across element
interfaces (both within-rank and cross-rank). All 231 Python tests are collected (230 pass and
the opt-in C++ preprocess smoke test skips by default). The current CTest layout registers 63
entries and requires an MPI-enabled build configuration.

Buried source support implemented (`source_z_m = None`→free surface, `float`→buried). Preprocessor auto-detects surface vs buried mode and excludes PML elements for buried sources.

**Full example validation suite** (`scripts/run_all_examples.sh`) runs the 3 canonical
examples end-to-end: `halfspace` (vs Lamb), `layer` (vs PyFK), `fullspace-cubic`
(solver/backend comparison; auto-skips without a GPU). All three selected to the CUDA
solver; last run (2026-08-05, RTX 5060 Ti): 3 PASSED via `elastic_cuda` — halfspace
scale=2.91/rel_l2=0.185, layer scale=2.60, fullspace Stokes corr=0.781 (non-fatal WARNING < 0.80). `viscoelastic_cuda` is now fixed and verified (2026-08-07) — see status table;

**Historical results below (amplitude and fitted-L2 interpretation superseded):**

Fullspace re-verified and refined 2026-08-07: mesh 18³ (1 km) → **24³ (750 m)** — recorded nodes 57.7 k → ~116 k, elements/λs 3.0 → 4.0, overall Stokes corr **0.781 → 0.843** (mean_l2 0.706; best-fit SEM/ana scale 0.34 ≈ 2.9×; shape L2 0.131) — NOTE the 0.79-era numbers were deflated by a stale hardcoded interior mask: analytical_compare.py used an 18³-era [3000,15000] m box that included ~2.25 km of real PML on each side at 24³/PML-7 (and on 18³ too). Fixed: interior box is now DERIVED from config.h5 pml\_\* attrs × true element size from model.h5 cell coords (not tile coords, which are cropped to the recording region); guard rejects whole-domain PML. Same-day correction: 0.7916 → 0.8430, far bin \[2,4)λs 0.496→0.613 (n=3). Residual gap to ~0.99: PML reflections (not window-isolatable in this box) + ~2.9× amplitude (uniform in distance; source injection verified: partition-of-unity Lagrange weights, same stf_values both sides). Mesh: PML 5→7 elements, source (9500)→(9375)³, tiles [2,2,3,3]; `compare.sh` runs `gf_postprocess_mpi` (peak 20.6 GB @18³ / 34.3 GB @24³, cap 56) + consistency guard (Stage 2.5) + hermetic cleanup; a conflicting `--source` is now a hard error. See `examples/fullspace-cubic/VERIFICATION.md`.
Grid-convergence study (2026-08-08, 18³/20³/24³, common receiver box [5500,12500] m via `--interior-box`): shape L2 (scale-invariant) converges monotonically 0.1225→0.1207→0.1061 (3.0→4.0 elem/λs, ~13% gain); raw mean_corr is noisy/non-monotonic (0.847/0.849/0.828 — PML-tail flooring + 50-receiver sampling noise drown the gain); best-fit scale is FLAT 0.340/0.340/0.336 → the ~2.9× amplitude factor does NOT converge with h (effective-source/convention offset, not a mesh error). Solver ~64/86/148 s/dir, postprocess peak 16.2/17.4/34.3 GB. Study finalized 2026-08-10: five fixed-receiver grids (18/20/22/24/28) via `examples/meshsize/gen_grids.py`, uncompressed HDF5, postprocess ≤64 GB budget (per-grid MPI ranks 12/12/4/4/3) — shape L2 flat 0.123-0.127 across 3.0-4.7 elem/λs (residual is PML/near-field, NOT resolution); ~3× scale flat 0.34; 28³ is the agreed ceiling. cgroup counts page cache: RSS sampling understates peak (22³@6ranks OOM at 64 GB) — a 16-tile guard now gates the comparison.

**Amplitude correction (2026-09-17):** the historical ~0.34 SEM/analytical scale was
caused by postprocess sharing one `node_count` across displacement, velocity, and
acceleration, dividing each by exactly three. Serial and MPI paths now use independent
per-field counts; corrected historical full-space scales are 1.014–1.038. Correlation is
unchanged. The historical fitted-L2 values are invalid because their denominator was not
scale-invariant; the formula and amplitude gate [0.8, 1.2] are now fixed. A fresh 20³ CUDA/MPI
run passed: default interior sample mean_corr=0.8573, raw rel_l2=0.4496, scale=1.019,
fitted rel_l2=0.3454; 64 fixed receivers mean_corr=0.8476, raw rel_l2=0.4629, scale=1.038,
fitted rel_l2=0.3458.

After fixing the postprocess mass-weighting and per-field count bugs and the Green tensor
index convention mismatch, scaled waveform correlation is 0.991 (halfspace) / 0.745
(layer). The former scales 2.95/2.60 become 0.983/0.867 after the exact ×3 correction.

| Solver variant | Multi-rank | DOF numbering | Status |
|---------------|------------|---------------|--------|
| CPU + MPI (elastic) | ✅ (16 ranks) | Global (ibool) | ✅ Verified — diagonals 1.01-1.03× ref |
| CPU + MPI (viscoelastic) | ✅ (16 ranks) | Global (ibool) | ✅ Verified — elastic limit rel_l2=0.0 |
| CUDA single (elastic) | N/A | Global (ibool) | ✅ Verified — rel_l2=5.1e-4 vs CPU 16-rank (fullspace, steps 400/700) |
| CUDA single (viscoelastic) | N/A | Global (ibool) | ✅ Verified (2026-08-07) — FIXED kernel shadowing bug: `element_cuda.cu` non-PML branch redeclared `double sigma[3][3]` (shadowing the outer array; Step A filled the inner copy that was discarded, scatter read uninitialized stack garbage → Q→∞ divergence, scaled rel_l2≈1.0/scale≈0). Removed the duplicate declaration (1-line fix). halfspace scale=2.91/rel_l2=0.185 vs Lamb (identical to elastic_cuda); GPU elastic-vs-visco bitwise-equal at step 500 (both halfspace & layer) |

**CUDA vs MPI-CPU fullspace consistency (2026-08-05):** verified on
`examples/fullspace-cubic` (direction=x, 800 steps, all-face
PML): strain rel_l2=1.7e-4 (step 400) / 5.1e-4 (step 700), Pearson corr

> 0.9999998 — PASS. Three real bugs fixed along the way:

1. `preprocess/partition.py`: exchange DOF patterns built from face-adjacent
   pairs only — nodes shared by 3+ ranks (edges/corners) were never
   exchanged. Rebuilt from node ownership (all co-owner rank pairs, sorted by
   global node id).
1. PML damping assembly: each rank assigned `last-local-cell-wins` damping at
   shared nodes, so ranks damped their own copies differently and diverged.
   Now "highest global cell id wins" via a packed `cell_id + damping/2`
   `exchange_halo_max` reduction (`forward/share/src/exchange.{cpp,hpp}`,
   `exchange_noop.cpp`) — identical to the single-rank/CUDA rule.
1. `compute_full_strain` (solver.cpp) double-offset bug: element base pointer
   included `+ node_idx*3` while stencil reads added their own offsets —
   wrong-node reads everywhere plus out-of-bounds heap reads at each rank's
   last element (the apparent exponential "strain explosion"; velocity/
   displacement were correct all along). Fixed by removing the offset.
   FIXED (2026-08-05): `gf_preprocess run` now accepts `--n-ranks N`
   (consumed in `run_main`, not forwarded to stage1) and `cli.py` always passes
   `config.py:n_ranks` — so `config.py` is the single source of truth and a
   1-rank (or any) control run needs no manual partition rebuild. The hardcoded
   `n_ranks=16` in `preprocess/cpp/config_user_fullspace.cpp` is now only the
   bare-invocation default. Comparison tool:
   `examples/fullspace-cubic/compare_solvers.py` (coordinate-
   aligned, rerunnable).
   **Postprocess MPI tile-parallel** (`gf_postprocess_mpi`): OOM fixed (memory
   redesigned from full-replication ~331 GB/16 ranks to tile-local ~17 GB).
   VERIFIED (2026-08-05, halfspace, 9 tiles): tiles round-robin across ranks
   (each written exactly once, any n_ranks); `mpirun -n 4` output numerically
   identical (bitwise) to serial `gf_postprocess` across all 9 tiles. See
   [`docs/design/postprocess-tile-parallel.md`](docs/design/postprocess-tile-parallel.md)
   and [`docs/deferred.md`](docs/deferred.md) §7.

**Code cleanup (2026-07-28):** removed residual debug code from
`preprocess/cpp/source_locator.cpp` (-31 lines), duplicate `#include` in
`postprocess/cpp/writer.hpp`, and orphan debug comments in `postprocess/cpp/main.cpp`
and `main_mpi.cpp`. `main.cpp`/`main_mpi.cpp` still share ~950 duplicated lines —
refactor deferred until multi-rank MPI is verified.

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
- **HDF5 写入不压缩 (强制)**: All `.h5` writes must explicitly disable compression. Python: every `create_dataset`/`require_dataset` call must pass `compression=None` explicitly (no gzip/shuffle/deflate, no omission). C++: dataset creation uses `H5P_DEFAULT` or chunking only (`H5Pset_chunk`); `H5Pset_deflate`/`H5Pset_gzip` etc. are forbidden. All compression was removed 2026-08-09 (see `docs/design-decisions.md` §10); re-enabling requires a design review and a coordinated update of this rule.
- **Run formatter before stage/commit**: Run `bash format.sh` before `git add` or `git commit`. It formats Python, Markdown, C/C++, CUDA, and CMake files.

## External Reference Codes

`external_reference_codes/` has SPECFEM3D Cartesian and Globe implementations
(read-only, untracked by git) — study SEM patterns only.
