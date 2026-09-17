# Examples

End-to-end computation examples for gf-calculation.

Each example is self-contained and demonstrates the computational pipeline:

```
mesh generation → preprocess (GLL + material + PML + SLS attenuation) →
forward solver → postprocess (Green's ftn extraction)
```

The preprocessor auto-injects the config's viscoelastic (SLS) parameters
(`q_mu`, `q_kappa`, `n_sls`) into `model.h5`. The solver to run is chosen by
the user inside each example's `forward.sh` (commented-out, switchable).

Green's function extraction uses configured shallow mesh vertices. No receivers.

## Half-Space

Homogeneous elastic half-space with a buried point force at 278 m depth.

| File | Purpose |
|------|---------|
| `halfspace/config.py` | Simulation configuration (Python config script) |
| `halfspace/mesh_gen.py` | Regular hex mesh generator (standalone) |
| `halfspace/compare.sh` | **Full validation pipeline** — SEM → analytic Lamb reference → comparison |
| `halfspace/reference.py` | Analytic Lamb (Johnson 1974) reference waveform (self-contained, no PYTHONPATH needed) |
| `halfspace/compare.py` | Compare reference vs SEM GreenFunctionLibrary result |
| `halfspace/setenv.sh` | Environment init (Spack MPI/Eigen/HDF5) |
| `halfspace/mesh.sh` | Stage 1: mesh generation |
| `halfspace/preprocess.sh` | Stage 2: GLL geometry, materials, PML, partition |
| `halfspace/forward.sh` | Stage 3: forward solver (3 force directions) — **solver switchable inside** (uncomment one) |
| `halfspace/postprocess.sh` | Stage 4: Green's function tile extraction |

**Quick start — full validation:**

```bash
# End-to-end: SEM pipeline → Lamb analytic reference → comparison plot
bash examples/halfspace/compare.sh

# Or step by step:
source examples/halfspace/setenv.sh
bash examples/halfspace/preprocess.sh   # Stage 1+2: mesh + preprocess
bash examples/halfspace/forward.sh      # Stage 3: forward solver
bash examples/halfspace/postprocess.sh  # Stage 4: Green's function extraction

# Generate analytic reference + compare (no PYTHONPATH needed)
# --source = displacement observation point; --receiver = point matching SEM source
python examples/halfspace/reference.py examples/halfspace/greenfun \
  --source 5556 5556 0 --receiver 5278 5278 278 --source-depth-m 278.0 \
  --output /tmp/lamb_ref.npz

python examples/halfspace/compare.py examples/halfspace/greenfun \
  --source 5556 5556 0 --receiver 5278 5278 278 \
  --reference /tmp/lamb_ref.npz --output /tmp/lamb_cmp.npz --fit-scale
```

**What it does:**

1. Generates a 18×18×9 regular hex mesh (2916 elements, 10km×10km×5km)
1. Runs preprocessor: GLL geometry, constant material (Vp=5000, Vs=3000, ρ=2700), PML boundaries, 16-rank METIS partition
1. Runs MPI forward solver in 3 directions (x, y, z) — 16 ranks, ~2-3 s/direction (CPU)
1. Extracts Green's functions into spatial tiles
1. Generates analytic Lamb (Johnson 1974) reference waveform
1. Compares SEM result with analytic reference (relative L² error, best-fit amplitude scaling)

**Output layout:**

```
examples/halfspace/
├── model.h5                  # Extended mesh (topology + GLL + materials + PML)
├── config.h5                 # Simulation parameters + STF
├── partitions/
│   └── partition_{r}.h5      # Per-rank local elements + exchange patterns
├── wavefields/
│   ├── x/record_*.h5         # shallow mesh-vertex strain, force x
│   ├── y/record_*.h5         # shallow mesh-vertex strain, force y
│   └── z/record_*.h5         # shallow mesh-vertex strain, force z
├── greenfun/
│   └── tile_x*_y*.h5         # horizontal Green tiles
├── lamb_reference.npz        # Analytic reference (500, 3, 3)
├── lamb_comparison.npz       # Comparison with SEM
└── vtk/                      # Visualization outputs
```

**Build requirements:**

```bash
# Python dependencies
uv sync --group dev

# C++ forward solver — all targets auto-detected:
cmake -B build && cmake --build build
# All binaries go to bin/

# MPI environment (if using Spack)
source env_setup.sh
```

## Layered Half-Space

Two-layer model (soft 500 m layer over stiff half-space) with PyFK reference.

| File | Purpose |
|------|---------|
| `layer/config.py` | SEM + PyFK configuration (depth-dependent vp/vs/rho) |
| `layer/mesh_gen.py` | Regular hex mesh generator (standalone) |
| `layer/reference.py` | PyFK layered reference Green tensor (needs Python 3.9 venv) |
| `layer/compare.py` | Compare reference vs SEM GreenFunctionLibrary result |
| `layer/compare.sh` | **Full validation pipeline** — SEM → PyFK reference → comparison |

**Quick start:**

```bash
# End-to-end: SEM pipeline → PyFK layered reference → comparison
bash examples/layer/compare.sh

# Or manually (after SEM pipeline has run):
# --source = displacement observation point; --receiver = point matching SEM source
examples/layer/.venv/bin/python examples/layer/reference.py examples/layer/greenfun \
  --source 5500 5000 0 --receiver 5000 5000 250 --output /tmp/layer_ref.npz

python examples/layer/compare.py examples/layer/greenfun \
  --source 5500 5000 0 --receiver 5000 5000 250 \
  --reference /tmp/layer_ref.npz --output /tmp/layer_cmp.npz --fit-scale
```

**Model:**

| Layer | Thickness (km) | Vs (km/s) | Vp (km/s) | ρ (g/cm³) |
|-------|---------------|-----------|-----------|-----------|
| 1 | 0.5 | 1.5 | 2.5 | 2.2 |
| 2 (∞) | 0.0 | 3.0 | 5.0 | 2.7 |

Material functions (`vp_m_s`, `vs_m_s`, `density_kg_m3`) are depth-dependent
piecewise functions compatible with the SEM preprocessor.

**PyFK environment setup:**

```bash
cd examples/layer
uv venv .venv --python 3.9
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install pyfk obspy h5py
```

## Canonical Examples & Solver Selection

Three canonical, self-contained examples replace the earlier
`{model}-{physics}-{runtime}-{language}` test-case matrix (GPU/CPU and solver
are no longer encoded in directory names):

| Case | Purpose |
|------|---------|
| `halfspace/` | Homogeneous half-space — vs analytic Lamb (Johnson 1974) reference |
| `layer/` | Two-layer half-space — vs PyFK reference |
| `fullspace-cubic/` | Full-space (all-PML) — **solver / backend (GPU vs CPU) comparison** |

All configs use **viscoelastic (SLS) parameters**: `q_mu`, `q_kappa`, `n_sls`
(and `f0_for_pml_hz`). The preprocessor auto-injects them into `model.h5`
(`field/cell/tau_*`), so the solver exercises the SLS code path. With the
default Q→∞ (elastic limit) visco output is bit-identical to elastic; set
`q_mu`/`q_kappa` to a finite value (e.g. `100.0`) for real attenuation.

**Solver selection** is done by the user, not by test-case naming. Edit
`examples/<case>/forward.sh` and uncomment ONE solver entry — options cover:

| Option | Solver | Backend |
|--------|--------|---------|
| (A) `gf_solver_viscoelastic_mpi` | viscoelastic (SLS) | CPU + MPI (default) |
| (B) `gf_solver_viscoelastic_cuda` | viscoelastic (SLS) | CUDA single-GPU |
| (C) `gf_solver_viscoelastic_mpi_cuda` | viscoelastic (SLS) | CUDA + MPI |
| (D) `gf_solver_elastic_mpi` | elastic | CPU + MPI |
| (E) `gf_solver_elastic_cuda` | elastic | CUDA single-GPU |
| (F) `gf_solver_elastic_mpi_cuda` | elastic | CUDA + MPI |
| (G) custom ranks | any `*_mpi` | override `config.py:n_ranks` |
| (H) custom path | any | build variant |

The chosen solver runs all 3 force directions. MPI rank count comes from
`config.py:n_ranks` by default.

**Usage:**

```bash
source scripts/env.sh && scripts/build.sh --backend cpu
bash examples/halfspace/compare.sh                # single case

bash scripts/run_all_examples.sh --dry-run        # list the 3 cases
bash scripts/run_all_examples.sh --case halfspace # run one case
bash scripts/run_all_examples.sh                  # run all cases
```

`fullspace-cubic` is the GPU vs CPU comparison case: its `compare.sh` runs the
CUDA solver (skips gracefully with "SKIP: no GPU available" when no GPU is
present), and `compare_solvers.py` aligns the CUDA and MPI-CPU strain fields
(rel_l2, Pearson correlation). See `fullspace-cubic/VERIFICATION.md`.

## Adding a New Example

1. Create `examples/<name>/` with:
   - `config.py` — Python config (see `preprocess/config_loader.py` for schema)
   - `mesh_gen.py` — Mesh generator
   - `reference.py` — Analytic/numerical reference (if applicable)
   - `compare.sh` — Orchestration script
1. Follow the halfspace example as a template
1. Update this README
