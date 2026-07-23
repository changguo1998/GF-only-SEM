# gf-calculation

3D elastic SEM wave-equation solver + strain Green's function extraction.

Python pre-processing, C++17 postprocess + MPI compute, HDF5 I/O, METIS partitioning.

## Pipeline

```
config.py + model.h5 → preprocess → partition_{r}.h5 + config.h5
                                   → forward solver (x3 directions)
                                   → wavefields/{x,y,z}/record_{r}.h5
                                   → postprocess → greenfun/tile*.h5
```

## Quick Start

````bash
# Environment
# Install dependencies
uv sync --group dev

# Set up environment (Spack + venv + PATH)
source scripts/env.sh

# Build all solvers and tools
scripts/build.sh

# Run a solver (interactive menu)
scripts/solver.sh

# Or run directly:
scripts/solver.sh elastic cpu mpi -- --direction x
scripts/solver.sh elastic cuda -- --direction x

## Forward Solvers

Six solver binaries, selected via `scripts/solver.sh` or directly:

| Binary | Physics | Backend | MPI |
|--------|---------|---------|-----|
| `gf_solver_elastic_mpi` | elastic | CPU | yes |
| `gf_solver_elastic_cuda` | elastic | CUDA | no |
| `gf_solver_elastic_mpi_cuda` | elastic | CUDA | yes |
| `gf_solver_viscoelastic_mpi` | viscoelastic | CPU | yes |
| `gf_solver_viscoelastic_cuda` | viscoelastic | CUDA | no |
| `gf_solver_viscoelastic_mpi_cuda` | viscoelastic | CUDA | yes |

- MPI and CUDA are auto-detected by CMake. Missing dependencies skip their targets.
- GPU auto-binds via `cudaSetDevice(rank % n_devices)`.
- When MPI ranks exceed GPUs (e.g. `-n 16` on 1-GPU host), solver warns, reduces to 1 rank per GPU, and redistributes partitions.

## Modules

| Module | Language | Purpose |
|--------|----------|---------|
| `preprocess/` | Python + C++17 | GLL geometry, material, PML, partition, config |
| `forward/` | C++17 | Elastic CG-SEM solver (libgf_elastic) + 3 MPI/CUDA executables |
| `forward/viscoelastic/` | C++17 | Viscoelastic SEM solver (SLS) — skeleton, implementation deferred |
| `postprocess/` | C++17 | Strain Green's function extraction (Python archived) |
| `tools/` | Python + C++17 | GMSH->HDF5 converter (Python); VTK tools (C++ primary, Python archived) |

## Key Commands

### Build

```bash
scripts/build.sh              # all targets (auto-detect CPU/CUDA)
scripts/build.sh cpu          # CPU only
scripts/build.sh cuda         # CPU + CUDA
scripts/build.sh -t gf_postprocess  # single target
````

### Run

```bash
scripts/solver.sh                       # interactive menu
scripts/solver.sh elastic cpu mpi -- --direction x
```

### Run

```bash
# Preprocess
python -m preprocess                          # reads config.py + model.h5 from CWD

# Forward (3 directions) — choose your binary:
mpirun -n $N_RANKS bin/gf_solver_elastic_mpi --direction x
mpirun -n $N_RANKS bin/gf_solver_elastic_mpi --direction y
mpirun -n $N_RANKS bin/gf_solver_elastic_mpi --direction z

# Or single-GPU (no MPI):
bin/gf_solver_elastic_cuda --direction x

# Or multi-GPU via MPI:
mpirun -n $N_RANKS bin/gf_solver_elastic_mpi_cuda --direction x

# Post-process
gf_postprocess model.h5 config.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/

# C++ accelerator thread count
export OMP_NUM_THREADS=8   # default: all available CPUs
```

### Install

```bash
scripts/install.sh                              # → /usr/local, auto-backend
scripts/install.sh /opt/gf-calculation          # custom prefix
scripts/install.sh ~/.local cpu                 # user install, CPU only
```

Binaries go to `$PREFIX/bin/`. Add to PATH and run:

```bash
export PATH="/opt/gf-calculation/bin:$PATH"
solver.sh                                       # interactive solver selector
solver.sh elastic cpu mpi -- --direction x
```

## Configuration

`config.py` (Python script, no YAML/TOML). See `examples/halfspace/config.py`.

Key fields: `polynomial_order`, `output_dt_s`, `total_duration_s`, `cfl_safety`,
`n_ranks`, `pml_thickness`, `source_x_m`/`source_y_m`, `record_depth_max_m`,
`tilex_elements`, `tiley_elements`, `stf_func(t_s)`, `vp_m_s`/`vs_m_s`/`density_kg_m3` callables.

## Design Highlights

- **No receivers** — shallow mesh-vertex recording, no CSV/search/interpolation
- **Timestep split** — `solver_dt` (CFL) + `output_dt_s` (snapshot interval)
- **Source direction** not in config — CLI `--direction {x,y,z}` per run
- **Elastic only** — SLS attenuation deferred
- **Full variable names** — `solver_dt`, `snapshot_stride`, `vertex_ids` (no abbreviations)
- **VTK output with GLL sub-cells** — mesh hexahedra supplemented with GLL-derived edge, face, and sub-volume cells for proper ParaView interpolation; cell data broadcast from parent hex to child GLL cells

## Testing

```bash
python -m pytest tests -q                          # Python (104)
ctest --test-dir build --output-on-failure          # C++ (Catch2)
bash examples/halfspace/run.sh                      # Full pipeline
```

## Documentation

| Document | Contents |
|----------|----------|
| `docs/design-decisions.md` | Architecture, schemas, rationale |
| `docs/math.md` | Full mathematical formulation |
| `preprocess/AGENTS.md` | Preprocess module |
| `forward/AGENTS.md` | Forward solver (CPU+GPU backend) |
| `postprocess/AGENTS.md` | Post-process module |
