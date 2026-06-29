# gf-calculation

3D elastic SEM wave-equation solver + strain Green's function extraction.

Python pre/post-processing, C++17 MPI compute, HDF5 I/O, METIS partitioning.

## Pipeline

```
config.py + mesh.h5 → preprocess → partition_{r}.h5 + config.h5
                                   → forward solver (x3 directions)
                                   → wavefields/{x,y,z}/record_{r}.h5
                                   → postprocess → greenfun/tile*.h5
```

## Quick Start

```bash
# Environment
uv sync --group dev
source env_setup.sh

# Build (CPU — default)
cmake -S . -B build -DGF_DEVICE_BACKEND=CPU && cmake --build build

# CUDA backend (requires CUDA toolkit, e.g. via Spack)
# source $HOME/.spack/share/spack/setup-env.sh
# spack load cuda
# cmake -S . -B build -DGF_DEVICE_BACKEND=CUDA && cmake --build build

# Run example
bash examples/halfspace/run.sh
```

## CUDA Backend

The element residual computation (K·u, the throughput bottleneck) supports
a GPU backend via `compute_element_residual<Backend>` template dispatch.

- **CPU** (default, `-DGF_DEVICE_BACKEND=CPU`): serial loop over elements
- **CUDA** (`-DGF_DEVICE_BACKEND=CUDA`): one GPU block per element, one thread per GLL node
- **HIP/SYCL**: deferred — same pattern

Build with `-DGF_DEVICE_BACKEND=CUDA`. See `docs/superpowers/design/gpu.md` for details.

## Modules

| Module | Language | Purpose |
|--------|----------|---------|
| `preprocess/` | Python + C++17 | GLL geometry, material, PML, partition, config |
| `forward/` | C++17 | Elastic CG-SEM solver (libgf) + MPI executable |
| `postprocess/` | Python | Strain Green's function extraction |
| `compress/` | C++17 | Header-only HDF5 compression utilities |
| `tools/` | Python | GMSH→HDF5 converter, VTK visualization with GLL sub-cell topology |

## Key Commands

### Build

```bash
# CPU
cmake -S . -B build -DGF_DEVICE_BACKEND=CPU && cmake --build build

# CUDA (after: spack load cuda)
cmake -S . -B build -DGF_DEVICE_BACKEND=CUDA && cmake --build build
```

### Run

```bash
# Preprocess
python -m preprocess                          # reads config.py + mesh.h5 from CWD

# Forward (3 directions)
mpirun -n $N_RANKS gf_solver --direction x
mpirun -n $N_RANKS gf_solver --direction y
mpirun -n $N_RANKS gf_solver --direction z

# Post-process
gf-postprocess mesh.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/

# C++ accelerator thread count
export OMP_NUM_THREADS=8   # default: all available CPUs
```

## Configuration

`config.py` (Python script, no YAML/TOML). See `examples/halfspace/config.py`.

Key fields: `polynomial_order`, `output_dt_s`, `total_duration_s`, `cfl_safety`,
`n_ranks`, `pml_thickness`, `source_x_m`/`source_y_m`, `record_depth_max_m`,
`green_tile_size_m`, `stf_func(t_s)`, `vp_m_s`/`vs_m_s`/`density_kg_m3` callables.

## Design Highlights

- **No receivers** — shallow mesh-vertex recording, no CSV/search/interpolation
- **Timestep split** — `solver_dt` (CFL) + `output_dt_s` (snapshot interval)
- **Source direction** not in config — CLI `--direction {x,y,z}` per run
- **Elastic only** — SLS attenuation deferred
- **Full variable names** — `solver_dt`, `snapshot_stride`, `vertex_ids` (no abbreviations)
- **VTK output with GLL sub-cells** — mesh hexahedra supplemented with GLL-derived edge, face, and sub-volume cells for proper ParaView interpolation; cell data broadcast from parent hex to child GLL cells

## Testing

```bash
python -m pytest tests -q                          # Python (96)
ctest --test-dir build --output-on-failure          # C++ (48)
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
