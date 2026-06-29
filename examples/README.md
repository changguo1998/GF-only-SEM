# Examples

End-to-end computation examples for gf-calculation.

Each example is self-contained and demonstrates the computational pipeline:

```
mesh generation → preprocess → forward solver
```

Green's function extraction uses configured shallow mesh vertices. No receivers.

## Half-Space

Homogeneous elastic half-space with a point force at the surface.

| File | Purpose |
|------|---------|
| `halfspace/config.py` | Simulation configuration (Python config script) |
| `halfspace/mesh_gen.py` | Regular hex mesh generator (standalone) |
| `halfspace/run.sh` | End-to-end pipeline script |
**Quick start:**

```bash
cd /path/to/gf-calculation
bash examples/halfspace/run.sh
```

**Visualization (after preprocess):**

```bash
cd examples/halfspace/
source setenv.sh          # set PATH to include tool entry points
mesh2vtk                  # → vtk/mesh.vtk (material fields + GLL points)
partition2vtk             # → vtk/partition_{r}.vtk (per-rank view)
```

**Visualization (after forward solver):**

```bash
wavefield2vtk             # → vtk/wavefield_{step}.vtk (cell-corner strain)
wavefield2vtk_detail      # → vtk/wavefield_{step}.vtk (full GLL point strain)
```

**Build requirements:**

```bash
# Python dependencies
uv sync --group dev

# C++ forward solver — all targets auto-detected:
cmake -B build && cmake --build build
# Or build individually:
#   cmake --build build --target gf_solver_mpi        # MPI + CPU
#   cmake --build build --target gf_solver_cuda        # CUDA single-GPU
#   cmake --build build --target gf_solver_mpi_cuda    # MPI + CUDA
# All binaries go to bin/

# MPI environment (if using Spack)
source env_setup.sh
```

**What it does:**

1. Generates a 10×10×5 regular hex mesh (500 elements, 10km×10km×5km)
1. Runs preprocessor: GLL geometry, constant material (Vp=5000, Vs=3000, ρ=2700), PML boundaries, 2-rank METIS partition
1. Runs forward solver in 3 directions (x, y, z) with MPI

**Output layout:**

```
examples/halfspace/
├── mesh.h5                  # Extended mesh (topology + GLL + materials + PML)
├── config.h5                # Simulation parameters + STF
├── partitions/
│   ├── partition_0.h5       # Rank 0: local elements + exchange patterns
│   └── partition_1.h5       # Rank 1
├── wavefields/
│   ├── x/record_*.h5        # shallow mesh-vertex strain, force x
│   ├── y/record_*.h5        # shallow mesh-vertex strain, force y
│   └── z/record_*.h5        # shallow mesh-vertex strain, force z
├── restart/
│   ├── x/restart_*.h5       # latest full-volume restart
│   ├── y/restart_*.h5
│   └── z/restart_*.h5
└── greenfun/
    └── tile_x*_y*.h5        # horizontal Green tiles
```

## Adding a New Example

1. Create `examples/<name>/` with:
   - `config.py` — Python config (see `preprocess/config_loader.py` for schema)
   - `run.sh` — Pipeline script
1. Follow the halfspace example as a template
1. Update this README
