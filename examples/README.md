# Examples

End-to-end computation examples for gf-calculation.

Each example is self-contained and demonstrates the computational pipeline:

```
mesh generation → preprocess → forward solver
```

Green's function extraction from snapshots operates on GLL nodes directly.
No receiver positions needed.

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

**Build requirements:**

```bash
# Python dependencies
uv sync --group dev

# C++ forward solver
cd build && cmake .. && make gf_solver

# MPI environment (if using Spack)
source env_setup.sh
```

**What it does:**

1. Generates a 10×10×5 regular hex mesh (500 elements, 10km×10km×5km)
2. Runs preprocessor: GLL geometry, constant material (Vp=5000, Vs=3000, ρ=2700), PML boundaries, 2-rank METIS partition
3. Runs forward solver in 3 directions (x, y, z) with MPI

**Output layout:**

```
examples/halfspace/output/
├── mesh.h5                  # Extended mesh (topology + GLL + materials + PML)
├── mesh_auxiliary.h5        # Auxiliary CSR adjacency relations
├── configs/
│   └── config.h5            # Simulation parameters + STF
├── partitions/
│   ├── partition_0.h5       # Rank 0: local elements + exchange patterns
│   └── partition_1.h5       # Rank 1
├── wavefields/
│   ├── x/record_*.h5        # Strain snapshots (force in x)
│   ├── y/record_*.h5        # Strain snapshots (force in y)
│   └── z/record_*.h5        # Strain snapshots (force in z)
```

## Adding a New Example

1. Create `examples/<name>/` with:
   - `config.py` — Python config (see `preprocess/config_loader.py` for schema)
   - `run.sh` — Pipeline script
2. Follow the halfspace example as a template
3. Update this README