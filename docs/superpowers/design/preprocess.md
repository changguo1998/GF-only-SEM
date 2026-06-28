# Preprocess Module — Technical Design

> Parent: [docs/design-decisions.md](../../design-decisions.md)
> Implementation plan: [docs/superpowers/plans/2026-06-08-preprocess.md](../plans/2026-06-08-preprocess.md)

## Goal

Python module. Reads mesh topology and `config.py`. Computes derived model data and the shallow recording map. Writes `mesh.h5`, `partition_{r}.h5`, and `config.h5`.

## Data Flow

```
mesh.h5 (/topology/) ─────────┐
config.py (script, importable) ─┤
                                ↓
                          preprocessor (Python [+ optional C++ accelerator])
                          ├── import config.py
                          ├── read mesh.h5 topology
                          ├── [C++ accelerator: GLL node coords, J, dξ/dx, mass, CFL h_min, PML ramp]
                          ├── compute GLL node coords per element (Python fallback)
                          ├── compute geometric quantities (J, dξ/dx) (Python fallback)
                          ├── call config.py material functions → GLL nodes
                          ├── compute lumped mass
                          ├── auto-detect boundary tags (surface level), set is_pml flags
                          ├── compute C-PML damping profiles, stretched-coordinate functions,
                          │   convolution coefficients (layer-based, face/edge/corner classification)
                          ├── compute cfl_dt = cfl_safety × h_min / vp_max (minimum GLL node spacing)
                          ├── derive solver_dt + snapshot_stride from output_dt_s and CFL
                          ├── derive nsteps = ceil(total_duration_s / solver_dt)
                          ├── pre-flight validation (mesh, material, CFL-derived stride, boundary, source, STF, storage)
                          ├── partition (METIS) + GLL node global numbering + exchange patterns
                          ├── evaluate STF over time range
                          ├── locate source elements (free surface only), compute Lagrange interpolation weights
                          ├── write GLL geometry + is_pml BACK to mesh.h5 (extends it)
                          ↓
                     mesh.h5 (extended) + partition_{r}.h5 + config.h5
                          │
                          ↓
                     forward solver (C++)
```

Optional output: `mesh_auxiliary.h5` (CSR adjacency for validation/acceleration).

## Architecture

Single Python CLI. Reads `mesh.h5` and `config.py` from CWD. No CLI args. No YAML/TOML. Loads config with `importlib`.

Outputs: extend `mesh.h5` and write one `partition_{r}.h5` per rank. No monolithic `model.h5`.

```
preprocess/
├── __init__.py
├── cli.py              — CLI entry point
├── accelerator.py      — optional C++ subprocess runner (GLL geom, CFL, PML)
├── config_loader.py     — importlib load config.py, validate
├── topology_reader.py   — read mesh.h5 /topology/
├── gll_geometry.py      — compute GLL node coords, jacobian, dξ/dx per element (Python fallback)
├── material.py          — evaluate config vp_m_s(x_m,y_m,z_m), vs_m_s(x_m,y_m,z_m), density_kg_m3(x_m,y_m,z_m) at GLL nodes
├── mass.py              — compute lumped mass (requires ρ from material step)
├── boundary_detector.py — auto boundary tagging (surface level), set is_pml flags
├── cpml.py              — C-PML: element type classification (face/edge/corner), damping profiles,
│                          stretched-coordinate functions (K, α), convolution coefficients
├── partition.py          — METIS partitioning + GLL node global numbering + exchange pattern precomputation
├── stf_evaluator.py     — evaluate stf_func() → time series array
├── source_locator.py    — locate source elements, compute natural coords + Lagrange interpolation weights
├── cfl_validator.py      — compute cfl_dt, derive solver_dt and snapshot_stride
├── preflight.py          — comprehensive pre-flight validation
├── partition_writer.py   — write partition_{r}.h5
└── config_writer.py     — write single config.h5 (rank-invariant, no direction)
├── cpp/
│   ├── CMakeLists.txt   — build target
│   └── main.cpp         — GLL geometry, CFL h_min, PML damping (no MPI)
```

## C++ Accelerator

The heaviest numerical computations (GLL geometry, CFL h_min, PML damping ramps)
can be offloaded to a standalone C++ executable via subprocess:

- **Binary**: `preprocess/cpp/gf_preprocess_cpp` (build manually, no MPI)
- **Dependencies**: HDF5, Eigen3 (same as forward solver)
- **Data flow**: reads mesh.h5 `/topology/`, writes results to `/field/element/`
- **Integration**: `accelerator.py` runs the binary, parses `H_MIN` from stdout,
  reads precomputed arrays from HDF5. Falls back to pure Python if binary absent.
- **CLI signature**:
  ```
  gf_preprocess_cpp <mesh.h5> <N> <cfl_safety> \
      <pml_xmin> <pml_xmax> <pml_ymin> <pml_ymax> <pml_zmin> <pml_zmax>
  ```
- **Build** (example):
  ```sh
  g++ -std=c++17 -O2 -march=native -I<eigen3>/include/eigen3 \
      -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
      -o gf_preprocess_cpp main.cpp -lhdf5 -lm
  ```

This is purely optional — the pipeline runs identically without it.

## Technology

- Python 3.10+
- numpy, h5py, scipy (interpolation), pytest
- METIS — called via ctypes or subprocess (partition step)
- Optional C++17 (HDF5, Eigen3) for heavy loops via subprocess
- No YAML/TOML dependency
- Elastic only — SLS attenuation deferred

## CLI

```
python -m preprocess
```

No arguments — reads `mesh.h5` and `config.py` from the current working directory.

| File | Description |
|------|-------------|
| `mesh.h5` | Converter output in CWD — read topology, write extended geometry + is_pml back |
| `config.py` | User's Python config script in CWD |
Output files are placed alongside the inputs:

- `config.h5` — single rank-invariant config
- `partitions/partition_{r}.h5` — per-rank partition files
- `mesh.h5` is extended in-place with `/field/element/` geometry and `is_pml`

## Config Script (`config.py`)

User writes importable `config.py`. Preprocess imports it, reads variables, and samples the STF over the full time range.

```python
# Example config.py — imported by preprocessor
title = "test_run"
polynomial_order = 5       # N (GLL order)

# Time
output_dt_s = 0.001        # user-specified snapshot interval (s)
total_duration_s = 10.0    # simulation duration (s)
cfl_safety = 0.5           # CFL safety factor (0 < cfl_safety < 1)

# Storage / output
snapshot_precision = "float32"  # "float32" or "float64"
storage_limit_gb = 100           # abort if estimated output exceeds this
record_depth_max_m = 50_000.0    # requested max recorded depth below free surface
restart_dt_s = 60.0              # overwrite latest restart every 60 s
green_tile_size_m = 50_000.0     # horizontal x/y Green-function tile width
strict_validation = True

# Material — callable functions evaluated at each GLL node
def vp_m_s(x_m, y_m, z_m):
    return 3000.0

def vs_m_s(x_m, y_m, z_m):
    return 1500.0

def density_kg_m3(x_m, y_m, z_m):
    return 2500.0

# PML thickness per face (zmin=0 because z_min is free surface)
pml_thickness = {"xmin": 3, "xmax": 3, "ymin": 3, "ymax": 3, "zmin": 0, "zmax": 3}

# Source position (z is auto-placed on free surface at z ≈ z_min)
source_x_m = 500.0
source_y_m = 500.0
# source_z auto-placed on top free surface (z ≈ z_min) by preprocessor

# Source direction is NOT in config.py — preprocessor generates one config.h5.
# Forward solver takes direction via CLI flag --direction {x,y,z}

# Output
n_ranks = 4                   # MPI ranks for partition

def stf_func(t_s):
    """Source time function. t_s in seconds, returns amplitude."""
    import numpy as np
    f0_hz = 5.0
    t0_s = 0.3
    a = np.pi * f0_hz * (t_s - t0_s)
    return (1 - 2 * a**2) * np.exp(-a**2)
```

### Config Validation

| Check | Rule |
|-------|------|
| polynomial_order | ≥ 1, integer |
| output_dt_s | > 0 |
| total_duration_s | > 0 |
| cfl_safety | 0 < cfl_safety < 1 |
| snapshot_precision | "float32" or "float64" |
| restart_dt_s | > 0 |
| record_depth_max_m | ≥ 0 and within domain depth |
| green_tile_size_m | > 0 |
| storage_limit_gb | > 0 |
| source position | source_x_m and source_y_m within xy domain bounds (auto-detected from mesh), z auto-placed on free surface |
| stf_func | callable, signature `(float) -> float`, returns finite non-NaN values for t ∈ [0, nsteps×solver_dt] |
| vp_m_s, vs_m_s, density_kg_m3 | callable, each signature `(float, float, float) -> float`, returns positive values |
| n_ranks | ≥ 1, integer |
| pml_thickness | dict with keys xmin, xmax, ymin, ymax, zmin, zmax; values ≥ 0 integers |
| strict_validation | bool, default True |

### Pre-flight Validation

Before writing output files, the preprocessor runs comprehensive validation:

| Category | Check | Failure Mode |
|----------|-------|-------------|
| Mesh | n_cell > 0, n_vertex > 0 | Empty mesh → abort |
| Mesh | All hex elements have 8 distinct vertices | Degenerate hex → abort |
| Mesh | det(J) > 0 at all GLL nodes | Inverted/tangled element → abort |
| Material | vp > 0, vs ≥ 0, density > 0 at all GLL nodes | Invalid material → abort |
| CFL | cfl_dt = cfl_safety × h_min / vp_max (h_min = minimum GLL node spacing) | — |
| CFL | Find smallest stride where output_dt_s / stride ≤ cfl_dt; set solver_dt and snapshot_stride | No stride ≤ MAX_STRIDE → abort with suggestion |
| Time | nsteps = ceil(total_duration_s / solver_dt); nsteps % snapshot_stride == 0; restart_stride = round(restart_dt_s / solver_dt) ≥ 1 | Invalid derived stride → abort |
| Boundary | Free surface detected at z ≈ z_min | No free surface → abort |
| Boundary | PML has ≥ 2 elements per absorbing face | Too thin PML → warn |
| Source | source_x_m, source_y_m within domain bounds | Outside domain → abort |
| STF | stf_func(t_s) returns finite, non-NaN for t_s ∈ [0, nsteps×solver_dt] | Bad STF → abort |
| Recording | record_depth_actual_m is a horizontal spectral-element face at or deeper than record_depth_max_m; at least one non-PML vertex selected | Invalid recording map → abort |
| Storage | Estimated disk usage ≤ storage_limit_gb | Exceeds limit → abort |
| Partition | All ranks have ≥ 1 element | Empty rank → abort |

## Processing Steps

### 1. Load Topology

Read mesh.h5 `/topology/` datasets into memory. All datasets follow X2Y naming, 1-based indexing, signed direction.

### 2. Compute GLL Node Geometry

For each element with 8 corner vertices and polynomial order N:

1. Compute (N+1)³ GLL node positions via the element's geometric mapping:
   `x(ξ,η,ζ) = Σ l_a(ξ)·l_b(η)·l_c(ζ)·x_abc` where `l_a` are GLL Lagrange basis functions and `x_abc` are corner coordinates.
1. Compute Jacobian: `dx/dξ` (3×3) via differentiation of the mapping.
1. `dξ/dx = (dx/dξ)^(-1)` — inverse Jacobian at each GLL node.
1. `det(J) = det(dx/dξ)` — integration weight factor.

Output: `/field/element/coords`, `/field/element/dxi_dx`, `/field/element/jacobian`.

Note: lumped mass is computed after material interpolation (step 3) because it requires ρ.

### 3. Evaluate Material at GLL Nodes

Call the user-defined functions from config.py at each GLL node position (computed in step 2):

- `vp_m_s(x_m, y_m, z_m)` → compressional wave speed
- `vs_m_s(x_m, y_m, z_m)` → shear wave speed
- `density_kg_m3(x_m, y_m, z_m)` → mass density

Output: `/field/element/vp`, `/field/element/vs`, `/field/element/density`.

### 4. Compute Lumped Mass

Using the density from step 3, compute lumped mass diagonal at each GLL node:
`mass_ijk = w_i·w_j·w_k·ρ_ijk·det(J_ijk)`

Output: `/field/element/mass`.

### 5. CFL Validation

After GLL geometry and material are known, derive the solver timestep from the user-facing snapshot interval:

1. Compute minimum GLL node spacing `h_min` across all elements (minimum Euclidean distance between adjacent GLL nodes in physical space)
1. Compute `vp_max = max(vp)` across all GLL nodes
1. Compute `cfl_dt = cfl_safety × h_min / vp_max`
1. Search `stride = 1..MAX_STRIDE` for the first value where `output_dt_s / stride ≤ cfl_dt`
1. Set `solver_dt = output_dt_s / stride` and `snapshot_stride = stride`
1. Set `restart_stride = round(restart_dt_s / solver_dt)` and validate `restart_stride >= 1`
1. Set `nsteps = ceil(total_duration_s / solver_dt)`
1. Print computed `cfl_dt`, `solver_dt`, `snapshot_stride`, `restart_stride`, and `nsteps` to stdout

Store time fields in `/simulation`. Forward integrates with `solver_dt`, writes strain every `snapshot_stride`, and overwrites restart every `restart_stride`.

### 6. Auto-Detect Boundary Tags

No GMSH physical groups. Boundary tags computed from surface face center geometry:

```
For each surface: check face center position
  z ≈ z_min      → boundary_tag = 1  (free surface)
  on domain bounds → boundary_tag = 2  (absorbing/PML)
  else              → boundary_tag = 0  (interior)
```

Domain bounds auto-detected from `vertex_to_coord`. Output: `/field/surface/boundary_tag`.

### 7. Compute C-PML Parameters (Layer-Based)

C-PML elements are identified by layer-based connectivity, not centroid position.
Classification is independent per direction: for each element, `d_x_active = distance_from_x_boundary < cpml_thickness.xmin OR xmax`.
Count active directions → 1=face, 2=edge, 3=corner.
Each direction's damping uses its own distance (can differ between e.g. x and y for a corner element).

1. Start from boundary faces with `boundary_tag = 2` (absorbing).
1. Walk inward through element connectivity: trace N layers of elements from each absorbing boundary face.
1. For each element, independently check each direction against its respective boundary distance.
1. Classify each C-PML element as **face** (damped in 1 direction), **edge** (damped in 2 directions), or **corner** (damped in 3 directions).
1. Compute directional damping profiles `d_x(ξ)`, `d_y(ξ)`, `d_z(ξ)` per GLL node — each direction's damping depends only on that direction's distance from the boundary.
1. Compute stretched-coordinate functions `K_x(ξ)`, `α_x(ξ)` (and similarly for y, z) per GLL node.
1. Precompute convolution coefficients `α_c`, `β_c`, `ā` from the damping profiles (second-order convolution scheme, per Wang et al. 2006).

Output: `/field/element/cpml/{cpml_type, d_x, d_y, d_z, K_x, K_y, K_z, alpha_x, alpha_y, alpha_z, conv_coef_*}`.

### 8. Pre-Flight Validation

Comprehensive validation before partition and writing. Runs as a checklist; with `strict_validation=True` errors abort the run, with `strict_validation=False` errors are logged as warnings and processing continues.

1. **Material**: `vp > 0`, `vs ≥ 0`, `density > 0` at all GLL nodes; `λ = ρ(vp² - 2vs²) > 0` (elastic stability); warn if `vs ≡ 0` anywhere
1. **Mesh quality**: `det(J) > 0` at all GLL nodes (no inverted elements); warn on extreme aspect ratios
1. **CFL/time**: validate `solver_dt ≤ cfl_dt`, integer strides, and `nsteps % snapshot_stride == 0`; log all derived values
1. **Boundary**: require free and absorbing surfaces; validate `pml_thickness`; warn if absorbing face has \<2 PML elements
1. **Source**: within xy domain bounds; Newton iteration found at least one containing element on free surface; sum of normalized Lagrange weights ≈ 1
1. **STF**: all values finite (no NaN/Inf); warn if non-zero DC component
1. **Partition**: `n_ranks ≤ n_cell` (pre-check before calling METIS)
1. **Recording map**: snap requested depth to `record_depth_actual_m`; mark non-PML elements/vertices above it.
1. **Storage**: estimate partitions + shallow strain + latest restart. Abort if > `storage_limit_gb`:
   - `snapshots_per_run = nsteps / snapshot_stride`
   - `strain_per_run_GB = snapshots_per_run × n_record_vertices × 6 × bytes_per_float / 1e9`
   - `restart_GB = n_cell × NGLL³ × 3 × 3 × 8 / 1e9 + pml_memory_GB`
   - `total_GB = strain_per_run_GB × 3 + restart_GB × 3 + partition_GB`
   - Print storage estimate to stdout

### 9. Partition (METIS) + GLL Node Global Numbering

1. Build dual graph: elements as nodes, shared faces as edges.
1. Call METIS `PartGraphKway` with `n_ranks`.
1. Validate: METIS returned valid partition (all ranks > 0 elements).
1. Assign `element_to_rank[n_cell]`.
1. For each rank, identify:
   - `local_element_ids`: owned element global IDs
   - `ghost_element_ids`: elements sharing a face with owned elements but owned by other ranks
   - `ghost_owners`: which rank owns each ghost
1. **GLL numbering**: assign 1-based global IDs; 0 = null. Build `gll_to_global[...]` (`ibool`). Match shared nodes by coordinate tolerance `1e-6 × min_element_size`.
1. For each neighbor rank, precompute face-pair exchange lists:
   - send: (owned_local_idx, face_idx) → (ghost_idx, ghost_face)
   - recv: ghost elements to receive into

Output: one `partition_{r}.h5` per rank with owned/ghost data, metadata, and `/recording/` map. See [mesh.md](mesh.md).

### 10. Build Shallow Recording Map

Green output is shallow mesh vertices, not full GLL. Preprocess builds the map once. Forward then writes with no topology search.

1. Read `record_depth_max_m` and `green_tile_size_m` from `config.py`.
1. Compute `target_z = zmin + record_depth_max_m` (z positive downward).
1. Set `record_depth_actual_m` to the first horizontal element face at or below `target_z`.
1. Select non-PML elements fully above that depth; no clipping.
1. Select unique mesh vertices attached to selected elements.
1. For each vertex, choose one owned source element and corner so forward writes it once.

Output in each `partition_{r}.h5`:

```
/recording/
  attrs: basis="mesh_vertices", record_depth_max_m, record_depth_actual_m,
         green_tile_size_m, excludes_pml=true
  save_element_mask          bool[n_local_elem]
  vertex_ids                 int64[n_record_vertices]
  source_element_local_index int32[n_record_vertices]
  source_corner_index        int32[n_record_vertices]
```

### 11. Evaluate STF

Call `config.stf_func(t_s)` at `t_s = 0, solver_dt, 2*solver_dt, ..., (nsteps-1)*solver_dt`.

Output: `stf[nsteps]` time series array → written to config.h5 `/source/`.

### 12. Locate Source Elements + Precompute Interpolation Weights

Source z = z_min (auto-placed on top free surface). Source is only specified by `source_x_m` and `source_y_m` in config.py.

1. Search only free-surface elements (those with a face on boundary_tag = 1) to locate all elements containing the source (source_x_m, source_y_m) projected onto z_min.
   The source may lie on a shared face, edge, or vertex of adjacent elements.
1. For each containing element, map source (x_s, y_s, z_s) to natural coordinates (ξ_s, η_s, ζ_s)
   via Newton iteration using precomputed dξ/dx.
1. Compute Lagrange interpolation weights: `w_ijk = l_i(ξ_s)·l_j(η_s)·l_k(ζ_s)` for each
   GLL node (i,j,k) in each containing element.
1. Normalize Lagrange weights: Σ w_ijk = 1 across all sharing surface elements.
1. Store element IDs, natural coordinates, and weights in config.h5 `/source/elements/`.

The forward solver multiplies these precomputed weights by STF amplitude and adds to the
residual — no runtime Newton iteration or element search needed.

### 13. Write Output Files

- **mesh.h5** — extended in-place with GLL geometry, `is_pml`, and boundary tags.
- **partition\_{r}.h5** — per-rank element data, GLL numbering, exchange patterns, metadata, and `/recording/` map
- **config.h5** — simulation, domain, source, weights, and STF. No force direction.
- **mesh_auxiliary.h5** (optional) — CSR adjacency relations

## HDF5 Output

### mesh.h5 (extended)

The preprocessor reads topology from `mesh.h5` and writes back:

- `/field/element/coords`, `/field/element/dxi_dx`, `/field/element/jacobian` — GLL geometry
- `/field/element/is_pml` — int8 flag per element (1=PML, 0=ordinary); preprocessing also uses it to build `/recording/` maps
- `/field/surface/boundary_tag` — surface boundary tags (0=interior, 1=free surface, 2=absorbing)

Full schema for `/field/` groups in [mesh.md](mesh.md).

Material, mass, C-PML, and partition data stay in per-rank `partition_{r}.h5` files.

### partition\_{r}.h5

One per MPI rank. Contains owned/ghost GLL fields, partition metadata, and `/recording/` map. Full schema: [mesh.md](mesh.md).

### config.h5

Single rank-invariant file shared across all 3 force direction runs. Direction is passed to the forward solver via `--direction {x,y,z}` CLI flag.

#### Schema

```
config.h5
├── /simulation/
│   ├── title                  : string
│   ├── polynomial_order       : int32            — N (GLL order)
│   ├── solver_dt              : float64          — auto-computed CFL timestep (Newmark loop)
│   ├── output_dt_s            : float64          — user-specified snapshot interval
│   ├── snapshot_stride        : int32            — solver steps per snapshot
│   ├── nsteps                 : int32            — derived total solver steps
│   ├── cfl_safety             : float64
│   ├── snapshot_precision     : string           — "float32" or "float64"
│   ├── restart_dt_s           : float64          — restart overwrite interval
│   ├── restart_stride         : int32            — solver steps per restart write
│   ├── record_depth_max_m     : float64          — requested shallow recording depth
│   ├── record_depth_actual_m  : float64          — snapped horizontal element-face depth
│   ├── green_tile_size_m      : float64          — horizontal postprocess tile width
│   └── storage_limit_gb       : int32            — abort if estimated storage exceeds this
│
├── /domain/
│   ├── xmin, xmax             : float64          — domain bounds
│   ├── ymin, ymax             : float64
│   ├── zmin, zmax             : float64          — z positive downward
│   └── pml_thickness          : int32[6]         — [xmin,xmax,ymin,ymax,zmin,zmax] in element layers
│
└── /source/
    ├── x, y                   : float64          — source position (z auto-placed on top free surface)
    ├── stf                     : float64[nsteps]  — precomputed STF time series (amplitude at t = n·solver_dt)
    ├── n_src_elements         : attr int32        — number of containing elements
    └── /elements/
        ├── element_ids        : int64[n_src_elements]         — global element IDs (1-based)
        ├── xi, eta, zeta      : float64[n_src_elements]       — natural coordinates in [-1, 1]
        └── weights            : float64[n_src_elements, NGLL, NGLL, NGLL] — Lagrange w_ijk (normalized Σw = 1)
```

Note: no `/attenuation/` group. Attenuation (SLS) is deferred to future work.
No `direction` attribute. Runtime `--direction` selects x/y/z; jobs share one `config.h5`.

## No Receivers

Preprocessor does not configure receivers. It builds a shallow mesh-vertex recording map from `record_depth_max_m`. Postprocess uses those vertices directly.

## No Per-Cell Material Tags

Material is at GLL nodes per element via config.py functions. Forward solver reads them directly — no interpolation at runtime.
