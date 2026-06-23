# Preprocess Module — Technical Design

> Parent: [docs/design-decisions.md](../../design-decisions.md)
> Implementation plan: [docs/superpowers/plans/2026-06-08-preprocess.md](../plans/2026-06-08-preprocess.md)

## Goal

Python module that reads mesh topology and a Python config script, computes all derived model data (GLL-node material, geometric quantities, partition, C-PML, source weights, STF evaluation, pre-flight validation) and writes extended `mesh.h5` (GLL geometry + is_pml) + per-rank partition files `partition_{r}.h5` + single `config.h5` (rank-invariant simulation + domain + source, shared by all 3 force directions).

## Data Flow

```
mesh.h5 (/topology/) ─────────┐
config.py (script, importable) ─┤
                                ↓
                          preprocessor (Python)
                          ├── import config.py
                          ├── read mesh.h5 topology
                          ├── compute GLL node coords per element
                          ├── compute geometric quantities (J, dξ/dx)
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
                     mesh.h5 (extended) + partition_{r}.h5 + configs/config.h5
                          │
                          ↓
                     forward solver (C++)
```

Optional output: `mesh_auxiliary.h5` (CSR adjacency for validation/acceleration).

## Architecture

Single Python module with a CLI entry point. The config file IS the configuration — no YAML/TOML parsing. The preprocessor uses `importlib` to load the user's config script as a Python module.

Output files: the preprocessor extends the input `mesh.h5` by appending `/field/element/` data, and writes one `partition_{r}.h5` per MPI rank. There is no monolithic `model.h5` — all data lives in mesh.h5 (extended) + partition files.

```
preprocess/
├── __init__.py
├── cli.py              — CLI entry point
├── config_loader.py     — importlib load config.py, validate
├── topology_reader.py   — read mesh.h5 /topology/
├── gll_geometry.py      — compute GLL node coords, jacobian, dξ/dx per element
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
└── config_writer.py     — write single configs/config.h5 (rank-invariant, no direction)
```

## Technology

- Python 3.10+
- numpy, h5py, scipy (interpolation), pytest
- METIS — called via ctypes or subprocess (partition step)
- No YAML/TOML dependency
- Elastic only — SLS attenuation deferred

## CLI

```
python -m preprocess mesh.h5 config.py
```

| Arg | Description |
|-----|-------------|
| `mesh.h5` | positional, converter output — read topology, write extended geometry + is_pml back |
| `config.py` | positional, user's Python config script |

Output files are automatically placed in convention-based directories:
- `configs/config.h5` — single rank-invariant config
- `partitions/partition_{r}.h5` — per-rank partition files
- mesh.h5 is extended in-place with `/field/element/` geometry and `is_pml`

## Config Script (`config.py`)

The user writes an importable Python module. The preprocessor imports it and extracts variables. The STF function is defined in the script — the preprocessor calls it over the full time range.

```python
# Example config.py — imported by preprocessor
title = "test_run"
polynomial_order = 5       # N (GLL order)

# Time
output_dt_s = 0.001        # user-specified snapshot interval (s)
total_duration_s = 10.0    # simulation duration (s)
cfl_safety = 0.5           # CFL safety factor (0 < cfl_safety < 1)

# Storage
snapshot_precision = "float32"  # "float32" or "float64"
storage_limit_gb = 100           # abort if estimated storage exceeds this
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
| Time | nsteps = ceil(total_duration_s / solver_dt); nsteps % snapshot_stride == 0 | Invalid derived stride → abort |
| Boundary | Free surface detected at z ≈ z_min | No free surface → abort |
| Boundary | PML has ≥ 2 elements per absorbing face | Too thin PML → warn |
| Source | source_x_m, source_y_m within domain bounds | Outside domain → abort |
| STF | stf_func(t_s) returns finite, non-NaN for t_s ∈ [0, nsteps×solver_dt] | Bad STF → abort |
| Storage | Estimated disk usage ≤ storage_limit_gb | Exceeds limit → abort |
| Partition | All ranks have ≥ 1 element | Empty rank → abort |

## Processing Steps

### 1. Load Topology

Read mesh.h5 `/topology/` datasets into memory. All datasets follow X2Y naming, 1-based indexing, signed direction.

### 2. Compute GLL Node Geometry

For each element with 8 corner vertices and polynomial order N:

1. Compute (N+1)³ GLL node positions via the element's geometric mapping:
   `x(ξ,η,ζ) = Σ l_a(ξ)·l_b(η)·l_c(ζ)·x_abc` where `l_a` are GLL Lagrange basis functions and `x_abc` are corner coordinates.
2. Compute Jacobian: `dx/dξ` (3×3) via differentiation of the mapping.
3. `dξ/dx = (dx/dξ)^(-1)` — inverse Jacobian at each GLL node.
4. `det(J) = det(dx/dξ)` — integration weight factor.

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
2. Compute `vp_max = max(vp)` across all GLL nodes
3. Compute `cfl_dt = cfl_safety × h_min / vp_max`
4. Search `stride = 1..MAX_STRIDE` for the first value where `output_dt_s / stride ≤ cfl_dt`
5. Set `solver_dt = output_dt_s / stride` and `snapshot_stride = stride`
6. Compute `nsteps = ceil(total_duration_s / solver_dt)` and adjust effective duration to `nsteps × solver_dt` if needed
7. Print computed `cfl_dt`, `solver_dt`, `snapshot_stride`, and `nsteps` to stdout

`solver_dt`, `output_dt_s`, derived `snapshot_stride`, and derived `nsteps` are stored in config.h5 `/simulation/`. The forward solver uses `solver_dt` for Newmark integration and writes snapshots every `snapshot_stride` solver steps.

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
2. Walk inward through element connectivity: trace N layers of elements from each absorbing boundary face.
3. For each element, independently check each direction against its respective boundary distance.
4. Classify each C-PML element as **face** (damped in 1 direction), **edge** (damped in 2 directions), or **corner** (damped in 3 directions).
5. Compute directional damping profiles `d_x(ξ)`, `d_y(ξ)`, `d_z(ξ)` per GLL node — each direction's damping depends only on that direction's distance from the boundary.
6. Compute stretched-coordinate functions `K_x(ξ)`, `α_x(ξ)` (and similarly for y, z) per GLL node.
7. Precompute convolution coefficients `α_c`, `β_c`, `ā` from the damping profiles (second-order convolution scheme, per Wang et al. 2006).

Output: `/field/element/cpml/{cpml_type, d_x, d_y, d_z, K_x, K_y, K_z, alpha_x, alpha_y, alpha_z, conv_coef_*}`.

### 8. Pre-Flight Validation

Comprehensive validation before partition and writing. Runs as a checklist; with `strict_validation=True` errors abort the run, with `strict_validation=False` errors are logged as warnings and processing continues.

1. **Material**: `vp > 0`, `vs ≥ 0`, `density > 0` at all GLL nodes; `λ = ρ(vp² - 2vs²) > 0` (elastic stability); warn if `vs ≡ 0` anywhere
2. **Mesh quality**: `det(J) > 0` at all GLL nodes (no inverted elements); warn on extreme aspect ratios
3. **CFL/time**: validate derived `solver_dt ≤ cfl_dt`; validate integer `snapshot_stride` and `nsteps % snapshot_stride == 0`; log cfl_dt, solver_dt, snapshot_stride
4. **Boundary**: at least one surface tagged free surface (1); at least one tagged absorbing (2); verify `pml_thickness` values ≤ actual element layers from each boundary; PML thickness ≥ 2 elements per absorbing face (warn if thinner)
5. **Source**: within xy domain bounds; Newton iteration found at least one containing element on free surface; sum of normalized Lagrange weights ≈ 1
6. **STF**: all values finite (no NaN/Inf); warn if non-zero DC component
7. **Partition**: `n_ranks ≤ n_cell` (pre-check before calling METIS)
8. **Storage estimation**: compute total estimated disk usage (partition files + expected snapshot files). Abort if > `storage_limit_gb`:
   - `snapshots_per_run = nsteps / snapshot_stride`
   - `strain_per_run_GB = snapshots_per_run × n_cell × NGLL³ × 6 × bytes_per_float / 1e9`
   - `restart_GB = n_cell × NGLL³ × 3 × 3 × 8 / 1e9`
   - `total_GB = strain_per_run_GB × 3 + restart_GB × 3 + partition_GB`
   - Print storage estimate to stdout

### 9. Partition (METIS) + GLL Node Global Numbering

1. Build dual graph: elements as nodes, shared faces as edges.
2. Call METIS `PartGraphKway` with `n_ranks`.
3. Validate: METIS returned valid partition (all ranks > 0 elements).
4. Assign `element_to_rank[n_cell]`.
5. For each rank, identify:
   - `local_element_ids`: owned element global IDs
   - `ghost_element_ids`: elements sharing a face with owned elements but owned by other ranks
   - `ghost_owners`: which rank owns each ghost
6. **GLL node global numbering**: assign a unique global node ID to every distinct GLL node on this rank (1-based, 0=null). Build `gll_to_global[n_elem_total, NGLL, NGLL, NGLL]` — the core CG-SEM assembly mapping (SPECFEM3D's `ibool` equivalent). Shared GLL nodes (within-rank and cross-rank) are identified by geometric coincidence with tolerance = `1e-6 × min_element_size`.
7. For each neighbor rank, precompute face-pair exchange lists:
   - send: (owned_local_idx, face_idx) → (ghost_idx, ghost_face)
   - recv: ghost elements to receive into

Output: one `partition_{r}.h5` per MPI rank, containing the local subset of all element data (owned + ghost) plus partition metadata (see [mesh.md](mesh.md)).

### 10. Evaluate STF

Call `config.stf_func(t_s)` at `t_s = 0, solver_dt, 2*solver_dt, ..., (nsteps-1)*solver_dt`.

Output: `stf[nsteps]` time series array → written to config.h5 `/source/`.

### 11. Locate Source Elements + Precompute Interpolation Weights

Source z = z_min (auto-placed on top free surface). Source is only specified by `source_x_m` and `source_y_m` in config.py.

1. Search only free-surface elements (those with a face on boundary_tag = 1) to locate all elements containing the source (source_x_m, source_y_m) projected onto z_min.
   The source may lie on a shared face, edge, or vertex of adjacent elements.
2. For each containing element, map source (x_s, y_s, z_s) to natural coordinates (ξ_s, η_s, ζ_s)
   via Newton iteration using precomputed dξ/dx.
3. Compute Lagrange interpolation weights: `w_ijk = l_i(ξ_s)·l_j(η_s)·l_k(ζ_s)` for each
   GLL node (i,j,k) in each containing element.
4. Normalize Lagrange weights: Σ w_ijk = 1 across all sharing surface elements.
5. Store element IDs, natural coordinates, and weights in config.h5 `/source/elements/`.

The forward solver multiplies these precomputed weights by STF amplitude and adds to the
residual — no runtime Newton iteration or element search needed.

### 12. Write Output Files

- **mesh.h5 (extended in-place)** — GLL geometry (`/field/element/coords`, `/field/element/dxi_dx`, `/field/element/jacobian`), PML flags (`/field/element/is_pml`), and boundary tags (`/field/surface/boundary_tag`) written back to input mesh.h5. This data is needed by the postprocess module for Newton iteration at receiver positions and PML exclusion.
- **partition_{r}.h5** — one per MPI rank, containing the local subset of element data (own + ghost), GLL global numbering, exchange patterns, and partition metadata
- **configs/config.h5** — simulation config, domain bounds, source (position + elements + weights), STF. No direction — direction is passed via CLI `--direction {x,y,z}` to the forward solver.
- **mesh_auxiliary.h5** (optional) — CSR adjacency relations

## HDF5 Output

### mesh.h5 (extended)

The preprocessor reads topology from `mesh.h5` and writes back:
- `/field/element/coords`, `/field/element/dxi_dx`, `/field/element/jacobian` — GLL node positions and geometric derivatives needed by postprocess for receiver interpolation
- `/field/element/is_pml` — int8 flag per element (1=PML, 0=ordinary) for postprocess PML exclusion
- `/field/surface/boundary_tag` — surface boundary tags (0=interior, 1=free surface, 2=absorbing)

Full schema for `/field/` groups in [mesh.md](mesh.md).

All other GLL-node data (material, mass, CPML) and partition data are NOT written to mesh.h5. They are distributed across per-rank `partition_{r}.h5` files.

### partition_{r}.h5

One per MPI rank. Contains the local subset (owned + ghost elements) of all GLL-node fields plus partition metadata. Full schema in [mesh.md](mesh.md).

### configs/config.h5

Single rank-invariant file shared across all 3 force direction runs. Direction is passed to the forward solver via `--direction {x,y,z}` CLI flag.

#### Schema

```
configs/config.h5
├── /simulation/
│   ├── title                  : string
│   ├── polynomial_order       : int32            — N (GLL order)
│   ├── solver_dt              : float64          — auto-computed CFL timestep (Newmark loop)
│   ├── output_dt_s            : float64          — user-specified snapshot interval
│   ├── snapshot_stride        : int32            — solver steps per snapshot
│   ├── nsteps                 : int32            — derived total solver steps
│   ├── cfl_safety             : float64
│   ├── snapshot_precision     : string           — "float32" or "float64"
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
Note: no `direction` attribute. Force direction is specified via `--direction` CLI flag at runtime. Three independent SLURM jobs share one config.h5 with different `--direction` values.

## No Receivers

Receivers are NOT configured in the preprocessor. Postprocess extracts receiver time series from snapshot files using mesh.h5 geometry.

## No Per-Cell Material Tags

Material is at GLL nodes per element via config.py functions. Forward solver reads them directly — no interpolation at runtime.