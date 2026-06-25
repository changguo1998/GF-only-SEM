# Design Decisions — Green's Function SEM Solver

This document records the architectural decisions made during the `/grill-me` session.
It is a working reference for development, kept under `docs/`.

Mathematical formulation for all methods below: [`docs/math.md`](math.md)

## 1. Physical Model

- **Method**: Continuous Galerkin Spectral Element Method (CG-SEM)
- **Geometry**: 3D Cartesian (start), spherical not planned
- **Physics**: Viscoelastic (deferred — elastic-only for initial implementation)
- **Attenuation**: Fixed Q (frequency-independent) modeled with standard linear solid (SLS) — deferred
- **Relaxation**: Per-GLL-node τ-method SLS parameters — deferred
- **Wave equation**: Second-order hyperbolic, forward time integration

## 2. Discretization

- **Element type**: Hexahedra, Gauss-Lobatto-Legendre (GLL) quadrature
- **Polynomial order**: N=3 (testing), N=5 (production)
- **NGLL**: N+1 = 4 (test) / 6 (prod)
- **Time integration**: Newmark explicit (2nd order predictor-corrector, β=0, γ=½)
- **CFL**: Conditional stability, standard SEM constraint
- **NGLL derivation**: Implicit in array shapes — no separate attribute needed

## 3. Boundary Conditions

- **Absorbing boundaries**: C-PML (convolutional Perfectly Matched Layer), matching SPECFEM3D
- **Domain**: Cartesian box with C-PML layers surrounding the physical domain
- **Coordinate convention**: z positive downward (seismology standard). z_min = top free surface, z_max = bottom.
- **PML thickness**: Configurable per face (default 3 elements)
- **PML precomputation**: All C-PML data (damping profiles d_x, d_y, d_z; stretched-coordinate functions K, α; convolution coefficients; element type tags face/edge/corner) precomputed by preprocessor — forward solver reads and applies directly
- **Boundary detection**: Auto, by geometry. z_min = free surface (z positive downward), other domain bounds = absorbing. Auto-detection: z ≈ z_min → free surface (tag=1), other domain bounds → absorbing (tag=2).
  No GMSH physical groups needed.

## 4. Sources

- **Source type**: Single force (point impulse)
- **Directions**: 3 orthogonal (x, y, z)
- **Green's tensor**: Full 3×3 strain GF requires 3 forward runs (one per orthogonal force direction x, y, z)
- **Injection**: Lagrange interpolation to surrounding GLL nodes (sub-node accuracy)
- **Source position**: User specifies source_x_m, source_y_m only. source_z is auto-placed on top free surface (z ≈ z_min) by preprocessor.
- **Source weights**: Precomputed by preprocessor (element list + (ξ_s, η_s, ζ_s) natural coordinates + Lagrange weights w_ijk per element) — forward solver just reads and distributes
- **Source interpolation weights**: Normalized across all sharing surface elements so Σ w_ijk = 1
- **STF**: External — user-defined Python function in config script, evaluated by preprocessor
- **STF evaluation**: Integer timesteps only. STF[n] = force amplitude at t = n·solver_dt
- **No inline STF types**: Single user function replaces all STF parameterization
- **Source direction**: NOT in config.py. Preprocessor writes one config.h5. Forward solver reads source direction from CLI `--direction` flag — three independent runs (x, y, z) managed by SLURM.

## 5. Architecture

- **Language (core)**: C++17
- **Build system**: CMake
- **Glue language**: Python
- **Project structure**:
  ```
  tools/           — GMSH → mesh.h5 converter (Python, format transport only)
  preprocess/      — Python: GLL geometry, material interpolation, partition, config
  forward/         — C++: core physics library (libgf) + MPI solver executable (elastic only)
  compress/        — C++: header-only checkpoint compression utilities
  postprocess/     — Python: strain GF extraction at GLL nodes, spatial tiling
  tests/
  external_reference_codes/
  ```

## 6. Mesh, I/O & Data Files

### File Pipeline

```
GMSH .msh → converter → mesh.h5 (topology only)
                         ↓
  mesh.h5 ─────────────────────┤
  config.py ───────────────────┤
                         ↓
                    preprocessor
                    ├── GLL geometry, dξ/dx → write back to mesh.h5 (extend)
                    ├── Material at GLL nodes
                    ├── lumped mass
                    ├── PML damping profiles
                    ├── Auto solver_dt from CFL + output_dt_s snapshot stride
                    ├── Comprehensive validation
                    ├── METIS partition
                    ├── STF time series
                    ↓
              mesh.h5 (extended: +coords +dxi_dx)
               partition_{0,1,...}.h5 (per-rank, local subset)
               configs/config.h5 (single, no direction)
                          ↓
                     forward solver --direction {x,y,z}
                         (reads partitions/partition_{r}.h5 + configs/config.h5)
                          ↓
                    wavefields/{x,y,z}/record_{r}.h5
                          ↓
                     postprocess (reads mesh.h5 for geometry)
                          ↓
                    greenfun/tile_{i}.h5
```

### File Purposes

| File | Producer | Consumer | Content |
|------|----------|----------|---------|
| mesh.h5 | converter → extended by preprocessor | preprocessor, postprocess | Topology + GLL coords + dxi_dx + jacobian + is_pml (geometry only, no material) |
| partition_{r}.h5 | preprocessor | forward | Per-rank local subset of all element data (coords, dxi_dx, jacobian, mass, vp, vs, density, cpml/*) + partition metadata (gll_to_global, exchange) |
| configs/config.h5 | preprocessor | forward | Simulation params, domain bounds, source position + precomputed STF + weights. No direction — passed via CLI `--direction`. |
| wavefields/{direction}/record_{r}.h5 | forward | postprocess | L2-smoothed strain at GLL nodes + (u,v,a) restart state, extendible time axis |
| mesh_auxiliary.h5 | preprocessor (optional) | validation | CSR adjacency relations |
| greenfun/tile_{i}.h5 | postprocess | user | Strain Green's functions at GLL nodes, tiled by element range |

### Design Rules

| Rule | Example |
|------|---------|
| X2Y naming for relations | `edge_to_vertex`, `cell_to_surface`, `dxi_dx` |
| 1-based indexing, 0 = null | element IDs, vertex IDs, surface IDs |
| Sign = direction (signed int) | `+edge_id` = positive traversal, `-edge_id` = reverse |
| Element-first layout | `[n_cell, NGLL, NGLL, NGLL, ...]` |
| NGLL = N+1 embedded in shapes | Derived from array dims, not a separate attribute |
| Python configs as importable scripts | No YAML/TOML — `config.py` imported by preprocessor |
| Model and config in separate files | partition_{r}.h5 = mesh data per rank; config.h5 = simulation parameters |

### Topology Hierarchy

```
vertex → edge → surface → cell
         (2 vertices)   (4 edges, quad)   (6 surfaces, hex)
```

### mesh.h5 Schema (Extended Geometry)

mesh.h5 holds topology and geometry only — no material, no partition data:

```
mesh.h5
├── /topology/                  ← copied from converter
│   ├── n_vertex, n_edge, n_surface, n_cell  : attr int64
│   ├── vertex_to_coord         : float64[n_vertex, 3]
│   ├── edge_to_vertex          : int64[n_edge, 2]
│   ├── surface_to_edge         : int64[n_surface, 4]
│   └── cell_to_surface         : int64[n_cell, 6]
│
└── /field/element/             ← GLL-node level [n_cell, NGLL, NGLL, NGLL, ...]
    ├── coords                  : float64[..., 3]   — GLL node (x,y,z)
    ├── jacobian                : float64[...]       — det(J)
    └── dxi_dx                  : float64[..., 3,3]  — ∂ξ_i/∂x_j
```

### partition_{r}.h5 Schema (Per-Rank)

Each partition file holds a local subset of all element data needed by one MPI rank:

```
partition_{r}.h5
├── /topology/                  ← local subset of topology
├── /field/element/*            ← local subset: coords, dxi_dx, jacobian, mass, vp, vs, density, cpml/*
│
└── /partition/
    ├── n_ranks                 : attr int32
    ├── n_elem_local            : attr int32
    ├── n_elem_ghost            : attr int32
    ├── n_global_nodes          : attr int64              — unique GLL nodes on this rank
    ├── local_element_ids       : int64[n_elem_local]
    ├── ghost_element_ids       : int64[n_ghost]
    ├── ghost_owners            : int32[n_ghost]
    ├── gll_to_global           : int64[n_elem_total, NGLL, NGLL, NGLL]  — ibool: local→global GLL node ID
    └── /exchange/              — precomputed face-pair lists per neighbor
```

### Record Format

One file per MPI rank per forward run. Extendible datasets for strain and restart state:

```
wavefields/{direction}/record_{r}.h5
├── attrs: rank, source_direction, ngll
├── local_element_ids  : int64[n_elem_local]
├── strain             : float32[n_snapshots, n_elem_local, NGLL, NGLL, NGLL, 6]
└── /restart/          — latest state (u, v, a) for resume
    ├── displacement   : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
    ├── velocity       : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
    └── acceleration   : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
```

### config.h5 Format

One file for all force directions. Force direction is passed as a CLI argument (`--direction {x,y,z}`) to the forward solver — not embedded in the file:

```
config.h5
├── /simulation/
│   ├── title                  : string
│   ├── polynomial_order       : int32
│   ├── solver_dt              : float64            — auto-computed CFL timestep (Newmark loop)
│   ├── output_dt_s            : float64            — user-specified snapshot interval
│   ├── snapshot_stride        : int32              — solver steps per snapshot (integer)
│   ├── nsteps                 : int32              — total solver steps (derived from total_duration_s)
│   ├── cfl_safety             : float64
│   ├── snapshot_precision     : string             — "float32" or "float64"
│   └── storage_limit_gb       : int32              — abort if estimated storage exceeds this
│
├── /domain/
│   ├── xmin, xmax, ymin, ymax, zmin, zmax  : float64
│   └── pml_thickness          : int32[6]    — xmin,xmax,ymin,ymax,zmin,zmax
│
└── /source/
    ├── x, y                   : float64            — source position (z auto-placed on top free surface)
    ├── stf                     : float64[nsteps]    — pre-evaluated STF time series (amplitude at t = n·solver_dt)
    ├── n_src_elements         : attr int32         — number of containing elements
    └── /elements/
        ├── element_ids        : int64[n_src_elements]   — global element IDs (1-based)
        ├── xi, eta, zeta      : float64[n_src_elements] — natural coordinates
        └── weights            : float64[n_src_elements, NGLL, NGLL, NGLL] — Lagrange w_ijk (normalized)
```

Note: no `/attenuation/` group. Attenuation (SLS) is deferred to future work.
Note: no `direction` attribute. Force direction is specified via `--direction` CLI flag at runtime.

### Green's Function Output

Strain Green's function library — stores strain components, not displacement.
Output is spatially tiled by lat/lon bounding boxes:

```
greenfun/
├── tile_0.h5    — attrs: minlat, maxlat, minlon, maxlon
├── tile_1.h5
└── ...
```

Each tile contains the strain Green's functions for all elements within its spatial range.

## 7. Preprocessor Decisions

- **Config format**: Importable Python script (not YAML/TOML). Config file IS the configuration. All dimensional fields carry SI-unit suffixes (`_m`, `_s`, `_m_s`, `_kg_m3`).
- **Material functions**: Callable Python functions with SI-unit suffixes `vp_m_s(x_m, y_m, z_m)`, `vs_m_s(x_m, y_m, z_m)`, `density_kg_m3(x_m, y_m, z_m)` in config.py — no separate binary model file.
- **Output**: partition_{r}.h5 (per-rank subset of element data) + single config.h5 (rank-invariant simulation + domain + source data).
- **Source direction**: NOT in config.py or config.h5. Preprocessor auto-generates one config.h5. Forward solver takes force direction via CLI `--direction {x,y,z}`.
- **Timestep split**: User specifies `output_dt_s` (snapshot interval) and `total_duration_s` in config.py. Preprocessor computes `cfl_dt = cfl_safety × h_min / vp_max` from minimum GLL node spacing and maximum vp, then derives `solver_dt` by searching for an integer stride such that `output_dt_s / stride ≤ cfl_dt`. The `snapshot_stride = output_dt_s / solver_dt` (integer). `nsteps = ceil(total_duration_s / solver_dt)`. Forward solver uses `solver_dt` in the Newmark loop and writes snapshots when `step % snapshot_stride == 0`.
- **Validation**: Comprehensive checks at preprocess time:
  - Mesh: n_cell > 0, non-degenerate hex elements, det(J) > 0 at all GLL nodes
  - Material: vp > 0, vs ≥ 0, density > 0 at all GLL nodes
  - CFL: solver_dt auto-derived from CFL constraint, snapshot_stride validated as integer
  - Boundary: Free surface detected at z ≈ z_min; PML has ≥ 2 elements per absorbing face (warn if thinner)
  - Source: x_m, y_m within domain bounds; stf_func returns finite non-NaN values over [0, nsteps×solver_dt]
  - Storage: estimated disk usage ≤ storage_limit_gb, abort if exceeded
  - Snapshot stride: nsteps % snapshot_stride == 0
- **Mesh output format**: Extended HDF5 — preprocessor writes GLL geometry (`coords`, `dxi_dx`, `jacobian`, `is_pml`) back to mesh.h5; all rank-local data to partition_{r}.h5.
- **STF precomputation**: stf_func(t_s) evaluated over full time range at solver_dt spacing, written as time series array to config.h5. Forward solver reads array — no runtime STF evaluation.
- **Mass computation**: After material interpolation (ρ needed for lumped mass).
- **CPML precomputation**: Layer-based — trace element connectivity from boundary faces inward. Classify each CPML element as face/edge/corner type. Precompute all C-PML arrays: damping profiles d_x, d_y, d_z per GLL node; stretched-coordinate functions K, α; convolution coefficients; element type tags. Written to partition_{r}.h5 `/field/element/cpml/`.
- **Partitioning**: METIS k-way partition + GLL node global numbering (ibool equivalent) + precomputed exchange patterns. Each rank gets its own partition_{r}.h5 with local subset.
- **Geometric precompute**: GLL coords, Jacobian, dξ/dx, lumped mass, C-PML arrays — all at GLL nodes. Written to partition_{r}.h5 `/field/element/`.
- **Source precompute**: Source z auto-placed on top free surface (z ≈ z_min). Element list, natural coordinates (ξ_s, η_s, ζ_s), Lagrange interpolation weights w_ijk for source injection. Weights normalized across sharing surface elements (Σ w_ijk = 1).
- **No receivers**: Postprocess operates on GLL-node strain directly — no receiver CSV, no receiver search, no position interpolation.
- **No inline STF**: User-defined function, evaluated over full time range.
- **DRY metric**: N/NGLL embedded in array shapes — no separate config attribute.
- **Elastic only**: SLS attenuation deferred to future work.

## 8. Forward Solver Decisions

- **Elastic only**: No SLS memory variables. Attenuation deferred to future work.
- **Matrix-free assembly**: No global system matrix. K·u computed element-by-element.
- **Global residual array**: Single `r[NDIM, NGLOB_AB]` indexed by global GLL node ID (ibrk). Assembly via `iglob(i,j,k,ispec)` mapping (grown) — element contributions accumulated additively, shared nodes implicitly summed.
- **Precomputed data**: All mesh-dependent quantities read from partition_{r}.h5 — no init phase.
- **Material**: Read at GLL nodes from partition_{r}.h5 — no runtime interpolation.
- **Source injection**: Precomputed Lagrange weights and element list from config.h5 — forward solver distributes STF amplitude to GLL nodes via stored weights. No runtime Newton iteration.
- **C-PML memory**: 21 rmemory arrays = 39 scalar values per GLL node per CPML element. Second-order recursive convolution (Wang et al. 2006, eq. 21, θ=1/8). Read from partition_{r}.h5.
- **C-PML**: Read all precomputed convolution coefficients from partition_{r}.h5. Apply C-PML memory variable update and acceleration correction per element — no runtime damping computation.
- **Shared node assembly**: Within-rank: implicit via global array accumulation. Cross-rank: MPI halo exchange using precomputed face-pair patterns — pack/unpack per face, assemble shared nodes.
- **Runtime loop**: Newmark predict → element residual (matrix-free K·u) → C-PML → source → MPI exchange → Newmark correct → strain compute (separate pass on corrected u) → snapshot write.
- **Strain computation**: Separate element pass after Newmark correct — computes ε = ½(∇u_new + ∇u_newᵀ) from corrected displacement field.
- **L2 strain smoothing**: After element-wise strain computation, global L2 projection onto continuous GLL nodal basis. ε_smooth = M⁻¹ · Σ ∫ N · ε_elem dΩ. Produces C⁰-continuous strain at shared nodes. Matches SPECFEM3D convention.
- **Strain in record**: L2-smoothed strain (not raw element strain). Stored as float32 (default) or float64 (configurable). Written when `step % snapshot_stride == 0`.
- **3 runs per source**: 3 orthogonal force directions (x, y, z), independent gf_solver invocations managed by SLURM. Single config.h5 shared across all 3 runs; force direction passed via CLI `--direction {x,y,z}`. Each run writes snapshots to its own directory `wavefields/{direction}/`.
- **Restart/resume**: Snapshots save corrected (u, v, a) state alongside strain. `--resume` flag restores state and continues the time loop — the `a` in snapshot is the corrected M⁻¹·r, directly usable for Newmark prediction.
- **Parallelism**: Pure MPI (one rank per core). Architecture leaves GPU/DCU kernel swap-in path for future acceleration — see [`design/gpu.md`](superpowers/design/gpu.md) for the device-abstraction design.
- **ibool/GLL node numbering**: Per-rank global GLL node IDs stored in partition_{r}.h5 `/partition/gll_to_global`, 1-based with 0=null. Interface node lists precomputed for MPI exchange.

## 9. Testing & Validation

- **C++ framework**: Catch2 (header-only via CMake FetchContent)
- **Python framework**: pytest + numpy.testing.assert_allclose
- **Testing tiers**:
  - **Unit (CI, always)**: Element assembly, Newmark step, PML variables, mesh loader, snapshot I/O, preprocessor validation (material, mesh, CFL, boundary)
  - **Integration (CI)**: Small forward run (N=3, few time steps), verify no crash, hash-check output
  - **Benchmark (Manual)**: Analytical benchmarks (homogeneous half-space, layered medium)
  - **Profile (Manual)**: Performance profiling, scalability check

## 10. Compression

- **Compression timing**: Inline (C++ writes compressed HDF5 during forward run)
- **Algorithm**: HDF5 built-in filters (zlib/gzip, lzf) + float32 storage
- **Chunking**: Element-first, chunk_size=64 along element dim, time dim chunk=1
- **No post-hoc compression pass**

## 11. External References

- **SPECFEM3D** (Cartesian): `external_reference_codes/specfem3d/` — reference implementation
- **SPECFEM3D Globe** (spherical): `external_reference_codes/specfem3d_globe/` — reference

Both are untracked (`*.gitignore`). Changes to them do not affect the repo.

## 12. Green's Function Pipeline

- **3 orthogonal force directions**: 3 independent forward runs (one per fx, fy, fz) produce the full 3×3 strain Green's tensor at a single source location.
- **Single source location**: One source position per GF computation. Multiple source locations require separate preprocessor + 3×N forward runs.
- **Postprocess alignment**: Postprocess validates solver_dt, nsteps, n_cell alignment across the 3 force-direction runs before extracting Green's functions.
- **PML exclusion**: PML elements excluded from the Green's function library — only physical-domain elements contribute.
- **Spatial tiling**: Green's function library is spatially tiled by element range. Each `greenfun/tile_{i}.h5` stores the full GLL-node Green's tensor for a contiguous block of elements.
- **Reciprocity**: Numerical source placed at top free surface. Strain recorded everywhere in the physical domain, enabling reciprocity-based interpretation.
