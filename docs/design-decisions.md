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
- **PML precompute**: Preprocess writes all C-PML arrays. Forward reads and applies them.
- **Boundary detection**: Auto, by geometry. z_min = free surface (z positive downward), other domain bounds = absorbing. Auto-detection: z ≈ z_min → free surface (tag=1), other domain bounds → absorbing (tag=2).
  No GMSH physical groups needed.

## 4. Sources

- **Source type**: Single force (point impulse)
- **Directions**: 3 orthogonal (x, y, z)
- **Green's tensor**: Full 3×3 strain GF requires 3 forward runs (one per orthogonal force direction x, y, z)
- **Injection**: Lagrange interpolation to surrounding GLL nodes (sub-node accuracy)
- **Source position**: User specifies source_x_m, source_y_m only. source_z is auto-placed on top free surface (z ≈ z_min) by preprocessor.
- **Source weights**: Preprocessor writes source elements, natural coords, and weights. Forward only distributes them.
- **Source interpolation weights**: Normalized across all sharing surface elements so Σ w_ijk = 1
- **STF**: External — user-defined Python function in config script, evaluated by preprocessor
- **STF evaluation**: Integer timesteps only. STF[n] = force amplitude at t = n·solver_dt
- **No inline STF types**: Single user function replaces all STF parameterization
- **Source direction**: Not in config. Forward gets `--direction`; SLURM runs x/y/z jobs.

## 5. Architecture

- **Language (core)**: C++17
- **Build system**: CMake
- **Glue language**: Python
- **Project structure**:
  ```
  tools/           — GMSH → model.h5 converter + VTK visualization tools (Python)
  preprocess/      — Python: GLL geometry, material interpolation, partition, config
  forward/         — C++: core physics library (libgf) + MPI solver executable (elastic only)
  compress/        — C++: header-only checkpoint compression utilities
  postprocess/     — Python: strain GF extraction at shallow mesh vertices
  tests/
  external_reference_codes/
  ```

## 6. Mesh, I/O & Data Files

### File Pipeline

```
GMSH .msh → converter → model.h5 (topology only)
                         ↓
  model.h5 ─────────────────────┤
  config.py ───────────────────┤
                         ↓
                    preprocessor
                    ├── GLL geometry, dξ/dx → write back to model.h5 (extend)
                    ├── Material at GLL nodes
                    ├── lumped mass
                    ├── PML damping profiles
                    ├── Auto solver_dt from CFL + output_dt_s snapshot stride
                    ├── Recording map: shallow, non-PML mesh vertices
                    ├── Comprehensive validation
                    ├── METIS partition
                    ├── STF time series
                    ↓
              model.h5 (extended: +coords +dxi_dx)
               partition_{0,1,...}.h5 (per-rank, local subset)
               config.h5 (single, no direction)
                          ↓
                     forward solver --direction {x,y,z}
                         (reads partitions/partition_{r}.h5 + config.h5)
                          ↓
                    wavefields/{x,y,z}/record_{r}_{step}.h5
                    restart/{x,y,z}/restart_{r}.h5
                          ↓
                     abb|                     postprocess (merge vertex strain — also reads config.h5)
                          ↓
                    greenfun/tile_x{i}_y{j}.h5
```

### File Purposes

| File | Producer | Consumer | Content |
|------|----------|----------|---------|
| model.h5 | converter → preprocessor | preprocessor, postprocess | Topology + GLL geometry + `is_pml`. No material. Postprocess uses `/topology/vertex_to_coord`. |
| partition\_{r}.h5 | preprocessor | forward | Per-rank element data, C-PML, partition metadata, and `/recording/` map |
| config.h5 | preprocessor | forward, postprocess | Simulation params, cadence, record depth, tile size, domain, source, STF, weights. No direction. |
| wavefields/{direction}/record\_{r}\_{step}.h5 | forward | postprocess | L2-smoothed strain at recorded vertices; one step per file |
| restart/{direction}/restart\_{r}.h5 | forward | forward (`--resume`) | Latest full-volume restart: u, v, a, C-PML memory, step/time |
| model_auxiliary.h5 | preprocessor (optional) | validation | CSR adjacency relations |
| greenfun/tile_x{i}\_y{j}.h5 | postprocess | user | Mesh-vertex strain Green tensors, x/y tiled |

### Design Rules

| Rule | Example |
|------|---------|
| X2Y naming for relations | `edge_to_vertex`, `cell_to_surface`, `dxi_dx` |
| 1-based indexing, 0 = null | element IDs, vertex IDs, surface IDs |
| Sign = direction (signed int) | `+edge_id` = positive traversal, `-edge_id` = reverse |
| Element-first layout | `[n_cell, NGLL, NGLL, NGLL, ...]` |
| NGLL = N+1 embedded in shapes | Derived from array dims, not a separate attribute |
| Python configs as importable scripts | No YAML/TOML — `config.py` imported by preprocessor |
| Model and config in separate files | partition\_{r}.h5 = mesh data per rank; config.h5 = simulation parameters |

### Topology Hierarchy

```
vertex → edge → surface → cell
         (2 vertices)   (4 edges, quad)   (6 surfaces, hex)
```

### model.h5 Schema (Extended Geometry)

model.h5 holds topology, geometry, and full material/field data at GLL nodes:

```
model.h5
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
    ├── dxi_dx                  : float64[..., 3,3]  — ∂ξ_i/∂x_j
    ├── mass                    : float64[...]       — lumped mass diagonal
    ├── vp, vs, density         : float64[...]       — material at GLL nodes
    ├── lambda, mu              : float64[...]       — elastic constants
    ├── damping                 : float64[...]       — PML damping profile
    ├── is_pml                  : int8[n_cell]       — PML flag
    └── tile_index              : int64[n_cell]      — tile ID or -1
```

These same field datasets are also written to `partition_{r}.h5` per rank.

### partition\_{r}.h5 Schema (Per-Rank)

Each partition file holds a local subset of all element data needed by one MPI rank:

```
partition_{r}.h5
├── /topology/                  ← local subset of topology
├── /field/element/*            ← local subset: coords, dxi_dx, jacobian, mass, vp, vs, density, lambda, mu, cpml/*
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

### Record and Restart Format

Forward writes shallow mesh-vertex records (strain + displacement/velocity/acceleration) and separate latest-only restarts.

```
wavefields/{direction}/record_{r}_{step}.h5
├── attrs: rank, source_direction, basis="mesh_vertices", record_depth_max_m,
│          record_depth_actual_m, excludes_pml=true
├── vertex_ids     : int64[n_record_vertices]             # global mesh vertex IDs
├── strain         : float32[1, n_record_vertices, 6]     # single step
├── displacement   : float32[1, n_record_vertices, 3]
├── velocity       : float32[1, n_record_vertices, 3]
└── acceleration   : float32[1, n_record_vertices, 3]

restart/{direction}/restart_{r}.h5
├── attrs: rank, source_direction, step, time_s, ngll
├── displacement      : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
├── velocity          : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
├── acceleration      : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
└── pml_memory_*      : float64[...]             # all C-PML state required for exact resume
```

### config.h5 Format

One file serves all force directions. Forward gets direction from `--direction {x,y,z}`:

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
│   ├── restart_dt_s           : float64            — latest-only restart overwrite interval
│   ├── restart_stride         : int32              — solver steps per restart write
│   ├── record_depth_max_m     : float64            — requested shallow recording depth
│   ├── record_depth_actual_m  : float64            — snapped spectral-element face depth
│   ├── nx_elements, ny_elements, nz_elements  : int64   — mesh grid dims
│   ├── pml_{x,y,z}{min,max}  : int64                — PML thickness in elements
│   ├── tilex_elements, tiley_elements : int64[n_tiles] - horizontal tile sizes in elements

│   ├── green_tile_size_m           : float64 (optional) — spatial tile size in meters; overrides element tiling
│   ├── log_stride             : int32              — progress-report interval in solver steps
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

Notes: no `/attenuation/`; SLS is deferred. No `direction`; runtime CLI sets it.

### Green's Function Output

Green library stores strain, not displacement. Tiles use element-index bins from `tilex_elements`/`tiley_elements` (default) or coordinate-index bins from `green_tile_size_m` (when set):

```
greenfun/
├── tile_x000_y000.h5
├── tile_x001_y000.h5
└── ...
```

Each tile stores recorded vertices in its x/y bounds for all saved depths. Green files store `vertex_ids`; coordinates stay in `model.h5`.

## 7. Preprocessor Decisions

- **Config format**: Importable Python script, not YAML/TOML. Dimensional fields use SI suffixes.
- **Material functions**: `config.py` defines `vp_m_s`, `vs_m_s`, and `density_kg_m3`. No separate binary model.
- **Output**: partition\_{r}.h5 (per-rank subset of element data) + single config.h5 (rank-invariant simulation + domain + source data).
- **Source direction**: NOT in config.py or config.h5. Preprocessor auto-generates one config.h5. Forward solver takes force direction via CLI `--direction {x,y,z}`.
- **Timestep split**: User sets `output_dt_s`, `restart_dt_s`, and `total_duration_s`. Preprocess computes `solver_dt`, `snapshot_stride`, and `restart_stride`. Forward writes strain/restart on those strides.
- **Validation**: Comprehensive checks at preprocess time:
  - Mesh: n_cell > 0, non-degenerate hex elements, det(J) > 0 at all GLL nodes
  - Material: vp > 0, vs ≥ 0, density > 0 at all GLL nodes
  - CFL: solver_dt auto-derived from CFL constraint; snapshot_stride and restart_stride validated as integers
  - Boundary: Free surface detected at z ≈ z_min; PML has ≥ 2 elements per absorbing face (warn if thinner)
  - Source: x_m, y_m within domain bounds; stf_func returns finite non-NaN values over [0, nsteps×solver_dt]
  - Storage: estimated disk usage ≤ storage_limit_gb, abort if exceeded
  - Snapshot/restart cadence: nsteps % snapshot_stride == 0; restart_stride >= 1
- **Mesh output**: Preprocess adds GLL geometry and `is_pml` to `model.h5`. Rank-local data and `/recording/` go to `partition_{r}.h5`.
- **STF precompute**: Preprocess samples `stf_func(t_s)` at `solver_dt` and writes an array. Forward does no STF eval.
- **Mass computation**: After material interpolation (ρ needed for lumped mass).
  - **CPML precompute**: Tag PML elements and write linear-ramp damping profile to `/field/element/damping`. Full C-PML (d/K/α per direction, convolution coefficients) is deferred.
- **Partitioning**: METIS k-way partition + GLL node global numbering (ibool equivalent) + precomputed exchange patterns. Each rank gets its own partition\_{r}.h5 with local subset.
- **Geometric precompute**: GLL coords, Jacobian, dξ/dx, lumped mass, C-PML arrays — all at GLL nodes. Written to partition\_{r}.h5 `/field/element/`.
- **Source precompute**: Put source on top free surface. Write source elements, natural coords, and normalized weights.
- **No receivers**: Postprocess uses recorded mesh vertices. No receiver CSV, search, or interpolation.
- **No inline STF**: User-defined function, evaluated over full time range.
- **DRY metric**: N/NGLL embedded in array shapes — no separate config attribute.
- **Elastic only**: SLS attenuation deferred to future work.

## 8. Forward Solver Decisions

- **Elastic only**: No SLS memory variables. Attenuation deferred to future work.
- **Matrix-free assembly**: No global system matrix. K·u computed element-by-element.
- **Global residual**: Single `r[NDIM, NGLOB_AB]` indexed by global GLL ID. Element contributions add into shared nodes.
- **Precomputed data**: All mesh-dependent quantities read from partition\_{r}.h5 — no init phase.
- **Material**: Read at GLL nodes from partition\_{r}.h5 — no runtime interpolation.
- **Source injection**: Precomputed Lagrange weights and element list from config.h5 — forward solver distributes STF amplitude to GLL nodes via stored weights. No runtime Newton iteration.
- **PML damping**: Simple linear-ramp damping applied to velocity: v ← v − d(node)·v. Precomputed damping profile read from partition. Full recursive-convolution C-PML (Wang et al. 2006, 39 memory variables) is deferred.
- **No runtime PML build**: Damping profile precomputed by preprocessor, read from partition at startup.
- **Shared nodes**: Within-rank sums via global array. Cross-rank sums via precomputed MPI face exchanges.
  9a4|- **Runtime loop**: Newmark predict → residual → PML damping → source → MPI exchange → Newmark correct → strain → output.
  c5d|- **Strain computation**: Per-vertex strain computed inline from corrected displacement at recorded mesh corners — reference gradient via derivative matrix, chain rule to physical gradient, symmetric Voigt strain. No separate element pass.
  fa6|- **No L2 smoothing in recording mode**: Strain is computed directly at recorded corner nodes (no global projection). Full-volume GLL strain (legacy fallback) uses direct per-node computation, not L2 projection.
  cee|- **Strain record**: Write per-vertex strain at recorded mesh corners only. float32 default, float64 optional.
- **3 runs per source**: Run x/y/z force jobs. One shared `config.h5`. Each writes `wavefields/{direction}/`.
- **Restart/resume**: Restart is separate and latest-only. It stores u/v/a, step/time, and C-PML memory. `--resume` continues from it.
- **Parallelism**: Pure MPI, one rank per core. GPU element residual works alongside MPI (GPU replaces only the element kernel; residual copied back to CPU for exchange); see [`gpu.md`](design/gpu.md).

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
- **Postprocess alignment**: Validate timing, basis, depth, and merged `vertex_ids` across x/y/z before assembly.
- **PML exclusion**: PML elements/vertices are excluded by the preprocessing recording map — only physical-domain shallow vertices contribute.
- **Element tiling**: `tilex_elements`/`tiley_elements` define x/y tile sizes in elements. Tiles partition the non-PML interior. Each tile stores mesh-vertex Green tensors for all recorded depths.
- **Reciprocity**: Source is on the top free surface. Strain records cover the configured shallow output volume.
