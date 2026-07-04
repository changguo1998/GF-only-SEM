# Mesh Module — Technical Design

> Parent: [../design-decisions.md](../design-decisions.md)

## Goal

External mesh conversion to internal HDF5 topology format. GMSH .msh → model.h5 (topology only, format transport).

Partitioning and all derived data are done by the preprocessor, not the mesh module.

## Data Flow

```
GMSH .msh v4.1
      │
      ▼
  converter (Python, format transport only — no computation)
      │
  model.h5  (/topology/ only — converter output)
      │
      ▼
  preprocessor  (interpolation, boundaries, geometric precompute, partition)
      │              extends model.h5 with /field/element/coords,
      │              /field/element/dxi_dx, /field/element/jacobian,
      │              /field/element/is_pml (PML flag for preprocess recording map)
  partition_{0}.h5, partition_{1}.h5, …  (per-rank partition files)
  config.h5  (simulation + domain + source, rank-invariant)
      │
      ▼
  forward solver
```

### Directory Layout

```
working_dir/
├── config.py                    — user's config script
├── model.h5                      — converter output + preprocessor extensions
├── configs/
│   └── config.h5                — single config file (rank-invariant, no direction)
├── partitions/
│   ├── partition_0.h5
│   ├── partition_1.h5
│   └── ...
├── wavefields/
│   ├── x/record_{r}_{step}.h5    — forward run fx, shallow mesh-vertex strain
│   ├── y/record_{r}_{step}.h5    — forward run fy
│   └── z/record_{r}_{step}.h5    — forward run fz
├── restart/
│   ├── x/restart_{r}.h5         — latest-only full-volume restart
│   ├── y/restart_{r}.h5
│   └── z/restart_{r}.h5
└── greenfun/
    ├── tile_x000_y000.h5
    ├── tile_x001_y000.h5
    └── ...
```

## Architecture

Single component:

1. **Python converter** (`tools/gmsh_to_hdf5.py`) — GMSH v4.1 → internal HDF5 topology. Format transport only. No computation, no partition.

Partitioning was moved to the preprocessor so all derived data (GLL node positions, material, geometric quantities) and partition share the same computation pass.

## Technology

- Python 3 + meshio (GMSH v4.1 reader)
- HDF5

## HDF5 Format — `model.h5` (converter + preprocessor output)

### Converter phase

The converter does NO computation. It reads GMSH and writes `/topology/` only.

### Preprocessor phase (extension)

Preprocess extends `model.h5` with geometry used by forward, validation, and vertex lookup:

```
model.h5  (preprocessor extensions)
└── /field/
    ├── /element/
    │   ├── coords     : float64[n_cell, NGLL, NGLL, NGLL, 3]   — GLL node (x, y, z) per element
    │   ├── dxi_dx     : float64[n_cell, NGLL, NGLL, NGLL, 3, 3] — ∂ξ_i/∂x_j per element
    │   ├── jacobian   : float64[n_cell, NGLL, NGLL, NGLL]      — det(J) per element
    │   ├── mass       : float64[n_cell, NGLL, NGLL, NGLL]      — lumped mass diagonal
    │   ├── vp         : float64[n_cell, NGLL, NGLL, NGLL]      — P-wave speed at GLL nodes
    │   ├── vs         : float64[n_cell, NGLL, NGLL, NGLL]      — S-wave speed at GLL nodes
    │   ├── density    : float64[n_cell, NGLL, NGLL, NGLL]      — density at GLL nodes
    │   ├── lambda     : float64[n_cell, NGLL, NGLL, NGLL]      — 1st Lamé parameter
    │   ├── mu         : float64[n_cell, NGLL, NGLL, NGLL]      — 2nd Lamé parameter (shear modulus)
    │   ├── damping    : float64[n_cell, NGLL, NGLL, NGLL]      — PML damping profile
    │   ├── is_pml     : int8[n_cell]                            — 1=PML element, 0=ordinary
    │   └── tile_index : int64[n_cell]                           — tile ID or -1 (PML / below recording depth)
    │
    └── /surface/
        └── boundary_tag : int64[n_surface]                     — 0=interior, 1=free surface, 2=absorbing
```

These `/field/` groups are written to both `model.h5` and `partition_{r}.h5`.
Material, C-PML metadata, per-rank partition metadata, and recording maps also live in `partition_{r}.h5`.

**`is_pml` flag**: preprocess marks absorbing-layer elements. Recording map excludes them.

### Design Rules (shared by model.h5 and partition\_{r}.h5)

| Rule | Example |
|------|---------|
| `X2Y` naming for relations | `edge_to_vertex`, `cell_to_surface` |
| 1-based indexing, 0 = null | `surface_to_cell` → `(0, 5)` means boundary |
| Sign = direction (signed int) | `+edge_id` = positive traversal, `-edge_id` = reverse |
**Note**: Edge/surface topology is for boundary tags and diagnostics. Forward uses elements and GLL nodes only. The CG-SEM matrix-free assembly uses GLL nodes and element-local indexing, not face/edge connectivity.

### Schema — converter phase

```
model.h5  (converter writes /topology/ only)
└── /topology/
    ├── n_vertex         : attr int64
    ├── n_edge           : attr int64
    ├── n_surface        : attr int64
    ├── n_cell           : attr int64
    │
    ├── vertex_to_coord  : float64[n_vertex, 3]           — (x, y, z), vertex ID = 1-based row index
    │
    ├── edge_to_vertex   : int64[n_edge, 2]               — (+v1, +v2), v1 < v2
    │                                                       positive direction = v1 → v2
    │
    ├── surface_to_edge  : int64[n_surface, 4]            — (±edge_id, ...) CCW loop
    │                                                       +edge = traverse in positive edge direction
    │                                                       -edge = traverse in reverse
    │
    └── cell_to_surface  : int64[n_cell, 6]               — (±surface_id, ...)
                                                              +surface = outward normal
                                                              -surface = inward normal
```

### Topology Hierarchy

```
vertex → edge → surface → cell
         (2 vertices)   (4 edges, quad face)   (6 surfaces, hex)
```

## HDF5 Format — `partition_{r}.h5` (per-rank partition files, preprocessor output)

Each MPI rank gets one partition file with local topology, element fields, and metadata. Forward reads one file per rank.

```
partition_{r}.h5
├── /topology/              ← local subset of model.h5 /topology/
│   ├── n_vertex, n_edge, n_surface, n_cell  : attr int64
│   ├── vertex_to_coord  : float64[n_vertex_local, 3]
│   ├── edge_to_vertex   : int64[n_edge_local, 2]
│   ├── surface_to_edge  : int64[n_surface_local, 4]
│   └── cell_to_surface  : int64[n_cell_local, 6]
│
├── /field/
│   ├── /element/           ← full-rank arrays [n_elem_total, NGLL, NGLL, NGLL, …]
│   │   ├── coords   : float64[n_elem_total, NGLL, NGLL, NGLL, 3]   — GLL node (x,y,z)
│   │   ├── dxi_dx : float64[n_elem_total, NGLL, NGLL, NGLL, 3, 3] — ∂ξ_i/∂x_j
│   │   ├── jacobian  : float64[n_elem_total, NGLL, NGLL, NGLL]       — det(J)
│   │   ├── mass      : float64[n_elem_total, NGLL, NGLL, NGLL]       — lumped mass diagonal
│   │   │
│   │   ├── vp, vs, density, lambda, mu : float64[n_elem_total, NGLL, NGLL, NGLL]
│   │   │
│   │   └── /cpml/                                    — C-PML precomputed arrays
│   │       ├── cpml_type       : int8[n_cell_local]    — 0=interior, 1=face, 2=edge, 3=corner
│   │       ├── d_x, d_y, d_z   : float64[…NGLL,NGLL]  — directional damping per GLL
│   │       ├── K_x, K_y, K_z   : float64[…NGLL,NGLL]  — stretched-coordinate κ
│   │       ├── alpha_x, alpha_y, alpha_z  : float64[…NGLL,NGLL]  — stretched-coordinate α
│   │       ├── conv_coef_alpha : float64[…NGLL,NGLL,3]  — convolution α_c per direction
│   │       ├── conv_coef_beta  : float64[…NGLL,NGLL,3]  — convolution β_c per direction
│   │       └── conv_coef_abar  : float64[…NGLL,NGLL,3]  — convolution ā per direction
│   │
│   └── /surface/
│       └── boundary_tag  : int32[n_surface_local]  — 0=interior, 1=free surface, 2=absorbing
│
└── /partition/
    ├── n_ranks                 : attr int32
    ├── element_to_rank         : int32[n_cell]         — full-rank METIS output
    │
    ├── local_element_ids       : int64[n_elem_local]   — owned element global IDs
    ├── ghost_element_ids       : int64[n_ghost]         — halo elements (other ranks)
    ├── ghost_owners            : int32[n_ghost]         — which rank owns each ghost
    ├── gll_to_global           : int64[n_elem_total, NGLL, NGLL, NGLL]
    │                               — GLL node (i,j,k) → global node ID
    │                               — n_elem_total = n_elem_local + n_ghost
    │                               — 1-based, 0 = null
    │
    └── /exchange/                                    — precomputed MPI patterns
        ├── n_neighbors      : attr int32
        ├── neighbors        : int32[n_neighbors]
        │
        ├── send_to_{neighbor}/
        │   ├── n_faces      : attr int32
        │   ├── elem_idx     : int32[n_faces]        — local element index (owned)
        │   ├── face_idx     : int8[n_faces]         — face 0-5 on that element
        │   ├── ghost_idx    : int32[n_faces]        — ghost element index to unpack into
        │   └── ghost_face   : int8[n_faces]         — face on the ghost element
        │
        └── recv_from_{neighbor}/
            ├── n_faces      : attr int32
            ├── ghost_idx    : int32[n_faces]
            └── ghost_face   : int8[n_faces]
```

### Design Notes — partition\_{r}.h5

- **NGLL** = N+1, embedded in array shapes — no separate attribute needed. Polynomial order N is known to all components via array shapes.
- All element-level fields use element-first layout: `[n_elem_total, NGLL, NGLL, NGLL, …]`
- C-PML arrays (`/field/element/cpml/`) are all precomputed by the preprocessor. Forward solver reads directly — no runtime PML damping computation.
- `gll_to_global` is the CG-SEM assembly map (`ibool`). It maps each local/ghost GLL node to a global ID. Shared nodes accumulate into the same ID.
- `n_elem_total = n_elem_local + n_ghost` — both owned and ghost elements share the same `gll_to_global` numbering space.

## Boundary Tags

```
surface boundary_tag: 0 = interior, 1 = free surface (z ≈ z_min), 2 = absorbing (other domain bounds)
```

Boundary detection is auto, by geometry. No GMSH physical groups needed. One free surface + five absorbing boundaries for this application.

## Design Notes

- **model.h5** gives postprocess vertex coordinates by `vertex_ids`. Converter writes `/topology/`; preprocess adds `/field/element/*` for forward checks.
- **partition\_{r}.h5** serves forward: field data, C-PML, metadata, and per-rank `/recording/` map.
- Postprocess does no element search or interpolation.

## File Layout

```
tools/
└── gmsh_to_hdf5.py          — GMSH v4.1 → model.h5 converter (Python, format transport only)
```

The mesh module is a Python converter only. All topology manipulation happens in the preprocessor.
