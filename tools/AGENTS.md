# tools/ — GMSH → HDF5 Mesh Converter

## Purpose

Converts GMSH `.msh` format to `mesh.h5` topology. Transport-only — no geometry computation.

## Files

| File | Responsibility |
|------|---------------|
| `gmsh_to_hdf5.py` | Read GMSH `.msh`, extract hexahedral topology, write `mesh.h5` with vertex/edge/surface/cell relations |

## Output Schema (mesh.h5 /topology/)

| Dataset | Shape | Description |
|---------|-------|-------------|
| `vertex_to_coord` | float64[n_vertex, 3] | Vertex coordinates |
| `edge_to_vertex` | int64[n_edge, 2] | Edge → 2 vertices (signed) |
| `surface_to_edge` | int64[n_surface, 4] | Surface → 4 edges (signed, CCW) |
| `cell_to_surface` | int64[n_cell, 6] | Cell → 6 surfaces (signed, inward/outward normal) |

Attributes: `n_vertex`, `n_edge`, `n_surface`, `n_cell`.

The preprocessor extends mesh.h5 with GLL geometry (`/field/element/coords`, `dxi_dx`, `jacobian`, `is_pml`).

## Tests

`tests/tools/test_gmsh_to_hdf5.py` — unit tests for extraction + I/O.
`tests/tools/test_gmsh_to_hdf5_integration.py` — end-to-end with GMSH-generated meshes.