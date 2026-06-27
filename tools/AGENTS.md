# tools/ — Mesh Conversion & VTK Visualization

## Purpose

Tools for mesh format conversion (GMSH → HDF5) and VTK visualization
of mesh, partition, and wavefield data. All tools are installed as
console_scripts (entry points) via the root `pyproject.toml` and are
available on `PATH` after sourcing `examples/halfspace/setenv.sh`.

## Files

| File | Console Script | Responsibility |
|------|---------------|---------------|
| `gmsh_to_hdf5.py` | — (imported) | Read GMSH `.msh`, extract hexahedral topology, write `mesh.h5` with vertex/edge/surface/cell relations |
| `mesh2vtk.py` | `mesh2vtk` | Convert `mesh.h5` + partitions/ to `mesh.vtk` with material fields |
| `partition2vtk.py` | `partition2vtk` | Convert per-rank partition files to `partition_{r}.vtk` for visualising METIS decomposition |
| `wavefield2vtk.py` | `wavefield2vtk` | Convert strain snapshots to per-timestep VTK (cell-corner data only) |
| `wavefield2vtk_detail.py` | `wavefield2vtk_detail` | Convert strain snapshots to per-timestep VTK with full GLL point data |

## VTK Output Format

All VTK writers produce **binary unstructured grid** files (`.vtk`),
viewable in ParaView / VisIt.

### Mesh & Partition Writers (`mesh2vtk`, `partition2vtk`)

When GLL coordinates exist in `mesh.h5` (`/field/element/coords`),
the writers produce **detail-mode** VTK:

| VTK Section | Contents |
|-------------|----------|
| `POINTS` | Mesh vertices (2601 for 16×16×8 mesh) + all/partition-local GLL points (125 per cell, NGLL=5) |
| `CELLS` | Hexahedra (type 12) referencing mesh vertices + VERTEX cells (type 1) for each GLL point |
| `CELL_DATA` | Per-hex fields: Vp, Vs, Density, Mass, PML_Damping, PML_flag (and Rank for partition) |
| `POINT_DATA` | Same fields, cell-averaged at GLL points; mesh vertices get 0 |

Fallback: if GLL coords are absent, writes mesh-vertices-only VTK (original behaviour).

### Wavefield Writers

**`wavefield2vtk`** — cell-corner mode:

- `POINTS` = mesh vertices only
- `CELL_DATA` = strain components averaged over GLL points per cell
- One `.vtk` file per timestep

**`wavefield2vtk_detail`** — full GLL mode:

- `POINTS` = mesh vertices + all GLL points
- `CELL_DATA` = PML_flag (hex cells only)
- `POINT_DATA` = strain at each GLL point (mesh vertices → 0)
- One `.vtk` file per timestep

## Usage

```bash
# After sourcing setenv.sh, all tools are on PATH:
cd examples/halfspace/
mesh2vtk                   # → vtk/mesh.vtk
partition2vtk              # → vtk/partition_{r}.vtk
wavefield2vtk              # → vtk/wavefield_{step}.vtk  (cell corners)
wavefield2vtk_detail       # → vtk/wavefield_{step}.vtk  (GLL points)
```

## Output Schema (mesh.h5 /topology/)

| Dataset | Shape | Description |
|---------|-------|-------------|
| `vertex_to_coord` | float64[n_vertex, 3] | Vertex coordinates |
| `edge_to_vertex` | int64[n_edge, 2] | Edge → 2 vertices (signed) |
| `surface_to_edge` | int64[n_surface, 4] | Surface → 4 edges (signed, CCW) |
| `cell_to_surface` | int64[n_cell, 6] | Cell → 6 surfaces (signed, inward/outward normal) |

Attributes: `n_vertex`, `n_edge`, `n_surface`, `n_cell`.

The preprocessor extends `mesh.h5` with GLL geometry (`/field/element/coords`,
`dxi_dx`, `jacobian`, `is_pml`).

## Tests

- `tests/tools/test_gmsh_to_hdf5.py` — unit tests for topology extraction + I/O
- `tests/tools/test_gmsh_to_hdf5_integration.py` — end-to-end with GMSH-generated meshes
