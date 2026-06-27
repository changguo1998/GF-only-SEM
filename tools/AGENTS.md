# tools/ — Mesh Conversion and VTK Visualization

## Purpose

Convert GMSH meshes to HDF5. Write VTK files for mesh, partitions, and wavefields.
Tools install as root `pyproject.toml` console scripts. After sourcing `examples/halfspace/setenv.sh`, they are on `PATH`.

## Files

| File | Script | Role |
|------|--------|------|
| `gmsh_to_hdf5.py` | — | Read `.msh`; write `mesh.h5` topology. |
| `mesh2vtk.py` | `mesh2vtk` | Write `mesh.vtk` with mesh and material fields. |
| `partition2vtk.py` | `partition2vtk` | Write `partition_{r}.vtk` for METIS partitions. |
| `wavefield2vtk.py` | `wavefield2vtk` | Write per-step VTK with cell-corner strain. |
| `wavefield2vtk_detail.py` | `wavefield2vtk_detail` | Write per-step VTK with full GLL point strain. |

## VTK Format

All writers produce binary unstructured-grid `.vtk` files for ParaView or VisIt.

### Mesh and Partition

If `/field/element/coords` exists, write detail-mode VTK:

| Section | Contents |
|---------|----------|
| `POINTS` | Mesh vertices plus GLL points. |
| `CELLS` | Hexahedra plus one `VERTEX` cell per GLL point. |
| `CELL_DATA` | Vp, Vs, density, mass, PML damping, PML flag, and rank when present. |
| `POINT_DATA` | Cell-averaged fields at GLL points; mesh vertices get 0. |

If GLL coords are absent, write mesh vertices only.

### Wavefield

`wavefield2vtk`:

- `POINTS`: mesh vertices only.
- `CELL_DATA`: strain averaged over GLL points per cell.
- One file per timestep.

`wavefield2vtk_detail`:

- `POINTS`: mesh vertices plus all GLL points.
- `CELL_DATA`: PML flag on hex cells.
- `POINT_DATA`: GLL strain; mesh vertices get 0.
- One file per timestep.

## Usage

```bash
cd examples/halfspace/
mesh2vtk
partition2vtk
wavefield2vtk
wavefield2vtk_detail
```

Outputs go under `vtk/`.

## mesh.h5 Topology Schema

| Dataset | Shape | Description |
|---------|-------|-------------|
| `vertex_to_coord` | float64[n_vertex, 3] | Vertex coordinates |
| `edge_to_vertex` | int64[n_edge, 2] | Signed edge vertices |
| `surface_to_edge` | int64[n_surface, 4] | Signed CCW surface edges |
| `cell_to_surface` | int64[n_cell, 6] | Signed cell surfaces |

Attributes: `n_vertex`, `n_edge`, `n_surface`, `n_cell`.

Preprocess extends `mesh.h5` with `/field/element/coords`, `dxi_dx`, `jacobian`, and `is_pml`.

## Tests

- `tests/tools/test_gmsh_to_hdf5.py`
- `tests/tools/test_gmsh_to_hdf5_integration.py`
