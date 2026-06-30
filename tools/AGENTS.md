# tools/ — Mesh Conversion and VTK Visualization

## Purpose

Convert GMSH meshes to HDF5. Write VTK files for mesh, partitions, and wavefields.
Python scripts install as root `pyproject.toml` console scripts. C++ accelerated
binaries build via CMake and land in `bin/` (`gf_model2vtk`, `gf_partition2vtk`,
`gf_wavefield2vtk`).

## Files

### Python (always available)

| File | Script | Role |
|------|--------|------|
| `gmsh_to_hdf5.py` | — | Read `.msh`; write `model.h5` topology. |
| `model2vtk.py` | `model2vtk` | Write `model.vtk` with mesh and material fields. |
| `partition2vtk.py` | `partition2vtk` | Write `partition_{r}.vtk` for METIS partitions. |
| `wavefield2vtk.py` | `wavefield2vtk` | Write per-step VTK with cell-corner strain. |
| `wavefield2vtk_detail.py` | `wavefield2vtk_detail` | Write per-step VTK with full GLL point strain. |

### C++ accelerated (built via CMake, `bin/gf_*2vtk`)

| Binary | Equivalent Python | Speed-up via |
|--------|------------------|--------------|
| `gf_model2vtk` | `model2vtk.py` | OpenMP (cells, vertices, GLL sub-cells) |
| `gf_partition2vtk` | `partition2vtk.py` | OpenMP (parallel VTK write per rank) |
| `gf_wavefield2vtk` | `wavefield2vtk.py` | OpenMP (per-vertex strain accumulation) |

C++ tools produce byte-identical VTK output. Fall back to Python if binary not found.
HDF5 C library is not thread-safe → all HDF5 reads serial, compute inside parallel regions.

## VTK Format

All writers produce binary unstructured-grid `.vtk` files for ParaView or VisIt.

### Mesh and Partition

If `/field/element/coords` exists, write detail-mode VTK with GLL sub-cells:

| Section | Contents |
|---------|----------|
| `POINTS` | Mesh vertices plus GLL points. |
| `CELLS` | Original hexahedra + GLL-derived edge (LINE), face (QUAD), and sub-volume (HEX) cells for proper ParaView interpolation. |
| `CELL_TYPES` | 12 (hex), 3 (line), 9 (quad). |
| `CELL_DATA` | Vp, Vs, density, mass, PML damping, PML flag, and rank when present. Hex values broadcast to child GLL cells via `gll_elem_map`. |
| `POINT_DATA` | Cell-averaged fields at GLL points; mesh vertices get 0. |

GLL sub-cell topology per hex element (NGLL=5 → 125 GLL points):

- 12×(NGLL−1) = 48 edge LINE cells
- 6×(NGLL−1)² = 96 face QUAD cells
- (NGLL−1)³ = 64 sub-volume HEX cells

If GLL coords are absent, write mesh vertices only (non-detail mode).

### Wavefield

`wavefield2vtk`:

- `POINTS`: mesh vertices only.
- `CELL_DATA`: strain averaged over GLL points per cell.
- One file per timestep.

`wavefield2vtk_detail`:

- `POINTS`: mesh vertices plus all GLL points.
- `CELLS`: hex cells + GLL-derived edge/face/sub cells.
- `CELL_DATA`: PML flag broadcast to all GLL sub-cells.
- `POINT_DATA`: GLL strain (18 fields: 6 Voigt × 3 directions); mesh vertices get 0.
- One file per timestep.

## Usage

```bash
cd examples/halfspace/
# Python (always available)
model2vtk
partition2vtk
wavefield2vtk

# C++ accelerated (after cmake --build, in PATH from setenv.sh)
gf_model2vtk
gf_partition2vtk
gf_wavefield2vtk
```

Outputs go under `vtk/`.

## Build

Built automatically as part of the project CMake:

```bash
cd build
cmake ..           # configures tools/cpp/
cmake --build .    # builds gf_model2vtk, gf_partition2vtk, gf_wavefield2vtk
```

## model.h5 Topology Schema

| Dataset | Shape | Description |
|---------|-------|-------------|
| `vertex_to_coord` | float64[n_vertex, 3] | Vertex coordinates |
| `edge_to_vertex` | int64[n_edge, 2] | Signed edge vertices |
| `surface_to_edge` | int64[n_surface, 4] | Signed CCW surface edges |
| `cell_to_surface` | int64[n_cell, 6] | Signed cell surfaces |

Attributes: `n_vertex`, `n_edge`, `n_surface`, `n_cell`.

Preprocess extends `model.h5` with `/field/element/coords`, `dxi_dx`, `jacobian`, and `is_pml`.

## Tests

- `tests/tools/test_gmsh_to_hdf5.py`
- `tests/tools/test_gmsh_to_hdf5_integration.py`
