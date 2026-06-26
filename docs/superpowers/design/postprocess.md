# Postprocess Module — Technical Design

> Parent: [docs/design-decisions.md](../../design-decisions.md)
> Implementation plan: [docs/superpowers/plans/2026-06-08-postprocess.md](../plans/2026-06-08-postprocess.md)

## Goal

Python module that reads HDF5 strain snapshots from the C++ SEM solver and mesh.h5 for GLL-node geometry, assembles the full 3×6 strain Green's tensor at every GLL node, and writes spatially tiled HDF5 output.

No receiver positions — output is the full GLL-node field.

## Context

Three independent forward runs (force directions x, y, z) produce 3 sets of snapshot files in separate directories (wavefields/x/, wavefields/y/, wavefields/z/). Postprocess reads these snapshot files together with mesh.h5 (topology + GLL geometry) and assembles the strain Green's tensor at all GLL nodes.

Green's function extraction requires these 3 forward runs per source location (3 orthogonal force directions x, y, z). Postprocess assembles the 3×6 strain Green's tensor from these snapshot sets, each containing 6 strain components per timestep (symmetric tensor in Voigt notation).

Strain is the primary scientific output — no displacement integration in postprocess.

## Data Flow

```
mesh.h5 (topology + /field/element/coords + dxi_dx + is_pml) ──┐
wavefields/x/record_{r}.h5 ────────────────────────────────────┤
wavefields/y/record_{r}.h5 ────────────────────────────────────┤
wavefields/z/record_{r}.h5 ────────────────────────────────────┤
                                                                ↓
                                                          postprocess (Python)
                                                          ├── read mesh.h5 for GLL coords, dxi_dx, is_pml
                                                          ├── read record files from 3 wavefields/ directories,
                                                          │   merge by global element ID
                                                          ├── validate time alignment across 3 record sets (dt, nsteps, n_cell)
                                                          ├── Green's tensor assembly (3 force directions → 3×6 strain components)
                                                          ↓
                                                     greenfun/tile_{i}.h5
```

mesh.h5 is produced by the converter (topology) and extended by the preprocessor (GLL coords, dxi_dx, is_pml flag).

Three forward runs produce 3 sets of per-rank record files in `wavefields/{x,y,z}/`. Each set contains `record_{r}.h5` for ranks 0..N-1, identified by directory name.

## Architecture

```
mesh.h5 + snapshot files
         │
         ▼
  CheckpointReader  — reads strain dataset + attrs from snapshot files
  GeometryReader    — reads /field/element/coords, dxi_dx from mesh.h5
         │
         ▼
  Time alignment validation  — NGLL, source_direction, n_cell via record + config.h5 metadata
         │
         ▼
  Green's tensor assembly  — 3 force directions × 6 strain components at every GLL node
         │
         ▼
  GFWriter  — writes greenfun/tile_{i}.h5 (tiled by element range)
```

### Element Data

The postprocess assembles Green's tensor directly at every GLL node — no
point-in-hexahedron search or interpolation needed.

### PML Exclusion

PML elements are excluded from the Green's function output. PML identification comes from mesh.h5 `/field/element/is_pml`.

## Constraints

- Python 3.10+
- Dependencies: numpy, h5py, scipy, click (CLI), pytest
- NGLL derived from mesh.h5 array shapes or snapshot strain shape
- No receiver positions — output is the full GLL-node field
- Hexahedral elements only (GMSH-generated)

## Input Files

### mesh.h5 — Geometry Source

Geometry comes from mesh.h5, which the converter creates (topology) and the preprocessor extends (GLL coords, dxi_dx, element flags).

| Dataset | Shape | Usage |
|---------|-------|-------|
| /topology/vertex_to_coord | float64[n_vertex, 3] | Hex corner coordinates |
| /topology/cell_to_surface | int64[n_cell, 6] | Element topology |
| /field/element/coords | float64[n_cell, NGLL, NGLL, NGLL, 3] | GLL node (x,y,z) per element |
| /field/element/dxi_dx | float64[n_cell, NGLL, NGLL, NGLL, 3,3] | Newton iteration for point-in-hex |
| /field/element/is_pml | int8[n_cell] | 1 = PML element, 0 = ordinary |
| /partition/n_ranks | attr int32 | Number of snapshot files to expect |
NGLL = N+1 is extracted from shape[1] of any element dataset.

### PML Exclusion

PML elements are excluded from the Green's function output. The `/field/element/is_pml` flag from mesh.h5 is used to skip PML elements during tensor assembly and tiling.

Elements with all surface boundaries tagged as absorbing (tag=2 in `/field/surface/boundary_tag`) are marked PML.

### Time Alignment Validation

Before processing, postprocess validates that all 3 snapshot sets (fx, fy, fz) have identical run metadata. The NGLL and local_element_ids count from rank 0 of each set are compared. Run metadata (solver_dt, nsteps, snapshot_stride, n_cell) is read from config.h5 and validated across the three direction runs. If any mismatch is found, postprocess aborts with an error message listing the mismatched values, for example:

```
Time alignment mismatch:
  fx: dt=0.01 nsteps=1000 n_cell=512
  fy: dt=0.01 nsteps=1000 n_cell=512
  fz: dt=0.02 nsteps=500  n_cell=512
ERROR: fz differs in dt, nsteps
```

### Strain: Smoothed

The strain in snapshot files is L2-smoothed — globally projected to be C⁰ continuous across element boundaries. This eliminates the element-boundary discontinuity inherent in SEM strain fields, so no strain averaging or handling of multi-valued GLL nodes is needed at shared faces. Postprocess interpolates the smoothed strain field directly.

### Snapshot Files — Strain Source

Per MPI rank, one file per forward run:

```
wavefields/{direction}/record_{r}.h5
├── attrs:
│   ├── rank               : int32
│   ├── source_direction   : string           # "x", "y", or "z"
│   └── ngll                : int32           # N+1
├── local_element_ids       : int64[n_elem_local]    ← 1-based global element IDs
├── strain                  : {precision}[n_snapshots, n_elem_local, NGLL, NGLL, NGLL, 6]
│                                                       └── εxx,εyy,εzz,εxy,εxz,εyz
└── /restart/               ← (u,v,a) state, used only for resume
    ├── displacement        : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
    ├── velocity            : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
    └── acceleration        : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
```

Postprocess combines all rank files: maps `local_element_ids` → global element IDs, merges strain into a unified `[n_snapshots, n_cell, NGLL, NGLL, NGLL, 6]` view indexed by global element ID.

### Spatial Tiling

For production runs, the full Green's tensor may not fit in memory (TB-scale). The postprocessor tiles the domain by element range and writes one Green's function file per tile.

Tiling is by contiguous element index ranges — each tile covers a batch of elements with their full GLL-node tensor data.

Two modes:

1. **Small domain** (< available RAM): assemble at all elements, write one tile
1. **Tiled domain**: partition by element range, process one batch at a time

### Green's Function Output

Strain Green's function library — tiled by element range:

```
greenfun/
├── tile_0.h5
│   ├── attrs:
│   │   ├── description         : string
│   │   ├── version             : string
│   │   ├── ncell               : int32          ← elements in this tile
│   │   ├── ngll                : int32          ← NGLL = N+1
│   │   └── elem_start          : int32
│   ├── /time/
│   │   ├── t                   : float64[nt]
│   │   ├── dt                  : float64
│   │   ├── nsteps              : int32
│   └── /field/
│       ├── greens_tensor       : float32[nt, n_local_cell, NGLL, NGLL, NGLL, 6, 3]
│       └── coords              : float32[n_local_cell, NGLL, NGLL, NGLL, 3]
├── tile_1.h5
└── ...
```

## CLI

```
gf-postprocess mesh.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ -o greenfun/
```

| Arg | Description |
|-----|-------------|
| `mesh.h5` | positional, geometry file (topology + GLL coords + dxi_dx + is_pml) |
| `--fx dir` | directory containing fx-direction record files |
| `--fy dir` | directory containing fy-direction record files |
| `--fz dir` | directory containing fz-direction record files |
| `-o dir` | output directory for Green's function tiles (default: greenfun/) |
| `--tile-elems N` | max elements per tile (default: 100) |

## File Layout

```
postprocess/
├── pyproject.toml
├── src/gf_post/
│   ├── __init__.py         — package exports, version
│   ├── reader.py           — CheckpointReader (strain) + GeometryReader (mesh.h5)
│   ├── geometry.py         — GLL: nodes, weights, Lagrange basis
│   ├── index.py            — Spatial index (KD-tree over element centroids)
│   ├── search.py           — Point-in-hexahedron (Newton iteration using dξ/dx)
│   ├── interpolate.py      — GLL interpolation of strain at arbitrary point
│   ├── assembly.py         — Green's tensor assembly from 3 runs
│   ├── writer.py           — Strain GF output HDF5 writer
│   └── cli.py              — CLI entry point
└── tests/
    ├── conftest.py         — Synthetic snapshot fixtures
    ├── test_reader.py, test_geometry.py, test_index.py, test_search.py
    ├── test_interpolate.py, test_assembly.py, test_writer.py, test_cli.py
```
