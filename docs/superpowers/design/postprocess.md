# Postprocess Module — Technical Design

> Parent: [docs/design-decisions.md](../../design-decisions.md)
> Implementation plan: [docs/superpowers/plans/2026-06-08-postprocess.md](../plans/2026-06-08-postprocess.md)

## Goal

Python module that reads HDF5 strain checkpoints from the C++ SEM solver and mesh.h5 for GLL-node geometry, locates receiver positions within mesh elements, performs GLL basis interpolation at receiver positions, and outputs strain Green's functions in HDF5.

## Context

The forward C++ solver checkpoints strain at all GLL nodes per MPI rank. Three independent forward runs (force directions x, y, z) produce 3 sets of checkpoint files in separate directories. Postprocess reads these checkpoints together with mesh.h5 (topology + GLL geometry) to extract strain time series at arbitrary receiver positions.

Green's function extraction requires these 3 forward runs per source location (3 orthogonal force directions x, y, z). Postprocess assembles the 3×6 strain Green's tensor from these checkpoint sets, each containing 6 strain components per timestep (symmetric tensor in Voigt notation).

Strain is the primary scientific output — no displacement integration in postprocess.

## Data Flow

```
mesh.h5 (topology + /field/element/coords + dxi_dx + is_pml) ──┐
wavefields/x/record_{r}.h5 ────────────────────────────────────┤  (force direction x, N rank files)
wavefields/y/record_{r}.h5 ────────────────────────────────────┤  (force direction y, N rank files)
wavefields/z/record_{r}.h5 ────────────────────────────────────┤  (force direction z, N rank files)
receivers.csv ─────────────────────────────────────────────────┤
                                                                ↓
                                                          postprocess (Python)
                                                          ├── read mesh.h5 for GLL coords, dξ/dx, is_pml
                                                          ├── read record files from 3 wavefields/ directories,
                                                          │   merge by global element ID
                                                          ├── validate time alignment across 3 record sets (dt, nsteps, n_cell)
                                                          ├── KD-tree spatial index over non-PML element centroids
                                                          ├── point-in-hexahedron search (Newton iteration with dξ/dx)
                                                          ├── GLL basis interpolation at receiver position
                                                          ├── Green's tensor assembly (3 force directions → 3×6 strain components)
                                                          ↓
                                                     greenfun/tile_{i}.h5
```

mesh.h5 is produced by the converter (topology) and extended by the preprocessor (GLL coords, dxi_dx, is_pml flag).

Three forward runs produce 3 sets of per-rank record files in `wavefields/{x,y,z}/`. Each set contains `record_{r}.h5` for ranks 0..N-1, identified by directory name.

## Architecture

```
mesh.h5 + checkpoint files
         │
         ▼
  CheckpointReader  — reads strain dataset + attrs from checkpoint files
  GeometryReader    — reads /field/element/coords, dxi_dx from mesh.h5
         │
         ▼
  Time alignment validation  — dt, nsteps, n_cell must match across 3 runs
         │
         ▼
  Spatial index (KD-tree)  — non-PML element centroids from GLL coords
         │
         ▼
  Point-in-hexahedron search (Newton iteration)  — uses dξ/dx
         │
         ▼
  GLL basis interpolation  — evaluate strain at receiver position
         │
         ▼  (repeat for 3 runs: force directions x, y, z)
Green's tensor assembly  — 3 force directions × 6 strain components per timestep
          │
          ▼
  GFWriter  — writes greenfun/tile_{i}.h5 (spatial tiles by lat/lon bounding box)
```

### Element Lookup Strategy

Receivers are arbitrary 3D points. Two-phase search:

1. **Coarse**: KD-tree over non-PML element centroids (computed from GLL node coords from mesh.h5) → candidate elements. For tiled mode, build one KD-tree per spatial block.
2. **Fine**: Newton iteration in natural coordinates (ξ, η, ζ) ∈ [-1, 1]³ using precomputed dξ/dx from mesh.h5 → exact containing element + natural coordinates.

Once the containing element is found, GLL Lagrange basis `l_i(ξ)·l_j(η)·l_k(ζ)` interpolates strain from the element's NGLL³ strain tensor to the receiver position.

Receivers placed in PML regions receive a "no containing element found" error — the KD-tree excludes PML elements entirely.

## Constraints

- Python 3.10+
- Dependencies: numpy, h5py, scipy, click (CLI), pytest
- NGLL derived from mesh.h5 array shapes or checkpoint strain shape
- Hexahedral elements only (GMSH-generated)
- Receivers are arbitrary 3D points (PML regions excluded)

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
| /partition/n_ranks | attr int32 | Number of checkpoint files to expect |

NGLL = N+1 is extracted from shape[1] of any element dataset.

### PML Exclusion

PML elements are excluded from the Green's function library. The spatial index (KD-tree) only indexes non-PML elements. PML identification comes from mesh.h5 `/field/element/is_pml`, a boolean flag set by the preprocessor: elements with all surface boundaries tagged as absorbing (tag=2 in `/field/surface/boundary_tag`) are marked PML.

Receivers placed in PML regions will produce a "no containing element found" error.

### Time Alignment Validation

Before processing any receivers, postprocess validates that all 3 checkpoint sets (fx, fy, fz) have identical time parameters. The `dt`, `nsteps`, and `n_cell` attributes from rank 0 of each set are compared. If any mismatch is found, postprocess aborts with an error message listing the mismatched values, for example:

    Time alignment mismatch:
      fx: dt=0.01 nsteps=1000 n_cell=512
      fy: dt=0.01 nsteps=1000 n_cell=512
      fz: dt=0.02 nsteps=500  n_cell=512
    ERROR: fz differs in dt, nsteps

### Strain: Smoothed

The strain in checkpoint files is L2-smoothed — globally projected to be C⁰ continuous across element boundaries. This eliminates the element-boundary discontinuity inherent in SEM strain fields, so no strain averaging or handling of multi-valued GLL nodes is needed at shared faces. Postprocess interpolates the smoothed strain field directly.

### Checkpoint Files — Strain Source

Per MPI rank, one file per forward run:

```
wavefields/{direction}/record_{r}.h5
├── attrs:
│   ├── rank               : int32
│   ├── source_direction   : int32           # 0=x, 1=y, 2=z
│   ├── dt                  : float64
│   ├── checkpoint_interval : int32
│   ├── nsteps              : int32
│   └── current_step        : int32
├── local_element_ids       : int64[n_elem_local]    ← 1-based global element IDs
├── strain                  : {precision}[n_checkpoints, n_elem_local, NGLL, NGLL, NGLL, 6]
│                                                       └── εxx,εyy,εzz,εxy,εxz,εyz
└── /restart/               ← (u,v,a) state, used only for resume
    ├── displacement        : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
    ├── velocity            : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
    └── acceleration        : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
```

Postprocess combines all rank files: maps `local_element_ids` → global element IDs, merges strain into a unified `[n_checkpoints, n_cell, NGLL, NGLL, NGLL, 6]` view indexed by global element ID.

### Spatial Tiling

For production runs, merging the full domain strain into memory is infeasible (TB-scale). The postprocessor optionally partitions the domain into spatial blocks (~50km × 50km in XY) and writes one Green's function library file per block. A receiver's GF is loaded by reading only the block file that contains its XY coordinates.

Spatial tiling is performed entirely by the postprocessor — the forward solver writes standard per-rank record files and knows nothing about blocks. For small domains where the full strain fits in memory, tiling is skipped automatically.

Three modes:
1. **Small domain** (< available RAM): merge all record files, interpolate all receivers, write one tile
2. **Tiled domain** (TB-scale): partition domain into XY blocks, process one block at a time — each block reads relevant checkpoint slices, interpolates receivers in that block, writes block_gf_{xmin}_{ymin}.h5
3. **Single receiver**: only load strain from the elements needed for that receiver — skip full domain merge and tiling entirely

### Receiver Input

```
receivers.csv
# Format: name,x,y,z
R001,1000.0,2000.0,3000.0
R002,1500.0,2500.0,3500.0
```

### Green's Function Output

Strain Green's function library — spatially tiled output, each tile covering a lat/lon bounding box:

```
greenfun/
├── tile_0.h5
│   ├── attrs:
│   │   ├── description         : string
│   │   ├── version             : string
│   │   ├── nreceivers          : int32
│   │   ├── ngll                : int32          ← NGLL = N+1
│   │   ├── minlat, maxlat      : float64        ← tile spatial extent
│   │   └── minlon, maxlon      : float64
│   ├── /receivers/
│   │   ├── positions           : float64[nrecv, 3]
│   │   ├── names               : string[nrecv]
│   │   └── element_ids         : int64[nrecv]   ← global element ID (1-based)
│   ├── /time/
│   │   ├── t                   : float64[nt]
│   │   └── dt                  : float64
│   └── /waveforms/
│       ├── fx/                                   # Force in +x
│       │   ├── recv_0001/
│       │   │   ├── strain_xx   : float64[nt]
│       │   │   └── ...
│       │   └── ...
│       ├── fy/                                   # Force in +y
│       └── fz/                                   # Force in +z
├── tile_1.h5
└── ...
```

Three modes:
1. **Small domain** (< available RAM): merge all record files, interpolate all receivers, write one tile
2. **Tiled domain** (TB-scale): tile domain into spatial blocks, process one block at a time
3. **Single receiver**: only load strain from the elements needed for that receiver

## CLI

```
python -m postprocess mesh.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ receivers.csv -o greenfun/
```

| Arg | Description |
|-----|-------------|
| `mesh.h5` | positional, geometry file (topology + GLL coords + dxi_dx + is_pml) |
| `--fx dir` | directory containing fx-direction record files |
| `--fy dir` | directory containing fy-direction record files |
| `--fz dir` | directory containing fz-direction record files |
| `receivers.csv` | positional, receiver positions |
| `-o dir` | output directory for Green's function tiles (default: greenfun/) |
| `--tile-size N` | receivers per spatial tile (default: 1000) |

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
    ├── conftest.py         — Synthetic checkpoint fixtures
    ├── test_reader.py, test_geometry.py, test_index.py, test_search.py
    ├── test_interpolate.py, test_assembly.py, test_writer.py, test_cli.py
```