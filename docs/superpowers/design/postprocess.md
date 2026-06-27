# Postprocess Module — Technical Design

> Parent: [docs/design-decisions.md](../../design-decisions.md)
> Implementation plan: [docs/superpowers/plans/2026-06-08-postprocess.md](../plans/2026-06-08-postprocess.md)

## Goal

Python module that reads shallow mesh-vertex strain snapshots from the three C++ SEM forward runs (`x`, `y`, `z` force directions), assembles the full 3×6 strain Green's tensor at recorded mesh vertices, and writes horizontally tiled HDF5 output.

No receiver positions — output is the configured shallow full-volume mesh-vertex field.

## Context

The forward solver still computes the full SEM domain at all GLL nodes. To keep the Green's function library tractable, forward records only a configured shallow volume:

- mesh vertices only (not interior GLL points),
- `depth <= record_depth_actual_m`, where the actual depth is snapped to a horizontal spectral-element face at or deeper than `record_depth_max_m`,
- non-PML only.

Three independent forward runs (force directions x, y, z) produce per-rank record files under `wavefields/{x,y,z}/`. Postprocess merges these files by global mesh vertex ID and stacks the 3 force directions into a 3×6 strain Green's tensor.

Strain is the primary scientific output — no displacement integration in postprocess.

## Data Flow

```
mesh.h5 (/topology/vertex_to_coord) ────────────────────────────┐
config.h5 (/simulation green_tile_size_m + timing metadata) ─────┤
wavefields/x/record_{r}.h5 ──────────────────────────────────────┤
wavefields/y/record_{r}.h5 ──────────────────────────────────────┤
wavefields/z/record_{r}.h5 ──────────────────────────────────────┤
                                                                 ↓
                                                           postprocess (Python)
                                                           ├── read mesh vertices by vertex_id
                                                           ├── read record files from 3 directions
                                                           ├── merge by global vertex ID
                                                           ├── validate timing, basis, depth, vertex sets
                                                           ├── stack 3 directions → [nt, n_vertex, 6, 3]
                                                           └── write horizontal tiles
                                                                 ↓
                                                         greenfun/tile_x{i}_y{j}.h5
```

## Architecture

```
mesh.h5 + config.h5 + shallow snapshot files
         │
         ▼
  RecordReader     — reads vertex_ids + strain dataset + attrs from record files
  GeometryReader   — reads /topology/vertex_to_coord from mesh.h5
         │
         ▼
  Validation        — source_direction, timing, basis, depth, vertex_id consistency
         │
         ▼
  Green's tensor assembly — 3 force directions × 6 strain components at mesh vertices
         │
         ▼
  GFWriter          — writes greenfun/tile_x{i}_y{j}.h5 by horizontal x/y bins
```

## Constraints

- Python 3.10+
- Dependencies: numpy, h5py, click, pytest
- No receiver positions, no receiver search, no point interpolation.
- Output basis is mesh vertices only: `basis = "mesh_vertices"`.
- The SEM compute basis remains GLL; only recorded Green's function output is downsampled to element-corner mesh vertices.
- PML vertices/elements are excluded by the forward recording map prepared during preprocessing.

## Input Files

### mesh.h5 — Vertex Coordinates

| Dataset | Shape | Usage |
|---------|-------|-------|
| `/topology/vertex_to_coord` | float64[n_vertex, 3] | Coordinates for recorded `vertex_ids` |

Postprocess does not need GLL coordinates, `dxi_dx`, or point-in-element search.

### config.h5 — Timing + Tiling Metadata

| Attribute | Usage |
|-----------|-------|
| `/simulation/solver_dt` | Solver timestep |
| `/simulation/output_dt_s` | Snapshot interval |
| `/simulation/snapshot_stride` | Solver steps per snapshot |
| `/simulation/nsteps` | Total solver steps |
| `/simulation/record_depth_max_m` | Requested maximum recorded depth |
| `/simulation/record_depth_actual_m` | Spectral-element face depth actually used |
| `/simulation/green_tile_size_m` | Horizontal tile width in x and y |

### Snapshot Files — Strain Source

Per MPI rank, one file per forward run:

```
wavefields/{direction}/record_{r}.h5
├── attrs:
│   ├── rank                    : int32
│   ├── source_direction        : string           # "x", "y", or "z"
│   ├── basis                   : string           # "mesh_vertices"
│   ├── record_depth_max_m      : float64
│   ├── record_depth_actual_m   : float64
│   └── excludes_pml            : bool
├── vertex_ids                  : int64[n_record_vertices]   # global mesh vertex IDs, 1-based
└── strain                      : {precision}[n_snapshots, n_record_vertices, 6]
                                      # εxx, εyy, εzz, εxy, εxz, εyz
```

Restart files are not postprocess inputs.

## Validation

Before processing, postprocess validates that all three direction sets have identical run metadata:

- `basis == "mesh_vertices"`,
- matching `record_depth_max_m` and `record_depth_actual_m`,
- matching `solver_dt`, `snapshot_stride`, and number of snapshots,
- matching `vertex_ids` set after merging all ranks,
- correct `source_direction` for each directory.

Mismatch aborts with a message listing the differing values.

## Green's Tensor Assembly

Input after merge:

```
strain_fx : [nt, n_vertex, 6]
strain_fy : [nt, n_vertex, 6]
strain_fz : [nt, n_vertex, 6]
```

Output:

```
greens_tensor : float32[nt, n_vertex, 6, 3]
```

`Shape[-2]` is strain component. `Shape[-1]` is force direction (`x`, `y`, `z`).

## Horizontal Tiling

Postprocess tiles by horizontal x/y bins using `green_tile_size_m` from `config.py` / `config.h5`.

For each recorded vertex coordinate `(x, y, z)`:

```
tile_x = floor((x - xmin) / green_tile_size_m)
tile_y = floor((y - ymin) / green_tile_size_m)
```

Each tile contains all recorded depths for the vertices whose x/y fall in the tile. Tiles do not split the time axis; HDF5 dataset chunking handles time-block access.

## Green's Function Output

```
greenfun/
├── tile_x000_y000.h5
├── tile_x001_y000.h5
└── ...
```

Each tile:

```
tile_x{i}_y{j}.h5
├── attrs:
│   ├── version                 : string
│   ├── basis                   : "mesh_vertices"
│   ├── tile_x_index            : int32
│   ├── tile_y_index            : int32
│   ├── x_min_m, x_max_m        : float64
│   ├── y_min_m, y_max_m        : float64
│   ├── z_min_m, z_max_m        : float64
│   ├── record_depth_max_m      : float64
│   ├── record_depth_actual_m   : float64
│   └── excludes_pml            : bool
├── /time/
│   ├── t                       : float64[nt]
│   ├── dt                      : float64
│   └── nsteps                  : int32
├── /mesh/
│   └── vertex_ids              : int64[n_tile_vertices]
└── /field/
    └── greens_tensor           : float32[nt, n_tile_vertices, 6, 3]
```

Coordinates are not duplicated in Green's function files. Consumers recover coordinates from `mesh.h5` using `vertex_ids`.

## CLI

```
gf-postprocess mesh.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ -o greenfun/
```

| Arg | Description |
|-----|-------------|
| `mesh.h5` | positional, topology file with `/topology/vertex_to_coord` |
| `--fx dir` | directory containing x-force record files |
| `--fy dir` | directory containing y-force record files |
| `--fz dir` | directory containing z-force record files |
| `-o dir` | output directory for Green's function tiles (default: `greenfun/`) |

Horizontal tile size comes from `config.h5` (`/simulation/green_tile_size_m`), not a CLI override.

## File Layout

```
postprocess/
├── pyproject.toml
├── src/gf_post/
│   ├── __init__.py         — package exports, version
│   ├── reader.py           — RecordReader + GeometryReader
│   ├── assembly.py         — Green's tensor assembly from 3 runs
│   ├── writer.py           — Strain GF output HDF5 writer
│   └── cli.py              — CLI entry point
└── tests/
    ├── conftest.py         — synthetic snapshot fixtures
    └── test_reader.py      — reader tests
```
