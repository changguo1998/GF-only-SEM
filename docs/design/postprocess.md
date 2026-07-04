# Postprocess Module — Technical Design

> Parent: [../design-decisions.md](../design-decisions.md)

## Goal

Read strain snapshots from three SEM runs (`x`, `y`, `z`). Build `3×6` strain Green tensors at recorded locations. Write self-contained horizontal HDF5 tiles — two editions: mesh-vertex (trilinear interpolation) and full-GLL (spectral interpolation).

No receivers. Output is the configured shallow, non-PML region. Each tile is fully standalone (no dependency on `model.h5` for reading). Tiles include mesh topology, material, travel times (future), and Green's tensor in time-series-first layout.

## Context

Forward computes the full GLL SEM domain. It records at two granularities controlled by `record_mode` in `config.py`:

- **`"vertices"`**: mesh vertices only (GLL corner nodes), depth ≤ `record_depth_actual_m`, no PML vertices. → compact record, supports mesh-only edition.
- **`"gll"`**: all NGLL³ GLL nodes of cells fully within recording depth (non-PML). → full record, supports both editions (mesh extracted from GLL corners).

Each force run writes rank files under `wavefields/{x,y,z}/`. Postprocess merges by global cell ID (gll mode) or vertex ID (vertices mode), then stacks force directions.

## Record File Format

### `record_mode = "vertices"`

```
record_{r}.h5
├── attrs: rank, source_direction, basis="mesh_vertices", excludes_pml, record_depth_*
├── /vertex_ids  : int64[n_vertices]                 # global, 1-based
├── /strain      : float32[n_snapshots, n_vertices, 6]
├── /displacement: float32[n_snapshots, n_vertices, 3]
├── /velocity    : float32[n_snapshots, n_vertices, 3]
└── /acceleration: float32[n_snapshots, n_vertices, 3]
```

### `record_mode = "gll"`

```
record_{r}.h5
├── attrs: rank, source_direction, basis="gll_nodes", excludes_pml, record_depth_*, NGLL
├── /cell_ids     : int64[n_cells]                    # global, 1-based
├── /strain       : float32[n_snapshots, n_cells, NGLL, NGLL, NGLL, 6]
├── /displacement : float32[n_snapshots, n_cells, NGLL, NGLL, NGLL, 3]
├── /velocity     : float32[n_snapshots, n_cells, NGLL, NGLL, NGLL, 3]
└── /acceleration : float32[n_snapshots, n_cells, NGLL, NGLL, NGLL, 3]
```

Restart files are not inputs.

## Data Flow

```
model.h5 (/topology, /field/element/{vp,vs,density,jacobian})
config.h5 (/simulation timing, tiles, source, record_mode)
wavefields/{x,y,z}/record_{r}.h5
         │
         ├── auto-detect record_mode
         │
         ├── Mesh-only edition (--mesh)
         │     • merge cell/vertex records
         │     • extract 8 corner GLL nodes per cell → dedup → vertices
         │     • extract corner material → per-vertex
         │     • assemble greens [n_vertex, 3, 6, nt]
         │     • bin cells by interior element index → tiles
         │     • build local cell_to_vertex (0-based)
         │     • recenter coords: x-sx, y-sy, z=z (origin at source horiz, z=0 vert)
         │     • write tile_x{i}_y{j}.h5
         │
         └── Full-GLL edition (--gll)
               • merge cell records
               • keep full per-GLL-node material
               • assemble greens [n_cell, NGLL, NGLL, NGLL, 3, 6, nt]
               • bin cells by interior element index → tiles
               • store GLL basis metadata (gll_xi, gll_weights)
               • recenter cell origins
               • write gll_tile_x{i}_y{j}.h5
```

## Architecture

```
RecordReader     — read record files, auto-detect format
GeometryReader   — read /topology, build cell_to_vertex from X2Y relations
MaterialReader   — read /field/element vp/vs/density at corner GLL nodes
ConfigReader     — read config.h5 (simulation, source, tiles, record_mode)
Validation       — check direction, timing, basis, consistent cells/vertices across runs
Assembly         — stack x/y/z strain → Green tensor (vertex or cell layout)
CellSelector     — identify cells fully in record zone; bin into tiles
MeshTileWriter   — write vertex-based tile_x{i}_y{j}.h5
GLLTileWriter    — write cell-based gll_tile_x{i}_y{j}.h5
CLI              — entry point with --mesh/--gll flags
```

## Tile Schemas

### Mesh-only: `tile_x{i}_y{j}.h5`

```
├── attrs: version="2.0", basis="mesh_vertices", corner_order="gmsh_hex"
│          tile_x_index, tile_y_index
│          x_min_m, x_max_m, y_min_m, y_max_m, z_min_m, z_max_m  (recentered)
│          source_x_m, source_y_m, source_z_m  (original frame, for traceability)
│          record_depth_max_m, record_depth_actual_m, excludes_pml=true
├── /time/dt          : float64            ← output_dt_s (snapshot interval)
├── /time/nt          : int32              ← number of snapshots (t = dt·arange(nt))
├── /mesh/vertex_coords  : float64[n_v, 3]        ← recentered
├── /mesh/cell_to_vertex : int32[n_c, 8]          ← local 0-based indices into vertex_*
├── /mesh/cell_bounds    : float64[n_c, 6]        ← recentered xmin,ymin,zmin,xmax,ymax,zmax
├── /material/vp  : float32[n_v]                  ← per-vertex (from corner GLL nodes)
├── /material/vs  : float32[n_v]
├── /material/rho : float32[n_v]
└── /field/greens_tensor : float32[n_v, 3, 6, nt]  ← chunk=(1, 3, 6, nt), gzip+shuffle
```

**Consumer interpolation (trilinear):**

1. Locate query `(xq, yq, zq)` → scan `cell_bounds` for containing cell.

1. `cell_to_vertex[cell_idx]` → 8 local vertex indices.

1. For each of 8 corners (GMSH order), compute local (xi, eta, zeta) in [0,1]³:

   | corner | (xi,eta,zeta) | trilinear weight |
   |--------|---------------|-------------------|
   | 0 | (0,0,0) | (1-xi)(1-eta)(1-zeta) |
   | 1 | (1,0,0) | xi(1-eta)(1-zeta) |
   | 2 | (1,1,0) | xi·eta·(1-zeta) |
   | 3 | (0,1,0) | (1-xi)eta(1-zeta) |
   | 4 | (0,0,1) | (1-xi)(1-eta)zeta |
   | 5 | (1,0,1) | xi(1-eta)zeta |
   | 6 | (1,1,1) | xi·eta·zeta |
   | 7 | (0,1,1) | (1-xi)eta·zeta |

1. `greens[query_point, dir, comp, :] = Σ_{c=0}^{7} greens[corner_idx[c], dir, comp, :] × weight[c]`.

   This is an approximation of the full SEM spectral solution. For exact spectral interpolation, use the full-GLL edition.

### Full-GLL: `greenfun/gll/gll_tile_x{i}_y{j}.h5`

```
├── attrs: version="2.0", basis="gll_nodes", NGLL, corner_order="gmsh_hex"
│          tile_x_index, tile_y_index
│          x_min_m, x_max_m, y_min_m, y_max_m, z_min_m, z_max_m  (recentered)
│          source_x_m, source_y_m, source_z_m  (original frame)
│          record_depth_max_m, record_depth_actual_m, excludes_pml=true
├── /time/dt        : float64
├── /time/nt        : int32
├── /mesh/gll_xi    : float64[NGLL]          ← GLL node coordinates in reference [-1, 1]
├── /mesh/gll_weights : float64[NGLL]        ← GLL quadrature weights (optional, for integrals)
├── /mesh/cell_origin : float64[n_c, 3]      ← recentered cell min-corner (xmin,ymin,zmin)
├── /mesh/cell_size   : float64[n_c, 3]      ← (dx, dy, dz) per cell
├── /material/vp  : float32[n_c, NGLL, NGLL, NGLL]
├── /material/vs  : float32[n_c, NGLL, NGLL, NGLL]
├── /material/rho : float32[n_c, NGLL, NGLL, NGLL]
└── /field/greens_tensor : float32[n_c, NGLL, NGLL, NGLL, 3, 6, nt]
    ← chunk=(1, NGLL, NGLL, NGLL, 1, 1, min(nt,256)), gzip+shuffle
```

**Consumer interpolation (spectral) — EXACT:**

1. Locate query `(xq, yq, zq)` → find cell via `cell_origin ≤ (xq,yq,zq) < cell_origin + cell_size`.
1. Compute reference coordinates: `xi = 2*(xq - cell_origin_x)/cell_size_x - 1`, similarly for eta, zeta. All in [-1, 1].
1. Evaluate Lagrange polynomials at (xi, eta, zeta):
   - `l_i(xi) = Π_{m≠i} (xi - gll_xi[m]) / (gll_xi[i] - gll_xi[m])` for all NGLL nodes.
   - Similarly `l_j(eta)`, `l_k(zeta)`.
1. Sum: `greens(query, dir, comp, :) = Σ_{i,j,k} greens[cell, i, j, k, dir, comp, :] × l_i(xi) × l_j(eta) × l_k(zeta)`.

This recovers the SEM spectral solution at any point within the cell exactly (machine-precision — no interpolation error).

Primer on GLL evaluation at arbitrary (xi, eta, zeta) is available in the math docs.

## Coordinate System

Tiles use a recentered coordinate frame:

- **Horizontal origin**: source location → `x' = x - source_x`, `y' = y - source_y`.
- **Vertical origin**: `z = 0` (free surface / datum) → `z' = z`.
- Source position in tile frame: `(0, 0, source_z)`.

All tile attributes and mesh datasets use recentered coordinates. Source position in original frame is stored as attrs `source_x_m`, `source_y_m`, `source_z_m` for traceability.

## Material

- **Mesh-only edition**: material extracted from corner GLL nodes of each cell. Since preprocess interpolates material continuously, values at a shared vertex are identical across sharing cells — no averaging needed.
- **Full-GLL edition**: material stored at all NGLL³ GLL nodes per cell, directly from `model.h5 /field/element/`.

## Travel Times

Deferred to future module. Placeholder datasets (`/travel_time/tp`, `/travel_time/ts`) will be added in that module at vertex or GLL-node granularity.

## Cell Selection & Tiling

A cell qualifies for output if:

1. **Fully recorded**: all NGLL³ GLL nodes (including all 8 corners) are in the recorded set.
1. **Non-PML**: interior element (i ≥ pml_xmin, i < nx - pml_xmax; similarly for y, z).
1. **Within record depth**: cell z-max ≤ record_depth_actual_m.

Qualifying cells are binned by interior element index `(i - pml_xmin, j - pml_ymin)` into tiles. Tile sizes come from `config.h5 /simulation/tilex_elements` and `tiley_elements` (element-count tiling).

Alternatively, when `green_tile_size_m` is set in `config.py` (`/simulation/green_tile_size_m` in `config.h5`), spatial binning replaces element binning: vertices are assigned to tiles by `floor((x - xmin) / green_tile_size_m)` and `floor((y - ymin) / green_tile_size_m)`. This produces spatially-uniform tiles independent of mesh discretization, with element-count tiling as the fallback when `green_tile_size_m` is absent or None.

A vertex on a tile boundary (shared by cells in adjacent tiles) is duplicated in both tiles — acceptable for fully standalone tile files. Spatial tiling produces tile boundaries at fixed coordinate intervals; vertices exactly on a boundary fall into the lower tile due to `floor()`.

## Cell-to-Vertex Derivation

`model.h5` topology stores X2Y relations (`cell_to_surface → surface_to_edge → edge_to_vertex`), not direct `cell_to_vertex`. Postprocess reconstructs `cell_to_vertex_global [n_total_cell, 8]` in GMSH canonical order using the same algorithm as `tools/wavefield2vtk.py::build_element_vertex_map`.

The derivation walks each cell's 6 signed surfaces, 4 signed edges per surface, 2 vertices per edge. Sign determines edge traversal direction. The `_HEX_FACES` template maps GMSH local vertex indices (0–7) to face vertex loops.

GLL corner index mapping for material extraction (GLL index = `(i*NGLL + j)*NGLL + k` with k-fast, i-slow):

| GMSH corner | GLL (i,j,k) | GLL flat index |
|-------------|-------------------|-------------------------|
| 0 | (0, 0, 0) | 0 |
| 1 | (NGLL-1, 0, 0) | (NGLL-1)·NGLL² |
| 2 | (NGLL-1, NGLL-1, 0) | (NGLL-1)·(NGLL²+NGLL) |
| 3 | (0, NGLL-1, 0) | (NGLL-1)·NGLL |
| 4 | (0, 0, NGLL-1) | NGLL-1 |
| 5 | (NGLL-1, 0, NGLL-1) | (NGLL-1)·(NGLL²+1) |
| 6 | (NGLL-1, NGLL-1, NGLL-1) | (NGLL-1)·(NGLL²+NGLL+1) |
| 7 | (0, NGLL-1, NGLL-1) | (NGLL-1)·(NGLL+1) |

## CLI

```
gf-postprocess model.h5 config.h5 \
    --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ \
    -o greenfun/ \
    [--mesh] [--gll]
```

| Arg | Meaning |
|-----|---------|
| `model.h5` | topology + field element data |
| `config.h5` | simulation params + source + tiles |
| `--fx/y/z dir` | force-direction record directories |
| `-o dir` | output dir (default: `greenfun/`) |
| `--mesh` | produce mesh-vertex edition |
| `--gll` | produce full-GLL edition |

At least one of `--mesh` or `--gll` must be specified. Error if neither given. Error if `--gll` requested but record files are in vertices mode (postprocess auto-detects format).

Tile sizes come from `config.h5`, not CLI.

## Output Layout

```
greenfun/
├── tile_x000_y000.h5
├── tile_x001_y000.h5
├── ...
└── gll/
    ├── gll_tile_x000_y000.h5
    ├── gll_tile_x001_y000.h5
    └── ...
```

## Validation

Abort if any direction set differs in:

- `basis` or `record_mode`,
- `record_depth_max_m` or `record_depth_actual_m`,
- `solver_dt`, `snapshot_stride`, snapshot count,
- cell/vertex set composition,
- expected `source_direction`.

## Assembly

**Mesh-only**: `[nt, n_v, 6]` × 3 directions → stack on force axis → `[nt, n_v, 3, 6]` → transpose to `[n_v, 3, 6, nt]`.

**Full-GLL**: `[nt, n_c, NGLL, NGLL, NGLL, 6]` × 3 directions → stack → `[nt, n_c, NGLL, NGLL, NGLL, 3, 6]` → transpose to `[n_c, NGLL, NGLL, NGLL, 3, 6, nt]`.

## Implementation Notes

- **Time-series-first layout**: time axis innermost → one vertex's (or one GLL node's) full time series is contiguous on disk. Chunked accordingly.
- **HDF5 compression**: gzip level 4 + shuffle filter. Time-series-first + shuffle = good compressibility for smooth waveforms.
- **Material extraction**: reads from `model.h5 /field/element/` at GLL corner indices. For mesh edition, one value per vertex (no averaging needed for continuous media).
- **Record mode detection**: postprocess inspects the first record file for presence of `cell_ids` (gll mode) or `vertex_ids` (vertices mode).

## Related Module Changes

Postprocess depends on these forward/record changes to support `record_mode = "gll"`:

| Module | Change |
|--------|--------|
| `preprocess/recording_map.py` | Support `record_mode="gll"` — identify cells in record zone, return cell-level recording map with all NGLL³ GLL nodes per cell |
| `preprocess/config_writer.py` | Write `/simulation/record_mode` attr to config.h5 |
| `forward/src/record.cpp` | `RecordWriter::write_step()` accepts cell-based GLL arrays `[n_cell, NGLL, NGLL, NGLL, comps]` alongside existing vertex-based arrays |
| `forward/src/solver.cpp` | `extract_recorded` lambda branches on `record_mode` — per-cell GLL extraction vs per-vertex extraction |
| `examples/halfspace/config.py` | Add `record_mode = "gll"` field |

These changes deliver the record file formats specified above. Postprocess auto-detects the format at runtime.

## Constraints

- Python 3.10+
- numpy, h5py, click, pytest
- No receivers, receiver search, or point interpolation (per design-decisions.md)
- Forward records recorded domain only (no PML, depth-bounded)
- Tile files fully standalone — no dependency on model.h5 for consumer reads

## Files

```
postprocess/
├── pyproject.toml
├── src/gf_post/
│   ├── __init__.py
│   ├── reader.py         — RecordReader, GeometryReader, ConfigReader, build_cell_to_vertex()
│   ├── assembly.py       — stack 3 force directions, transpose to time-series-first
│   ├── mesh_writer.py    — MeshTileWriter (vertex-based tile_x{i}_y{j}.h5)
│   ├── gll_writer.py     — GLLTileWriter (cell-based gll_tile_x{i}_y{j}.h5)
│   ├── material.py       — read vp/vs/density from model.h5, extract corner/per-GLL values
│   ├── cell_selector.py  — identify qualifying cells, bin into tiles
│   └── cli.py
└── tests/
    ├── conftest.py
    ├── test_reader.py
    ├── test_assembly.py
    ├── test_mesh_writer.py
    ├── test_gll_writer.py
    └── test_cell_selector.py
```

## Design Decisions

1. **Two editions**: mesh-only (trilinear, compact) + full-GLL (spectral, exact). User chooses per invocation. Recording at GLL level always supports both.
1. **record_mode in config**: `"gll"` is the recommended mode — produces larger record files but unlocks both editions. `"vertices"` is lighter but limits output to mesh-only.
1. **Standalone tiles**: no global vertex/cell IDs stored. cell_to_vertex uses local 0-based indices. Coords recentered. Material embedded. Full self-containment.
1. **Trilinear as approximation**: mesh edition uses trilinear interpolation of 8 corner values. Sub-cell spectral content is lost. For exact SEM evaluation, use full-GLL edition.
1. **Time-series-first**: `[..., nt]` innermost, chunked for one-location reads. `dt` stored as scalar + `nt` as length — consumer derives time array.
1. **Coordinate recentering**: horizontal origin at source, vertical origin at z=0. Recentered coords stored in tiles. Original source position in attrs for traceability.
