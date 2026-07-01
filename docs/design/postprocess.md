# Postprocess Module — Technical Design

> Parent: [../design-decisions.md](../design-decisions.md)
> Plan: ~~`docs/superpowers/plans/2026-06-08-postprocess.md`~~ (deleted)

## Goal

Read strain snapshots from three SEM runs (`x`, `y`, `z`). Merge by mesh vertex. Build `3×6` strain Green tensors. Write horizontal HDF5 tiles.

No receivers. Output is the configured shallow, non-PML mesh-vertex field.

## Context

Forward computes the full GLL SEM domain. It records only a small output set:

- mesh vertices only,
- `depth <= record_depth_actual_m`,
- no PML vertices.

`record_depth_actual_m` is the first horizontal element face at or below `record_depth_max_m`.

Each force run writes rank files under `wavefields/{x,y,z}/`. Postprocess merges them by global `vertex_id`, then stacks force directions into `[nt, n_vertex, 6, 3]`.

Postprocess outputs strain only. It does not integrate displacement.

## Data Flow

```
model.h5 (/topology/vertex_to_coord) ───────────────┐
config.h5 (/simulation timing + tile elements) ┤
wavefields/x/record_{r}.h5 ────────────────────────┤
wavefields/y/record_{r}.h5 ────────────────────────┤
wavefields/z/record_{r}.h5 ────────────────────────┤
                                                    ↓
postprocess
├── read vertex coordinates
├── read 3 direction record sets
├── merge by global vertex_id
├── validate timing, basis, depth, vertices
├── stack → [nt, n_vertex, 6, 3]
└── write greenfun/tile_x{i}_y{j}.h5
```

## Architecture

```
RecordReader   — read attrs, vertex_ids, strain
GeometryReader — read /topology/vertex_to_coord
Validation     — check direction, timing, basis, depth, vertex set
Assembly       — stack x/y/z strain into Green tensor
GFWriter       — write horizontal x/y tiles
```

## Constraints

- Python 3.10+
- numpy, h5py, click, pytest
- No receivers, receiver search, or point interpolation.
- Output basis: `basis = "mesh_vertices"`.
- Compute basis remains GLL.
- Preprocess/forward exclude PML from the recording map.

## Inputs

### `model.h5`

| Dataset | Shape | Use |
|---------|-------|-----|
| `/topology/vertex_to_coord` | float64[n_vertex, 3] | coordinates for `vertex_ids` |

Postprocess does not need GLL geometry, `dxi_dx`, or element search.

### `config.h5`

| Field | Use |
|-------|-----|
| `/simulation/solver_dt` | solver timestep |
| `/simulation/output_dt_s` | snapshot interval |
| `/simulation/snapshot_stride` | steps per snapshot |
| `/simulation/nsteps` | total steps |
| `/simulation/record_depth_max_m` | requested depth |
| `/simulation/record_depth_actual_m` | snapped depth |
| `/simulation/tilex_elements` | x tile sizes in elements (int64 array) |
| `/simulation/tiley_elements` | y tile sizes in elements (int64 array) |
| `/simulation/nx_elements` | total elements in x |
| `/simulation/pml_{x,y,z}{min,max}` | PML thickness in elements |

### Record files

One file per rank per force direction:

```
wavefields/{direction}/record_{r}.h5
├── attrs: rank, source_direction, basis="mesh_vertices",
│          record_depth_max_m, record_depth_actual_m, excludes_pml=true
├── vertex_ids : int64[n_record_vertices]        # global, 1-based
└── strain     : {precision}[n_snapshots, n_record_vertices, 6]
                 # εxx, εyy, εzz, εxy, εxz, εyz
```

Restart files are not inputs.

## Validation

Abort if any direction set differs in:

- `basis`,
- `record_depth_max_m` or `record_depth_actual_m`,
- `solver_dt`, `snapshot_stride`, or snapshot count,
- merged `vertex_ids`,
- expected `source_direction`.

Error messages list the differing values.

## Assembly

Input:

```
strain_fx : [nt, n_vertex, 6]
strain_fy : [nt, n_vertex, 6]
strain_fz : [nt, n_vertex, 6]
```

Output:

```
greens_tensor : float32[nt, n_vertex, 6, 3]
```

Axis `6` = strain component. Axis `3` = force direction (`x`, `y`, `z`).

## Horizontal Tiling

Tile sizes come from `config.h5` (`/simulation/tilex_elements`, `tiley_elements`), which give
tile sizes in horizontal element counts. Combined with mesh grid dimensions and PML thickness,
each vertex's element index determines its tile assignment.

Each tile keeps all saved times and depths for vertices in that x/y bin.

## Output

```
greenfun/
├── tile_x000_y000.h5
├── tile_x001_y000.h5
└── ...
```

Tile schema:

```
tile_x{i}_y{j}.h5
├── attrs: version, basis="mesh_vertices", tile_x_index, tile_y_index,
│          x_min_m, x_max_m, y_min_m, y_max_m, z_min_m, z_max_m,
│          record_depth_max_m, record_depth_actual_m, excludes_pml
├── /time/t  : float64[nt]
├── /time/dt : float64
├── /time/nsteps : int32
├── /mesh/vertex_ids : int64[n_tile_vertices]
└── /field/greens_tensor : float32[nt, n_tile_vertices, 6, 3]
```

Tiles do not duplicate coordinates. Consumers read coordinates from `model.h5` by `vertex_ids`.

## CLI

```
gf-postprocess model.h5 config.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ -o greenfun/
```

| Arg | Meaning |
|-----|---------|
| `model.h5` | topology file with `/topology/vertex_to_coord` |
| `--fx dir` | x-force records |
| `--fy dir` | y-force records |
| `--fz dir` | z-force records |
| `-o dir` | output dir, default `greenfun/` |

Tile size comes from `config.h5`, not CLI.

## File Layout

```
postprocess/
├── pyproject.toml
├── src/gf_post/
│   ├── __init__.py
│   ├── reader.py      — RecordReader + GeometryReader
│   ├── assembly.py    — stack 3 runs
│   ├── writer.py      — Green tile writer
│   └── cli.py
└── tests/
    ├── conftest.py
    └── test_reader.py
```
