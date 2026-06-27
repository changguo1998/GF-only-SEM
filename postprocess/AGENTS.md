# postprocess/ — AGENTS.md

## Purpose

Read shallow mesh-vertex strain snapshots from 3 runs (`x`, `y`, `z`). Build `3×6` strain Green tensors at recorded vertices. Write horizontal HDF5 tiles.

No receivers. Output is the configured shallow mesh-vertex field.

## Files

| File | Responsibility |
|------|----------------|
| `reader.py` | read rank record files, mesh vertex coords, merge by `vertex_id` |
| `assembly.py` | stack 3 force directions into Green tensor |
| `writer.py` | write `greenfun/tile_x{i}_y{j}.h5` |
| `cli.py` | CLI entry point |

## Data Flow

```
mesh.h5 (/topology/vertex_to_coord) ───┐
config.h5 (timing + tile size) ────────┤
wavefields/{x,y,z}/record_{r}.h5 ──────┤
                                       ↓
merge by global vertex_id
→ validate timing/depth/vertex sets
→ assemble [nt, n_vertex, 6, 3]
→ write horizontal x/y tiles
```

## CLI

```bash
gf-postprocess mesh.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ -o greenfun/
```

Tile size comes from `config.h5` (`/simulation/green_tile_size_m`).

## Modes

1. Small: merge all records in RAM.
1. Production: stream by tile/time chunk.

## Tests

`tests/` has 19 reader tests. Assembly, writer, and CLI still need mesh-vertex tile tests.

## Design Doc

[`docs/superpowers/design/postprocess.md`](../docs/superpowers/design/postprocess.md)
