# postprocess/ — Python Strain Green's Function Extraction

## Purpose

Reads shallow mesh-vertex strain snapshots from 3 forward runs (fx, fy, fz).
Assembles the full 3×6 strain Green's tensor at recorded mesh vertices,
outputs horizontally tiled HDF5.

No receiver positions — output is the configured shallow full-volume mesh-vertex field.

## Files

| File | Responsibility |
|------|---------------|
| `reader.py` | `RecordReader` (per-rank mesh-vertex strain records) + `GeometryReader` (mesh.h5 vertex coordinates) + `merge_records()` |
| `assembly.py` | Green's tensor assembly from 3 force directions × 6 strain components |
| `writer.py` | HDF5 Green's function tile writer (`greenfun/tile_x{i}_y{j}.h5`) |
| `cli.py` | CLI entry point |

## Data Flow

```
mesh.h5 (/topology/vertex_to_coord) ───┐
config.h5 (timing + tile metadata) ─────┤
wavefields/{x,y,z}/record_{r}.h5 ──────┤  (3 direction sets, N rank files each)
                                       ↓
    merge_records (per-rank → unified by global vertex ID)
    → Time/depth/vertex-set validation across 3 direction runs
    → Green's tensor assembly (3×6) at recorded mesh vertices
    → GFWriter (horizontal x/y tiles)
```

## CLI

```bash
gf-postprocess mesh.h5 \
    --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ \
    -o greenfun/
```

Horizontal tile size comes from `config.h5` (`/simulation/green_tile_size_m`), not CLI.

## Modes

1. **Small domain** (< RAM): merge all records, assemble all recorded vertices, write tiles
1. **Streaming/tiled** (production): process horizontal tiles/time chunks without materializing all data

## Tests

`tests/` — 19 tests currently covering reader module (RecordReader, GeometryReader, merge_records).
assembly, writer, and CLI need updates/tests for mesh-vertex horizontal tiling.

## Design Doc

`docs/superpowers/design/postprocess.md`
