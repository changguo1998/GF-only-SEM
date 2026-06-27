# postprocess/ — Python Strain Green's Function Extraction

## Purpose

Reads strain snapshots from 3 forward runs (fx, fy, fz) + mesh.h5 geometry.
Assembles the full 3×6 strain Green's tensor at every GLL node,
outputs spatially tiled HDF5.

No receiver positions — output is the full GLL-node field.

## Files

| File | Responsibility |
|------|---------------|
| `reader.py` | `RecordReader` (strain dataset, per-rank) + `GeometryReader` (mesh.h5 coords, dξ/dx, is_pml) + `merge_records()` |
| `assembly.py` | Green's tensor assembly from 3 force directions × 6 strain components |
| `writer.py` | HDF5 Green's function tile writer (`greenfun/tile_{i}.h5`) |
| `cli.py` | CLI entry point |

## Data Flow

```
mesh.h5 (GLL coords, dξ/dx, is_pml) ──┐
wavefields/{x,y,z}/record_{r}.h5 ─────┤  (3 direction sets, N rank files each)
                                      ↓
    merge_records (per-rank → unified by global element ID)
    → Time alignment validation across 3 direction runs
    → Green's tensor assembly (3×6) at all GLL nodes
    → GFWriter (spatially tiled by element range)
```

## CLI

```bash
gf-postprocess mesh.h5 \
    --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ \
    -o greenfun/
```

## Modes

1. **Small domain** (< RAM): merge all records, assemble at all GLL nodes, one tile
1. **Tiled** (TB-scale): tile by element range, one tile file per block

## Tests

`tests/` — 19 tests covering reader module (RecordReader, GeometryReader, merge_records).
assembly, writer, and CLI have zero tests (gap).

## Design Doc

`docs/superpowers/design/postprocess.md`
