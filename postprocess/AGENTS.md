# postprocess/ — Python Strain Green's Function Extraction

## Purpose

Reads strain snapshots from 3 forward runs (fx, fy, fz) + mesh.h5 geometry.
Locates receivers in mesh elements, interpolates strain via GLL basis,
assembles full 3×6 strain Green's tensor, outputs spatially tiled HDF5.

## Files

| File | Responsibility |
|------|---------------|
| `reader.py` | `CheckpointReader` (strain dataset) + `GeometryReader` (mesh.h5 coords, dξ/dx, is_pml) |
| `geometry.py` | GLL nodes, weights, Lagrange basis (1D + 3D tensor product) |
| `index.py` | KD-tree spatial index over non-PML element centroids |
| `search.py` | Point-in-hexahedron Newton iteration using dξ/dx |
| `interpolate.py` | GLL basis interpolation of strain at receiver position |
| `assembly.py` | Green's tensor assembly from 3 force directions × 6 strain components |
| `writer.py` | HDF5 Green's function tile writer (`greenfun/tile_{i}.h5`) |
| `cli.py` | CLI entry point |

## Data Flow

```
mesh.h5 (GLL coords, dξ/dx, is_pml) ──┐
wavefields/{x,y,z}/record_{r}.h5 ─────┤  (3 direction sets, N rank files each)
receivers.csv ────────────────────────┤
                                      ↓
    CheckpointReader (merge per-rank by global element ID)
    → Time alignment validation across 3 direction runs
    → KD-tree spatial index (non-PML centroids)
    → Point-in-hexahedron search (Newton, dξ/dx)
    → GLL strain interpolation at receiver
    → Green's tensor assembly (3×6)
    → GFWriter (spatially tiled HDF5)
```

## CLI

```bash
python -m postprocess mesh.h5 \
    --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ \
    receivers.csv -o greenfun/
```

## Modes

1. **Small domain** (< RAM): merge all records, interpolate all receivers, one tile
2. **Tiled** (TB-scale): XY blocks, one block at a time, per-block GF file
3. **Single receiver**: load only needed elements, skip merge

## Tests

`tests/` — 46 tests covering reader, geometry, index, search, interpolation, assembly, writer.

## Design Doc

`docs/superpowers/design/postprocess.md`