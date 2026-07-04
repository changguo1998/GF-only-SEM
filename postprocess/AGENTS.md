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
model.h5 (/topology/vertex_to_coord) ───┐
config.h5 (timing + tile size) ────────┤
wavefields/{x,y,z}/record_{r}_{step}.h5 ─┤ (per-step files)
                                         ↓
merge by global vertex_id
→ validate timing/depth/vertex sets
→ assemble [nt, n_vertex, 6, 3]
→ write horizontal x/y tiles
```

## CLI

```bash
gf-postprocess model.h5 config.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ -o greenfun/
```

Per-step record files (`record_{r}_{step}.h5`) are auto-discovered in each wavefield directory.

Tile sizes come from `config.h5` (`/simulation/tilex_elements`, `tiley_elements`) or `green_tile_size_m` (optional spatial tile size).

## Modes

1. Small: merge all records in RAM.
1. Production: stream by tile/time chunk.

## Tests

`postprocess/tests/` has tests covering reader, writer, and conftest.

## C++ Accelerator

Binary `gf_postprocess` in `postprocess/cpp/` provides an accelerated C++17 edition
of the postprocessing pipeline. Built via CMake, lands in `bin/gf_postprocess`.

### Pipeline (matches Python `gf_post`)

```
model.h5 + config.h5
→ read config (/simulation/ attrs + tile arrays)
→ read mesh (/topology/vertex_to_coord + /domain/ bounds)
→ discover record_{r}_{step}.h5 per direction (--fx, --fy, --fz)
→ per-step: merge strain by global vertex_id across ranks
→ assemble Green's tensor [nt, n_vertex, 6, 3]
→ subset to recorded vertices
→ bin vertices into tiles (element-count or spatial)
→ write tile_x{i}_y{j}.h5 (gzip+shuffle compressed float32)
```

### CLI

```bash
gf_postprocess <model.h5> <config.h5> --fx <dir> --fy <dir> --fz <dir> -o <dir>
```

Output is byte-identical to Python `gf_post` (vertex IDs + Green's tensor values match exactly).
C++ adds HDF5 gzip level 4 + shuffle compression (transparent to readers).

### Files

| File | Responsibility |
|------|----------------|
| `reader.hh` | HDF5 readers: config, model, record discovery and per-file scatter |
| `writer.hh` | HDF5 tile writer with element-count and spatial binning |
| `main.cpp` | CLI entry point, pipeline orchestration, machine-parseable stats |
| `CMakeLists.txt` | CMake build (HDF5 + OpenMP) |

### Build

Built automatically as part of the project CMake. Target: `gf_postprocess`.

```bash
cd build
cmake ..
cmake --build . --target gf_postprocess
```

### Performance

~0.4s for halfspace example (500 steps × 3 directions, 845 recorded vertices,
25 output tiles) vs Python ~several seconds.

## Design Doc

[`../docs/design/postprocess.md`](../docs/design/postprocess.md)
