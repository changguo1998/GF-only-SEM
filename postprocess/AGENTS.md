# postprocess/ — AGENTS.md

## Purpose

Read shallow GLL-node strain snapshots from 3 runs (`x`, `y`, `z`). Build `3×6` strain Green tensors at recorded nodes. Write horizontal HDF5 tiles.

No receivers. Output is the configured shallow GLL-node field.

## Files

| File | Responsibility |
|------|----------------|
| `reader.hpp` | HDF5 readers: config, model, record discovery and per-file scatter |
| `writer.hpp` | HDF5 tile writer with element-count and spatial binning |
| `main.cpp` | 串行/MPI 共用的 tile 分批流程、CLI 和统计输出 |
| `common.hpp` | 参数解析、质量读取、STF 降采样和字段平均等共用辅助函数 |
| `CMakeLists.txt` | 构建串行目标；检测到 MPI 时额外构建 MPI 目标 |
| `_archive/` | Archived Python implementation (reference only) |

## Data Flow

```
model.h5 (/topology/vertex_to_coord) ───┐
config.h5 (timing + tile size) ────────┤
partitions/partition_{r}.h5:/recording ┤ (static recording layout)
wavefields/{x,y,z}/record_{r}_{step}.h5 ─┤ (per-step files)
                                         ↓
rank 0 rebuilds output-rank layouts and indexes direction files
→ rank 0 bins cells; MPI broadcasts shared indexes
→ rank 0 writes one compact `wavefields/tile_indexes/tile_index_xNNN_yNNN.h5` per tile
→ each worker reads its assigned tile indexes
→ extract and assemble one tile at a time
→ write horizontal x/y tiles
```

## CLI

```bash
gf_postprocess model.h5 config.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ -o greenfun/
```

Per-step record files (`record_{r}_{step}.h5`) are auto-discovered in each wavefield directory.
Tile sizes come from `config.h5` (`/simulation/tilex_elements`, `tiley_elements`) or `green_tile_size_m` (optional spatial tile size).

两个目标编译同一个 `main.cpp`。`gf_postprocess` 以单 worker 串行处理全部 tile；
`gf_postprocess_mpi` 在 MPI ranks 间轮转分配 tile。未检测到 MPI 时仅构建串行目标。

Output is byte-identical to the Python reference (vertex IDs + Green's tensor values match exactly).

## Pipeline

```
model.h5 + config.h5
→ read config (/simulation/ attrs + tile arrays)
→ read mesh (/topology/vertex_to_coord + /domain/ bounds)
→ discover record_{r}_{step}.h5 per direction (--fx, --fy, --fz)
→ rank 0 reads `/recording` maps from source partitions and rebuilds output-rank layouts
→ rank 0 writes compact per-tile node, record-point, and mass indexes
→ each worker reads its assigned tile index files
→ each worker extracts assigned tiles without rebuilding indexes
→ mass-lumped L2 project strain; count-average continuous vector fields
→ assemble Green's tensor [nt, n_tile_node, 6, 3]
→ write uncompressed tile_x{i}_y{j}.h5 (precision follows config snapshot_precision)
```

## Build

Built automatically as part of the project CMake. Target: `gf_postprocess`.

```bash
cd build
cmake ..
cmake --build . --target gf_postprocess
```

检测到 MPI 时可构建并行目标：

```bash
cmake --build . --target gf_postprocess_mpi
```

See [`../docs/design/postprocess-tile-parallel.md`](../docs/design/postprocess-tile-parallel.md) for the shared serial/MPI design.

## Performance

峰值字段内存受单个 tile 大小约束。串行目标顺序处理所有 tile；MPI 目标轮转分配 tile，
每个 rank 同样只保留当前 tile 的字段。共享索引仅由 rank 0 构建一次；其他 rank 接收
广播，并只为自己负责的 tile 构建一次 tile 专属索引。

## Tests

The shared serial/MPI pipeline was verified on the complete halfspace records (500 output
steps, three force directions): serial and 4-rank MPI generated 16 tiles, with all 160
datasets and all attributes bit-identical. Runtime was 158.4 s serial and 61.1 s MPI.
The partition-only layout and compact tile-index schema were reverified on the 20³ fullspace
records (800 output steps, 62,073 nodes, 16 tiles): serial 93.1 s, MPI-4 33.9 s; all 160 output
datasets and attributes matched both paths and the previous dense-index baseline exactly. Compact
indexes retain 148,120 of 1,792,000 dense entries (8.266%).
The archived Python implementation (`_archive/`) includes pytest tests for the reference code.

## Design Doc

[`../docs/design/postprocess.md`](../docs/design/postprocess.md)
