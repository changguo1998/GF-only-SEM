# postprocess/ — AGENTS.md

## Purpose

Read production compact-strain or Debug full-domain snapshots from 3 runs (`x`, `y`, `z`), build
`3×6` strain Green tensors at recorded nodes, and write horizontal HDF5 tiles.

No receivers. Output is the configured shallow GLL-node field.

## Files

| File | Responsibility |
|------|----------------|
| `reader.hpp` | HDF5 readers: config, model, record discovery and per-file scatter |
| `writer.hpp` | HDF5 tile writer with element-count and spatial binning |
| `main.cpp` | 串行/MPI 共用的 worker 局部 record 复用、组装和统计流程 |
| `common.hpp` | 参数解析、rank 重叠调度、质量读取和字段辅助函数 |
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
→ assign tiles by record-rank overlap and balanced work
→ each worker reads a record once and scatters it to all assigned matching tiles
→ write horizontal x/y tiles
```

## CLI

```bash
gf_postprocess model.h5 config.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ -o greenfun/
```

Per-step record files (`record_{r}_{step}.h5`) are auto-discovered in each wavefield directory.
Tile sizes come from `config.h5` (`/simulation/tilex_elements`, `tiley_elements`) or `green_tile_size_m` (optional spatial tile size).

两个目标编译同一个 `main.cpp`。`gf_postprocess` 以单 worker 串行处理全部 tile；
`gf_postprocess_mpi` 按 record-rank 重叠度聚类 tile，并在 MPI workers 间平衡工作量。
每个 tile 仍只有一个写入者。未检测到 MPI 时仅构建串行目标。

Output is byte-identical to the Python reference (vertex IDs + Green's tensor values match exactly).

## Pipeline

```
model.h5 + config.h5
→ read config (/simulation/ attrs + tile arrays)
→ read mesh (/topology/vertex_to_coord + /domain/ bounds)
→ discover record_{r}_{step}.h5 per direction (--fx, --fy, --fz)
→ rank 0 reads `/recording` maps from source partitions and rebuilds output-rank layouts
→ map recording cells to compact records or indices in each Debug full-domain record
→ rank 0 writes compact per-tile node, record-point, and mass indexes
→ each worker reads its assigned tile index files
→ each worker extracts assigned tiles without rebuilding indexes
→ mass-lumped L2 project strain; Debug only: count-average continuous vector fields
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

每个 worker 在共享的字段预算内尽量把所属 tile 放入同一批；总预算默认 32 GiB，可用
`GF_POST_MEMORY_GB`（正整数，单位 GiB）覆盖。一次读取 record、分发到该批全部相关 tile，
方向局部字段仍一次只保留一个方向。共享索引仅由 rank 0
构建一次；tile 索引只保留确有贡献的 record ranks。20³/MPI-4 的4个tile均可放入一批，
因此同一 record 在每个 worker 内只读取一次；更大任务超出预算时自动分批重读。

## Tests

The shared serial/MPI pipeline was verified on the complete halfspace records (500 output
steps, three force directions): serial and 4-rank MPI generated 16 tiles, with all 160
datasets and all attributes bit-identical. Runtime was 158.4 s serial and 61.1 s MPI.
The partition-only layout and compact tile-index schema were reverified on the 20³ fullspace
records (800 output steps, 62,073 nodes, 16 tiles): serial 93.1 s, MPI-4 33.9 s; all 160 output
datasets and attributes matched both paths and the previous dense-index baseline exactly. Compact
indexes retain 148,120 of 1,792,000 dense entries (8.266%).
The worker-local record-reuse pipeline was verified on the same 20³ records with MPI-4 on
2026-09-23: runtime fell from the 34.3 s profile baseline to 19.0 s; each worker opened 2,400
files and performed 9,600 dataset reads instead of 9,600 and 38,400. HDF5 reads fell from
59.187% to 17.693% of cumulative worker time. An 8-step serial/MPI-4 regression produced
bit-identical values and attributes across all 16 tiles and 160 datasets.
The final pipeline also passed a fresh finite-Q full-flow run: preprocess 2,880 elements; CUDA
x/y/z each completed 275 solver steps and wrote 55 snapshots; serial (0.4 s) and MPI-2 (0.3 s)
postprocess outputs matched exactly across all 4 tiles, 40 datasets, and attributes, with no
NaN/Inf.
The archived Python implementation (`_archive/`) includes pytest tests for the reference code.

## Design Doc

[`../docs/design/postprocess.md`](../docs/design/postprocess.md)
