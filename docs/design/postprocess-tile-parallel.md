# Postprocess Tile-Parallel (MPI) - Design Document

> **状态（2026-08-05）：已验证。** 内存从完整复制（16 ranks 约 331 GB）改为
> tile 局部提取（16 ranks 约 17 GB）。tile 轮转分配以及 1-rank/4-rank 与串行后处理的
> 运行结果一致性均已验证。

## Goal

将后处理 tile 写入任务分配给 MPI ranks，避免任一进程分配完整的
`[n_steps, n_recorded, ...]` 数组。每个 tile 仅由一个 rank 写入；一个 rank 可处理多个
tile，多余 ranks 提前退出。

## Problem (Fixed)

The original `main_mpi.cpp` replicated `merge_direction()` on every rank. Each
rank built the **complete** merged dataset for all 3 force directions before the
tile-write distribution happened. With the fullspace-cubic config (35,937 unique
GLL nodes, 800 steps, 16 ranks):

| Array | Size per rank |
|-------|--------------|
| 3× direction strain/disp/vel/acc | ~62 GB |
| greens_subset + disp/vel/acc_subset | ~21 GB |
| **Total per rank** | **~20.7 GB** |
| **16 ranks total** | **~331 GB** |

Machine capacity: 125 GB RAM + 119 GB swap = 244 GB. The 331 GB demand caused
OOM + swap thrash, rendering the system unresponsive and forcing a reboot.

Root cause: `merge_direction()` (line 566-568) ran on all 16 ranks, but only the
final tile-write (line 811) was distributed. All memory was allocated before the
rank guard at line 811.

## Architecture (Implemented)

Three-phase design that bounds per-rank memory to tile-local data:

```text
Phase 1 - merge_metadata() x3 (all ranks, cheap ~6 MB total)
    Build GLL-node union + cell metadata. NO per-step field arrays.

Phase 2 - binning (all ranks, metadata only)
    Bin recording cells into tiles. Ranks >= n_tiles exit HERE. Surviving
    ranks own tile positions rank, rank + nranks, ...

Phase 3 - extract_tile_fields() + assembly (surviving ranks, ~1 GB each)
    Each rank reads record files but accumulates field data ONLY for its
    tile's nodes. All sharing cells are still iterated so mass-weighted /
    count-based averaging stays correct. One direction at a time to bound
    peak memory.
```

### Memory comparison

| | Old (full replication) | New (tile-local) |
|---|---|---|
| Per-rank peak | ~20.7 GB | ~1.1 GB |
| 16-rank total | ~331 GB (OOM) | ~17 GB (fits in RAM) |

## Key Data Structures

```cpp
// Per-file mapping: local GLL node id -> global merged index
struct FileMapping {
    RecordFileInfo info;
    std::vector<int32_t> local_to_global;
    std::vector<int32_t> cell_gll_idx;
    std::vector<int64_t> rec_cell_model_idx;
    hsize_t n_rec_cell, nnodes;
};

// Merged metadata for one force direction (NO per-step field arrays)
struct MergedMetadata {
    std::vector<double> gll_node_coords;           // [n_unique_gll, 3]
    std::vector<int64_t> gll_node_ids;             // [n_unique_gll]
    std::vector<int32_t> cell_gll_node_index;      // [n_rec_cell_merged * n_node]
    std::vector<int64_t> recording_cell_model_index;
    std::vector<FileMapping> file_maps;
    std::vector<StepGroup> groups;
    int64_t n_unique_gll, n_steps, n_node_per_cell, n_rec_cell_merged;
    bool has_displacement, has_velocity, has_acceleration;
};

// Per-direction tile-local field arrays (second pass, tile-local only)
struct DirFields {
    std::vector<double> strain;        // [n_steps, n_local, 6]
    std::vector<double> displacement;  // [n_steps, n_local, 3]
    std::vector<double> velocity;      // [n_steps, n_local, 3]
    std::vector<double> acceleration;  // [n_steps, n_local, 3]
};
```

## Key Functions

### `merge_metadata(dir_path) -> MergedMetadata`

First pass over all record files in a direction. Builds:

- Global GLL-node union (dedup by `gll_node_ids` across rank files)
- Merged `cell_gll_node_index` and `recording_cell_model_index`
- `file_maps` (per-file `local_to_global` index) and `groups` (per-step file
  groupings, used by Phase 3)
- Optional field presence flags (`has_displacement`, etc.)

Cost: ~2 MB per direction. Safe to replicate on every rank.

### `extract_tile_fields(meta, tile_local_index, n_local, ...) -> DirFields`

Second pass over record files. For each step, for each file, reads strain and
optional disp/vel/acc. Accumulates **only** tile-local nodes (via
`tile_local_index`), but iterates **all** recording cells so shared-node
averaging sees every contributing element.

- **Strain**: mass-weighted average (`node_weight_sum` per tile-local node)
- **Disp/vel/acc**: count-based average (`node_count` per tile-local node)

Cost: ~1 GB per rank (n_steps × n_local × 15 doubles peak, one direction at a time).

### `main()` flow

```text
MPI init + 200ms stagger
read config, model, cell_mass
merge_metadata x3 (fx, fy, fz)          -- Phase 1, all ranks
verify n_steps + GLL node consistency
build time_arr, downsample STF
bin cells into tiles                    -- Phase 2, all ranks
if mpi_rank >= n_tiles: exit            -- rank guard BEFORE field alloc
build tile-local node set + tile_local_index
for dir in {fx, fy, fz}:                -- Phase 3, one direction at a time
    fields = extract_tile_fields(meta[dir], ...)
    assemble strain -> tile_greens[dir]
    assemble disp/vel/acc -> tile_*[dir]
write_tile(...)
MPI finalize + print stats
```

### Tensor layouts (must match serial `main.cpp`)

- `tile_greens`: `[n_steps, n_local, dir(3), comp(6)]` — 18 doubles per (s, li)
- `tile_displacement/velocity/acceleration`: `[n_steps, n_local, comp(3), dir(3)]` — 9 per (s, li)

## Averaging Correctness

Each recording cell is a SEM element. A GLL node on a cell boundary is shared by
up to 8 neighboring elements (2×2×2 in 3D). The merge must average all
contributing elements:

- **Strain** uses mass-weighted averaging: `strain[li] = Σ(strain_c × mass_c) / Σ(mass_c)`.
  This is the L2 projection onto GLL nodes.
- **Displacement/velocity/acceleration** use count-based averaging:
  `field[li] = Σ(field_c) / count`. CG-SEM enforces continuity at shared nodes,
  so all contributing elements give identical values; count averaging is exact and
  avoids spurious mass scaling.

In the tile-local extraction, `tile_local_index[global_idx]` returns -1 for
non-tile-local nodes, so they are skipped during accumulation. But the cell loop
still iterates all cells — a tile-local node on a tile boundary still receives
contributions from all its sharing cells, even if those cells belong to a
different tile. This guarantees byte-identical results vs. the serial version.

## MPI Strategy

- **Round-robin ownership**: rank `r` processes tile positions
  `r, r + n_ranks, r + 2 * n_ranks, ...`. Every tile is written exactly once
  for any positive rank count; ranks beyond `n_tiles` exit after Phase 2.
- **200ms per-rank stagger** (`usleep`) avoids HDF5 metadata contention when
  multiple ranks open the same record files concurrently.
- **MPI over OpenMP**: with a non-threadsafe HDF5, per-process file handles sidestep
  HDF5's lack of thread safety entirely.
- **Serial version is thread-parallel too (2026-08-07)**: `gf_postprocess` links the
  Spack `hdf5 +threadsafe` build, so its tile write loop now runs on a small thread
  pool — each thread writes its own distinct tile file (no shared HDF5 handles).
  `GF_POST_WRITE_THREADS` caps the pool (1 = serial fallback). Verified bit-identical
  to the serial output on all 9 halfspace tiles (datasets + attrs). Caveat: the
  threadsafe build's global internal lock serializes concurrent chunked H5Dwrite, so
  the gain saturates around 4 threads (~1.4× on the write phase) — MPI process
  parallelism remains the scaling path for large models.
- No shared mutable state between ranks — each rank builds its tile arrays from
  scratch in its own address space.

## Implementation Status

**已完成并验证：**

1. ✅ `merge_metadata()` — Phase 1, cheap metadata-only merge (all ranks)
1. ✅ `extract_tile_fields()` — Phase 3, tile-local field extraction
1. ✅ Rank exit guard moved before field allocation (fixes OOM)
1. ✅ Lambda refactoring for disp/vel/acc read + assembly loops
1. ✅ Build passes (`gf_postprocess_mpi`)
1. ✅ Memory: 331 GB → 17 GB (16 ranks)
1. ✅ Round-robin tile ownership for arbitrary positive rank counts
1. ✅ Halfspace runtime verification: 1-rank and 4-rank MPI outputs are numerically
   bit-identical to serial output across every dataset and attribute of all 9 tiles

**可选的后续清理：**

- 两条合并流程有意保持独立：串行路径完整复制字段，MPI 路径提取 tile 局部字段。若未来
  引入第三种后端，可将约 500 行结构相似代码移入共用库；当前正确性和维护不依赖此重构。
- 不要求 HDF5 文件的序列化字节完全一致；数据集值与属性位级一致，这才是使用方可见的
  契约。
