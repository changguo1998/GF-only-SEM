# Postprocess Tile-Batched Serial/MPI - Design Document

> **状态（2026-09-23）：已验证。** 串行和 MPI 目标共用同一个 `main.cpp` tile 分批流程。
> 串行目标使用一个 worker，MPI 目标轮转分配 tile。内存从完整复制（16 ranks 约
> 331 GB）降至 tile 局部提取（16 ranks 约 17 GB）。

## Goal

以分批 tile 流程避免任一进程分配完整的 `[n_steps, n_recorded, ...]` 数组。每个 tile
仅由一个 worker 写入；串行构建按顺序处理，MPI 构建在 ranks 间轮转分配，多余 ranks
提前退出。

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

Three-phase design that bounds per-worker memory to tile-local data:

```text
Phase 1 - shared index preparation (rank 0 only)
    Scan three direction file lists once. Read partition recording maps, then build
    the output-rank layouts, GLL union, direct record paths, mass indexes, and tile bins.
    Broadcast the completed shared indexes to MPI workers.

Phase 2 - assigned tile index preparation (one owner per tile)
    Rank 0 writes one compact index file per tile. Workers own positions rank,
    rank + worker_count, ... and read their index files before field I/O.

Phase 3 - extract_tile_fields() + assembly (surviving workers, ~1 GB each)
    Each worker reads record files but accumulates field data ONLY for its
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
// Per-rank mapping: local GLL node id -> global merged index
struct RankMapping {
    int rank;
    std::vector<int32_t> local_to_global;
    std::vector<int32_t> cell_gll_idx;
    std::vector<int64_t> cell_point_mass_idx;
    hsize_t n_rec_cell, nnodes;
};

// Shared layout metadata (NO per-step field arrays)
struct LayoutMetadata {
    std::vector<double> gll_node_coords;           // [n_unique_gll, 3]
    std::vector<int64_t> gll_node_ids;             // [n_unique_gll]
    std::vector<int32_t> cell_gll_node_index;      // [n_rec_cell_merged * n_node]
    std::vector<RankMapping> rank_maps;
    int64_t n_unique_gll, n_node_per_cell, n_rec_cell_merged;
};

struct DirectionRecords {
    std::vector<int> steps;
    std::vector<std::vector<std::string>> record_paths; // [step][rank]
    bool has_displacement, has_velocity, has_acceleration;
};

struct TilePlan {
    std::vector<int32_t> rank_ids;
    std::vector<int64_t> rank_entry_offsets;
    std::vector<int64_t> record_cell_point_index;
    std::vector<int32_t> tile_local_node_index;
    std::vector<int64_t> cell_mass_index;
    std::vector<int32_t> cell_gll_index;
    std::vector<int64_t> node_ids;
    std::vector<double> node_coords;
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

### `merge_partition_metadata(partition_dir, records) -> LayoutMetadata`

Rank 0 reads each output rank's source partition range and builds:

- Global GLL-node union (dedup by `gll_node_ids` across partition recording maps)
- Merged `cell_gll_node_index`
- `rank_maps` with per-rank `local_to_global`, cell-node, and mass indexes
- Direct per-direction `(step, rank) → record path` tables

MPI broadcasts this result; other workers do not repeat layout reads or index construction.

### `extract_tile_fields(layout, records, tile_plan, ...) -> DirFields`

Second pass over field-only record files. For each step, for each rank, reads strain and
optional disp/vel/acc. Accumulates only tile-local nodes by iterating the compact
`record_cell_point_index` entries. The index includes every element copy contributing to a
tile-local node, including copies from cells assigned to adjacent tiles.

- **Strain**: mass-weighted average (`node_weight_sum` per tile-local node)
- **Disp/vel/acc**: count-based average (`node_count` per tile-local node)

Cost: ~1 GB per worker (n_steps × n_local × 15 doubles peak, one direction at a time).

### `main()` flow

```text
optional MPI init + 200ms stagger
set worker_rank=0, worker_count=1 when MPI is disabled
read config, model, cell_mass
rank 0: scan directions + rebuild output-rank layouts from partitions
rank 0: build record-path, mass, and tile-bin indexes
rank 0: write one compact index file per tile
MPI: broadcast shared indexes
verify output steps
build time_arr, downsample STF
if worker_rank >= n_tiles: exit         -- guard BEFORE field alloc
read every assigned TilePlan             -- Phase 2, one owner per tile
for tile_plan in assigned plans:
    for dir in {fx, fy, fz}:            -- Phase 3, one direction at a time
        fields = extract_tile_fields(layout, records[dir], tile_plan, ...)
        assemble strain -> tile_greens[dir]
        assemble disp/vel/acc -> tile_*[dir]
    write_tile(...)
optional MPI finalize + print stats
```

### Tensor layouts

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

Index construction scans every recording cell. For each tile it retains all cell-point copies
whose global node belongs to that tile, including copies from cells assigned to an adjacent tile.
Extraction therefore traverses only valid compact entries while preserving every contribution at
tile boundaries. Serial and MPI builds use identical numerical operations.

## Serial/MPI Strategy

- **Single source:** both targets compile `main.cpp`. `GF_POST_MPI` only enables
  MPI initialization, staggering, and finalization.
- **Serial ownership:** `worker_rank=0`, `worker_count=1`, so one worker processes
  every tile sequentially.
- **Round-robin ownership**: rank `r` processes tile positions
  `r, r + n_ranks, r + 2 * n_ranks, ...`. Every tile is written exactly once
  for any positive rank count; ranks beyond `n_tiles` exit after Phase 2.
- **200ms per-rank stagger** (`usleep`) avoids HDF5 metadata contention when
  multiple ranks open the same record files concurrently.
- **MPI over OpenMP**: with a non-threadsafe HDF5, per-process file handles sidestep
  HDF5's lack of thread safety entirely.
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
1. ✅ Serial and MPI targets compile the same `main.cpp` pipeline
1. ✅ 2026-09-21 complete halfspace regression (500 output steps, three force
   directions): serial and 4-rank MPI generated 16 tiles; all 160 datasets and all
   attributes were bit-identical. Runtime: 158.4 s serial, 61.1 s MPI (2.59× speedup)
1. 2026-09-23 partition-only layout and schema-v2 tile indexes verified on the complete 20³
   fullspace records (800 steps, 62,073 nodes, 16 tiles): serial 93.1 s, MPI-4 33.9 s. All 160
   datasets and attributes matched each other and the previous dense-index output exactly.
   Compact files retain 148,120 of 1,792,000 dense entries (8.266%).

HDF5 files need not be byte-identical at the serialization level. Dataset values
and attributes are the consumer-visible contract.
