# Postprocess Tile-Batched Serial/MPI - Design Document

> **状态（2026-09-23）：已验证。** 串行和 MPI 目标共用同一个 `main.cpp` worker
> 局部流程。tile 按 record-rank 重叠度聚类并平衡工作量；每个 worker 对一个 record
> 只读取一次，再分发给其负责的全部相关 tile。

## Goal

消除“每处理一个 tile 就完整重读一次 record”的冗余，同时保持每个 tile 仅由一个
worker 写入。MPI 调度尽量把依赖相同 record ranks 的 tile 聚到同一 worker，并以 tile
工作量限制负载偏差；无任务的 workers 在字段分配前退出。

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

Three-phase design that trades bounded worker-local output memory for fewer record reads:

```text
Phase 1 - shared index preparation (rank 0 only)
    Scan three direction file lists once. Read partition recording maps, then build
    the output-rank layouts, GLL union, direct record paths, mass indexes, and tile bins.
    Broadcast the completed shared indexes to MPI workers.

Phase 2 - assignment and tile index preparation (one owner per tile)
    Rank 0 writes one compact index file per tile. Zero-entry ranks are omitted.
    A greedy scheduler balances estimated work while preferring already-needed
    record ranks, then broadcasts the ownership table.

Phase 3 - extract_worker_tile_fields() + assembly
    For each direction and step, a worker opens every needed record rank once,
    reads each enabled field once, and scatters it to all matching local tiles.
    Final arrays for a memory-bounded tile batch remain resident; temporary fields
    retain only one direction at a time.
```

### Memory and I/O comparison

| 20³, 800 steps | Previous one-tile loop | Worker-local reuse |
|---|---:|---:|
| Final arrays retained per MPI-4 worker | One tile | Four tiles, ~5.3 GB |
| One-direction temporary fields | One tile | Four tiles, ~1.8 GB |
| Estimated array peak per worker | ~1.8 GB | ~7.1 GB |
| Record opens per worker | 9,600 | 2,400 |
| Dataset reads per worker | 38,400 | 9,600 |

The historical full-replication design remains removed: it required about 20.7 GB per rank and
331 GB at 16 ranks. Current memory is proportional to tiles in one worker batch, not to a complete
copy on every MPI rank. The aggregate field-array budget defaults to 32 GiB and can be overridden
with `GF_POST_MEMORY_GB`; it is divided among active workers. Assignments exceeding one worker's
share are split into additional tile batches. This preserves
the larger-grid memory ceiling at the cost of one record pass per batch.

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

// Per-direction fields for every tile assigned to this worker
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

### `extract_worker_tile_fields(records, tile_plans, routes, ...) -> vector<DirFields>`

Second pass over field-only record files. For each step and needed record rank, reads strain and,
in Debug builds, displacement/velocity/acceleration once. Prebuilt worker routes identify every
assigned tile and compact entry range consuming that rank. The index includes every element copy
contributing to a tile-local node, including copies from cells assigned to adjacent tiles.

- **Strain**: mass-weighted average (`node_weight_sum` per tile-local node)
- **Disp/vel/acc**: count-based average (`node_count` per tile-local node)

I/O cost is one record read per `(worker, direction, step, record-rank)`, independent of how many
tiles on that worker use it.

### `main()` flow

```text
optional MPI init + 200ms stagger
set worker_rank=0, worker_count=1 when MPI is disabled
read config, model, cell_mass
rank 0: scan directions + rebuild output-rank layouts from partitions
rank 0: build record-path, mass, and tile-bin indexes
rank 0: write one compact index file per tile, omitting zero-entry ranks
rank 0: assign tiles by rank overlap + balanced estimated work
MPI: broadcast shared indexes
verify output steps
build time_arr, downsample STF
if worker owns no tile: exit             -- guard BEFORE field alloc
read every assigned TilePlan             -- Phase 2, one owner per tile
pack assigned plans into memory-bounded batches
for tile batch:
    allocate final arrays for batch tiles
    build record-rank -> tile-entry routes
    for dir in {fx, fy, fz}:             -- Phase 3, one direction at a time
        fields = extract_worker_tile_fields(records[dir], batch plans, routes, ...)
        assemble each tile's direction into its final tensors
    for tile_plan in batch:
        write_tile(...)
optional MPI finalize + print stats
```

### Tensor layouts

- `tile_greens`: `[n_steps, n_local, comp(6), dir(3)]` — 18 doubles per (s, li)
- Debug only: `tile_displacement/velocity/acceleration`:
  `[n_steps, n_local, comp(3), dir(3)]` — 9 per (s, li)

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
- **Serial ownership:** `worker_rank=0`, `worker_count=1`, so one worker owns every tile.
- **Overlap-aware MPI ownership:** tiles are sorted by estimated work. Among workers within a
  10% per-tile balance window, the scheduler chooses the worker already requiring the greatest
  record-rank read weight. Every tile is written exactly once.
- **200ms per-rank stagger** (`usleep`) avoids HDF5 metadata contention when
  multiple ranks open the same record files concurrently.
- **MPI over OpenMP**: with a non-threadsafe HDF5, per-process file handles sidestep
  HDF5's lack of thread safety entirely.
- No shared mutable state between ranks: each worker builds and writes only its assigned tiles.

## Implementation Status

**已完成并验证：**

1. `merge_partition_metadata()` — Phase 1 metadata-only merge on rank 0
1. `extract_worker_tile_fields()` — one read per worker record and multi-tile scatter
1. Rank exit guard before field allocation
1. Overlap-aware, load-balanced tile ownership for arbitrary positive worker counts
1. Halfspace runtime verification: 1-rank and 4-rank MPI outputs are numerically
   bit-identical to serial output across every dataset and attribute of all 9 tiles
1. Serial and MPI targets compile the same `main.cpp` pipeline
1. 2026-09-21 complete halfspace regression (500 output steps, three force
   directions): serial and 4-rank MPI generated 16 tiles; all 160 datasets and all
   attributes were bit-identical. Runtime: 158.4 s serial, 61.1 s MPI (2.59× speedup)
1. 2026-09-23 partition-only layout and schema-v2 tile indexes verified on the complete 20³
   fullspace records (800 steps, 62,073 nodes, 16 tiles): serial 93.1 s, MPI-4 33.9 s. All 160
   datasets and attributes matched each other and the previous dense-index output exactly.
   Compact files retain 148,120 of 1,792,000 dense entries (8.266%).
1. 2026-09-23 worker-local reuse verification on the same 20³ records: MPI-4 runtime 19.0 s
   versus the 34.3 s profile baseline. Each worker opened 2,400 files and read 9,600 datasets;
   HDF5 reading fell from 59.187% to 17.693% of cumulative worker time. An 8-step regression
   produced identical serial/MPI values and attributes in all 16 tiles and 160 datasets. The
   complete output passed the fullspace analytical check with mean correlation 0.9742,
   scale 0.999, and scale-fitted relative L2 0.1551.
1. A fresh finite-Q full-flow regression completed preprocess (2,880 elements), CUDA x/y/z
   forward runs (275 steps and 55 snapshots each), then serial and MPI-2 postprocess. All 4 tiles,
   40 datasets, and attributes matched exactly; every floating dataset was finite. Postprocess
   wall times were 0.4 s serial and 0.3 s MPI-2.

HDF5 files need not be byte-identical at the serialization level. Dataset values
and attributes are the consumer-visible contract.
