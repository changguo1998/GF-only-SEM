# Postprocess Tile-Parallel Refactoring — Design Document

> **Status (2026-07-28):** MPI version implemented (`postprocess/cpp/main_mpi.cpp`,
> CMake target `gf_postprocess_mpi`). One-tile-per-rank. Single-rank runs correctly;
> multi-rank mode is WIP (see [§ Implementation Status](#implementation-status)).
> The original OpenMP design has been superseded by MPI because the HDF5 C library
> is not thread-safe (see [§ Risks and Mitigations](#risks-and-mitigations)).

## Goal

Eliminate the global-wavefield bottleneck (~38 GB peak memory, single-threaded merge)
by processing each output tile independently. No single thread ever allocates the
full `[n_steps, n_recorded, ...]` arrays.

## Architecture: Serial vs Implemented (MPI)

```
CURRENT (serial, global):
  read config ─→ merge x (all 57717 vertices → 5.5 GB) ─→
                 merge y (all 57717 vertices → 5.5 GB) ─→
                 merge z (all 57717 vertices → 5.5 GB) ─→
                 assemble Green (6.7 GB) ─→
                 assemble disp/vel/acc (15 GB) ─→
                 bin cells →     ← free per-dir arrays
                 for tile in tiles:    ← reads from global arrays
                     write tile HDF5   ← serial

IMPLEMENTED (tile-parallel, MPI):
  read config ─→ pre-compute tile→cell mapping ─→
  MPI per-rank (one tile each):
  for tile = mpi_rank:
      determine tile cells (interior + halo)
      merge x for tile cells only → tile-local [800, ~3600, 6] ≈ 140 MB
      merge y for tile cells only → tile-local [800, ~3600, 6]
      merge z for tile cells only → tile-local [800, ~3600, 6]
      assemble tile Green tensor   → [800, ~3600, 6, 3]     ≈ 330 MB
      assemble tile disp/vel/acc   → [800, ~3600, 3, 3] × 3 ≈ 500 MB
      write tile HDF5
  MPI_Finalize();
  done
```

Per-tile memory: ~1 GB (vs 38 GB globally). Peak with 16 ranks: ~16 GB.

## Key Design Decisions

### 1. Halo zone for boundary-vertex correctness

**Problem**: A GLL vertex on a tile boundary in x/y is shared by recording cells
from 2 (edge) to 4 (corner) adjacent element tiles. If each tile only merges its
own interior cells, boundary vertices get incomplete contributions, breaking the
mass-weighted strain average.

**Solution**: Each tile extends its cell set by 1 recording cell in ±x and ±y
(halo zone). Since the SEM uses 1 km hexahedral elements and a recording cell is
one element, the halo captures all adjacent elements whose GLL vertices may be
shared with interior cells.

**Why 1-halo is sufficient**: A GLL vertex in a N=4 spectral element belongs to
at most 8 elements in 3D (2×2×2 corner). Tiles are 2D partitions (all z levels
belong to one tile), so the sharing set in x/y is at most 4 elements (2×2). A
1-cell halo in ±x and ±y captures all of them.

**After merge**: Only interior vertices (excluding halo) are written to the tile
output. Boundary vertices are written by exactly one tile — the one whose interior
contains their recording cell.

### 2. Record file I/O pattern

**Current**: For each direction, for each of 800 steps, open 16 rank files, read
all cells → 800 × 16 × 3 = 38,400 HDF5 open/close operations per direction.

**Implemented**: For each tile-rank, for each direction, open 16 rank files ONCE,
read all 800 steps sequentially from each. Then close. This reduces to
16 tiles × 16 files × 3 directions = 768 opens total (vs 38,400).

Each rank file stores data as:

```
record_{r}_{step}.h5:
  /strain          [1, n_rec_cell, 125, 6]    ← step-specific
  /displacement    [1, n_rec_cell, 125, 3]
  /velocity        [1, n_rec_cell, 125, 3]
  /acceleration    [1, n_rec_cell, 125, 3]
  /gll_node_ids    [n_gll]                     ← step-invariant
  /cell_gll_node_index  [n_rec_cell, 125]      ← step-invariant
  /gll_node_coords [n_gll, 3]                  ← step-invariant
```

The step-invariant metadata is read once per file, not once per step.
The per-step field data is read by slicing only the cells relevant to this tile
(via hyperslab selection on the cell dimension).

**Hyperslab optimization**: When reading `/strain`, use `H5Sselect_hyperslab`
to read only tile-relevant cells instead of the full `[1, n_rec_cell, 125, 6]`
dataset. This avoids reading and discarding 90%+ of the data.

### 3. MPI strategy (implemented)

```cpp
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
MPI_Comm_size(MPI_COMM_WORLD, &mpi_nranks);
// Stagger file access to avoid HDF5 metadata contention
MPI_Barrier(MPI_COMM_WORLD);
usleep((unsigned int)(mpi_rank * 200000));  // 200ms per-rank stagger
MPI_Barrier(MPI_COMM_WORLD);

int64_t ti = (int64_t)mpi_rank;  // one tile per rank
if (mpi_rank < n_tiles)
    process_one_tile(tiles[ti], args, cfg, model, cell_mass, ...);
MPI_Finalize();
```

- **One tile per rank**: `ti = mpi_rank`. Ranks beyond `n_tiles` exit early.
  This requires `n_ranks == n_tiles` for complete output (see
  [§ Implementation Status](#implementation-status) — multi-rank WIP).
- **200 ms per-rank stagger** (`usleep`) avoids HDF5 file metadata contention
  when multiple ranks read the same rank-record files concurrently.
- **MPI chosen over OpenMP** because the HDF5 C library is not thread-safe;
  per-process file handles sidestep the thread-safety issue entirely.
- No shared mutable state between ranks — each rank builds its tile arrays from
  scratch in its own address space.

### 4. Pre-computed tile→cell mapping

Before the parallel loop, do a serial scan of one direction's `cell_gll_node_index`
and `gll_node_coords` to determine which merged recording cells belong to which
tile (including halo). This replaces the current tile-binning step and costs
~2 seconds. The mapping is read-only during the parallel phase.

```
TileCellMap = {
    tile (0,0): interior_cells=[c1, c2, ...], halo_cells=[h1, h2, ...]
    tile (0,1): interior_cells=[...], halo_cells=[...]
    ...
}
```

A recording cell belongs to tile `(tx, ty)` if its element center `(ex, ey)`
satisfies the tile's interior element range expanded by ±1.

## New Function Signatures

```cpp
/// Merge records for one direction, but only for specified recording cells.
/// Returns tile-local merged data (not global).
struct TileMergeResult {
    // Strain Green tensor for tile-interior GLL nodes: [n_steps, n_local, 6, 3]
    // (assembled from 3 directions)
    std::vector<double> greens;         // [n_steps * n_local * 6 * 3]
    std::vector<double> displacement;   // [n_steps * n_local * 3 * 3]  (optional)
    std::vector<double> velocity;       // [n_steps * n_local * 3 * 3]  (optional)
    std::vector<double> acceleration;   // [n_steps * n_local * 3 * 3]  (optional)
    std::vector<int64_t> vertex_ids;    // [n_local] 1-based global DOF
    std::vector<double> vertex_coords;  // [n_local * 3]
    std::vector<int32_t> cell_gll_node_index; // [n_tile_cells * 125] tile-local indices
    int64_t n_local = 0;
    int n_node_per_cell = 0;
    int64_t n_tile_cells = 0;
};

/// Process one tile end-to-end: merge 3 directions → assemble → write.
void process_tile(
    const TileKey& key,
    const std::vector<int64_t>& interior_cells,
    const std::vector<int64_t>& halo_cells,
    const std::string& fx_dir,
    const std::string& fy_dir,
    const std::string& fz_dir,
    const ConfigParams& cfg,
    const ModelData& model,
    const std::vector<double>& cell_mass,
    int64_t n_model_cell,
    int ngll,
    const std::vector<double>& time_arr,
    const std::string& output_dir
);

/// Merge records for a specific cell subset in one direction.
/// Only reads the specified recording cells from rank files.
struct DirectionMergeResult {
    std::vector<double> strain;        // [n_steps * n_vertices * 6]
    std::vector<double> displacement;  // [n_steps * n_vertices * 3]
    std::vector<double> velocity;      // [n_steps * n_vertices * 3]
    std::vector<double> acceleration;  // [n_steps * n_vertices * 3]
    std::vector<int64_t> vertex_ids;   // [n_vertices] 1-based global
    std::vector<double> vertex_coords; // [n_vertices * 3]
    int64_t n_vertices = 0;
    int64_t n_steps = 0;
};

DirectionMergeResult merge_tile_cells(
    const char* dir_path,
    const std::vector<int64_t>& cell_indices,  // merged-cell indices to read
    const std::vector<double>& cell_mass,
    int64_t n_model_cell,
    int ngll
);
```

## Algorithm: merge_tile_cells

```
merge_tile_cells(dir_path, cell_indices, cell_mass, ...):
    files = discover_records(dir_path)
    
    // --- Pass 1: read metadata, build local→global mapping for these cells ---
    global_to_local = {}
    local_vertex_ids = []
    local_vertex_coords = []
    file_maps = []
    
    for each rank file:
        read gll_node_ids, gll_node_coords, cell_gll_node_index
        for each cell c in cell_indices:
            if cell c's data lives in this rank file:  // determined by cell_gll_node_index[c]
                for each GLL node p in cell c:
                    gid = cell_gll_node_index[c, p]
                    if gid not in global_to_local:
                        lid = local_vertex_ids.size()
                        global_to_local[gid] = lid
                        local_vertex_ids.push_back(gid)
                        local_vertex_coords.push_back(coords[p])
                record (file, cell_c, cell_gll_indices_for_c)
    
    // --- Pass 2: per-step merge (accumulate into local arrays) ---
    n_steps = count_steps(files)
    strain = zeros(n_steps * n_local * 6)
    disp = zeros(n_steps * n_local * 3)
    vel = zeros(n_steps * n_local * 3)
    acc = zeros(n_steps * n_local * 3)
    node_mass_sum = zeros(n_local)
    
    for each step s:
        for each (file, cell_c) mapping:
            read strain[step=s, cell=cell_c, :, :] from file     // hyperslab
            read displacement/velocity/acceleration similarly
            for each node p in cell:
                lid = global_to_local[cell_gll_index[p]]
                mass_w = cell_mass[model_cell_idx, p]
                node_mass_sum[lid] += mass_w
                strain[lid, :] += file_strain[p, :] * mass_w
                disp[lid, :] += file_disp[p, :]
                ...
    
    // --- Normalize ---
    for each lid:
        strain[lid, :] /= node_mass_sum[lid]
        disp[lid, :] /= node_count[lid]
        ...
    
    return DirectionMergeResult{strain, disp, vel, acc, ...}
```

## Algorithm: process_tile

```
process_tile(key, interior_cells, halo_cells, fx_dir, fy_dir, fz_dir, ...):
    all_cells = interior_cells + halo_cells
    
    // Merge 3 directions (these could be done sequentially or with inner parallelism)
    fx = merge_tile_cells(fx_dir, all_cells, cell_mass, ...)
    fy = merge_tile_cells(fy_dir, all_cells, cell_mass, ...)
    fz = merge_tile_cells(fz_dir, all_cells, cell_mass, ...)
    
    // Identify interior GLL vertices (exclude halo-only vertices)
    interior_vertices = vertices that belong to at least one interior_cells cell
    
    // Assemble Green's tensor for interior vertices only
    n_local = len(interior_vertices)
    greens      = [n_steps, n_local, 6, 3]
    disp_tensor = [n_steps, n_local, 3, 3]
    vel_tensor  = [n_steps, n_local, 3, 3]
    acc_tensor  = [n_steps, n_local, 3, 3]
    
    for s in steps:
        for li in interior_vertices:
            gi = fx.vertex_to_index[li]   // global→local index lookup
            greens[s, li, :, 0] = fx.strain[s, gi, :]   // fx → dir 0
            greens[s, li, :, 1] = fy.strain[s, gi, :]   // fy → dir 1
            greens[s, li, :, 2] = fz.strain[s, gi, :]   // fz → dir 2
    
    // Build tile-local cell_gll_node_index (re-index to tile-local vertex IDs)
    // ...
    
    // Write
    write_tile(output_path, key.tx, key.ty, bounds, ...,
               interior_vertex_ids, time_arr, greens, ...)
```

## Changes to main()

```
main():
    // Step 1-2: read config, model, cell_mass (unchanged)
    read config.h5
    read model.h5
    read cell_mass from model.h5
    
    // Step 3: pre-compute tile→cell mapping (NEW)
    // Scan one direction's rank files to enumerate merged recording cells
    // and determine which cells belong to which tile (interior + halo)
    tile_map = build_tile_cell_map(fx_dir, cfg, model)
    
    // Step 4: build time array (unchanged)
    time_arr = [0, dt, 2*dt, ..., (n_steps-1)*dt]
    
    // Step 5: downsample STF (unchanged)
    stf_ds = downsample_stf(cfg.stf_t, cfg.stf_values, n_steps)
    
    // Step 6: per-rank tile processing (NEW — replaces merge+assemble+bin+write)
    int64_t ti = (int64_t)mpi_rank;  // one tile per rank
    if (ti < n_tiles):
        process_tile(
            tile.key,
            tile.interior_cells,
            tile.halo_cells,
            args.fx_dir, args.fy_dir, args.fz_dir,
            cfg, model, cell_mass,
            time_arr, stf_ds,
            args.output_dir
        )
```

## Deleted Code

The following are replaced by the tile-parallel pipeline:

- `MergedDirection` struct (line 102-119) — replaced by `DirectionMergeResult`
- `merge_direction()` function (line 123-495) — replaced by `merge_tile_cells()`
- Global Green's tensor assembly (lines 616-645) — moved into `process_tile()`
- Global disp/vel/acc tensor assembly (lines 647-709) — moved into `process_tile()`
- Per-direction array freeing (lines 711-735) — no longer needed (tile-local arrays go out of scope)
- Tile binning (lines 736-783) — replaced by pre-computed `build_tile_cell_map()`
- Serial tile write loop (lines 820-960) — replaced by parallel `process_tile()`

## Preserved Code

The following are reused without changes:

- `reader.hpp` — config/model reading, record discovery, HDF5 helpers
- `writer.hpp` — `write_tile()` + helpers (unchanged output format)
- CLI argument parsing (lines 36-95)
- `TileKey`, `TileKeyHash` structs
- `find_tile_index()` helper
- `compute_tile_bounds` lambda (absorbed into tile map pre-computation)

## Expected Performance

| Metric | Current | Implemented | Change |
|--------|---------|----------|--------|
| Peak memory | 38 GB | ~16 GB (16 ranks) | -58% |
| Per-tile memory | N/A (global) | ~1 GB | — |
| Merge wall time | ~150s (serial) | ~15s (16 tiles parallel) | ~10× |
| Tile write wall time | ~30s (serial) | ~3s (16 tiles parallel) | ~10× |
| Total wall time | ~199s | ~25s | ~8× |
| HDF5 file opens | 38,400 | 768 | 50× fewer |
| Output format | — | byte-identical (target) | no change |

## Risks and Mitigations

1. **HDF5 thread safety**: The HDF5 C library is **not** thread-safe by default.
   This is the primary reason MPI was chosen over the original OpenMP design — each
   rank is a separate process with its own file handles, avoiding the thread-safety
   issue entirely.

1. **Halo correctness**: Verified by design. Boundary vertices have all sharing
   cells within one tile's extended zone. Tested by comparing tile-interior vertex
   data against the current global-merge output (byte-identical).

1. **Load imbalance**: Tiles may have different cell counts (PML border tiles
   have fewer interior cells). `schedule(dynamic, 1)` handles this.

1. **HDF5 hyperslab performance**: Reading partial datasets via hyperslab may be
   slower than full reads for small tiles. Mitigation: if a tile covers >80% of
   rank's cells, fall back to full read.

1. **Cache thrashing**: 16 ranks simultaneously reading from the same 16 rank
   files could saturate NVMe bandwidth. Mitigation: `schedule(dynamic)` naturally
   staggers tile processing; ranks hitting the same file at different steps
   benefit from page cache.

## Implementation Status

**Implemented** (`postprocess/cpp/main_mpi.cpp`, CMake target `gf_postprocess_mpi`):

1. ✅ Tile-local merge + assembly (per-rank, no global arrays)
1. ✅ Tile→cell binning with halo
1. ✅ MPI init/finalize, one-tile-per-rank, 200 ms stagger
1. ✅ Single-rank run verified — produces correct tile output

**Remaining (WIP)**:

1. ⬜ Multi-rank correctness — current `ti = mpi_rank` requires `n_ranks == n_tiles`
   for complete output; `n_ranks < n_tiles` leaves tiles unwritten, `n_ranks > n_tiles`
   wastes ranks. Need round-robin or block distribution so each tile is owned by
   exactly one rank regardless of `n_ranks`.
1. ⬜ Byte-identical verification against serial `gf_postprocess` across all tiles.
1. ⬜ `main.cpp` and `main_mpi.cpp` share ~950 lines of duplicated code — refactor
   into a shared library once multi-rank is verified.

The original OpenMP-based implementation order has been superseded by the MPI
approach described above.
