# Postprocess Module — Technical Design

> Parent: [../design-decisions.md](../design-decisions.md)

## Goal

Read production compact-strain or Debug full-domain GLL snapshots from three SEM runs (x, y, z),
project strain onto unique global GLL nodes with lumped-mass weights, build 3×6 strain Green's
tensors, and write horizontal HDF5 tiles.

No receivers. Output is the configured shallow, non-PML region.

## Context

Preprocess writes each partition's recording map. Production forward records already follow that
compact map; Debug forward records write every local element. Postprocess reconstructs output-rank
layouts, maps recording cells directly or to Debug full-domain indices, prepares one compact index
file per tile, performs the strain projection, and assembles the full Green's tensor (3 force
directions × 6 strain components).

## Data Flow

```
model.h5 (/field/cell/mass, /domain/ bounds)
config.h5 (/simulation/ attrs, tile arrays)
wavefields/{x,y,z}/record_{r}_{step}.h5
partitions/partition_{r}.h5:/recording
         │
         ├── Read config, mesh
         ├── Rank 0 discovers direction files and reconstructs output-rank layouts
         ├── Resolve compact production or full-domain Debug record cells
         ├── Rank 0 builds shared indexes and tile bins; MPI broadcasts them
         ├── Rank 0 writes one compact index file per tile
         ├── Cluster tiles by record-rank overlap and balance worker load
         ├── Each worker reads every needed record once and scatters it to its tiles
         ├── Per-step: lumped-mass project strain onto tile-local GLL nodes
         ├── Debug only: count-average continuous vector fields
         ├── Assemble Green's tensor [nt, n_recorded, 6, 3]
         ├── Bin recorded GLL nodes and whole cells into tiles
         └── Write tile_x{i}_y{j}.h5
```

## Architecture

C++17 header-only design. `gf_postprocess` and `gf_postprocess_mpi` compile the same
worker-local record-reuse pipeline in `main.cpp`; the `GF_POST_MPI` build enables
rank-overlap-aware scheduling, while the serial build uses one worker. Without MPI, only the
serial target is built. See [`postprocess-tile-parallel.md`](postprocess-tile-parallel.md).

| File | Role |
|------|------|
| `cpp/main.cpp` | 串行/MPI 共用的 tile 分批、合并、组装与写入流程 |
| `cpp/common.hpp` | 两种构建共用的参数解析和辅助函数 |
| `cpp/reader.hpp` | HDF5 readers: config, model, record discovery and per-file scatter |
| `cpp/writer.hpp` | HDF5 tile writer with element-count and spatial binning |

## CLI

```bash
gf_postprocess model.h5 config.h5 \
    --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ \
    -o greenfun/
```

| Arg | Meaning |
|-----|---------|
| `model.h5` | Mesh with `/field/cell/mass`, GLL geometry, and `/domain/` bounds |
| `config.h5` | Simulation params, source, tiles |
| `--fx/y/z dir` | Force-direction record directories |
| `-o dir` | Output dir (default: `greenfun/`) |

Per-step record files (`record_{r}_{step}.h5`) are auto-discovered via POSIX glob in each wavefield
directory. Tile sizes come from `config.h5` (`/simulation/tilex_elements`/`tiley_elements` for
element-count tiling, or `green_tile_size_m` for spatial tiling).

## Record Merging

`partition_{r}.h5:/recording` stores `gll_node_ids`, `gll_node_coords`,
`cell_gll_node_index`, `rec_cell_local`, and recorded model-cell indices. Each record stores one
snapshot and the continuous source-partition range merged into its output rank. Production records
contain compact strain; Debug records contain full-domain dynamic fields.
Merge process:

1. Group `record_{r}_{step}.h5` files by step across all ranks.
1. Read each output rank's source partitions, reproduce the solver's ordered merge, build the
   unique global GLL-node union, and resolve each `rec_cell_local` directly in production records
   or as an index into a Debug full-domain record.
1. Build direct `(step, rank) → record path`, cell-point mass, tile-bin, and tile-local lookup
   tables before reading any field dataset.
1. Under MPI, rank 0 builds and broadcasts shared indexes, then assigns tiles by record-rank
   overlap subject to a 10% per-tile load-balance window.
1. For each direction and step, a worker opens each required record rank once and scatters its
   selected recording cells to every assigned tile with matching compact entries. Debug
   full-domain records use HDF5 hyperslabs to avoid loading unrelated cells.
1. Accumulate every element-local strain copy with its cell lumped mass.
1. Divide each global node by its accumulated mass.
1. In Debug builds only, count-average displacement, velocity, and acceleration independently;
   CG-SEM makes their shared-node copies identical.

生产 record 通过 `cell_scope="recording_cells"` 标识并直接读取。旧版紧凑 record 没有该
属性，仍按原顺序读取，因此已有波场无需重算。Debug record 通过
`cell_scope="all_local_cells"` 标识，并在此阶段应用记录区索引。

Record ranks with no contribution to a tile are omitted from that tile index and are not opened by
its worker.

## Per-Tile Index Contract

Before any field dataset is processed, rank 0 recreates
`wavefields/tile_indexes/` and writes exactly one `tile_index_xNNN_yNNN.h5` for every nonempty
output tile. Schema version 2 stores only valid record cell-points:

| Item | Meaning |
| `rank_ids` | Output record ranks represented in the file |
| `rank_entry_offsets` | CSR-style boundaries into the three entry arrays |
| `record_cell_point_index` | Flattened `(record_cell, GLL_point)` index within that rank's record |
| `tile_local_node_index` | Destination node within the output tile |
| `cell_mass_index` | Flattened `model.h5:/field/cell/mass` lookup |
| `gll_node_ids`, `gll_node_coords` | Tile-local node identity and coordinates |
| `cell_gll_node_index` | Tile output cells mapped to tile-local nodes |

This file is derived data: partitions remain the authoritative static layout, while tile index
files prevent every worker and every timestep from rebuilding or scanning dense rank-wide maps.
The `rank_ids` list contains only ranks with at least one valid entry.

### Strain Projection Decision

The default and retained method is the recording-domain mass-lumped discrete L2 projection:

```
projected_strain[I] = sum(cell_mass[e,I] * element_strain[e,I])
                    / sum(cell_mass[e,I])
```

It is selected because supported velocity and density models are smooth, the output library
requires one continuous value per global GLL node, and the method matches the GLL collocation and
diagonal-mass discretization used by the solver. It only reconciles copies at the same node; it
does not mix distinct neighboring nodes or apply a tunable low-pass filter.

SPECFEM receiver strain remains element-local and is interpolated inside the selected element.
That is the raw reference representation, but it does not provide the unique continuous GLL field
required by this library. A consistent-mass L2 solve and Gaussian or Laplacian smoothing are not
used by default; they add cost or alter the Green function's spatial spectrum without demonstrated
accuracy benefit. The bottom face of the recording region has incomplete element support, so
production configurations should record at least one cell deeper than the maximum query depth.

## Green's Tensor Assembly

Three direction merges produce `fx_subset`, `fy_subset`, `fz_subset` (each `[nt, n_recorded, 6]`).
Assembly stacks them along the force-direction axis:

```
greens_subset[nt, n_recorded, 6, 3]
  greens[:, :, :, 0] = fx_subset  (force x → column 0)
  greens[:, :, :, 1] = fy_subset  (force y → column 1)
  greens[:, :, :, 2] = fz_subset  (force z → column 2)
```

Each recorded GLL node stores 3 force directions × 6 strain components = 18 values per timestep.
Storage layout: time outermost, then GLL node, then component, then direction.

## Tiling

Two tiling modes, selected by config:

### Element-count tiling (default)

Recorded cells are binned by element index. Uses `tilex_elements` and `tiley_elements` from
`config.h5`; each cell and all of its GLL nodes stay in one tile.
PML region excluded via `pml_xmin/pml_xmax/pml_ymin/pml_ymax`.

### Spatial tiling (`green_tile_size_m`)

When `green_tile_size_m > 0` in config, cells are binned by spatial position:

```
tile_x = floor((x - xmin) / green_tile_size_m)
tile_y = floor((y - ymin) / green_tile_size_m)
```

Produces spatially-uniform tiles independent of mesh discretization.

## Output Schema

One file per tile:

```
greenfun/tile_x000_y000.h5
├── attrs:
│   ├── version           : "1.0.0"
│   ├── basis             : "gll"
│   ├── tile_x_index, tile_y_index : int32
│   ├── x_min_m, x_max_m, y_min_m, y_max_m, z_min_m, z_max_m : float64
│   ├── record_depth_max_m, record_depth_actual_m : float64
│   └── excludes_pml      : int32 (1)
├── /time/
│   ├── t                 : float64[nt]         (time array)
│   └── attrs: dt, nsteps
├── /mesh/
│   ├── gll_node_ids      : int64[n_local]     (1-based global IDs)
│   ├── gll_node_coords   : float64[n_local, 3]
│   └── cell_gll_node_index : int32[n_cell, NGLL³]
└── /field/
    └── greens_tensor     : float32[nt, n_local, 6, 3]
        uncompressed (compression disabled 2026-08-09); chunked (1, n, comp, comp)
```

Production tiles contain only `greens_tensor`. Debug tiles additionally contain
`displacement_tensor`, `velocity_tensor`, and `acceleration_tensor`, each shaped
`[nt, n_local, 3, 3]`; `greens_quantities` lists the fields present.

Tiles are self-contained for GLL interpolation: they include node IDs, coordinates, and the
cell-to-node map.

## Build

```bash
cd build
cmake ..
cmake --build . --target gf_postprocess      # serial
cmake --build . --target gf_postprocess_mpi   # verified MPI variant
scripts/build.sh --debug -t gf_postprocess    # diagnostic binary in bin-debug/
```

Dependencies: HDF5 C library (system). The serial `gf_postprocess` needs no MPI; the MPI variant `gf_postprocess_mpi` additionally requires an MPI implementation.

## Performance

Workers share a field-array budget, divided by the active worker count. It defaults to 32 GiB and
can be overridden with the positive-integer `GF_POST_MEMORY_GB` environment variable. Each worker
packs as many assigned tiles as fit its share; only oversized assignments require multiple passes
over records. The 20³ fullspace profile (800 steps, 62,073 unique nodes, 16 tiles, MPI-4) improved from
34.3 s to 19.0 s after worker-local record reuse. Per worker, file opens fell from 9,600 to
2,400 and dataset reads from 38,400 to 9,600. Across the four workers, HDF5 reading fell from
59.187% to 17.693% of cumulative process time. The tradeoff is that a worker retains the final
arrays for all assigned tiles; with four equal tiles in this case, final arrays plus one direction
of temporary fields are approximately 7.1 GB per worker.

## Validation

Abort if:

- Number of steps differs across x/y/z directions.
- No record files found in any direction directory.
- No recorded GLL nodes in the combined set.

Warn if recorded GLL-node sets differ across directions.

## Output Stats

Machine-parseable stats printed to stdout at completion:

```
STAT_NSTEPS=500
STAT_NVERTEX=7168
STAT_NRECORDED=845
STAT_NTILES=25
STAT_ELAPSED_S=0.4
```

## Constraints

- C++17 (primary implementation)
- Python 3.10+ (archived reference in `_archive/`)
- HDF5 C library
- Postprocess itself has no receivers, receiver search, or point interpolation; the separate
  `greenfun` reader performs cell lookup and GLL interpolation
- Forward records compact strain in production or full-domain fields in Debug; postprocess projects
  strain onto global GLL nodes
- Tile files contain the GLL coordinates and cell-to-node map required for interpolation

## File Layout

```
postprocess/
├── CMakeLists.txt              (builds cpp/)
├── cpp/
│   ├── CMakeLists.txt          (builds gf_postprocess + gf_postprocess_mpi)
│   ├── main.cpp                (shared serial/MPI tile-batched pipeline)
│   ├── common.hpp              (shared CLI and field helpers)
│   ├── reader.hpp               (config, model, record readers)
│   └── writer.hpp               (tile writer + binning)
└── _archive/                   (archived Python reference)
    ├── src/gf_post/*.py
    └── tests/
```
