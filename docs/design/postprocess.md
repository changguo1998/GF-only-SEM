# Postprocess Module — Technical Design

> Parent: [../design-decisions.md](../design-decisions.md)

## Goal

Read shallow element-local GLL field snapshots from three SEM runs (x, y, z). Merge per-rank
records, project strain onto unique global GLL nodes with lumped-mass weights, build 3×6 strain
Green's tensors, and write horizontal HDF5 tiles.

No receivers. Output is the configured shallow, non-PML region.

## Context

Preprocess writes each partition's recording map; forward writes field-only per-step records plus
the source-partition range used by each output rank. Postprocess reconstructs the output-rank layouts,
prepares one compact index file per tile, performs the strain projection, and assembles the full
Green's tensor (3 force directions × 6 strain components).

## Data Flow

```
model.h5 (/field/cell/mass, /domain/ bounds)
config.h5 (/simulation/ attrs, tile arrays)
wavefields/{x,y,z}/record_{r}_{step}.h5
partitions/partition_{r}.h5:/recording
         │
         ├── Read config, mesh
         ├── Rank 0 discovers direction files and reconstructs output-rank layouts
         ├── Rank 0 builds shared indexes and tile bins; MPI broadcasts them
         ├── Rank 0 writes one compact index file per tile
         ├── Each worker reads its assigned tile indexes
         ├── Per-step: lumped-mass project strain onto global GLL nodes
         ├── Count-average continuous vector fields
         ├── Assemble Green's tensor [nt, n_recorded, 6, 3]
         ├── Bin recorded GLL nodes and whole cells into tiles
         └── Write tile_x{i}_y{j}.h5
```

## Architecture

C++17 header-only design. `gf_postprocess` and `gf_postprocess_mpi` compile the same
tile-batched pipeline in `main.cpp`; the `GF_POST_MPI` build enables round-robin MPI
scheduling, while the serial build uses one worker. Without MPI, only the serial target
is built. See [`postprocess-tile-parallel.md`](postprocess-tile-parallel.md).

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
`cell_gll_node_index`, and recorded model-cell indices. Each record stores one snapshot's
element-local fields and the continuous source-partition range merged into its output rank.
Merge process:

1. Group `record_{r}_{step}.h5` files by step across all ranks.
1. Read each output rank's source partitions, reproduce the solver's ordered merge, build the
   unique global GLL-node union, and remap indices.
1. Build direct `(step, rank) → record path`, cell-point mass, tile-bin, and tile-local lookup
   tables before reading any field dataset.
1. Under MPI, rank 0 builds and broadcasts shared indexes; each tile-local lookup is built only by
   the worker that owns that tile.
1. For each step, accumulate every element-local strain copy with its cell lumped mass.
1. Divide each global node by its accumulated mass.
1. Count-average displacement, velocity, and acceleration independently; CG-SEM makes their
   shared-node copies identical.

Ranks with zero recorded cells produce empty files and are handled transparently.

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

Tiles are self-contained for GLL interpolation: they include node IDs, coordinates, and the
cell-to-node map.

## Build

```bash
cd build
cmake ..
cmake --build . --target gf_postprocess      # serial
cmake --build . --target gf_postprocess_mpi   # verified MPI variant
```

Dependencies: HDF5 C library (system). The serial `gf_postprocess` needs no MPI; the MPI variant `gf_postprocess_mpi` additionally requires an MPI implementation.

## Performance

~0.4s for the historical halfspace example (500 steps × 3 directions, 845 recorded nodes,
25 output tiles).

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
- Forward records shallow element-local GLL fields; postprocess projects strain onto global GLL nodes
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
