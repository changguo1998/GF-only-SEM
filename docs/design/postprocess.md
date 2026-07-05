# Postprocess Module — Technical Design

> Parent: [../design-decisions.md](../design-decisions.md)

## Goal

Read shallow mesh-vertex strain snapshots from three SEM runs (x, y, z). Built as a C++17 binary.
Merge per-rank records. Build 3×6 strain Green's tensors at recorded vertices. Write horizontal HDF5 tiles.

No receivers. Output is the configured shallow, non-PML region.

## Context

Forward writes per-rank record files: each MPI rank produces `record_{r}_{step}.h5` with vertex-level
strain at its recorded mesh vertices. Postprocess merges these by global vertex ID, then assembles
the full Green's tensor (3 force directions × 6 strain components).

## Data Flow

```
model.h5 (/topology/vertex_to_coord, /domain/ bounds)
config.h5 (/simulation/ attrs, tile arrays)
wavefields/{x,y,z}/record_{r}_{step}.h5
         │
         ├── Read config, mesh
         ├── Discover per-step record files in each direction dir
         ├── Per-step: merge strain by vertex_id across ranks
         ├── Build recorded vertex list (intersection of x/y/z masks)
         ├── Subset strain to recorded vertices
         ├── Assemble Green's tensor [nt, n_recorded, 6, 3]
         ├── Bin recorded vertices into tiles (element-count or spatial)
         └── Write tile_x{i}_y{j}.h5
```

## Architecture

C++17 header-only design. Single binary `gf_postprocess` (built via CMake, target `gf_postprocess`,
lands in `bin/gf_postprocess`). No compiled library — all logic in `main.cpp`, `reader.hh`, `writer.hh`.

| File | Role |
|------|------|
| `cpp/main.cpp` | CLI, pipeline orchestration, merge, assembly, subset, binning |
| `cpp/reader.hh` | HDF5 readers: config, model, record discovery and per-file scatter |
| `cpp/writer.hh` | HDF5 tile writer with element-count and spatial binning |

## CLI

```bash
gf_postprocess model.h5 config.h5 \
    --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ \
    -o greenfun/
```

| Arg | Meaning |
|-----|---------|
| `model.h5` | Mesh with `/topology/vertex_to_coord` and `/domain/` bounds |
| `config.h5` | Simulation params, source, tiles |
| `--fx/y/z dir` | Force-direction record directories |
| `-o dir` | Output dir (default: `greenfun/`) |

Per-step record files (`record_{r}_{step}.h5`) are auto-discovered via POSIX glob in each wavefield
directory. Tile sizes come from `config.h5` (`/simulation/tilex_elements`/`tiley_elements` for
element-count tiling, or `green_tile_size_m` for spatial tiling).

## Record Merging

Each record file stores `vertex_ids` (1-based global mesh vertex IDs) and `strain` for a single
snapshot on one MPI rank. Merge process:

1. Group `record_{r}_{step}.h5` files by step across all ranks.
1. For each step, allocate a full `[n_vertex, 6]` array, zero-initialized.
1. Read each rank's file, scatter strain to global array by `vertex_id - 1`.
1. Warn if a vertex appears in multiple ranks' files for the same step.
1. Track which vertices were recorded (vertex mask).

Ranks with zero recorded vertices (no shallow elements) produce empty files — handled transparently.

## Green's Tensor Assembly

Three direction merges produce `fx_subset`, `fy_subset`, `fz_subset` (each `[nt, n_recorded, 6]`).
Assembly stacks them along the force-direction axis:

```
greens_subset[nt, n_recorded, 6, 3]
  greens[:, :, :, 0] = fx_subset  (force x → column 0)
  greens[:, :, :, 1] = fy_subset  (force y → column 1)
  greens[:, :, :, 2] = fz_subset  (force z → column 2)
```

Each recorded vertex stores 3 force directions × 6 strain components = 18 values per timestep.
Storage layout: time outermost, then vertex, then component, then direction.

## Tiling

Two tiling modes, selected by config:

### Element-count tiling (default)

Vertex binned by element index. Uses `tilex_elements` and `tiley_elements` from `config.h5`.
Vertex's element index computed from its physical coordinates and uniform element size.
PML region excluded via `pml_xmin/pml_xmax/pml_ymin/pml_ymax`.

### Spatial tiling (`green_tile_size_m`)

When `green_tile_size_m > 0` in config, vertices binned by spatial position:

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
│   ├── basis             : "mesh_vertices"
│   ├── tile_x_index, tile_y_index : int32
│   ├── x_min_m, x_max_m, y_min_m, y_max_m, z_min_m, z_max_m : float64
│   ├── record_depth_max_m, record_depth_actual_m : float64
│   └── excludes_pml      : int32 (1)
├── /time/
│   ├── t                 : float64[nt]         (time array)
│   └── attrs: dt, nsteps
├── /mesh/
│   └── vertex_ids        : int64[n_local]     (1-based global IDs)
└── /field/
    └── greens_tensor     : float32[nt, n_local, 6, 3]
        compressed with gzip level 4 + shuffle
```

Tiles include `vertex_ids` only. Coordinates stay in `model.h5` (not duplicated).

## Build

```bash
cd build
cmake ..
cmake --build . --target gf_postprocess
```

Dependencies: HDF5 C library (system). No MPI, no OpenMP required.

## Performance

~0.4s for halfspace example (500 steps × 3 directions, 845 recorded vertices, 25 output tiles).

## Validation

Abort if:

- Number of steps differs across x/y/z directions.
- No record files found in any direction directory.
- No recorded vertices in the combined mask.

Warn if recorded vertex sets differ across directions.

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
- No receivers, receiver search, or point interpolation
- Forward records shallow mesh-vertex strain only — postprocess operates on merged vertices, not GLL nodes
- Tile files store `vertex_ids`; coordinates remain in `model.h5`

## File Layout

```
postprocess/
├── CMakeLists.txt              (builds cpp/)
├── cpp/
│   ├── CMakeLists.txt          (builds gf_postprocess)
│   ├── main.cpp                (CLI, pipeline)
│   ├── reader.hh               (config, model, record readers)
│   └── writer.hh               (tile writer + binning)
└── _archive/                   (archived Python reference)
    ├── src/gf_post/*.py
    └── tests/
```
