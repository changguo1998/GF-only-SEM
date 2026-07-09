# Green's Function Reader Module — Technical Design

> Parent: [../design-decisions.md](../design-decisions.md)
> Related: [postprocess.md](postprocess.md)

## Goal

Provide a Python library and CLI that reads precomputed Green's function tiles
and returns the Green's function at a requested (source, receiver) coordinate
pair, honoring the elasticity reciprocity convention:

- The SEM **source loading position** corresponds to a real-world **receiver**
  (station).
- The SEM **recorded mesh vertices** correspond to real-world potential
  **sources**.

The reader indexes multiple Green's function runs (one per SEM source location)
and answers queries by: (1) locating the run whose SEM source matches the query
receiver, then (2) interpolating at the query source among that run's recorded
vertices.

## Context

`postprocess` writes self-contained HDF5 tiles:

```
greenfun_library/src_XXXX/tile_xNNN_yNNN.h5
```

Each tile stores a strain Green's tensor `greens_tensor[nt, n_local, 6, 3]`
(6 strain components × 3 force directions) at recorded shallow mesh vertices,
plus `vertex_ids`. Currently:

- Tile vertex coordinates live only in `model.h5` — copying a tile loses them.
- Only strain is extracted; forward records also contain displacement.
- Source xyz is not stored in the tile; it lives in `config.h5 /source` attrs.

This design makes each tile fully self-contained and extends `postprocess` to
also extract the displacement Green's tensor.

## Self-Containance Principle

All metadata required to interpret a tile lives inside the tile HDF5 file. No
external manifest or index file is a source of truth — any such file is a
rebuildable cache. Copying a single tile (or any subset) never loses
association information.

## Tile Schema (revised)

```
tile_xNNN_yNNN.h5
├── attrs:
│   ├── version            : "1.0.0"
│   ├── basis              : "mesh_vertices"
│   ├── tile_x_index, tile_y_index : int32
│   ├── x_min_m, x_max_m, y_min_m, y_max_m, z_min_m, z_max_m : float64
│   ├── record_depth_max_m, record_depth_actual_m : float64
│   ├── excludes_pml       : int32
│   ├── source_xyz_m       : float64[3]            # NEW: SEM source (= real receiver) xyz [m]
│   ├── source_directions  : str[3] = {"x","y","z"} # NEW
│   └── greens_quantities  : str = "strain,displacement"  # NEW
├── /time/
│   └── t                  : float64[nt]
├── /mesh/
│   ├── vertex_ids         : int64[n_local]
│   └── vertex_coords      : float64[n_local, 3]    # NEW: self-contained coords [m]
└── /field/
    ├── greens_tensor       : float32[nt, n_local, 6, 3]   (strain, gzip 4 + shuffle)
    └── displacement_tensor : float32[nt, n_local, 3, 3]   # NEW (displacement, gzip 4 + shuffle)
```

`displacement_tensor[t, i, c, d]` = displacement component `c` from force
direction `d` at recorded vertex `i`, time `t`. Layout mirrors
`greens_tensor` with 3 components instead of 6.

## Library Layout on Disk

```
greenfun_library/                  # arbitrary library root
└── src_XXXX/                      # one postprocess output per SEM source
    ├── tile_x000_y000.h5
    └── ...
src_YYYY/
    └── ...
_greenfun_index.h5                 # rebuildable cache (see below)
```

## Library Index Cache

The in-memory index (which run has which SEM source, which tile covers which
bounds) is rebuilt by scanning tiles on first launch. To avoid re-scanning on
every launch, the index is persisted as a rebuildable HDF5 cache:

```
_greenfun_index.h5
├── attrs:
│   ├── library_hash   : str     # blake2b state hash
│   ├── build_time     : str     # ISO timestamp
│   ├── n_sources, n_tiles : int
│   └── version        : "1.0"
├── /sources
│   ├── source_id      : int32[n_src]
│   ├── dir_path       : str[n_src]       (relative to root, e.g. "src_0000")
│   ├── source_xyz_m   : float64[n_src, 3]
│   └── n_tiles        : int32[n_src]
└── /tiles
    ├── source_id      : int32[n_tiles]
    ├── rel_path       : str[n_tiles]     (e.g. "src_0000/tile_x002_y003.h5")
    ├── tile_ij        : int32[n_tiles, 2]
    └── bounds_m       : float64[n_tiles, 6] (xmin,xmax,ymin,ymax,zmin,zmax)
```

### Hash algorithm

- **Content:** `(relative_path, size_bytes, mtime_ns)` for every
  `**/tile_*.h5` under the library root, sorted by path.
- **Hash:** `hashlib.blake2b` (standard library, ~3× faster than SHA-256),
  hex digest.
- **Cost:** only `os.stat` per file (no HDF5 opens); thousands of files
  complete in < 100 ms.

### Startup flow

```
GreenFunctionLibrary.__init__(root):
  1. glob root/**/tile_*.h5  →  stat each  →  compute current_hash
  2. read _greenfun_index.h5 library_hash
  3. match?  →  load index datasets into memory, build KDTree on source_xyz
  4. mismatch / missing?  →  open each tile, read source_xyz_m + bounds attrs,
                              rebuild index → write _greenfun_index.h5 → build KDTree
```

- Cache hit: no tile HDF5 opens; only stat + read the index file.
- Cache miss: rebuild (open all tiles, read attrs), write cache.
- Tamper recovery: if a tile changed content but kept size+mtime, the user
  deletes `_greenfun_index.h5` to force a rebuild.

The index file does not match the `tile_*.h5` glob, so copying a tile subset
naturally drops the cache — which is correct, since the cache is rebuildable.

## postprocess Changes (C++)

Three files in `postprocess/cpp/`:

### reader.hh

1. `ConfigParams` gains `double source_x_m, source_y_m, source_z_m`;
   `read_config` reads `/source` attrs `x`, `y`, `z`.
1. `MergedDirection` gains `std::vector<float> displacement;  // [n_steps, n_vertex, 3]`;
   `read_record_into` reads the `displacement` dataset `[1, n_local, 3]`
   (same `vertex_ids` as strain) and scatters by vertex_id.

### main.cpp

3. Assemble `disp_subset [n_steps, n_recorded, 3, 3]` alongside the existing
   `greens_subset [n_steps, n_recorded, 6, 3]`. Force directions merge:
   force-x → column 0, force-y → column 1, force-z → column 2.
1. Tile extraction loop builds both strain and displacement subsets per tile.

### writer.hh

5. `write_tile` gains parameters `source_xyz_m[3]`, `vertex_coords[n_local, 3]`,
   `displacement_tensor[n_steps, n_local, 3, 3]`.
1. Writes:
   - attrs: `source_xyz_m` (float64[3]), `source_directions` (str[3]),
     `greens_quantities` ("strain,displacement")
   - `/mesh/vertex_coords` (float64[n_local, 3], from model
     `vertex_to_coord` indexed by vertex_id)
   - `/field/displacement_tensor` (float32[nt, n_local, 3, 3], gzip 4 + shuffle)

### Backward compatibility

If a record file lacks the `displacement` dataset (legacy format), skip
displacement extraction and set `greens_quantities = "strain"`; do not write
`displacement_tensor`. postprocess must not require the new format.

## Python Reader Module

New root-level package `greenfun/` (alongside `preprocess/`, `forward/`,
`postprocess/`):

```
greenfun/
├── __init__.py          # exports GreenFunctionLibrary
├── library.py           # GreenFunctionLibrary: index + KDTree on SEM source xyz
├── source_run.py        # SourceRun: single greenfun run, tile loading + vertex KDTree
├── interpolator.py      # trilinear interpolation (structured hex mesh, 8 corners)
├── index_cache.py       # _greenfun_index.h5 read/write + blake2b hash check
└── query.py             # GreenQuery result dataclass + CLI entry gf_greenquery
```

### Core API

```python
class GreenFunctionLibrary:
    def __init__(self, root: str | Path, rebuild_index: bool = False): ...

    def query(
        self,
        source_xyz: ArrayLike,     # real source [x, y, z] in meters
        receiver_xyz: ArrayLike,   # real receiver (station) [x, y, z] in meters
        quantity: str = "strain",  # "strain" | "displacement" | "both"
    ) -> GreenQuery:
        """
        1. Find nearest SEM source to receiver_xyz via library KDTree → SourceRun
        2. Interpolate at source_xyz among that SourceRun's recorded vertices
        3. Return GreenQuery
        """

    def query_batch(
        self,
        sources: ArrayLike,        # [n_src, 3]
        receivers: ArrayLike,      # [n_rec, 3]
        quantity: str = "strain",
    ) -> dict[tuple[int, int], GreenQuery]: ...
```

```python
class SourceRun:
    """Single greenfun run (one SEM source). Lazy tile loading."""
    def __init__(self, dir_path: Path, source_xyz: np.ndarray): ...
    def load_vertex_index(self): ...           # read all tile vertex_coords, build KDTree
    def interpolate(self, xyz, quantity) -> np.ndarray:  # [nt, comp, force_dir]
```

```python
class GreenQuery:
    source_xyz: np.ndarray          # [3]
    receiver_xyz: np.ndarray        # [3]
    sem_source_xyz: np.ndarray      # [3] actually matched SEM source
    time: np.ndarray                # [nt]
    strain: np.ndarray | None       # [nt, 6, 3]
    displacement: np.ndarray | None # [nt, 3, 3]
    n_tiles_used: int
    interpolation_used: bool        # True = interpolated, False = exact vertex hit
```

### Reciprocity mapping

- `receiver_xyz` (real station) selects **which** greenfun run (SEM source
  location nearest to it).
- `source_xyz` (real source) is looked up among that run's **recorded
  vertices** (SEM records), interpolated if off-grid.

### Trilinear interpolation

Recorded vertices lie on a regular Cartesian hex mesh. Use
`scipy.spatial.KDTree` to find the 8 corner vertices (2×2×2 cube) around a
query point, compute local coordinates `(α, β, γ) ∈ [0, 1]³`, and weight each
corner's Green tensor value by the trilinear weight. The interpolation is
applied independently per `(t, component, force_direction)`.

Out-of-range queries (depth > `record_depth_max_m`, or inside PML) raise
`ValueError` with a range hint.

### CLI

```bash
gf_greenquery library/ \
    --source 5000 5000 100 \
    --receiver 5000 5000 0 \
    --quantity both \
    --output result.h5   # optional; default prints a summary
```

The CLI is a thin wrapper over `GreenFunctionLibrary.query()`; both the
function-call and CLI paths share one core implementation.

## Build, Dependencies, Integration

### Python package

- New root package `greenfun/`.
- Dependencies: `h5py` (existing), `numpy` (existing), `scipy` (KDTree, **new**).
- Root `pyproject.toml` `[project.scripts]` adds
  `gf_greenquery = "greenfun.query:main"`.
- `[tool.setuptools.packages.find]` include adds `"greenfun*"`.

### C++ postprocess build

- No new dependency; changes stay within `postprocess/cpp/`.
- `postprocess/CMakeLists.txt` already builds `gf_postprocess`; no new target.

### Documentation

- This file (`docs/design/greenfun.md`).
- `greenfun/AGENTS.md` (new) — module description following project convention.
- Root `AGENTS.md` module table gains a `greenfun/` row.
- `docs/deferred.md` unchanged; GLL-point wavefield interpolation is recorded
  here as a future enhancement (see below).

## Future Enhancement: GLL-Point Wavefield Interpolation

The current design interpolates on recorded mesh vertices (trilinear). If
forward is extended to output the full GLL-point wavefield, the reader can
detect a tile's `basis` attribute and switch to GLL-basis interpolation for
higher accuracy. This is deferred; the trilinear path is the v1 default and
degrades gracefully when only vertex data exists.

## Testing

New `tests/greenfun/` (pytest, shared infrastructure):

| File | Coverage |
|------|----------|
| `test_index_cache.py` | blake2b hit/miss; add/remove tile triggers rebuild; missing cache rebuilds; mtime change triggers rebuild |
| `test_source_run.py` | single-run load; self-contained vertex_coords; exact vertex hit returns that vertex's values; cross-tile vertex merge |
| `test_interpolator.py` | trilinear recovers an analytic linear field exactly; 8-corner equal weight at cube center; out-of-range raises; on-vertex degrades to exact |
| `test_library.py` | multiple SourceRuns routed correctly by receiver_xyz; KDTree nearest match; `query_batch` multi-pair results |
| `test_reciprocity.py` | reciprocity numerical check: `G(x←y)` and `G(y←x)` agree within the same library (symmetric fixture) |
| `test_cli.py` | `gf_greenquery` subprocess; `--output result.h5` writes; stdout summary format |

C++ side: add Catch2 cases in `tests/` verifying postprocess tiles contain
`source_xyz_m`, `vertex_coords`, and `displacement_tensor` with correct shapes.
