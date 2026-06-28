# Task 3: Postprocess — Spatial Tile Making

## Summary

Change postprocess tiling from element-count-based to spatial `green_tile_size_m`-based horizontal tiling. Update tile schema to match the design doc.

## Files to Modify

1. `postprocess/src/gf_post/cli.py` — remove `--tile-elems`, read `green_tile_size_m` from config.h5
1. `postprocess/src/gf_post/writer.py` — change tiling to spatial bins, update schema
1. `postprocess/src/gf_post/reader.py` — read `record_depth_max_m`/`record_depth_actual_m`/`green_tile_size_m` from config.h5
1. `postprocess/tests/conftest.py` — add `green_tile_size_m` to synthetic config
1. `postprocess/tests/test_reader.py` — update tests
1. **Create** `postprocess/tests/test_writer.py` — test spatial tiling

## Design

From `docs/superpowers/design/postprocess.md` "Horizontal Tiling":

### Tiling algorithm

Read `green_tile_size_m` from `config.h5` `/simulation/` attrs.

For each mesh vertex (from mesh.h5 `/topology/vertex_to_coord`):

```
tile_x = floor((x - xmin) / green_tile_size_m)
tile_y = floor((y - ymin) / green_tile_size_m)
```

Each tile keeps all saved times and depths for vertices in that x/y bin.

### Output schema

```
greenfun/
├── tile_x000_y000.h5
├── tile_x001_y000.h5
└── ...
```

Tile schema:

```
tile_x{i}_y{j}.h5
├── attrs: version, basis="mesh_vertices",
│          tile_x_index, tile_y_index,
│          x_min_m, x_max_m, y_min_m, y_max_m,
│          z_min_m, z_max_m,
│          record_depth_max_m, record_depth_actual_m,
│          excludes_pml
├── /time/t  : float64[nt]
├── /time/dt : float64
├── /time/nsteps : int32
├── /mesh/vertex_ids : int64[n_tile_vertices]
└── /field/greens_tensor : float32[nt, n_tile_vertices, 6, 3]
```

Key differences from current:

- Tile naming: `tile_x{i}_y{j}.h5` instead of `tile_{idx}.h5`
- Tiling by spatial bin, not element range
- `vertex_ids` dataset instead of `coords` — consumers read coordinates from `mesh.h5`
- Tensor order: `[nt, n_vertices, 6, 3]` — vertices only, no GLL dimensions
- Rich attrs with spatial bounds and recording params

### Reader changes

`reader.py` needs a method to read config.h5 simulation attrs:

```python
class ConfigReader:
    def __init__(self, path: str):
        ...
    @property
    def green_tile_size_m(self) -> float:
        ...
    @property
    def record_depth_max_m(self) -> float:
        ...
    @property
    def record_depth_actual_m(self) -> float:
        ...
```

Or alternatively, add a helper function that reads these from config.h5.

### CLI changes

```python
@click.command()
@click.argument("mesh", type=click.Path(exists=True))
@click.argument("config", type=click.Path(exists=True))  # NEW: config.h5 required
@click.option("--fx", ...)
@click.option("--fy", ...)
@click.option("--fz", ...)
@click.option("-o", "--output-dir", default="greenfun")
def main(mesh, config, fx, fy, fz, output_dir):
    # Read green_tile_size_m from config.h5
    # Use spatial tiling instead of --tile-elems
```

### Writer changes

```python
class GFWriter:
    @staticmethod
    def write(
        output_dir: str | Path,
        vertex_coords: npt.NDArray[np.float64],  # [n_vertex, 3] from mesh.h5
        vertex_ids: npt.NDArray[np.int64],        # [n_vertex] global vertex IDs
        time: npt.NDArray[np.float64],            # [nt]
        dt: float,
        greens: npt.NDArray[np.float64],          # [nt, n_vertex, 6, 3]
        green_tile_size_m: float,
        domain_bounds: dict[str, float],
        record_depth_max_m: float,
        record_depth_actual_m: float,
    ) -> List[Path]:
        # Spatial tiling by green_tile_size_m
        # Write tile_x{i}_y{j}.h5 per bin
```

## Constraints

- Tiles do not duplicate coordinates — consumers read from `mesh.h5` via `vertex_ids`
- Tile naming uses zero-padded 3-digit indices: `tile_x000_y000.h5`
- Empty bins produce no file (skip bins with 0 vertices)
- Postprocess still reads full per-rank record files and merges them — the tiling change is only in the output stage
- The tensor shape is `[nt, n_tile_vertices, 6, 3]` — no GLL dimensions (mesh vertices only)

## Tests

Create `postprocess/tests/test_writer.py`:

1. Test spatial tiling produces correct number of tiles for known mesh size
1. Test tile naming matches `tile_x{i}_y{j}.h5` pattern
1. Test tile schema (attrs, datasets present)
1. Test empty bin produces no file
1. Test vertex_ids in each tile match expected spatial bin
