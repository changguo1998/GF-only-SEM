# Task 1: Recording Map in Preprocess

## Summary

Build a shallow mesh-vertex recording map in preprocess. The map selects non-PML mesh vertices within `record_depth_max_m`, so the forward solver writes strain only at those shallow mesh corners — not full GLL volume.

## Files to Create/Modify

1. `preprocess/config_loader.py` — add `record_depth_max_m`, `green_tile_size_m` to REQUIRED_KEYS
1. **Create** `preprocess/recording_map.py` — new module: `build_recording_map()`
1. `preprocess/model_writer.py` — add `/recording/` group to `partition_{r}.h5` writer
1. `preprocess/config_writer.py` — add `record_depth_max_m`, `record_depth_actual_m`, `green_tile_size_m` to `/simulation/` attrs
1. `preprocess/cli.py` — call `build_recording_map()` before writing partitions, pass result to writer
1. `preprocess/preflight.py` — add recording-map validation check
1. `examples/halfspace/config.py` — add `record_depth_max_m`, `green_tile_size_m`
1. **Create** `tests/preprocess/test_recording_map.py`

## Design

From `docs/superpowers/design/preprocess.md` §10:

### Config fields (new required)

```python
record_depth_max_m = 2000.0   # float64 — record all vertices at or above zmin + this
green_tile_size_m = 1000.0    # float64 — horizontal tile width for postprocess
```

### Recording map algorithm

```
1. Read record_depth_max_m and green_tile_size_m from config
2. Compute target_z = zmin + record_depth_max_m  (z positive downward)
3. Set record_depth_actual_m to the first horizontal element face at or below target_z
   (snap to nearest element face, report actual depth)
4. Select non-PML elements fully above that depth; no partial clipping
5. Select unique mesh vertices attached to selected elements
6. For each vertex, choose one owned source element and corner so forward writes it once
```

### Output schema in `partition_{r}.h5`

```
/recording/
  attrs:
    basis              = "mesh_vertices"  (string)
    record_depth_max_m = float64
    record_depth_actual_m = float64
    green_tile_size_m  = float64
    excludes_pml       = true  (bool)
  save_element_mask          bool[n_local_elem]            — which local elements to record
  vertex_ids                 int64[n_record_vertices]       — global mesh vertex IDs (1-based)
  source_element_local_index int32[n_record_vertices]      — local index of owning element
  source_corner_index        int8[n_record_vertices]       — corner index (0-7) within that element
```

### config.h5 additions

In `/simulation/` attrs, add:

- `record_depth_max_m` (float64)
- `record_depth_actual_m` (float64)
- `green_tile_size_m` (float64)

## Interfaces

### `build_recording_map()` signature

```python
def build_recording_map(
    topology: TopologyData,
    boundary_tag: npt.NDArray[np.int64],
    domain_bounds: dict[str, float],
    is_pml: npt.NDArray[np.bool_ | np.int8],
    record_depth_max_m: float,
    green_tile_size_m: float,
    element_to_rank: npt.NDArray[np.int32] | None = None,
    per_rank: dict | None = None,
) -> dict:
```

Returns dict with:

- `record_depth_actual_m`: float
- `per_rank_recording`: dict[int, dict] — per-rank recording data:
  - `save_element_mask`: list[bool] n_local_elem
  - `vertex_ids`: list[int] global vertex IDs
  - `source_element_local_index`: list[int]
  - `source_corner_index`: list[int]

### Integration in `model_writer.py`

`_write_partition_files()` gets a new optional `recording_map` dict. When present, writes `/recording/` group in each `partition_{r}.h5`.

### Integration in `cli.py`

After partition step, call `build_recording_map()`. Pass result to `write_model()` and `write_config()`.

## Constraints

- No GLL interior points — only mesh corners (8 per hex, global vertex IDs from topology)
- PML elements are excluded entirely from recording
- Depth snaps to the nearest element face boundary (not user value)
- If `record_depth_max_m` covers the full domain (exceeds zmax-zmin), all non-PML vertices are recorded
- Forward solver reads `vertex_ids` + `source_element_local_index` + `source_corner_index` to extract strain at corners only

## Tests

Create `tests/preprocess/test_recording_map.py`:

1. Test recording map on small regular mesh — verify correct vertices selected
1. Test PML exclusion — PML-tagged elements not in recording
1. Test depth limit — vertices above depth included, below excluded
1. Test record_depth_actual_m snapping — snaps to element face
1. Test full-domain coverage — all non-PML vertices when depth exceeds domain
1. Test per-rank output shape matches partition — verify save_element_mask length = n_local_elem
