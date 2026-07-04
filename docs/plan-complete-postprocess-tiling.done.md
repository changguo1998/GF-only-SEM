# ✓ Complete: Postprocess Spatial Tiling + Fix Assembly Bug

## Motivation

Current postprocess uses element-count-based tiling (`tilex_elements`/`tiley_elements`).
Design doc describes spatial `green_tile_size_m` tiling. Also `assemble_greens_tensor()`
expects GLL-level input shapes but CLI passes vertex-level data — latent crash.

**Status: IMPLEMENTED** (see summary below).

## Changes

### 1. Fix `assemble_greens_tensor()` — vertex-level assembly

**File:** `postprocess/src/gf_post/assembly.py`

Current signature expects `[nt, n_cell, NGLL, NGLL, NGLL, 6]` (GLL-level).
CLI passes `[nt, n_vertices, 6]` (vertex-level). Change to:

```python
def assemble_greens_tensor(
    waveforms: dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Assemble strain Green's tensor from 3 force-direction strain fields.

    Args:
        waveforms: dict with keys "fx", "fy", "fz".
            Each value has shape [nt, n_vertices, 6] (vertex-level strain).
    Returns:
        [nt, n_vertices, 6, 3] Green's tensor.
    """
    nt, n_vertices, ncomp = waveforms["fx"].shape
    tensor = np.zeros((nt, n_vertices, ncomp, 3), dtype=np.float64)
    tensor[:, :, :, 0] = waveforms["fx"]
    tensor[:, :, :, 1] = waveforms["fy"]
    tensor[:, :, :, 2] = waveforms["fz"]
    return tensor
```

### 2. Add `green_tile_size_m` to config schema

**Files:** `examples/halfspace/config.py`, `preprocess/config_loader.py`,
`preprocess/config_writer.py`, `preprocess/preflight.py`

Add optional `green_tile_size_m` (float, meters). Backward compat:
if absent, fall back to element-count tiling (`tilex_elements`/`tiley_elements`).

`config_loader.py` — add `green_tile_size_m` as optional key (not required).
`config_writer.py` — write `/simulation/green_tile_size_m` dataset if present.
`preflight.py` — validate `green_tile_size_m > 0` when set.

### 3. Spatial tiling in `GFWriter.write()`

**File:** `postprocess/src/gf_post/writer.py`

Add new `green_tile_size_m: float | None = None` parameter.
When set, replace element-count binning with spatial binning:

```
tile_x = floor((x - xmin) / green_tile_size_m)
tile_y = floor((y - ymin) / green_tile_size_m)
```

Keep element-count path as fallback when `green_tile_size_m` is None.

### 4. Update `ConfigReader`

**File:** `postprocess/src/gf_post/reader.py`

Add `green_tile_size_m` property (read from `/simulation/` attrs, return None if absent).

### 5. Update CLI

**File:** `postprocess/src/gf_post/cli.py`

Read `green_tile_size_m` from config. Pass to `GFWriter.write()`.
When `green_tile_size_m` is set, skip element-based args (`nx_elements`, `ny_elements`,
`pml_thickness`, `tilex_elements`, `tiley_elements`).

### 6. Update tests

**File:** `postprocess/tests/test_writer.py`

Add test `test_spatial_tiling_single_tile` — 2 vertices in same spatial bin.
Add test `test_spatial_tiling_two_tiles` — vertices in different bins.
Add test `test_spatial_tiling_empty_bin` — no file written for empty bin.

**File:** `postprocess/tests/conftest.py`

Add `green_tile_size_m` to synthetic config fixture.

**File:** `tests/preprocess/test_recording_map.py` — no change needed
(recording map is depth-based, tiling is postprocess-only).

## Files to modify (summary)

| File | Change |
|------|--------|
| `postprocess/src/gf_post/assembly.py` | Fix shape unpack for vertex-level [nt, n_v, 6] |
| `postprocess/src/gf_post/writer.py` | Add spatial `green_tile_size_m` tiling |
| `postprocess/src/gf_post/reader.py` | Add `green_tile_size_m` property |
| `postprocess/src/gf_post/cli.py` | Read and pass `green_tile_size_m` |
| `preprocess/config_loader.py` | Add `green_tile_size_m` optional key |
| `preprocess/config_writer.py` | Write `green_tile_size_m` if present |
| `preprocess/preflight.py` | Validate `green_tile_size_m` |
| `examples/halfspace/config.py` | Add `green_tile_size_m` field |
| `postprocess/tests/test_writer.py` | Add spatial tiling tests |
| `postprocess/tests/conftest.py` | Add `green_tile_size_m` to fixture |

## Not in scope

- Two-edition output (mesh-only + GLL) — deferred, design in `docs/design/postprocess.md`
- `assemble_greens_tensor` for GLL-level — not needed until GLL edition is implemented
