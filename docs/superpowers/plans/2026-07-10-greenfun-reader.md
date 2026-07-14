# Greenfun Reader Module Parallel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. This plan is organized as dependency waves so independent agents can work in parallel without editing the same files.

**Goal:** Build `greenfun/` Python reader support and extend `gf_postprocess` tiles so Green's functions can be queried by source/receiver coordinates using reciprocity.

**Architecture:** C++ postprocess writes self-contained HDF5 tiles. Python indexes tile libraries with a rebuildable `_greenfun_index.h5`, routes receiver coordinates to nearest SEM source run, and interpolates requested source coordinates on mesh vertices.

**Tech Stack:** C++17 + HDF5, Python 3.13 + numpy + h5py + scipy, pytest, Catch2.

## Global Constraints

- No JSON/text manifest files; durable metadata must live in HDF5.
- Tiles must be self-contained: source xyz, vertex coords, time, strain, displacement when available.
- `_greenfun_index.h5` is cache only; tiles are source of truth.
- Cache hash uses `os.stat`: relative path, size, `mtime_ns`, then blake2b.
- Coordinates are local SEM Cartesian meters.
- Reciprocity: API source = recorded mesh vertex coordinate; API receiver = SEM source location.
- Full physical variable names in code; no short names for scientific quantities.
- Execute in isolated worktrees for parallel agents.
- Do not commit this plan unless explicitly requested.

______________________________________________________________________

## Parallel Dependency Graph

```text
Wave 0: Contract scaffold
  ├─ Wave 1A: C++ tile schema
  ├─ Wave 1B: Python index cache
  ├─ Wave 1C: Python interpolator
  └─ Wave 1D: Packaging/docs scaffold

Wave 2A: Python SourceRun loader      depends on 1B + 1C
Wave 2B: C++ postprocess integration  depends on 1A

Wave 3: GreenFunctionLibrary routing  depends on 2A
Wave 4: CLI + final integration       depends on 2B + 3 + 1D
```

## Conflict-Avoidance Rules

- One agent owns each file listed in its task.
- Shared files are edited only in Wave 0, Wave 1D, or Wave 4.
- If using git worktrees, merge in this order: Wave 0 → Wave 1A/1B/1C/1D → Wave 2A/2B → Wave 3 → Wave 4.
- If two branches touch the same file unexpectedly, stop and request integration review.

______________________________________________________________________

## Wave 0: Contract Scaffold (single owner)

**Purpose:** Create minimal stable interfaces so Python work can proceed in parallel.

**Files owned:**

- Create: `greenfun/__init__.py`
- Create: `greenfun/query.py`
- Create: `tests/greenfun/conftest.py`

**Deliverables:**

- `GreenQuery` dataclass with fields:
  - `source_xyz_m`, `receiver_xyz_m`, `sem_source_xyz_m`, `time`, `strain`, `displacement`, `source_run_dir`, `interpolation`
- Minimal `greenfun/__init__.py` exports.
- Synthetic tile fixture writer used by all Python tests.

**Validation:**

```bash
python - <<'PY'
from greenfun.query import GreenQuery
print(GreenQuery)
PY
pytest --collect-only tests/greenfun -q
```

**Commit:**

```bash
git add greenfun/__init__.py greenfun/query.py tests/greenfun/conftest.py
git commit -m "feat: add greenfun API scaffold"
```

______________________________________________________________________

## Wave 1A: C++ Self-Contained Tile Schema

**Can run in parallel with:** Wave 1B, 1C, 1D.

**Files owned:**

- Modify: `postprocess/cpp/reader.hh`
- Modify: `postprocess/cpp/writer.hh`
- Modify: `postprocess/cpp/main.cpp`
- Modify: `tests/CMakeLists.txt`
- Create: `tests/test_postprocess_tile.cpp`

**Required changes:**

- Read `/source` attrs `x`, `y`, `z` from `config.h5`.
- Read `/displacement` from record files when present.
- Assemble displacement Green tensor `[nt, n_recorded, 3, 3]`.
- Write tile attrs:
  - `source_xyz_m`
  - `source_directions = "x,y,z"`
  - `greens_quantities = "strain"` or `"strain,displacement"`
- Write tile datasets:
  - `/mesh/vertex_coords`
  - `/field/displacement_tensor` when available
- Preserve backward compatibility for old strain-only records.

**Validation:**

```bash
cd forward
cmake -B build -DGF_DEVICE_BACKEND=CPU
cmake --build build --target test_postprocess_tile
ctest --test-dir build -R test_postprocess_tile --output-on-failure
```

Then:

```bash
cd examples/halfspace
bash postprocess.sh
python - <<'PY'
import glob, h5py
path = sorted(glob.glob('output/greens/tile_*.h5'))[0]
with h5py.File(path, 'r') as h5:
    assert 'source_xyz_m' in h5.attrs
    assert 'greens_quantities' in h5.attrs
    assert '/mesh/vertex_coords' in h5
    assert '/field/greens_tensor' in h5
print('schema ok:', path)
PY
```

**Commit:**

```bash
git add postprocess/cpp/reader.hh postprocess/cpp/writer.hh postprocess/cpp/main.cpp tests/CMakeLists.txt tests/test_postprocess_tile.cpp
git commit -m "feat: write self-contained greenfun tiles"
```

______________________________________________________________________

## Wave 1B: Python Index Cache

**Can run in parallel with:** Wave 1A, 1C, 1D.

**Depends on:** Wave 0 fixture.

**Files owned:**

- Create: `greenfun/index_cache.py`
- Create: `tests/greenfun/test_index_cache.py`

**Required API:**

- `CACHE_FILENAME = "_greenfun_index.h5"`
- `SourceIndexEntry`
- `TileIndexEntry`
- `LibraryIndex`
- `compute_library_hash(root_path)`
- `scan_tiles(root_path)`
- `load_or_rebuild_index(root_path, rebuild=False)`

**Required behavior:**

- Hash only tile path/stat metadata.
- Cache source entries and tile entries in `_greenfun_index.h5`.
- Rebuild cache when missing, invalid version, or hash mismatch.

**Validation:**

```bash
pytest tests/greenfun/test_index_cache.py -v
```

**Commit:**

```bash
git add greenfun/index_cache.py tests/greenfun/test_index_cache.py
git commit -m "feat: cache greenfun library index"
```

______________________________________________________________________

## Wave 1C: Python Trilinear Interpolator

**Can run in parallel with:** Wave 1A, 1B, 1D.

**Files owned:**

- Create: `greenfun/interpolator.py`
- Create: `tests/greenfun/test_interpolator.py`

**Required API:**

- `TrilinearInterpolator(vertex_coords)`
- `interpolate(point_xyz_m, values)` where `values.shape[0] == n_vertex`

**Required behavior:**

- Supports trailing tensor dimensions unchanged.
- Exact vertex query returns exact vertex value.
- Interior cube query uses trilinear weights.
- Out-of-bounds query raises `ValueError`.

**Validation:**

```bash
pytest tests/greenfun/test_interpolator.py -v
```

**Commit:**

```bash
git add greenfun/interpolator.py tests/greenfun/test_interpolator.py
git commit -m "feat: interpolate greenfun mesh vertices"
```

______________________________________________________________________

## Wave 1D: Packaging and Module Docs

**Can run in parallel with:** Wave 1A, 1B, 1C.

**Files owned:**

- Modify: `pyproject.toml`
- Modify: root `AGENTS.md`
- Create: `greenfun/AGENTS.md`

**Required changes:**

- Add dependency: `scipy>=1.11`.
- Add package discovery: `greenfun*`.
- Add console script: `gf_greenquery = "greenfun.query:main"`.
- Add `greenfun/` row to root module table.
- Add module-specific conventions in `greenfun/AGENTS.md`.

**Validation:**

```bash
python - <<'PY'
import tomllib
from pathlib import Path
cfg = tomllib.loads(Path('pyproject.toml').read_text())
assert 'scipy>=1.11' in cfg['project']['dependencies']
assert cfg['project']['scripts']['gf_greenquery'] == 'greenfun.query:main'
print('packaging ok')
PY
```

**Commit:**

```bash
git add pyproject.toml AGENTS.md greenfun/AGENTS.md
git commit -m "feat: register greenfun package"
```

______________________________________________________________________

## Wave 2A: SourceRun Tile Loader

**Can run in parallel with:** Wave 2B.

**Depends on:** Wave 1B, Wave 1C.

**Files owned:**

- Create: `greenfun/source_run.py`
- Create: `tests/greenfun/test_source_run.py`

**Required API:**

- `SourceRun(root_path, source_entry, tile_entries)`
- `load()`
- `query(source_xyz_m, quantity="both") -> GreenQuery`

**Required behavior:**

- Load all tiles for one SEM source run lazily.
- Deduplicate shared boundary vertices by `vertex_id`.
- Preserve time axis consistency.
- Return strain, displacement, or both.
- Use `TrilinearInterpolator` for source-coordinate lookup.

**Validation:**

```bash
pytest tests/greenfun/test_source_run.py -v
```

**Commit:**

```bash
git add greenfun/source_run.py tests/greenfun/test_source_run.py
git commit -m "feat: load greenfun source runs"
```

______________________________________________________________________

## Wave 2B: C++ Integration Verification

**Can run in parallel with:** Wave 2A.

**Depends on:** Wave 1A.

**Files owned:**

- No new source files unless fixing Wave 1A defects.
- Optional: add a small Python schema check under `tests/greenfun/test_postprocess_schema.py` only if needed.

**Required checks:**

- Halfspace postprocess emits self-contained tiles.
- Existing strain tensor shape remains unchanged.
- Displacement tensor appears when records contain displacement.

**Validation:**

```bash
cd examples/halfspace
bash postprocess.sh
python - <<'PY'
import glob, h5py
paths = sorted(glob.glob('output/greens/tile_*.h5'))
assert paths
with h5py.File(paths[0], 'r') as h5:
    assert h5['field/greens_tensor'].shape[2:] == (6, 3)
    assert h5['mesh/vertex_coords'].shape[1] == 3
print('postprocess integration ok')
PY
```

**Commit:**

Only commit if this task changes files.

______________________________________________________________________

## Wave 3: GreenFunctionLibrary Reciprocity Routing

**Depends on:** Wave 2A.

**Files owned:**

- Create: `greenfun/library.py`
- Modify: `greenfun/__init__.py`
- Create: `tests/greenfun/test_library.py`

**Required API:**

- `GreenFunctionLibrary(root_path, rebuild_index=False)`
- `query(source_xyz_m, receiver_xyz_m, quantity="both") -> GreenQuery`
- `query_batch(source_xyz_m, receiver_xyz_m, quantity="both") -> list[GreenQuery]`

**Required behavior:**

- Build KDTree from SEM source coordinates in index.
- Route receiver coordinate to nearest SEM source run.
- Query selected `SourceRun` at source coordinate.
- Preserve reciprocity naming in returned `GreenQuery`.

**Validation:**

```bash
pytest tests/greenfun/test_library.py -v
```

**Commit:**

```bash
git add greenfun/library.py greenfun/__init__.py tests/greenfun/test_library.py
git commit -m "feat: query greenfun library by reciprocity"
```

______________________________________________________________________

## Wave 4: CLI and Final Integration

**Depends on:** Wave 1D, Wave 2B, Wave 3.

**Files owned:**

- Modify: `greenfun/query.py`
- Create: `tests/greenfun/test_cli.py`
- Modify integration files only if merge conflicts require it.

**Required CLI:**

```bash
gf_greenquery LIBRARY_ROOT \
  --source X Y Z \
  --receiver X Y Z \
  --quantity strain|displacement|both \
  --output result.npz \
  [--rebuild-index]
```

**Required output `.npz` keys:**

- Always: `time`, `source_xyz_m`, `receiver_xyz_m`, `sem_source_xyz_m`
- Optional: `strain`, `displacement`

**Validation:**

```bash
pytest tests/greenfun -v
python -m pip install -e .
gf_greenquery --help
```

Then run C++ schema test:

```bash
cd forward
cmake --build build --target test_postprocess_tile
ctest --test-dir build -R test_postprocess_tile --output-on-failure
```

Final formatter:

```bash
bash format.sh
git status --short
git diff --stat
```

**Commit:**

```bash
git add greenfun/query.py tests/greenfun/test_cli.py
git commit -m "feat: add greenfun query CLI"
```

If formatter changes files:

```bash
git add .
git commit -m "style: format greenfun implementation"
```

______________________________________________________________________

## Suggested Parallel Dispatch

After Wave 0 is merged, dispatch four agents:

1. **Agent C++ Tile Schema** → Wave 1A
1. **Agent Index Cache** → Wave 1B
1. **Agent Interpolator** → Wave 1C
1. **Agent Packaging Docs** → Wave 1D

After Wave 1 branches merge, dispatch two agents:

1. **Agent SourceRun** → Wave 2A
1. **Agent C++ Integration** → Wave 2B

Then run sequentially:

1. Wave 3 library routing
1. Wave 4 CLI/final integration

______________________________________________________________________

## Final Acceptance Checklist

- `pytest tests/greenfun -v` passes.
- `ctest --test-dir forward/build -R test_postprocess_tile --output-on-failure` passes.
- `examples/halfspace/postprocess.sh` writes tiles with `source_xyz_m` and `/mesh/vertex_coords`.
- `_greenfun_index.h5` rebuilds when tile stat hash changes.
- `gf_greenquery --help` imports and displays CLI help.
- `bash format.sh` exits 0.
- No JSON/text metadata manifests are added.

______________________________________________________________________

## Execution Handoff

Plan rewritten for parallel execution and saved to:

`docs/superpowers/plans/2026-07-10-greenfun-reader.md`

Recommended execution mode after approval: subagent-driven development with isolated worktrees, one agent per Wave 1 domain.
