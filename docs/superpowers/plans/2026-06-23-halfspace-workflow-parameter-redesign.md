# Halfspace Workflow Parameter Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `output_dt` into solver-internal `solver_dt` (auto-computed from CFL) and user-facing `output_dt_s` (snapshot interval), scale halfspace workflow to 500k elements, add SI-unit suffixes, and unify checkpoint→snapshot naming.

**Architecture:** `config_loader.py` validates new fields. `cfl_validator.py` computes `solver_dt` by searching `output_dt_s / stride ≤ cfl_dt`. `cli.py` derives `nsteps` and `snapshot_stride` and wires them through the pipeline. `config_writer.py` writes the new `/simulation/` schema. `preflight.py` validates stride integer-ness. Tests update fixtures, add new tests, and scale the halfspace workflow test.

**Tech Stack:** Python 3.10+, numpy, h5py, pytest

## Global Constraints

- SI-unit suffixes on config fields: `_m`, `_s`, `_m_s`, `_kg_m3`, `_hz`
- Element counts and dimensionless values: no suffix
- `pml_thickness` values: no suffix (element count)
- Remove: `nsteps`, `cfl_threshold`, `checkpoint_interval` from user config
- Rename: `checkpoint_precision` → `snapshot_precision`
- Rename callables: `vp`→`vp_m_s`, `vs`→`vs_m_s`, `density`→`density_kg_m3`
- New required field: `total_duration_s`
- `output_dt_s / solver_dt` must be integer (within `1e-12` tolerance)
- `MAX_STRIDE = 100` for solver_dt search
- `nsteps % snapshot_stride == 0`
- Round `total_duration_s / solver_dt` up for `nsteps`; warn if adjusted
- All storage estimates use `nsteps / snapshot_stride` snapshots (not `ceil(nsteps/interval)+1`)

______________________________________________________________________

````

I'll now write the full implementation plan with all tasks, exact code, and test expectations. The plan will be comprehensive — covering the 8 Python file changes, all test updates, and the C++ forward solver modifications needed.</think>This is a large plan (14 tasks across Python, C++, and docs). Let me write it efficiently.

<｜DSML｜tool_calls>
<｜DSML｜invoke name="write">
<｜DSML｜parameter name="content" string="true"># Halfspace Workflow Parameter Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `output_dt` into auto-computed `solver_dt` (CFL) and user-facing `output_dt_s` (snapshot interval). Scale halfspace workflow to 500k elements. Add SI-unit suffixes. Unify checkpoint→snapshot naming.

**Architecture:** `config_loader.py` validates renamed fields with unit suffixes and new `total_duration_s`. `cfl_validator.py` replaces `validate_cfl()` with `compute_solver_dt()` that searches `output_dt_s / stride ≤ cfl_dt`. `cli.py` derives `nsteps` and `snapshot_stride`. `config_writer.py` writes new `/simulation/` schema. `preflight.py` validates stride integer-ness and renames storage estimate. Tests update fixtures, add new test file for CFL validator, and scale the halfspace workflow test. C++ forward solver reads `solver_dt` and uses `snapshot_stride` for output writes.

**Tech Stack:** Python 3.10+, numpy, h5py, pytest, C++17/MPI/Eigen3/HDF5

## Global Constraints

- SI-unit suffixes: `_m`, `_s`, `_m_s`, `_kg_m3`, `_hz`; element counts/dimensionless: no suffix
- Remove from user config: `nsteps`, `cfl_threshold`, `checkpoint_interval`
- Rename: `checkpoint_precision` → `snapshot_precision`
- Rename callables: `vp`→`vp_m_s`, `vs`→`vs_m_s`, `density`→`density_kg_m3`
- New required field: `total_duration_s` (float)
- `output_dt_s / solver_dt` must be integer (tolerance `1e-12`)
- `MAX_STRIDE = 100` for solver_dt search
- `nsteps % snapshot_stride == 0` validated
- Round `nsteps = ceil(total_duration_s / solver_dt)`, adjust `total_duration_s = nsteps × solver_dt`
- Storage: `nsnapshots = nsteps / snapshot_stride` replaces old `ceil(nsteps/checkpoint_interval)+1`
- Halfspace workflow: 100×100×50 elements (10×10×5 km, 100 m cubes)

## Tasks

### Task 1 — Update `config_loader.py`

**File:** `preprocess/config_loader.py`

**Changes:**
1. `REQUIRED_KEYS`: rename fields with unit suffixes
   - `output_dt` → `output_dt_s`
   - Add `total_duration_s`
   - Remove `nsteps`, `cfl_threshold`, `checkpoint_interval`
   - Rename `checkpoint_precision` → `snapshot_precision`
2. `REQUIRED_CALLABLES`: rename `vp` → `vp_m_s`, `vs` → `vs_m_s`, `density` → `density_kg_m3`
3. `_validate_checkpoint_precision()` → rename to `_validate_snapshot_precision()`
4. Drop removed fields from `_validate_range()`
5. Update import paths and any references to old names

**Checklist:**
- [ ] Update `REQUIRED_KEYS` set
- [ ] Update `REQUIRED_CALLABLES` set
- [ ] Rename validation function
- [ ] Update `_validate_range()` to exclude removed fields
- [ ] Update any error messages referencing old field names

### Task 2 — Rewrite `cfl_validator.py`

**File:** `preprocess/cfl_validator.py`

**Changes:**
1. Remove `validate_cfl()` function entirely
2. Keep `compute_cfl_dt(cfl_safety, h_min, vp_max)` — unchanged
3. Add `compute_solver_dt(output_dt_s, cfl_dt, max_stride=100)`:
   - Search `stride=1..max_stride`: if `output_dt_s / stride ≤ cfl_dt`, break
   - Return `(solver_dt=output_dt_s/stride, snapshot_stride=stride)`
   - If no stride found, raise ValueError with descriptive message
4. Export both functions in `__init__.py` (or `__all__`)

**Checklist:**
- [ ] Remove `validate_cfl()`
- [ ] Add `compute_solver_dt()` with stride search
- [ ] Export new function
- [ ] Update module docstring

### Task 3 — Update `stf_evaluator.py`

**File:** `preprocess/stf_evaluator.py`

**Changes:**
- `evaluate_stf()` signature unchanged — it takes `(stf_func, dt, nsteps)`.
- The caller (`cli.py`) will now pass `solver_dt` as `dt` instead of `output_dt_s`.
- No code changes needed in this file, but verify the existing interface works with solver_dt.

**Checklist:**
- [ ] Verify interface works with `solver_dt` as `dt` parameter
- [ ] Ensure `t_s = k * dt` still produces correct values for `stf_func(t_s)`

### Task 4 — Update `config_writer.py`

**File:** `preprocess/config_writer.py`

**Changes:**
1. `_write_simulation()` signature: add `solver_dt` and `snapshot_stride` parameters
2. Write new fields to `/simulation/`:
   - `solver_dt` (float64)
   - `output_dt_s` (float64) — from config
   - `snapshot_stride` (int32)
   - `nsteps` (int32) — from preflight
   - `snapshot_precision` (string) — renamed from `checkpoint_precision`
3. Remove old fields: `dt`, `cfl_threshold`, `checkpoint_interval`
4. Rename: `checkpoint_precision` → `snapshot_precision`
5. Update type hints and docstring

**Checklist:**
- [ ] Update `_write_simulation()` signature
- [ ] Write new `/simulation/` attributes
- [ ] Remove old attributes
- [ ] Rename attribute references
- [ ] Update docstring

### Task 5 — Update `preflight.py`

**File:** `preprocess/preflight.py`

**Changes:**
1. `run_preflight()`: replace CFL ratio check with `nsteps % snapshot_stride == 0` validation
2. `_check_storage()`: rename `checkpoint_interval` → `snapshot_stride`, compute `nsnapshots = nsteps / snapshot_stride`
3. Update any variable names referencing old field names
4. Update error messages referencing old field names

**Checklist:**
- [ ] Replace CFL ratio check
- [ ] Update `_check_storage()` parameter names
- [ ] Update variable names
- [ ] Update error messages

### Task 6 — Update `cli.py`

**File:** `preprocess/cli.py`

**Changes:**
1. After CFL computation, call `compute_solver_dt()` to get `(solver_dt, snapshot_stride)`
2. Derive `nsteps = ceil(total_duration_s / solver_dt)`
3. Adjust `total_duration_s = nsteps * solver_dt` if needed; warn if adjusted
4. Pass `solver_dt` and `snapshot_stride` to preflight, stf_evaluator, config_writer
5. Use `config.output_dt_s`, `config.source_x_m`, `config.source_y_m`
6. Replace `config.cfl_threshold` usage with `solver_dt`-based logic

**Checklist:**
- [ ] Import `compute_solver_dt` from `cfl_validator`
- [ ] Add solver_dt derivation after CFL step
- [ ] Add nsteps derivation with ceil and warning
- [ ] Wire values to all downstream calls
- [ ] Update field name references

### Task 7 — Update `tests/preprocess/conftest.py`

**File:** `tests/preprocess/conftest.py`

**Changes:**
- Update `config_dict` to use new field names with unit suffixes
- Update `mock_config_module()` to use new field names
- Add `total_duration_s` to mock configs
- Remove `nsteps`, `cfl_threshold`, `checkpoint_interval`
- Rename `checkpoint_precision` → `snapshot_precision`
- Rename `vp`→`vp_m_s`, `vs`→`vs_m_s`, `density`→`density_kg_m3`

**Checklist:**
- [ ] Update `config_dict`
- [ ] Update `mock_config_module()`
- [ ] Update all fixture values

### Task 8 — Update `tests/preprocess/test_config_loader.py`

**File:** `tests/preprocess/test_config_loader.py`

**Changes:**
- Rewrite all test config strings with new field names
- Add test for `total_duration_s` as required field
- Add test for missing `total_duration_s` raising error
- Update assertion references to new field names
- Update import paths if needed

**Checklist:**
- [ ] Update all test config dictionaries
- [ ] Add `total_duration_s` validation tests
- [ ] Update assertions
- [ ] Update imports

### Task 9 — New file `tests/preprocess/test_cfl_validator.py`

**File:** `tests/preprocess/test_cfl_validator.py`

**Content:**
- Test `compute_cfl_dt()` with known inputs
- Test `compute_solver_dt()`:
  - Exact integer stride (output_dt_s=0.01, cfl_dt=0.00173 → stride=6)
  - stride=1 case (output_dt_s < cfl_dt)
  - No stride found within limit (output_dt_s too large)
  - Edge: output_dt_s exactly equals cfl_dt
  - Edge: floating point tolerance with `1e-12`

**Checklist:**
- [ ] Create test file
- [ ] Test `compute_cfl_dt()`
- [ ] Test `compute_solver_dt()` happy path
- [ ] Test `compute_solver_dt()` edge: stride=1
- [ ] Test `compute_solver_dt()` error: no stride found
- [ ] Test `compute_solver_dt()` edge: exact equality

### Task 10 — Update `tests/preprocess/test_config_writer.py`

**File:** `tests/preprocess/test_config_writer.py`

**Changes:**
- Update `_make_mock_config()` with new field names
- Update assertions for new `/simulation/` attributes
- Update for renamed `checkpoint_precision` → `snapshot_precision`
- Update for removed old attributes

**Checklist:**
- [ ] Update `_make_mock_config()`
- [ ] Update assertions for new schema
- [ ] Update attribute name references

### Task 11 — Update `tests/workflows/test_halfspace_workflow.py`

**File:** `tests/workflows/test_halfspace_workflow.py`

**Changes:**
- Rewrite `_make_config_module()` with new fields:
  - `output_dt_s` (not `output_dt`)
  - `total_duration_s` (new, e.g. 5.0)
  - Remove `nsteps`, `cfl_threshold`, `checkpoint_interval`
  - `snapshot_precision` (not `checkpoint_precision`)
  - `source_x_m`, `source_y_m`
  - `vp_m_s()`, `vs_m_s()`, `density_kg_m3()`
- Use `create_regular_hex_mesh(nx=100, ny=100, nz=50, lx=10000, ly=10000, lz=5000)`
- Update all assertions for new derived values:
  - `nsteps = ceil(5.0 / solver_dt)`
  - `nsnapshots = nsteps / snapshot_stride`
  - Storage estimates use new fields

**Checklist:**
- [ ] Rewrite config module creation
- [ ] Scale mesh to 500k elements
- [ ] Update assertions
- [ ] Update any workflow-specific logic

### Task 12 — Update C++ forward solver

**Files:**
- `forward/src/io.cpp` — config.json reader
- `forward/src/solver.cpp` — Newmark loop
- `forward/src/record.cpp` — snapshot writer
- `forward/include/forward/config.hpp` — corresponding header types

**Changes:**
1. Config reader: read `solver_dt` (not `dt`), `snapshot_stride`, `output_dt_s`, `snapshot_precision`
2. Newmark loop: use `solver_dt` for predictor/corrector; write snapshot when `step % snapshot_stride == 0`
3. Record writer: rename internal "checkpoint" → "snapshot"
4. Storage calc: use `nsteps / snapshot_stride`

**Checklist:**
- [ ] Update config reader in `io.cpp`
- [ ] Update config struct in `config.hpp`
- [ ] Update Newmark loop in `solver.cpp`
- [ ] Update record writer in `record.cpp`
- [ ] Update storage estimate calculations

### Task 13 — Docs updates

**Files:**
- `docs/design-decisions.md` — Update dt section, config field names
- `docs/superpowers/design/preprocess.md` — Update config example, CFL section, config.h5 schema
- `docs/superpowers/design/forward.md` — Update Newmark loop dt source, config.h5 reader, checkpoint→snapshot

**Checklist:**
- [ ] Update `docs/design-decisions.md`
- [ ] Update `docs/superpowers/design/preprocess.md`
- [ ] Update `docs/superpowers/design/forward.md`

### Task 14 — Integration test

**Changes:**
- Run the full halfspace workflow end-to-end
- Verify config.h5 contains expected schema
- Verify solver_dt, snapshot_stride, nsteps are consistent
- Verify storage estimates match expectations

**Checklist:**
- [ ] Run `test_halfspace_workflow.py` — pass
- [ ] Run config_loader tests — pass
- [ ] Run cfl_validator tests — pass
- [ ] Run config_writer tests — pass
- [ ] Run full test suite — pass

---```
````
