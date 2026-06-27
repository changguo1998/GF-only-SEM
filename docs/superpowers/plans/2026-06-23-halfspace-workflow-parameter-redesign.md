# Halfspace Workflow Parameter Redesign — Implementation Plan

> Use `subagent-driven-development` or `executing-plans` to run tasks. Check boxes track progress.

**Goal:** Split solver timestep from snapshot interval, add SI-unit config names, scale halfspace workflow, and rename checkpoint output to snapshot output.

**Design:** `docs/superpowers/specs/2026-06-23-halfspace-workflow-parameter-redesign-design.md`.

## Architecture

- `config_loader.py`: validate new names and `total_duration_s`.
- `cfl_validator.py`: compute `solver_dt` and `snapshot_stride`.
- `cli.py`: derive `nsteps` and pass timing through pipeline.
- `config_writer.py`: write new `/simulation/` schema.
- `preflight.py`: validate stride and storage.
- Tests: update fixtures, add CFL tests, scale workflow test.
- Forward C++: read `solver_dt`; write snapshots by `snapshot_stride`.

## Constraints

- Use SI suffixes: `_m`, `_s`, `_m_s`, `_kg_m3`, `_hz`.
- No suffix for counts or dimensionless fields.
- Remove user config fields: `nsteps`, `cfl_threshold`, `checkpoint_interval`.
- Rename `checkpoint_precision` → `snapshot_precision`.
- Rename callables: `vp` → `vp_m_s`, `vs` → `vs_m_s`, `density` → `density_kg_m3`.
- Add required `total_duration_s`.
- Require integer `output_dt_s / solver_dt` within `1e-12`.
- Use `MAX_STRIDE = 100`.
- Require `nsteps % snapshot_stride == 0`.
- Round `nsteps = ceil(total_duration_s / solver_dt)`; warn if duration changes.
- Estimate storage with `nsteps / snapshot_stride` snapshots.
- Halfspace target: `100 × 100 × 50` elements, 100 m cubes.

## Task 1 — `config_loader.py` ✓

- [x] Update required keys with SI suffixes.
- [x] Add `total_duration_s`.
- [x] Remove `nsteps`, `cfl_threshold`, and `checkpoint_interval`.
- [x] Rename `checkpoint_precision` to `snapshot_precision`.
- [x] Rename material callables.
- [x] Update validation names and errors.

## Task 2 — `cfl_validator.py` ✓

- [x] Keep `compute_cfl_dt()`.
- [x] Remove `validate_cfl()`.
- [x] Add `compute_solver_dt(output_dt_s, cfl_dt, max_stride=100)`.
- [x] Search stride 1..100 for `output_dt_s / stride <= cfl_dt`.
- [x] Return `(solver_dt, snapshot_stride)`.
- [x] Raise clear error if no stride works.

## Task 3 — `stf_evaluator.py` ✓

- [x] Keep signature `(stf_func, dt, nsteps)`.
- [x] Ensure caller passes `solver_dt` as `dt`.
- [x] Verify `t_s = k * dt` remains correct.

## Task 4 — `config_writer.py` ✓

- [x] Add `solver_dt` and `snapshot_stride` to `_write_simulation()`.
- [x] Write `solver_dt`, `output_dt_s`, `snapshot_stride`, `nsteps`, and `snapshot_precision`.
- [x] Remove old `dt`, `cfl_threshold`, and `checkpoint_interval`.
- [x] Update types and docstrings.

## Task 5 — `preflight.py` ✓

- [x] Replace CFL-ratio check with stride validation.
- [x] Validate `nsteps % snapshot_stride == 0`.
- [x] Rename storage variables from checkpoint to snapshot.
- [x] Use `nsnapshots = nsteps / snapshot_stride`.

## Task 6 — `cli.py` ✓

- [x] Import `compute_solver_dt`.
- [x] Compute `solver_dt` and `snapshot_stride` after CFL.
- [x] Derive and possibly round `nsteps`.
- [x] Warn if effective duration changes.
- [x] Pass new values to preflight, STF evaluator, and config writer.
- [x] Use `source_x_m`, `source_y_m`, and `output_dt_s`.

## Task 7 — Preprocess Fixtures ✓

**File:** `tests/preprocess/conftest.py`

- [x] Update config fixtures to new names.
- [x] Add `total_duration_s`.
- [x] Remove old time/checkpoint fields.
- [x] Rename material callables.

## Task 8 — Config Loader Tests ✓

**File:** `tests/preprocess/test_config_loader.py`

- [x] Rewrite config strings with new names.
- [x] Add required-field tests for `total_duration_s`.
- [x] Update assertions and imports.

## Task 9 — CFL Validator Tests ✓

**File:** `tests/preprocess/test_cfl_validator.py`

- [x] Test `compute_cfl_dt()`.
- [x] Test integer stride search.
- [x] Test stride 1.
- [x] Test exact equality with CFL.
- [x] Test no valid stride.
- [x] Test floating tolerance.

## Task 10 — Config Writer Tests ✓

**File:** `tests/preprocess/test_config_writer.py`

- [x] Update mock config.
- [x] Assert new `/simulation/` attrs.
- [x] Assert removed attrs are absent.
- [x] Assert `snapshot_precision` is written.

## Task 11 — Halfspace Workflow Test

**File:** `tests/workflows/test_halfspace_workflow.py`

- [ ] Use new config names.
- [ ] Use `total_duration_s`.
- [ ] Use `snapshot_precision`.
- [ ] Use `source_x_m`, `source_y_m`.
- [ ] Use `vp_m_s`, `vs_m_s`, `density_kg_m3`.
- [ ] Build `100 × 100 × 50` mesh over `10 × 10 × 5 km`.
- [ ] Assert derived `solver_dt`, `snapshot_stride`, `nsteps`, and storage.

> **Note:** Workflow test was deleted in commit 9c430ec (moved mesh dims to config.py). Needs recreation.

## Task 12 — C++ Forward Solver ✓

- [x] Read `solver_dt`, `snapshot_stride`, `output_dt_s`, and `snapshot_precision`.
- [x] Use `solver_dt` in Newmark.
- [x] Write snapshots when `step % snapshot_stride == 0`.
- [x] Rename internal checkpoint terms to snapshot.
- [x] Use `nsteps / snapshot_stride` for storage estimates.

## Task 13 — Docs

- [ ] Update `docs/design-decisions.md`.
- [ ] Update `docs/superpowers/design/preprocess.md`.
- [ ] Update `docs/superpowers/design/forward.md`.

## Task 14 — Integration

- [ ] Run halfspace workflow.
- [ ] Verify `config.h5` schema.
- [ ] Verify timing consistency.
- [ ] Verify storage estimates.
- [ ] Run related Python tests.
- [ ] Run full suite.