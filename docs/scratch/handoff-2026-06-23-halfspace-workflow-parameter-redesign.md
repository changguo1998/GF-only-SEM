# Handoff: Halfspace Workflow Parameter Redesign

> Generated 2026-06-23. Continuation of brainstorming + plan-writing session.

## State

### Completed

**Design spec** fully written and user-approved:
`docs/superpowers/specs/2026-06-23-halfspace-workflow-parameter-redesign-design.md`

Covers:

- Timestep split: `solver_dt` (auto from CFL) + `output_dt_s` (snapshot interval), `output_dt_s = N × solver_dt`
- SI-unit suffixes on all config fields (`_m`, `_s`, `_m_s`, `_kg_m3`, `_hz`)
- Checkpoint → snapshot naming unification
- Full 500k-element halfspace config (10×10×5 km, 100m elements, N=4, vp=5km/s, etc.)
- config.h5 schema changes
- Change scope: 6 Python files, 3 C++ files, 4 test files, 3 doc files

### In Progress

**Implementation plan** started but only header written:
`docs/superpowers/plans/2026-06-23-halfspace-workflow-parameter-redesign.md`

The plan has the goal, architecture, tech stack, and global constraints. Zero tasks written. Editor was failing on the `---` anchor.

### Not Started

All implementation. No code touched yet.

## What The Plan Needs (Task Decomposition)

Refer to the spec file for full details. Task structure (14 tasks):

**Python preprocess changes (Tasks 1-6):**

| # | File | What |
|---|------|------|
| 1 | `preprocess/config_loader.py` | Update `REQUIRED_KEYS` (rename fields with suffixes, remove `nsteps`/`cfl_threshold`/`checkpoint_interval`, add `total_duration_s`). Update `REQUIRED_CALLABLES` (`vp`→`vp_m_s`, etc). Rename `_validate_checkpoint_precision`→`_validate_snapshot_precision`. Drop removed fields from `_validate_range`. |
| 2 | `preprocess/cfl_validator.py` | Replace `validate_cfl()` with `compute_solver_dt()`: search `stride=1..MAX_STRIDE(100)` for smallest integer where `output_dt_s/stride ≤ cfl_dt`. Return `(solver_dt, snapshot_stride)`. Keep `compute_cfl_dt()`. |
| 3 | `preprocess/stf_evaluator.py` | `evaluate_stf()` unchanged — it's just `stf_func, dt, nsteps`. The caller passes `solver_dt` now instead of `output_dt_s`. Only the CLI call site changes. |
| 4 | `preprocess/config_writer.py` | `_write_simulation()`: write `solver_dt`, `output_dt_s`, `snapshot_stride` instead of `dt`, `cfl_threshold`, `checkpoint_interval`. Rename `checkpoint_precision`→`snapshot_precision`. Signature takes `solver_dt` and `snapshot_stride` as new params. |
| 5 | `preprocess/preflight.py` | `run_preflight()`: replace CFL ratio check with `nsteps % snapshot_stride == 0` validation. `_check_storage()`: rename params, compute `nsnapshots = nsteps/snapshot_stride`. |
| 6 | `preprocess/cli.py` | After CFL step: call `compute_solver_dt()` to get `solver_dt, snapshot_stride`. Derive `nsteps = ceil(total_duration_s / solver_dt)`. Adjust `total_duration_s` if needed. Pass `solver_dt, snapshot_stride` to preflight, stf_evaluator, config_writer. Use `config.output_dt_s`, `config.source_x_m`, `config.source_y_m`. |

**Test updates (Tasks 7-11):**

| # | File | What |
|---|------|------|
| 7 | `tests/preprocess/conftest.py` | `config_dict` and `mock_config_module` — rename fields, add `total_duration_s`, remove old fields. `vp`→`vp_m_s`, etc. |
| 8 | `tests/preprocess/test_config_loader.py` | Rewrite all test config strings with new field names. Add `total_duration_s` test. Test missing `total_duration_s`. |
| 9 | `tests/preprocess/test_cfl_validator.py` | **New file**. Test `compute_cfl_dt()`, `compute_solver_dt()` with various output_dt_s values. |
| 10 | `tests/preprocess/test_config_writer.py` | Update `_make_mock_config()` with new fields. Update assertions for new `/simulation/` attrs. |
| 11 | `tests/workflows/test_halfspace_workflow.py` | Rewrite `_make_config_module()` with new fields. Use `create_regular_hex_mesh(nx=100, ny=100, nz=50, lx=10000, ly=10000, lz=5000)`. Update all assertions. |

**C++ forward solver (Task 12):**

| # | File | What |
|---|------|------|
| 12a | `forward/src/io.cpp` | Read `solver_dt` (not `dt`), `snapshot_stride`, `output_dt_s`, `snapshot_precision` from config.h5 `/simulation/`. |
| 12b | `forward/src/solver.cpp` | Newmark loop: use `solver_dt` for predictor/corrector. Write snapshot when `step % snapshot_stride == 0`. |
| 12c | `forward/src/record.cpp` | Rename internal "checkpoint" concepts to "snapshot". |

**Docs (Task 13):**

| # | File | What |
|---|------|------|
| 13a | `docs/design-decisions.md` | Update dt section (~line 237), config field table (~line 192). |
| 13b | `docs/superpowers/design/preprocess.md` | Update config example, CFL section, config.h5 schema. |
| 13c | `docs/superpowers/design/forward.md` | Update Newmark loop dt source, config.h5 reader, checkpoint→snapshot. |

## Key Design Decisions (from approved spec)

- `solver_dt` computed by searching: `for stride=1..100: if output_dt_s/stride ≤ cfl_dt: break`
- `nsteps = ceil(total_duration_s / solver_dt)`; warn if `total_duration_s` adjusted
- `snapshot_stride = output_dt_s / solver_dt` (must be integer)
- `cfl_threshold` removed — obsolete when solver_dt IS the CFL dt
- `checkpoint_interval` removed — replaced by derived `snapshot_stride`
- Storage: 500k elem × 125 GLL × 6 components × 4 bytes = 1.43 GB/snapshot; 500 snapshots × 3 dirs ≈ 2.2 TB; `storage_limit_gb = 2500`

## Suggested Skills

The next agent should invoke:

1. **writing-plans** — to finish the implementation plan (or continue from the partial plan file, adding tasks 1-14 with exact code)
1. **brainstorming** — if any parameter decisions still need clarification
1. After plan: **subagent-driven-development** or **executing-plans** for implementation

## Related Artifacts

- Spec: `docs/superpowers/specs/2026-06-23-halfspace-workflow-parameter-redesign-design.md`
- Partial plan: `docs/superpowers/plans/2026-06-23-halfspace-workflow-parameter-redesign.md` (header only, no tasks)
- Project context: `AGENTS.md`, `docs/design-decisions.md`
- Design docs: `docs/superpowers/design/preprocess.md`, `docs/superpowers/design/forward.md`
