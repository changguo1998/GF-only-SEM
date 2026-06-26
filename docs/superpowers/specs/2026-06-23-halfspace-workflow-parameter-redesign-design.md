# Halfspace Workflow Parameter Redesign + Timestep Split

> Date: 2026-06-23
> Status: approved

## Motivation

The existing halfspace workflow test (`tests/workflows/test_halfspace_workflow.py`)
uses a 1-element (4×4×2 km) mesh that conflates smoke-test and production
purposes. The `output_dt` parameter is overloaded as both the simulation
timestep and the output interval, with no separation between solver stability
and snapshot frequency. The checkpoint naming is inconsistent with the
output-dt concept. This design separates concerns, scales the test to a
production-representative mesh, and establishes consistent naming.

## Goals

1. Split `output_dt` into two independent parameters:
   - `solver_dt` — auto-computed from CFL, used by the forward solver's Newmark loop
   - `output_dt` — user-specified snapshot interval (must be integer multiple of solver_dt)
1. Scale the halfspace workflow to 500k-element mesh (10×10×5 km, 100 m elements)
1. Unify "checkpoint" and "snapshot" naming → "snapshot" everywhere
1. Add SI-unit suffixes to all config fields
1. Remove `cfl_threshold` (obsolete with auto solver_dt) and `nsteps` (derived from total_duration)

## Config Schema (Final)

```python
title = "halfspace_10x10x5"

# Solver
polynomial_order = 4

# Time
output_dt_s       = 0.01    # snapshot interval (s)
total_duration_s  = 5.0     # simulation duration (s)
cfl_safety        = 0.5     # solver_dt = cfl_safety × h_min / vp_max

# Source — center of free surface
source_x_m = 5000.0
source_y_m = 5000.0

# Material — homogeneous halfspace
def vp_m_s(x_m, y_m, z_m):        return 5000.0
def vs_m_s(x_m, y_m, z_m):        return 3000.0
def density_kg_m3(x_m, y_m, z_m): return 2700.0

# STF — Ricker wavelet
def stf_func(t_s):
    import numpy as np
    f0_hz = 2.0; t0_s = 1.0
    a = np.pi * f0_hz * (t_s - t0_s)
    return (1 - 2 * a**2) * np.exp(-a**2)

# PML — 5 element layers, free surface at zmin
pml_thickness = {"xmin": 5, "xmax": 5, "ymin": 5, "ymax": 5, "zmin": 0, "zmax": 5}

# Partition
n_ranks = 16

# Storage
snapshot_precision = "float32"
storage_limit_gb   = 2500
```

## Unit Suffix Convention

| Category | Suffix | Example |
|----------|--------|---------|
| Distance/position | `_m` | `source_x_m`, material functions `(x_m, y_m, z_m)` |
| Time/duration | `_s` | `output_dt_s`, `total_duration_s`, `stf_func(t_s)` |
| Velocity | `_m_s` | `vp_m_s()`, `vs_m_s()` |
| Density | `_kg_m3` | `density_kg_m3()` |
| Frequency | `_hz` | `f0_hz` (local), `t0_s` (local) |
| Element count | no suffix | `pml_thickness` values |
| Dimensionless | no suffix | `polynomial_order`, `cfl_safety`, `n_ranks` |
| Prefixed units | keep | `storage_limit_gb`, `snapshot_precision` |

## Derived Parameters (Auto-Computed)

| Derived | Formula | Value |
|---------|---------|-------|
| `h_min` | Minimum GLL node spacing across all elements | ~17.3 m |
| `solver_dt_cfl` | `cfl_safety × h_min / vp_max` | 0.5 × 17.3 / 5000 = 0.00173 s |
| `solver_dt` | Search: `output_dt_s / stride ≤ solver_dt_cfl` | 0.01 / 6 = 0.001667 s |
| `snapshot_stride` | `output_dt_s / solver_dt` (integer) | 6 |
| `nsteps` | `total_duration_s / solver_dt` | 5.0 / 0.001667 = 3000 |
| Effective duration | `nsteps × solver_dt` | 3000 × 0.001667 = 5.0 s |
| Snapshots per run | `nsteps / snapshot_stride` | 3000 / 6 = 500 |

## solver_dt Selection Algorithm

```
solver_dt_cfl = cfl_safety × h_min / vp_max

for stride = 1, 2, 3, ... up to MAX_STRIDE (100):
    solver_dt = output_dt_s / stride
    if solver_dt ≤ solver_dt_cfl:
        snapshot_stride = stride
        break

if stride > MAX_STRIDE:
    error: "output_dt_s too large for CFL limit — increase cfl_safety or reduce output_dt_s"
```

Constraint: `nsteps % snapshot_stride == 0` (last solver step lands on a snapshot).

## Mesh

| Parameter | Value |
|-----------|-------|
| Domain | 10 km × 10 km × 5 km |
| Element size | 100 m cubes |
| Elements | 100 × 100 × 50 = 500,000 |
| Interior elements | 90 × 90 × 45 = 364,500 |
| PML elements | 135,500 (27%) |

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Polynomial order `N` | 4 |
| GLL nodes per element | 125 |
| Total GLL nodes | 62,500,000 |
| STF | Ricker, f0 = 2 Hz, t0 = 1.0 s |
| λ_min at 2 Hz | 5000 / 2 = 2500 m → 25 elements/λ |
| Free surface | zmin (z = 0 km) |
| PML faces | xmin, xmax, ymin, ymax, zmax |
| PML layers per face | 5 |

## Memory & Storage Estimate

### Per-rank memory (n_ranks=16, ~31k elements/rank)

At float64 (precomputed fields):

- Mesh fields: ~0.6 GB/rank
- Runtime u,v,a,r: ~0.7 GB/rank
- Strain: ~0.4 GB/rank
- Total: ~1.7 GB/rank

### Disk storage (snapshot_precision=float32)

| Item | Size |
|------|------|
| Strain per snapshot | 500k × 125 × 6 × 4 B = 1.43 GB |
| Snapshots per run | 500 |
| One direction | 715 GB |
| × 3 directions | ~2.15 TB |
| Restart + partitions | ~15 GB |
| Total | ~2.2 TB |

## Config Field Changes (From Current)

| Old | New | Action |
|-----|-----|--------|
| `nsteps` | — | Removed. Derived from `total_duration_s / solver_dt`. |
| `cfl_threshold` | — | Removed. solver_dt is auto-computed, no ratio to validate. |
| `checkpoint_interval` | — | Removed. Replaced by `snapshot_stride` (derived). |
| `checkpoint_precision` | `snapshot_precision` | Renamed. |
| `output_dt` | `output_dt_s` | Renamed with unit suffix. Changed meaning: snapshot interval (not solver timestep). |
| — | `total_duration_s` | New. Total simulation duration in seconds. |
| — | `solver_dt` | New. Auto-computed CFL timestep (written to config.h5). |
| — | `snapshot_stride` | New. Solver steps per snapshot (derived, written to config.h5). |
| `source_x` | `source_x_m` | Unit suffix. |
| `source_y` | `source_y_m` | Unit suffix. |
| `vp()` | `vp_m_s()` | Unit suffix. |
| `vs()` | `vs_m_s()` | Unit suffix. |
| `density()` | `density_kg_m3()` | Unit suffix. |
| `stf_func(t)` | `stf_func(t_s)` | Unit suffix on parameter. |

## config.h5 `/simulation/` Schema Changes

```
/simulation/  OLD                          /simulation/  NEW
├── dt           → REMOVED       ├── solver_dt          (NEW, float64)
├── nsteps       → KEPT          ├── output_dt_s        (NEW, float64)
├── cfl_threshold → REMOVED      ├── snapshot_stride    (NEW, int32)
├── checkpoint_interval → REMOVED├── snapshot_precision (renamed from checkpoint_precision)
├── checkpoint_precision → RENAMED├── nsteps            (derived, int32)
                                   ├── polynomial_order
                                   ├── cfl_safety
                                   └── storage_limit_gb
```

## Files Changed

### Python (preprocess)

| File | Change |
|------|--------|
| `config_loader.py` | `REQUIRED_KEYS`: rename fields with unit suffixes; add `total_duration_s`; remove `nsteps`, `cfl_threshold`, `checkpoint_interval`; rename `checkpoint_precision` → `snapshot_precision`. `REQUIRED_CALLABLES`: rename `vp`→`vp_m_s`, `vs`→`vs_m_s`, `density`→`density_kg_m3`. |
| `cfl_validator.py` | Replace `validate_cfl()` with `compute_solver_dt()` — search for stride giving `output_dt_s / stride ≤ solver_dt_cfl`. Returns `(solver_dt, snapshot_stride)`. |
| `preflight.py` | Replace CFL ratio check with snapshot-stride validation. Validate `nsteps % snapshot_stride == 0`. |
| `config_writer.py` | Write `solver_dt`, `output_dt_s`, `snapshot_stride` to `/simulation/`. Rename `checkpoint_precision` → `snapshot_precision`. |
| `stf_evaluator.py` | Evaluate `stf_func(t_s)` at `solver_dt` intervals over `nsteps`. |
| `cli.py` | Derive `nsteps = total_duration_s / solver_dt`. Wire `solver_dt` and `snapshot_stride` through pipeline. |

### C++ (forward solver)

| File | Change |
|------|--------|
| config reader | Read `solver_dt` (not `dt`) for timestep; read `output_dt_s`, `snapshot_stride` for output logic; read `snapshot_precision`. |
| Newmark loop | Use `solver_dt` for predictor/corrector step. |
| Snapshot writer | Write snapshot when `step % snapshot_stride == 0`. Append strain to record file. Rename internal "checkpoint" → "snapshot". |
| Storage calc | Use `nsteps / snapshot_stride` for expected snapshot count. |

### Tests

| File | Change |
|------|--------|
| `tests/workflows/test_halfspace_workflow.py` | Rewrite `_make_config_module()` with new field names, unit suffixes, 500k-element mesh. |
| `tests/preprocess/test_config_loader.py` | Add tests for `total_duration_s`, derived `nsteps` validation, removed fields. |
| `tests/preprocess/test_cfl_validator.py` | Tests for integer stride solver_dt search. |
| `tests/workflows/regular_hex_mesh.py` | Verify `create_regular_hex_mesh(nx=100, ny=100, nz=50, lx=10000, ly=10000, lz=5000)` works. |

### Docs

| File | Change |
|------|--------|
| `docs/design-decisions.md` | Update dt section, config field names, remove CFL threshold. |
| `docs/superpowers/design/preprocess.md` | Update config example, CFL section, config.h5 schema, checkpoint→snapshot. |
| `docs/superpowers/design/forward.md` | Update Newmark loop dt source, writer naming, config.h5 reader. |

## Edge Cases

| Condition | Handling |
|-----------|----------|
| `output_dt_s < solver_dt_cfl` | `stride=1`, `solver_dt=output_dt_s`. Warn: solver runs finer than CFL requires. |
| No stride found within `MAX_STRIDE` | Error: suggest smaller `output_dt_s` or larger elements. |
| `total_duration_s / solver_dt` not integer | Round `nsteps` up; adjust `total_duration_s = nsteps × solver_dt`. Warn user. |
| `nsteps % snapshot_stride ≠ 0` | Assertion error (should be impossible by construction). |
| PML thickness exceeds available elements | Preprocess boundary detector validates; abort with message. |
