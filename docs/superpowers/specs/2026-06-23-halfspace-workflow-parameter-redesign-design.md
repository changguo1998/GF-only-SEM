# Halfspace Workflow Parameter Redesign + Timestep Split

> Date: 2026-06-23
> Status: approved

## Motivation

The halfspace workflow used a 1-element smoke mesh. It also used `output_dt` for both solver stability and snapshot cadence. That mixed two concerns. Checkpoint naming also conflicted with output naming.

This design splits solver timestep from output interval, scales the workflow, and standardizes names.

## Goals

1. Split time config:
   - `solver_dt`: derived from CFL; used by Newmark.
   - `output_dt_s`: user snapshot interval; integer multiple of `solver_dt`.
1. Scale halfspace workflow to 500k elements.
1. Rename checkpoint output to snapshot output.
1. Add SI-unit suffixes to config fields.
1. Remove obsolete `cfl_threshold` and user-provided `nsteps`.

## Final Config Schema

```python
title = "halfspace_10x10x5"
polynomial_order = 4

output_dt_s = 0.01
total_duration_s = 5.0
cfl_safety = 0.5

source_x_m = 5000.0
source_y_m = 5000.0

def vp_m_s(x_m, y_m, z_m): return 5000.0
def vs_m_s(x_m, y_m, z_m): return 3000.0
def density_kg_m3(x_m, y_m, z_m): return 2700.0

def stf_func(t_s):
    import numpy as np
    f0_hz = 2.0
    t0_s = 1.0
    a = np.pi * f0_hz * (t_s - t0_s)
    return (1 - 2 * a**2) * np.exp(-a**2)

pml_thickness = {"xmin": 5, "xmax": 5, "ymin": 5, "ymax": 5, "zmin": 0, "zmax": 5}
n_ranks = 16
snapshot_precision = "float32"
storage_limit_gb = 2500
```

## Unit Suffix Rules

| Kind | Suffix | Example |
|------|--------|---------|
| Position/distance | `_m` | `source_x_m` |
| Time | `_s` | `output_dt_s` |
| Velocity | `_m_s` | `vp_m_s()` |
| Density | `_kg_m3` | `density_kg_m3()` |
| Frequency | `_hz` | `f0_hz` |
| Counts | none | `n_ranks` |
| Dimensionless | none | `cfl_safety` |
| Prefixed units | keep | `storage_limit_gb` |

## Derived Time Fields

| Field | Formula | Example |
|-------|---------|---------|
| `h_min` | minimum GLL spacing | ~17.3 m |
| `solver_dt_cfl` | `cfl_safety × h_min / vp_max` | 0.00173 s |
| `solver_dt` | first `output_dt_s / stride ≤ solver_dt_cfl` | 0.001667 s |
| `snapshot_stride` | `output_dt_s / solver_dt` | 6 |
| `nsteps` | `ceil(total_duration_s / solver_dt)` | 3000 |
| snapshots | `nsteps / snapshot_stride` | 500 |

Algorithm:

```text
for stride in 1..100:
    solver_dt = output_dt_s / stride
    if solver_dt <= solver_dt_cfl:
        snapshot_stride = stride
        break
```

If no stride works, reduce `output_dt_s` or increase mesh size. Require `nsteps % snapshot_stride == 0`.

## Mesh

| Parameter | Value |
|-----------|-------|
| Domain | 10 km × 10 km × 5 km |
| Element size | 100 m cubes |
| Elements | 100 × 100 × 50 = 500,000 |
| Interior | 90 × 90 × 45 = 364,500 |
| PML | 135,500 elements |

## Simulation

| Parameter | Value |
|-----------|-------|
| Polynomial order | 4 |
| GLL nodes/element | 125 |
| Total GLL nodes | 62,500,000 |
| STF | Ricker, 2 Hz, t0 = 1 s |
| λ_min at 2 Hz | 2500 m, 25 elements/λ |
| Free surface | zmin |
| PML faces | xmin, xmax, ymin, ymax, zmax |
| PML layers | 5 |

## Memory and Storage

Per rank at 16 ranks and float64 precomputed fields:

- Mesh fields: ~0.6 GB.
- Runtime `u,v,a,r`: ~0.7 GB.
- Strain: ~0.4 GB.
- Total: ~1.7 GB/rank.

Disk with float32 snapshots:

| Item | Size |
|------|------|
| Strain/snapshot | 1.43 GB |
| Snapshots/run | 500 |
| One direction | 715 GB |
| Three directions | ~2.15 TB |
| Restart + partitions | ~15 GB |
| Total | ~2.2 TB |

## Field Changes

| Old | New | Action |
|-----|-----|--------|
| `nsteps` | — | derive from duration and solver_dt |
| `cfl_threshold` | — | remove |
| `checkpoint_interval` | — | replace with `snapshot_stride` |
| `checkpoint_precision` | `snapshot_precision` | rename |
| `output_dt` | `output_dt_s` | rename; means snapshot interval |
| — | `total_duration_s` | add |
| — | `solver_dt` | derive and write to config.h5 |
| — | `snapshot_stride` | derive and write to config.h5 |
| `source_x`, `source_y` | `source_x_m`, `source_y_m` | add suffix |
| `vp`, `vs`, `density` | `vp_m_s`, `vs_m_s`, `density_kg_m3` | add suffix |
| `stf_func(t)` | `stf_func(t_s)` | add suffix |

## config.h5 `/simulation/`

Remove: `dt`, `cfl_threshold`, `checkpoint_interval`.

Add/write:

- `solver_dt`
- `output_dt_s`
- `snapshot_stride`
- `snapshot_precision`
- `nsteps`
- `polynomial_order`
- `cfl_safety`
- `storage_limit_gb`

## Files Changed

### Python

| File | Change |
|------|--------|
| `config_loader.py` | new field names and callable names |
| `cfl_validator.py` | `compute_solver_dt()` stride search |
| `preflight.py` | stride and storage validation |
| `config_writer.py` | new `/simulation/` attrs |
| `stf_evaluator.py` | sample at `solver_dt` |
| `cli.py` | derive and pass `solver_dt`, `snapshot_stride`, `nsteps` |

### C++

| Area | Change |
|------|--------|
| config reader | read `solver_dt`, `output_dt_s`, `snapshot_stride`, `snapshot_precision` |
| Newmark | use `solver_dt` |
| record writer | snapshot naming and stride writes |
| storage | use `nsteps / snapshot_stride` |

### Tests

| File | Change |
|------|--------|
| halfspace workflow | new config names and 500k-element mesh |
| config loader tests | new required fields and removed fields |
| CFL validator tests | stride search |
| regular mesh helper | verify 100×100×50 mesh |

## Edge Cases

| Condition | Handling |
|-----------|----------|
| `output_dt_s < solver_dt_cfl` | use stride 1; warn that solver is finer than required |
| no stride ≤ 100 | error; reduce `output_dt_s` or coarsen mesh |
| duration not integer | round `nsteps` up and warn |
| `nsteps % snapshot_stride != 0` | abort; should not happen by construction |
| PML too thick | boundary validation aborts |
