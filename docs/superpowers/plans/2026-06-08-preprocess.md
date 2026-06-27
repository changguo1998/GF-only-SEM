# Preprocess Module Implementation Plan

> Use `subagent-driven-development` or `executing-plans` to run tasks. Check boxes track progress.

**Goal:** Build Python preprocess. It reads `mesh.h5` and `config.py`, computes derived SEM data, and writes extended `mesh.h5`, `partition_{r}.h5`, and `config.h5`.

**Design:** `docs/superpowers/design/preprocess.md`.

## Config Script

`config.py` is the config. No YAML/TOML. Preprocess imports it with `importlib` and samples `stf_func(t_s)` over the full time range.

Required concepts:

- polynomial order, output cadence, total duration, CFL safety, precision, storage limit, rank count
- PML thickness by face
- source x/y position; z is the free surface
- material callables at GLL nodes
- STF callable

No receivers. Domain bounds come from `mesh.h5`. Force direction is a forward CLI argument.

## Config Validation

| Check | Rule |
|-------|------|
| `polynomial_order` | integer ≥ 1 |
| time fields | positive |
| `cfl_safety` | `0 < cfl_safety < 1` |
| `snapshot_precision` | `float32` or `float64` |
| `storage_limit_gb` | positive |
| source | x/y in domain; z on free surface |
| `stf_func` | callable `(float) -> float` |
| material | callable `(x, y, z) -> float` |
| `n_ranks` | integer ≥ 1 |
| `pml_thickness` | required face keys; integer values ≥ 0 |

## SLS Fitting — Deferred

SLS attenuation is not in the first milestone. Future work computes per-GLL-node `τ_σ` and `τ_ε` from Q values and writes:

```
/field/element/tau_sigma[n_cell, NGLL, NGLL, NGLL, n_sls]
/field/element/tau_epsilon[n_cell, NGLL, NGLL, NGLL, n_sls]
```

## Task 1 — Scaffold and Test Fixtures

**Files:** `preprocess/__init__.py`, `preprocess/cli.py`, `preprocess/config_loader.py`, `tests/preprocess/*`

- [x] Create package and test directories.
- [x] Add pytest fixtures for mock config modules and temp dirs.
- [x] Commit.

## Task 2 — Config Loader

**Files:** `preprocess/config_loader.py`, `tests/preprocess/test_config_loader.py`

- [x] Load Python config via `importlib`.
- [x] Validate required fields and callables.
- [x] Test valid config, missing field, and invalid direction/fields.
- [x] Commit.

## Task 3 — Topology Reader

**Files:** `preprocess/topology_reader.py`, `tests/preprocess/test_topology_reader.py`

- [x] Read `/topology/` from `mesh.h5`.
- [x] Preserve X2Y naming, 1-based IDs, and signed direction.
- [x] Test vertex, edge, surface, and cell relations.
- [x] Commit.

## Task 4 — GLL Geometry

**Files:** `preprocess/gll_geometry.py`, `tests/preprocess/test_gll_geometry.py`

- [x] Compute GLL coordinates for each 8-corner hex.
- [x] Compute Jacobian, `dxi_dx`, and mass inputs.
- [x] Test cube interpolation and determinant.
- [x] Commit.

## Task 5 — 3D Model Loader

**Files:** `preprocess/model_loader.py`, `tests/preprocess/test_model_loader.py`

- [x] Load/interpolate material to GLL nodes.
- [x] Current implementation returns constants.
- [x] Real binary model format remains deferred.
- [x] Commit.

## Task 6 — Boundary Detector

**Files:** `preprocess/boundary_detector.py`, `tests/preprocess/test_boundary_detector.py`

- [x] Detect free surface and absorbing faces from geometry.
- [x] Write `boundary_tag`: 0 interior, 1 free surface, 2 absorbing.
- [x] Test regular box tags.
- [x] Commit.

## Task 7 — SLS Parameters — Deferred

**Files:** `preprocess/sls.py`, `tests/preprocess/test_sls.py`

- [ ] Implement τ-method. Deferred.
- [ ] Test relaxation fit. Deferred.

## Task 8 — PML Profiles

**Files:** `preprocess/pml.py`, `tests/preprocess/test_pml.py`

- [x] Compute per-GLL damping in PML layers.
- [x] Interior damping is zero.
- [x] Test interior vs PML nodes.
- [x] Commit.

## Task 9 — METIS Partitioning

**Files:** `preprocess/partition.py`, `tests/preprocess/test_partition.py`

- [x] Build element dual graph.
- [x] Partition with METIS.
- [x] Compute ghosts, owners, and exchange face lists.
- [x] Test assignment, ghosts, and exchanges.
- [x] Commit.

## Task 10 — STF Evaluator

**Files:** `preprocess/stf_evaluator.py`, `tests/preprocess/test_stf_evaluator.py`

- [x] Sample `stf_func(t)` at solver timesteps.
- [x] Return time and value arrays.
- [x] Test shape and values.
- [x] Commit.

## Task 11 — Writers

**Files:** `preprocess/model_writer.py`, `preprocess/config_writer.py`, writer tests

- [x] Extend `mesh.h5` with fields.
- [x] Write per-rank `partition_{r}.h5`.
- [x] Write `config.h5` simulation, domain, and source groups.
- [x] Test output schemas.
- [x] Commit.

## Task 12 — CLI

**File:** `preprocess/cli.py`

- [x] Implement `python -m preprocess`.
- [x] Read `mesh.h5` and `config.py` from CWD.
- [x] Run all preprocess steps.
- [x] Commit.

## Final Layout

```
preprocess/
├── cli.py                 — entry point
├── config_loader.py        — load and validate config.py
├── topology_reader.py      — read mesh topology
├── gll_geometry.py         — GLL geometry
├── material.py             — material at GLL nodes
├── mass.py                 — lumped mass
├── boundary_detector.py    — boundary and PML flags
├── pml.py                  — PML coefficients
├── cfl_validator.py        — CFL and solver_dt
├── preflight.py            — validation
├── partition.py            — METIS and exchange maps
├── stf_evaluator.py        — STF samples
├── partition_writer.py     — partition_{r}.h5
└── config_writer.py        — config.h5
```
