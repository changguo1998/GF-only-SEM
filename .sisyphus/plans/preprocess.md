# Preprocess Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python `preprocess/` module that reads mesh topology (mesh.h5) and a Python config script (config.py), computes all derived model data (GLL-node material via config.py functions, geometric quantities, partition, C-PML, source weights, STF evaluation, pre-flight validation), and writes extended `mesh.h5` + `partition_{r}.h5` + `configs/config.h5`.

> **Design**: Technical decisions (C-PML formulas, config.h5 schema, boundary detection, partition) are documented in [`docs/superpowers/design/preprocess.md`](../../docs/superpowers/design/preprocess.md). This file contains only the implementation plan.

> **Note**: SLS attenuation is deferred to future work. The preprocessor does NOT compute SLS parameters in the initial milestone.

---

## Python Config Script (`config.py`)

The config file IS the configuration — no YAML/TOML parsing. The preprocessor uses `importlib` to load the user's config script as a Python module. The STF function is defined inline — the preprocessor calls it over the full time range.

```python
# Example config.py — imported by preprocessor
title = "test_run"
polynomial_order = 5       # N
output_dt = 0.001            # user-specified time step (s)
nsteps = 10000
cfl_safety = 0.5
cfl_threshold = 1.0          # max output_dt / cfl_dt ratio before abort
checkpoint_interval = 100
checkpoint_precision = "float32"  # "float32" or "float64"
storage_limit_gb = 100
n_ranks = 4

# PML thickness per face
pml_thickness = {"xmin": 3, "xmax": 3, "ymin": 3, "ymax": 3, "zmin": 0, "zmax": 3}

# Source
source_x = 500.0
source_y = 500.0
# source_z auto-placed on free surface; no source_direction (auto-gen by preprocessor)
# Force direction passed to forward solver via CLI --direction {x,y,z}

# Material model — callables evaluated per GLL node
def vp(x, y, z): return 3000.0
def vs(x, y, z): return 1500.0
def density(x, y, z): return 2500.0

def stf_func(t):
    import numpy as np
    f0 = 5.0
    t0 = 0.3
    return (1 - 2 * (np.pi * f0 * (t - t0))**2) * np.exp(-(np.pi * f0 * (t - t0))**2)
```

### Config Validation

| Check | Rule |
|-------|------|
| polynomial_order | ≥ 1, integer |
| output_dt | > 0 |
| nsteps | > 0 |
| cfl_safety | 0 < cfl_safety < 1 |
| cfl_threshold | > 0 |
| checkpoint_precision | "float32" or "float64" |
| storage_limit_gb | > 0 |
| source position | within domain bounds, z auto-placed on free surface |
| stf_func | callable, signature `(float) -> float` |
| vp, vs, density | callable, signature `(x, y, z) -> float` |
| pml_thickness | dict with keys xmin,xmax,ymin,ymax,zmin,zmax, int ≥ 0 |
| n_ranks | int ≥ 1 |
| title | string, non-empty |

---

## SLS Relaxation Parameter Fitting (τ-method) — DEFERRED

Viscoelastic attenuation (SLS τ-method) is deferred to future work.
All SLS-related tasks in this plan are marked DEFERRED.

---

## TODOs

- [ ] 1. Project scaffolding and test infrastructure

  **Files:**
  - Create: `preprocess/__init__.py`, `preprocess/cli.py`, `preprocess/config_loader.py`
  - Create: `tests/preprocess/__init__.py`, `tests/preprocess/conftest.py`

  **Steps:**
  - [ ] Step 1: Create package init and test init (`mkdir`, `touch`)
  - [ ] Step 2: Write `conftest.py` — mock config module fixture with `config_dict()` and `mock_config_module()` returning `ModuleType` with mock `stf_func`, `vp/vs/density` callables, `pml_thickness`, `n_ranks`, etc.
  - [ ] Step 3: Commit

  **Commit**: YES — `feat(preprocess): project scaffolding and test conftest`

---

- [ ] 2. Config Loader

  **Files:**
  - Create: `preprocess/config_loader.py`
  - Create: `tests/preprocess/test_config_loader.py`

  Use `importlib` to load Python config script, validate all required fields.

  - [ ] Step 1: Write test — load mock config, verify attributes, verify stf_func callable
  - [ ] Step 2: Write test — missing field raises ValidationError
  - [ ] Step 3: Write test — invalid values raise ValidationError
  - [ ] Step 4: Implement `load_config(path) -> ModuleType`
  - [ ] Step 5: Commit

  **Commit**: YES — `feat(preprocess): config loader with importlib`

---

- [ ] 3. Topology Reader

  **Files:**
  - Create: `preprocess/topology_reader.py`
  - Create: `tests/preprocess/test_topology_reader.py`

  Read `/topology/` datasets from mesh.h5. X2Y naming, 1-based indexing, signed direction.

  - [ ] Step 1: Write test — read synthetic mesh.h5, verify vertex_to_coord shape, n_vertex attr
  - [ ] Step 2: Write test — verify edge_to_vertex (signed int), surface_to_edge (CCW), cell_to_surface
  - [ ] Step 3: Implement `read_topology(path) -> TopologyData`
  - [ ] Step 4: Commit

  **Commit**: YES — `feat(preprocess): topology reader for mesh.h5`

---

- [ ] 4. GLL Geometry

  **Files:**
  - Create: `preprocess/gll_geometry.py`
  - Create: `tests/preprocess/test_gll_geometry.py`

  For each element with 8 corners and polynomial order N, compute (N+1)³ GLL node positions, Jacobian, dξ/dx, and lumped mass diagonal.

  - [ ] Step 1: Write test — GLL node coords for a cube element (should interpolate corners)
  - [ ] Step 2: Write test — det(J) for unit cube = 1/8 of cube volume
  - [ ] Step 3: Implement `compute_gll_geometry(topology, N)` → coords, jacobian, dxi_dx, mass
  - [ ] Step 4: Commit

  **Commit**: YES — `feat(preprocess): GLL geometry (coords, jacobian, dxi_dx, mass)`

---

- [ ] 5. Material Evaluation at GLL Nodes

  **Files:**
  - Create: `preprocess/material.py`
  - Create: `tests/preprocess/test_material.py`

  Evaluate `config.vp(x,y,z)`, `config.vs(x,y,z)`, `config.density(x,y,z)` at all GLL node positions.

  - [ ] Step 1: Write test — homogeneous material returns uniform arrays
  - [ ] Step 2: Implement `evaluate_material(gll_coords, config_module)` → vp, vs, density arrays at GLL nodes
  - [ ] Step 3: Commit

  **Commit**: YES — `feat(preprocess): material evaluation at GLL nodes`

---

- [ ] 6. Boundary Detector

  **Files:**
  - Create: `preprocess/boundary_detector.py`
  - Create: `tests/preprocess/test_boundary_detector.py`

  Auto-detect boundary tags from surface face center geometry. No GMSH physical groups.

  - [ ] Step 1: Write test — z_max surface → tag 1 (free surface), other bounds → tag 2 (absorbing), interior → tag 0
  - [ ] Step 2: Write test — detect PML elements: walk N layers inward from each absorbing boundary face
  - [ ] Step 3: Implement `detect_boundaries(topology, domain_bounds)` → boundary_tag[n_surface], is_pml[n_cell]
  - [ ] Step 4: Commit

  **Commit**: YES — `feat(preprocess): automatic boundary detection + PML element tagging`

---

- [ ] 7. SLS Parameter Computation — DEFERRED

  Viscoelastic attenuation (SLS τ-method) is deferred to future work.

  **Files:**
  - Create: `preprocess/sls.py` (placeholder)

  - [ ] (Future) Implement τ-parameter fitting for constant-Q approximation
  - [ ] (Future) Write per-GLL-node SLS arrays to partition_{r}.h5

  **Commit**: NO

---

- [ ] 8. C-PML Damping Profiles and Convolution Coefficients

  **Files:**
  - Create: `preprocess/pml.py`
  - Create: `tests/preprocess/test_pml.py`

  Compute all C-PML data per GLL node per PML element: damping profiles,
  stretched-coordinate factors, frequency-shift profiles, element type
  classification, and convolution coefficients.

  > **See design doc** `docs/superpowers/design/forward.md` (CPML Memory Variables section)
  > for exact formulas. Key references:
  > - `pml_set_local_dampingcoeff.f90` (SPECFEM3D) — damping profile computation
  > - `prepare_timerun.F90` (SPECFEM3D) — convolution coefficient precomputation
  > - `pml_par.f90` (SPECFEM3D) — array declarations

  **Sub-step A: Classify PML element types**
  - Use boundary tags + element adjacency: for each absorbing boundary face,
    walk N layers inward (N = pml_thickness for that face)
  - For each element, independently check each direction against its
    respective boundary distance
  - Classify: face (1 direction), edge (2 directions), corner (3 directions)
  - Assign `cpml_type` per element (1=face, 2=edge, 3=corner)
  - Determine `CPML_regions` code (1-7: X_ONLY, Y_ONLY, Z_ONLY,
    XY_ONLY, XZ_ONLY, YZ_ONLY, XYZ)

  **Sub-step B: Compute damping profiles per GLL node**
  For each CPML element, for each GLL node:

  ```
  For each active direction:
    dist = |coord - interface| / CPML_width    // normalized [0, 1]

    d_x = -((NPOWER + 1) · vp_max · log(CPML_Rcoef) / (2 · CPML_width_x))
           · dist^(1.2 · NPOWER)               // NPOWER=1, Rcoef=0.001

    K_x = K_MIN + (K_MAX - K_MIN) · dist       // K_MIN = K_MAX = 1 (constant)

    alpha_x = ALPHA_MAX_x · (1 - dist)          // where ALPHA_MAX depends on direction
    ALPHA_MAX_x = π · f0_PML · 0.9
    ALPHA_MAX_y = π · f0_PML · 1.0
    ALPHA_MAX_z = π · f0_PML · 1.1
  ```

  Inactive directions for face/edge elements: d=0, K=1, alpha=0.

  **Sub-step C: Compute convolution coefficients**
  For each GLL node per CPML element, precompute:

  ```
  beta = alpha + d / K    (per direction)

  // Recursive convolution coefficients (for each of alpha_x/y/z and beta_x/y/z)
  compute_convolution_coef(b):
    temp = exp(-0.5 · b · dt)
    coef0 = temp · temp
    coef1 = (1 - temp) / b        (or Taylor expansion for small b)
    coef2 = coef1 · temp
    → returns (coef0, coef1, coef2)

  → conv_coef_alpha[9]   = coef0,1,2 for alpha_x, alpha_y, alpha_z
  → conv_coef_beta[9]    = coef0,1,2 for beta_x, beta_y, beta_z
  ```

  Then compute accel-update coefficients Ā₁…Ā₅ via `l_parameter_computation()`
  and strain-update coefficients A₆…A₁₇ via `lijk_parameter_computation()`
  (see SPECFEM3D `prepare_timerun.F90` lines 840-880 and `pml_compute_accel_contribution.f90`
  `l_parameter_computation` for the full formulas).

  **Output arrays** (written to partition_{r}.h5 `/field/element/cpml/`):

  | Array | Shape | Description |
  |-------|-------|-------------|
  | cpml_type | int8[n_elem_local] | 1=face, 2=edge, 3=corner |
  | d_x, d_y, d_z | float64[n_elem_local, NGLL, NGLL, NGLL] | Damping profiles |
  | K_x, K_y, K_z | float64[n_elem_local, NGLL, NGLL, NGLL] | K=1 if no stretch |
  | alpha_x, alpha_y, alpha_z | float64[n_elem_local, NGLL, NGLL, NGLL] | Frequency shift |
  | conv_coef_alpha | float64[9, n_elem_local, NGLL, NGLL, NGLL] | coef0,1,2 for α_x,α_y,α_z |
  | conv_coef_beta | float64[9, n_elem_local, NGLL, NGLL, NGLL] | coef0,1,2 for β_x,β_y,β_z |
  | conv_coef_abar | float64[5, n_elem_local, NGLL, NGLL, NGLL] | Ā₁…Ā₅ for accel update |
  | conv_coef_strain | float64[18, n_elem_local, NGLL, NGLL, NGLL] | A₆…A₁₇ for strain update |

  - [ ] Step 1: Write test — interior element has cpml_type=0 (no PML), PML face has 1 active direction, edge has 2, corner has 3
  - [ ] Step 2: Write test — d_x > 0 near boundary, decays inward; K_x = 1 everywhere; alpha_x proportional to (1 - dist)
  - [ ] Step 3: Write test — convolution coefficients for known alpha/beta produce stable recursive update
  - [ ] Step 4: Implement `classify_pml_elements(topology, boundary_tag, pml_thickness)` → cpml_type, CPML_regions
  - [ ] Step 5: Implement `compute_damping_profiles(cpml_elements, gll_coords, domain_bounds, cpml_type, vp_max, f0_pml)` → d/K/alpha arrays
  - [ ] Step 6: Implement `compute_convolution_coefficients(d, K, alpha, dt, cpml_type)` → conv_coef_alpha, conv_coef_beta, conv_coef_abar, conv_coef_strain
  - [ ] Step 7: Commit

  **Commit**: YES — `feat(preprocess): C-PML profiles and convolution coefficients`

---

- [ ] 9. Partition (METIS)

  **Files:**
  - Create: `preprocess/partition.py`
  - Create: `tests/preprocess/test_partition.py`

  Build dual graph (elements as nodes, shared faces as edges), call METIS k-way, compute exchange patterns.

  - [ ] Step 1: Write test — partition N elements into K ranks, verify each element assigned exactly one rank
  - [ ] Step 2: Write test — verify ghost elements identified (shared faces across rank boundaries)
  - [ ] Step 3: Write test — verify exchange face-pair lists for neighbor ranks
  - [ ] Step 4: Implement `partition(topology, gll_coords, n_ranks)` → element_to_rank, per-rank data
  - [ ] Step 5: Commit

  **Commit**: YES — `feat(preprocess): METIS partition with exchange patterns`

---

- [ ] 10. STF Evaluator

  **Files:**
  - Create: `preprocess/stf_evaluator.py`
  - Create: `tests/preprocess/test_stf_evaluator.py`

  Evaluate `config.stf_func(t)` at t = 0, dt, 2*dt, ..., (nsteps-1)*dt.

  - [ ] Step 1: Write test — evaluate mock stf_func, verify output shape matches nsteps
  - [ ] Step 2: Implement `evaluate_stf(stf_func, dt, nsteps)` → stf_t[nsteps], stf_values[nsteps]
  - [ ] Step 3: Commit

  **Commit**: YES — `feat(preprocess): STF time series evaluator`

---

- [ ] 11. Model & Config Writers

  **Files:**
  - Create: `preprocess/partition_writer.py`
  - Create: `preprocess/config_writer.py`
  - Create: `tests/preprocess/test_partition_writer.py`
  - Create: `tests/preprocess/test_config_writer.py`

  Write all computed data to:
  1. `mesh.h5` — extended in-place with `/field/element/coords`, `/field/element/dxi_dx`, `/field/element/jacobian`, `/field/element/is_pml`, `/field/surface/boundary_tag`
  2. `partition_{r}.h5` — per-rank subset of topology + field/element (coords, jacobian, dxi_dx, mass, vp, vs, density, cpml/*) + partition metadata (local/ghost IDs, gll_to_global, exchange patterns)
  3. `configs/config.h5` — single rank-invariant file with exact schema below

  > **config.h5 schema** (see [`docs/superpowers/design/preprocess.md`](../../docs/superpowers/design/preprocess.md) for the full specification):
  > ```
  > /simulation/
  >   title, polynomial_order, dt, nsteps, cfl_safety, cfl_threshold,
  >   checkpoint_interval, checkpoint_precision, storage_limit_gb
  > /domain/
  >   xmin, xmax, ymin, ymax, zmin, zmax, pml_thickness[6]
  > /source/
  >   x, y, stf[nsteps],
  >   n_src_elements (attr),
  >   /elements/  element_ids, xi, eta, zeta, weights
  > ```
  > No direction in config.h5 — CLI `--direction` flag.
  > No attenuation group — SLS deferred.

  - [ ] Step 1: Write test — verify mesh.h5 contains `/field/element/` + `/field/surface/` groups
  - [ ] Step 2: Write test — verify partition_{r}.h5 files contain topology + field/element + partition metadata
  - [ ] Step 3: Write test — verify config.h5 contains `/simulation/`, `/domain/`, `/source/` with correct datasets
  - [ ] Step 4: Implement `write_partition(path, rank_data, fields, topology)` — schema per design doc partition_{r}.h5 section
  - [ ] Step 5: Implement `write_config(path, config, domain_bounds, stf_values, source_info)` — schema per above
  - [ ] Step 6: Write source locator: find containing elements for source_x/y on free surface, compute Newton iteration for (ξ,η,ζ) within [-1,1]³, compute normalized Lagrange weights
  - [ ] Step 7: Commit

  **Commit**: YES — `feat(preprocess): partition_writer, config_writer, and source locator`

---

- [ ] 12. CFL Validator

  **Files:**
  - Create: `preprocess/cfl_validator.py`
  - Create: `tests/preprocess/test_cfl_validator.py`

  Compute CFL-limited timestep: `cfl_dt = cfl_safety × h_min / vp_max` where `h_min` is minimum GLL node spacing.

  - [ ] Step 1: Write test — uniform mesh → cfl_dt = expected value
  - [ ] Step 2: Implement `compute_cfl_dt(gll_coords, vp, cfl_safety)` → cfl_dt, h_min
  - [ ] Step 3: Implement `validate_dt(output_dt, cfl_dt, cfl_threshold)` → PASS/FAIL with suggested max dt
  - [ ] Step 4: Commit

  **Commit**: YES — `feat(preprocess): CFL validation`

---

- [ ] 13. Pre-Flight Validation Module

  **Files:**
  - Create: `preprocess/preflight.py`
  - Create: `tests/preprocess/test_preflight.py`

  Comprehensive validation checklist before partition and writing.

  Checks: material positivity, det(J) > 0, CFL pass, boundary completeness, source found, STF finite, n_ranks ≤ n_cell, storage estimate.

  - [ ] Step 1: Write test — all checks pass for valid configuration
  - [ ] Step 2: Write test — invalid material triggers error
  - [ ] Step 3: Implement `preflight_check(config, topology, geometry, material, boundary, source_info, cfl_result)` → validation report
  - [ ] Step 4: Connect into CLI processing pipeline
  - [ ] Step 5: Commit

  **Commit**: YES — `feat(preprocess): pre-flight validation module`

---

- [ ] 14. CLI Entry Point

  **Files:**
  - Modify: `preprocess/cli.py`
  - Create: `tests/preprocess/test_cli.py`

  CLI: `python -m preprocess mesh.h5 config.py`

  - [ ] Step 1: Implement orchestration — load config → read topology → compute GLL geometry → evaluate material → detect boundaries → compute C-PML profiles → validate → partition → evaluate STF → locate source → write partition + config
  - [ ] Step 2: Commit

  **Commit**: YES — `feat(preprocess): CLI entry point with full pipeline`

---

## Final File Layout

```
preprocess/
├── __init__.py
├── cli.py              — CLI entry point with pipeline orchestration
├── config_loader.py    — importlib load config.py, validate
├── topology_reader.py  — read mesh.h5 /topology/
├── gll_geometry.py     — compute GLL node coords, jacobian, dξ/dx
├── material.py         — evaluate config vp/vs/density at GLL nodes
├── boundary_detector.py — auto boundary tagging + PML element flagging
├── pml.py              — C-PML damping profiles + convolution coefficients
├── cfl_validator.py    — compute cfl_dt, validate output_dt
├── preflight.py        — comprehensive pre-flight validation
├── partition.py        — METIS partitioning + exchange patterns
├── stf_evaluator.py    — evaluate stf_func() → time series
├── partition_writer.py — write partition_{r}.h5
├── config_writer.py    — write configs/config.h5 (full schema per design doc)
└── sls.py              — placeholder (deferred)
```