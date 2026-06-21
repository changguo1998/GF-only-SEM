# Preprocess Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python `preprocess/` module that reads mesh topology (mesh.h5) and a Python config script (config.py), computes all derived model data (GLL-node material via config.py functions, geometric quantities, partition, C-PML, source weights, STF evaluation, pre-flight validation), and writes extended `mesh.h5` + `partition_{r}.h5` + `configs/config.h5`.

> **Design**: Technical decisions (data flow, domain objects, HDF5 output schema, validation rules, SLS parameter computation) are documented in [`docs/superpowers/design/preprocess.md`](../design/preprocess.md). This file contains only the implementation plan.

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
| source position | within domain bounds (auto-detected from mesh), z auto-placed on free surface |
| stf_func | callable, signature `(float) -> float` |
| vp, vs, density | callable, each signature `(float, float, float) -> float` |
| n_ranks | ≥ 1, integer |
| pml_thickness | dict with keys xmin, xmax, ymin, ymax, zmin, zmax; values ≥ 0 integers |

**No receivers in config.** Receivers are handled by postprocess from checkpoint files.
**No explicit domain bounds in config.** Domain bounds are auto-detected from `mesh.h5` topology.
**No STF type enum.** STF is user-defined `stf_func(t)` in the config script.
**No inline material.** Material comes from 3D model binary, interpolated to GLL nodes.

---

## SLS Relaxation Parameter Fitting (τ-method) — DEFERRED

SLS attenuation is deferred to future work. Elastic-only for the initial milestone.

For each GLL node with per-node (q_kappa, q_mu) and global (f_min, f_max, n_sls), the τ-method computes per-GLL-node τ_σ and τ_ε so that Q(ω) ≈ constant across [f_min, f_max].

The approach follows Blanch et al. (1995) / SPECFEM conventions:

1. Compute logarithmic spacing of relaxation frequencies: ω_l = 2π * f_max * (f_min/f_max)^(l/(n_sls-1)) for l = 0..n_sls-1
2. Set τ_σ_l = 1/ω_l (stress relaxation times)
3. Solve for weights via least-squares fit across the band
4. Derive τ_ε_l from τ_σ_l and Q at each GLL node

Since the 3D model interpolation to GLL nodes provides per-node Q values, the τ-method operates at GLL-node granularity — not per-element or per-material-tag.

Output: `/field/element/tau_sigma` and `/field/element/tau_epsilon`, shape `[n_cell, NGLL, NGLL, NGLL, n_sls]`.

---

## Task Breakdown

### Task 1: Project scaffolding and test infrastructure

**Files:**
- Create: `preprocess/__init__.py`
- Create: `preprocess/cli.py`
- Create: `preprocess/config_loader.py`
- Create: `tests/preprocess/__init__.py`
- Create: `tests/preprocess/conftest.py`

- [x] **Step 1: Create package init and test init**

```bash
mkdir -p preprocess tests/preprocess
touch preprocess/__init__.py tests/preprocess/__init__.py
```

- [x] **Step 2: Write conftest.py — mock config module fixture**

```python
# tests/preprocess/conftest.py
import pytest
import sys
from types import ModuleType
from pathlib import Path


@pytest.fixture
def config_dict():
    """Returns a dict that can be used to build a mock config module."""
    return {
        "title": "test_run",
        "polynomial_order": 3,
        "dt": 0.001,
        "nsteps": 100,
        "cfl_safety": 0.5,
        "checkpoint_interval": 10,
        "model_path": "dummy_model.dat",
        "pml_thickness": {"xmin": 3, "xmax": 3, "ymin": 3, "ymax": 3, "zmin": 3, "zmax": 3},
        "n_sls": 3,
        "f_min": 0.1,
        "f_max": 10.0,
        "source_x": 500.0,
        "source_y": 500.0,
        "source_z": 500.0,
        "source_direction": "x",
        "n_ranks": 4,
    }


@pytest.fixture
def mock_config_module(config_dict):
    """Creates a mock Python module with config attributes + stf_func."""
    def stf_func(t):
        import numpy as np
        return np.exp(-t**2)

    mod = ModuleType("mock_config")
    for key, val in config_dict.items():
        setattr(mod, key, val)
    mod.stf_func = stf_func
    return mod


@pytest.fixture
def tmp_dir():
    """Temporary directory that auto-cleans after test."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
```

- [x] **Step 3: Commit**

---

### Task 2: Config Loader

**Files:**
- Create: `preprocess/config_loader.py`
- Create: `tests/preprocess/test_config_loader.py`

Use `importlib` to load a Python config script, validate all required fields.

- [x] **Step 1: Write test** — load mock config, verify attributes accessible, verify stf_func is callable
- [x] **Step 2: Write test** — missing required field raises ValidationError
- [x] **Step 3: Write test** — invalid source_direction raises ValidationError
- [x] **Step 4: Implement `load_config(path) -> ModuleType`** — import via importlib, validate all required fields per table above
- [x] **Step 5: Commit**

---

### Task 3: Topology Reader

**Files:**
- Create: `preprocess/topology_reader.py`
- Create: `tests/preprocess/test_topology_reader.py`

Read `/topology/` datasets from mesh.h5 into memory. X2Y naming, 1-based indexing, signed direction.

- [x] **Step 1: Write test** — read synthetic mesh.h5, verify vertex_to_coord shape, n_vertex attr
- [x] **Step 2: Write test** — verify edge_to_vertex (signed int), surface_to_edge (CCW), cell_to_surface
- [x] **Step 3: Implement `read_topology(path) -> TopologyData`** — read all `/topology/` datasets
- [x] **Step 4: Commit**

---

### Task 4: GLL Geometry

**Files:**
- Create: `preprocess/gll_geometry.py`
- Create: `tests/preprocess/test_gll_geometry.py`

For each element with 8 corner vertices and polynomial order N, compute (N+1)³ GLL node positions via geometric mapping, Jacobian, dξ/dx, and lumped mass diagonal.

- [x] **Step 1: Write test** — GLL node coords for a cube element (should interpolate corners)
- [x] **Step 2: Write test** — det(J) for unit cube should equal 1/8 of cube volume
- [x] **Step 3: Implement `compute_gll_geometry(topology, N)`** — returns coords, jacobian, dxi_dx, mass arrays
- [x] **Step 4: Commit**

---

### Task 5: 3D Model Loader

**Files:**
- Create: `preprocess/model_loader.py`
- Create: `tests/preprocess/test_model_loader.py`

Load binary 3D model (format TBD — placeholder for now). Interpolate Vp, Vs, density, Qκ, Qμ from model grid to GLL nodes.

- [x] **Step 1: Write test** — homogeneous model returns constant values at all GLL nodes
- [x] **Step 2: Implement `load_and_interpolate(model_path, gll_coords)`** — returns vp, vs, density, q_kappa, q_mu arrays (element-first, shape `[n_cell, NGLL, NGLL, NGLL]`)
- [x] **Step 3: Commit** — noted as placeholder until 3D model format is finalized (STILL PLACEHOLDER -- model_loader.py returns constant values, no real 3D model format support)

---

### Task 6: Boundary Detector

**Files:**
- Create: `preprocess/boundary_detector.py`
- Create: `tests/preprocess/test_boundary_detector.py`

Auto-detect boundary tags from surface face center geometry. No GMSH physical groups needed.

- [x] **Step 1: Write test** — z_max surface → tag 1 (free surface), x/y/z min → tag 2 (absorbing), interior → tag 0
- [x] **Step 2: Implement `detect_boundaries(topology, domain_bounds)`** — returns `boundary_tag[n_surface]` (0=interior, 1=free surface, 2=absorbing)
- [x] **Step 3: Commit**

---

### Task 7: SLS Parameter Computation — DEFERRED

Viscoelastic attenuation (SLS τ-method) is deferred to future work. Skip this task.

**Files:**
- Create: `preprocess/sls.py`
- Create: `tests/preprocess/test_sls.py`

τ-method: compute τ_σ and τ_ε per GLL node from per-node Q values and global (f_min, f_max, n_sls).

- [x] **Step 1: Write test** — constant Q produces consistent τ values, τ_σ < τ_ε
- [ ] **Step 2: Write test** -- n_sls=3 produces 3 mechanism pairs (DEFERRED -- SLS not implemented)
- [ ] **Step 3: Implement compute_sls_parameters(q, f_min, f_max, n_sls)** -- returns tau_sigma[n_sls], tau_epsilon[n_sls] per GLL node (DEFERRED -- SLS not implemented)
- [ ] **Step 4: Commit** (DEFERRED -- SLS parameter computation not implemented, no sls.py file exists)

---

### Task 8: PML Damping Profiles

**Files:**
- Create: `preprocess/pml.py`
- Create: `tests/preprocess/test_pml.py`

Compute damping coefficient profile across GLL nodes within PML elements, based on element centroid position + pml_thickness per face.

- [x] **Step 1: Write test** — interior element has damping = 0, PML-adjacent element has damping > 0
- [x] **Step 2: Implement `compute_pml_damping(topology, gll_coords, pml_thickness, domain_bounds)`** — returns damping array (element-first, `[n_cell, NGLL, NGLL, NGLL]`)
- [x] **Step 3: Commit**

---

### Task 9: Partition (METIS)

**Files:**
- Create: `preprocess/partition.py`
- Create: `tests/preprocess/test_partition.py`

Build dual graph (elements as nodes, shared faces as edges), call METIS k-way, compute exchange patterns.

- [x] **Step 1: Write test** — partition N elements into K ranks, verify each element assigned exactly one rank
- [x] **Step 2: Write test** — verify ghost elements identified (shared faces across rank boundaries)
- [x] **Step 3: Write test** — verify exchange face-pair lists for neighbor ranks
- [x] **Step 4: Implement `partition(topology, gll_coords, n_ranks)`** — returns element_to_rank, per-rank data (local IDs, ghost IDs, ghost owners, exchange patterns)
- [x] **Step 5: Commit**

---

### Task 10: STF Evaluator

**Files:**
- Create: `preprocess/stf_evaluator.py`
- Create: `tests/preprocess/test_stf_evaluator.py`

Evaluate `config.stf_func(t)` at t = 0, dt, 2*dt, ..., (nsteps-1)*dt.

- [x] **Step 1: Write test** — evaluate mock stf_func, verify output shape matches nsteps
- [x] **Step 2: Implement `evaluate_stf(stf_func, dt, nsteps)`** — returns `stf_t[nsteps]`, `stf_values[nsteps]`
- [x] **Step 3: Commit**

---

### Task 11: Model & Config Writers

**Files:**
- Create: `preprocess/model_writer.py`
- Create: `preprocess/config_writer.py`
- Create: `tests/preprocess/test_model_writer.py`
- Create: `tests/preprocess/test_config_writer.py`

Write all computed data to mesh.h5 (extended in-place), partition_{r}.h5, and configs/config.h5.

- [x] **Step 1: Write test** — verify mesh.h5 contains topology + field/element + field/surface groups, verify partition_{r}.h5 files contain field/element + partition metadata
- [x] **Step 2: Write test** — verify config.h5 contains simulation + domain + source groups
- [x] **Step 3: Implement `write_model(path, topology, fields, partition)`** — schema per design
- [x] **Step 4: Implement `write_config(path, config, domain_bounds, stf_t, stf_values)`** — schema per design
- [x] **Step 5: Commit**

---

### Task 12: CLI Entry Point

**Files:**
- Modify: `preprocess/cli.py`

CLI: `python -m preprocess mesh.h5 config.py`

- [x] **Step 1: Implement `main()`** — parse args, orchestrates all 9 processing steps
- [x] **Step 2: Commit**

---

## File Layout (Final)

```
preprocess/
├── __init__.py
├── cli.py              — CLI entry point
├── config_loader.py     — importlib load config.py, validate
├── topology_reader.py   — read mesh.h5 /topology/
├── gll_geometry.py      — compute GLL node coords, jacobian, dξ/dx per element
├── material.py          — evaluate config vp/vs/density functions at GLL nodes
├── mass.py              — compute lumped mass
├── boundary_detector.py — auto boundary tagging (surface level), set is_pml flags
├── pml.py               — compute PML damping profiles, convolution coefficients
├── cfl_validator.py     — compute cfl_dt, validate output_dt against threshold
├── preflight.py         — comprehensive pre-flight validation
├── partition.py         — METIS partitioning + exchange pattern precomputation
├── stf_evaluator.py     — evaluate stf_func() → time series array
├── partition_writer.py  — write partition_{r}.h5
└── config_writer.py     — write configs/config.h5
```