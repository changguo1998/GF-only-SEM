# Deferred & Delayed Designs and Plans

This file consolidates all features, designs, and implementation tasks that are
explicitly deferred, delayed, or marked "work on later." When work resumes on
any of these items, pull details from the referenced plan/design files.

---

## 1. SLS Viscoelastic Attenuation

**Status:** Deferred (elastic-only for initial milestone)

The SLS (Standard Linear Solid) attenuation model is the largest deferred
feature. It spans both preprocess and forward modules.

### Preprocess side
- **Plan:** `docs/superpowers/plans/2026-06-08-preprocess.md` → Task 7
- **Design:** `docs/superpowers/design/preprocess.md` → SLS Relaxation Parameter Fitting section
- **Source file:** `preprocess/sls.py` — does not exist
- **Tests:** `tests/preprocess/test_sls.py` — does not exist

**What's needed:**
- τ-method: compute τ_σ_l and τ_ε_l per GLL node from per-node (q_kappa, q_mu)
  and global (f_min, f_max, n_sls)
- Output: `/field/element/tau_sigma` and `/field/element/tau_epsilon`,
  shape `[n_cell, NGLL, NGLL, NGLL, n_sls]`

### Forward side
- **Plan:** `docs/superpowers/plans/2026-06-08-forward.md` → Task 5
- **Design:** `docs/superpowers/design/forward.md`
- **Source:** `forward/src/viscoelastic.cpp` — does not exist
- **Tests:** `forward/tests/test_viscoelastic.cpp` — does not exist

**What's needed:**
- SLS memory variable arrays: `[n_cell, NGLL³, n_sls]`
- Stress update using precomputed τ_σ, τ_ε per GLL node
- Viscoelastic stress contribution added to residual

---

## 2. Compression Benchmark Tool

**Status:** Not implemented

- **Plan:** `docs/superpowers/plans/2026-06-08-compress.md` → Task 7
- **Design:** `docs/superpowers/design/compress.md`
- **Source:** `compress/benchmark/CompressionBenchmark.cpp` — does not exist
- **CMake:** `compress/benchmark/CMakeLists.txt` — does not exist

**What's needed:**
Standalone CLI tool that writes/reads HDF5 datasets with various compression
configurations (none, LZF, zlib levels 1-9) at both float32 and float64
precision. Produces a table comparing file size, write time, read time, and
round-trip error. Uses production-relevant dataset sizes (N=5, ~1000 elements).

---

## 3. Compress Test Fixes

**Status:** `tests/test_compress.cpp` exists but tests fail due to tolerance issues

- **Plan:** `docs/superpowers/plans/2026-06-08-compress.md` → Task 6
- **Design:** `docs/superpowers/design/compress.md`

**What's needed:**
The `write_checkpoint()` signature in the test does not match the actual
implementation in `CheckpointWriter.h`. The round-trip tests need to be
rewritten to match the current API. See the plan file for the intended
function signatures and test cases.

---

## 4. Compress Integration into Forward Solver

**Status:** Not integrated — `gf_compress` is not linked in `forward/CMakeLists.txt`

- **Plan:** `docs/superpowers/plans/2026-06-08-compress.md` → Task 8
- **Design:** `docs/superpowers/design/compress.md`

**What's needed:**
- Add `target_link_libraries(libgf PRIVATE gf_compress)` in `forward/CMakeLists.txt`
- The forward solver currently uses compress headers directly (via `#include "gf/CompressionFilter.h"`)
  but does not link the interface library

---

## 5. 3D Model Binary Format (Material Interpolation)

**Status:** Placeholder — `model_loader.py` returns constant values only

- **Plan:** `docs/superpowers/plans/2026-06-08-preprocess.md` → Task 5
- **Design:** `docs/superpowers/design/preprocess.md`

**What's needed:**
Define and implement a binary 3D model file format. The preprocessor's
`load_and_interpolate()` function currently returns homogeneous material
constants regardless of input. A real implementation must read gridded
Vp, Vs, density, Qκ, Qμ and interpolate to GLL node positions.

---

## 6. GPU/DCU Device Abstraction

**Status:** Design-only, no implementation

- **Design:** `docs/superpowers/design/gpu.md`

Template-polymorphic device abstraction for future GPU/DCU acceleration.
No source files exist for this work.

---

## 7. Preprocess Commit Placeholders

**Status:** Several preprocess tasks are uncommitted per plan notes

- Topology reader (Task 3): "DEFERRED -- SLS parameter computation not implemented, no sls.py file exists"
- GLL geometry (Task 4): same
- SLS (Task 7): "DEFERRED -- SLS parameter computation not implemented, no sls.py file exists"

These are likely already committed but the plan file commit markers were never
updated. Verify with `git log --oneline`.

---

## Summary

| Item | Module | Priority | Effort |
|------|--------|----------|--------|
| SLS attenuation | preprocess + forward | High (physics completeness) | Large |
| Compression benchmark | compress | Low (profiling tool) | Small |
| Compress test fixes | compress | Medium | Small |
| Compress→forward link | forward | Low (no behavioral change) | Trivial |
| 3D model format | preprocess | High (input pipeline) | Medium |
| GPU abstraction | forward | Future | Very Large |