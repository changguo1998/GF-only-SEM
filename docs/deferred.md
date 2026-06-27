# Deferred Designs and Plans

This file lists known deferred work. When work resumes, use the linked design or plan.

______________________________________________________________________

## 1. SLS Viscoelastic Attenuation

**Status:** Deferred. Current solver is elastic-only.

SLS spans preprocess and forward.

### Preprocess

- **Plan:** `docs/superpowers/plans/2026-06-08-preprocess.md` → Task 7
- **Design:** `docs/superpowers/design/preprocess.md` → SLS fitting
- **Source:** `preprocess/sls.py` — missing
- **Tests:** `tests/preprocess/test_sls.py` — missing

Needed:

- Compute per-GLL-node `τ_σ_l` and `τ_ε_l` from `q_kappa`, `q_mu`, `f_min`, `f_max`, and `n_sls`.
- Write `/field/element/tau_sigma` and `/field/element/tau_epsilon` with shape `[n_cell, NGLL, NGLL, NGLL, n_sls]`.

### Forward

- **Plan:** `docs/superpowers/plans/2026-06-08-forward.md` → Task 5
- **Design:** `docs/superpowers/design/forward.md`
- **Source:** `forward/src/viscoelastic.cpp` — missing
- **Tests:** `forward/tests/test_viscoelastic.cpp` — missing

Needed:

- SLS memory arrays: `[n_cell, NGLL³, n_sls]`.
- Stress update using precomputed `τ_σ` and `τ_ε`.
- Add viscoelastic stress to residual.

______________________________________________________________________

## 2. Compression Benchmark Tool

**Status:** Not implemented.

- **Plan:** `docs/superpowers/plans/2026-06-08-compress.md` → Task 7
- **Design:** `docs/superpowers/design/compress.md`
- **Source:** `compress/benchmark/CompressionBenchmark.cpp` — missing
- **CMake:** `compress/benchmark/CMakeLists.txt` — missing

Needed: CLI that writes and reads HDF5 datasets with none, LZF, and zlib 1–9, at float32 and float64. Report size, write time, read time, and round-trip error. Use production-like sizes.

______________________________________________________________________

## 3. Compress Test Fixes

**Status:** `tests/test_compress.cpp` exists, but tests fail due to tolerance/API mismatch.

- **Plan:** `docs/superpowers/plans/2026-06-08-compress.md` → Task 6
- **Design:** `docs/superpowers/design/compress.md`

Needed: rewrite round-trip tests to match `CheckpointWriter.h` and current `write_checkpoint()` signatures.

______________________________________________________________________

## 4. Compress Integration into Forward

**Status:** Not integrated. `gf_compress` is not linked in `forward/CMakeLists.txt`.

- **Plan:** `docs/superpowers/plans/2026-06-08-compress.md` → Task 8
- **Design:** `docs/superpowers/design/compress.md`

Needed:

- Add `target_link_libraries(libgf PRIVATE gf_compress)`.
- Keep direct includes such as `#include "gf/CompressionFilter.h"`.

______________________________________________________________________

## 5. 3D Model Binary Format

**Status:** Placeholder. `model_loader.py` returns constants.

- **Plan:** `docs/superpowers/plans/2026-06-08-preprocess.md` → Task 5
- **Design:** `docs/superpowers/design/preprocess.md`

Needed: define a binary model format. Read gridded Vp, Vs, density, Qκ, and Qμ. Interpolate them to GLL nodes.

______________________________________________________________________

## 6. GPU/DCU Device Abstraction

**Status:** Design only.

- **Design:** `docs/superpowers/design/gpu.md`

Needed: template-polymorphic backend for future CUDA/HIP/SYCL element kernels.

______________________________________________________________________

## 7. Preprocess Plan Markers

**Status:** Some plan commit markers look stale.

- Topology reader, GLL geometry, and SLS tasks mention SLS deferral.
- Work may already be committed.

Needed: verify with `git log --oneline` and update plan markers if useful.

______________________________________________________________________

## Summary

| Item | Module | Priority | Effort |
|------|--------|----------|--------|
| SLS attenuation | preprocess + forward | High | Large |
| Compression benchmark | compress | Low | Small |
| Compress test fixes | compress | Medium | Small |
| Compress-forward link | forward | Low | Tiny |
| 3D model format | preprocess | High | Medium |
| GPU abstraction | forward | Future | Very large |
