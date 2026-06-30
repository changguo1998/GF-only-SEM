# Deferred Designs and Plans

This file lists known deferred work. When work resumes, use the linked design or plan.

______________________________________________________________________

## 1. SLS Viscoelastic Attenuation

**Status:** Deferred. Current solver is elastic-only.

SLS spans preprocess and forward.

### Preprocess

- **Plan:** initial implementation complete — SLS Task 7 not started
- **Design:** [`design/preprocess.md`](design/preprocess.md) → SLS fitting
- **Source:** `preprocess/sls.py` — missing
- **Tests:** `tests/preprocess/test_sls.py` — missing

Needed:

- Compute per-GLL-node `τ_σ_l` and `τ_ε_l` from `q_kappa`, `q_mu`, `f_min`, `f_max`, and `n_sls`.
- Write HDF5 datasets:

```
/field/element/tau_sigma[n_cell, NGLL, NGLL, NGLL, n_sls]
/field/element/tau_epsilon[n_cell, NGLL, NGLL, NGLL, n_sls]
```

### Forward

- **Plan:** initial implementation complete — SLS Task 5 not started
- **Design:** [`design/forward.md`](design/forward.md)
- **Source:** `forward/src/viscoelastic.cpp` — missing
- **Tests:** `forward/tests/test_viscoelastic.cpp` — missing

Needed:

- SLS memory arrays: `[n_cell, NGLL³, n_sls]`.
- Stress update using precomputed `τ_σ` and `τ_ε`.
- Add viscoelastic stress to residual.

______________________________________________________________________

## 2. Compression Benchmark Tool

**Status:** Not implemented.

- **Plan:** ~~`docs/superpowers/plans/2026-06-08-compress.md`~~ (deleted)
- **Design:** [`design/compress.md`](design/compress.md)
- **Source:** `compress/benchmark/CompressionBenchmark.cpp` — missing
- **CMake:** `compress/benchmark/CMakeLists.txt` — missing

Needed: CLI that writes and reads HDF5 datasets with none, LZF, and zlib 1–9, at float32 and float64. Report size, write time, read time, and round-trip error. Use production-like sizes.

______________________________________________________________________

## 3. Compress Test Fixes (Resolved)

**Status:** All compress tests pass (21,546 assertions, 9 test cases). Only issue: HDF5 LZF
filter 32000 not registered, so LZF round-trip is not exercised — the filter is expected to
be registered by the HDF5 runtime (plugin path). No test failures.

**Resolution:** Tests now match `CheckpointWriter.h` and current `write_checkpoint()`
signatures. Defect was fixed in a prior commit.

______________________________________________________________________

## 4. Compress Integration into Forward

**Status:** Not integrated. `gf_compress` is not linked in `forward/CMakeLists.txt`.

- **Plan:** ~~`docs/superpowers/plans/2026-06-08-compress.md`~~ (deleted)
- **Design:** [`design/compress.md`](design/compress.md)

Needed:

- Add `target_link_libraries(libgf PRIVATE gf_compress)`.
- Keep direct includes such as `#include "gf/CompressionFilter.h"`.

______________________________________________________________________

## 5. GPU/DCU Device Abstraction

**Status:** Implemented (CUDA backend).

- **Design:** [`design/gpu.md`](design/gpu.md)
- **Implementation:** `forward/include/gf/backend.hpp`, `forward/src/element_cuda.cu`, `forward/src/element_cpu.cpp`

Template-polymorphic `compute_element_residual<Backend>` with CPU and CUDA backends.
Batched API (`n_elem` parameter) for GPU throughput. Persistent device memory manager
(`cuda_device_manager.hpp`). See GPU design doc for details and future optimizations.

HIP/SYCL backends deferred — same pattern.

______________________________________________________________________

## Summary

| Item | Module | Priority | Effort |
|------|--------|----------|--------|
| SLS attenuation | preprocess + forward | High | Large |
| Compression benchmark | compress | Low | Small |
|| Compress test fixes | compress | Resolved | — |
|| Compress-forward link | forward | Low | Tiny |
| GPU abstraction | forward | Done | Large |
