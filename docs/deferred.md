# Deferred Designs and Plans

This file lists known deferred work. When work resumes, use the linked design or plan.

______________________________________________________________________

## 1. SLS Viscoelastic Attenuation

**Status:** Deferred. Current solver is elastic-only.

SLS spans preprocess and forward.

### Preprocess

Needed:

- Compute per-GLL-node τ_σ_l and τ_ε_l from q_kappa, q_mu, f_min, f_max, and n_sls.
- Write HDF5 datasets:
  ```
  /field/element/tau_sigma[n_cell, NGLL, NGLL, NGLL, n_sls]
  /field/element/tau_epsilon[n_cell, NGLL, NGLL, NGLL, n_sls]
  ```

### Forward

Needed:

- SLS memory arrays per element: [n_cell, NGLL³, n_sls].
- Stress update using precomputed τ_σ and τ_ε.
- Add viscoelastic stress to residual.

______________________________________________________________________

## 2. Compress Module Integration

**Status:** Not integrated. The `gf_compress` header-only library is not linked in `forward/CMakeLists.txt`.

- **Design:** [`design/compress.md`](design/compress.md)

Needed:

- Add `target_link_libraries(libgf PRIVATE gf_compress)`.
- Keep direct includes such as `#include "gf/CompressionFilter.h"`.

______________________________________________________________________

## 3. Full C-PML Implementation

**Status:** Deferred. Current solver uses simple linear-ramp damping.

Needed:

- Full recursive-convolution C-PML (Wang et al. 2006, θ=1/8) with 39 memory variables per GLL node.
- d/K/α damping profiles per direction.
- Second-order convolution coefficients.
- Memory arrays for convolution state.

Current implementation: simple linear-ramp `v ← v - d(node)·v`. Precomputed `damping` profile read from `partition_{r}.h5`.

______________________________________________________________________

## 4. Compression Benchmark Tool

**Status:** Not implemented.

- **Design:** [`design/compress.md`](design/compress.md)

Needed: CLI that writes and reads HDF5 datasets with none, LZF, and zlib 1–9, at float32 and float64. Report size, write time, read time, and round-trip error.

______________________________________________________________________

## 5. GPU/DCU Backends (HIP, SYCL)

**Status:** CUDA backend implemented. HIP and SYCL backends deferred.

Same pattern as CUDA: add tag struct, source file, CMake branch. See [`design/gpu.md`](design/gpu.md).

______________________________________________________________________

## Summary

| Item | Module | Priority | Effort |
|------|--------|----------|--------|
| SLS attenuation | preprocess + forward | High | Large |
| Full C-PML | preprocess + forward | Medium | Large |
| Compress integration | forward | Low | Tiny |
| Compression benchmark | compress | Low | Small |
| HIP/SYCL backends | forward | Low | Medium |
