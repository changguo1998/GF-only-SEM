# Deferred Designs and Plans

This file lists known deferred work. When work resumes, use the linked design or plan.

______________________________________________________________________

## 1. SLS Viscoelastic Attenuation

**Status:** Deferred. Current `forward/` solver is elastic-only.
A `forward/viscoelastic/` skeleton exists for future SLS implementation.

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

**Status:** Not integrated. The `gf_compress` header-only library is not linked in `forward/share/CMakeLists.txt`.

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

## 6. SEM Force Normalization — Amplitude Discrepancy

**Status:** Deferred. SEM displacement is systematically larger than analytic/PyFK references.

Both example pipelines run end-to-end, but the SEM result has a residual amplitude
error that points to a force-normalization issue in the source application path:

- **Halfspace** (Lamb analytic reference): SEM displacement is ~3× larger than the
  analytic reference after the convolution-truncation fix (`mode='same'`→`mode='full'`).
  Off-diagonal components (F_z→u_x, F_x→u_z) are severely underestimated (0.07–0.13×),
  suggesting a radiation-pattern or free-surface treatment issue.
- **Layer** (PyFK reference): SEM displacement is ~3×10⁵ larger than the PyFK reference
  after aligning coordinates and adding STF convolution.

### Investigation targets

1. `forward/share/src/source.cpp` — `PointForceSource::apply()`: verify GLL quadrature
   weights integrate to 1 (force → RHS normalization).
1. `forward/share/src/assembly.cpp` — `add_source_to_rhs()`: confirm the source term
   scaling matches a unit point force (1 N).
1. `postprocess/` — Green's function extraction: check for any unintended scaling or
   truncation factor in displacement → Green tensor assembly.
1. PyFK unit convention: confirm displacement output is in metres for 1 N force
   (static-limit check: u_z = F/(4πμr) gives ~2.9×10⁻¹⁴ m for the layer geometry).

### Context

- Convolution truncation bug (`mode='same'`) fixed in both `reference.py` files.
- Coordinate convention unified across `reference.py` and `compare.py`
  (`--source` = observation point, `--receiver` = source-match point).
- Commit `49d3455` documents the pipeline alignment; amplitude fix is the next step.

## Summary

| Item | Module | Priority | Effort |
|------|--------|----------|--------|
| SLS attenuation | preprocess + forward/viscoelastic | High | Large |
| Compress integration | forward/ | Low | Tiny |
| Compression benchmark | compress | Low | Small |
| HIP/SYCL backends | forward/elastic/ | Low | Medium |
| Force normalization | forward/share + postprocess | High | Medium |
