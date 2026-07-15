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

## 6. SEM Amplitude & Radiation Pattern Discrepancy

**Status:** Partially resolved. PyFK force unit fixed (dyne→Newton). Residual
~3× diagonal and off-diagonal radiation pattern issues remain.

### Resolved: PyFK force unit (commit `8e4cd8e`)

PyFK uses CGS internally: `sf` source amplitude in dyne, displacement in cm.
The `1e-15` scaling in `SourceModel._update_source_mechanism` converts the
user-specified `force_amplitude` to internal `m0`. Setting `FORCE_AMPLITUDE=1e5`
makes PyFK compute for 1 N (1 N = 1e5 dyne). The `_sync_trace_to_time`
function already converts cm→m (×0.01).

Static calibration confirmed: PyFK static for `force_amp=1.0` (1 dyne) matches
the analytic formula u = F/(4πμr) within 14%.

### Verified correct: SEM force normalization

1. `source_locator.py` — GLL Lagrange basis weights, normalized to sum=1
   across shared elements. For a source on a shared edge of 4 elements, each
   contributes ¼ weight; `scatter_to_rank` accumulates to full weight at
   shared nodes. ✓
1. `preprocess/cpp/main.cpp` + `cli.py:448` — mass = ρ·J·w_i·w_j·w_k
   (geometric part from C++, density multiplied in Python). ✓
1. `solver.cpp` — Newmark-β (β=¼, γ=½), standard implementation. ✓
1. Total force = `stf_val × Σ(weights)` = 1.0 N. ✓

### Remaining issues

**Issue A — Diagonal ~3× discrepancy** (both halfspace and layer):

SEM displacement is 1.7–3× larger than analytic/PyFK references on diagonal
components (F_z→u_z, F_x→u_x, F_y→u_y). Possible causes:

- Near-field effects (source-receiver distance ~1 wavelength at f0=2 Hz)
- PML reflections adding energy
- Source at shared element edge (4 elements meet at xi=±1, eta=±1)

**Issue B — Off-diagonal radiation pattern** (both halfspace and layer):

F_z→u_x and F_x→u_z are severely underestimated in SEM (0.09–0.18× of
reference). The vertical-force horizontal radiation and horizontal-force
vertical radiation are too weak. Possible causes:

- Free surface boundary condition (stress-free enforced via weak form)
- P-SV conversion at the free surface not correctly captured
- Source at shared edge affecting horizontal radiation pattern

### Next investigation steps

1. Move source to element interior (not shared edge) to isolate edge effects
1. Increase source-receiver distance to far-field to isolate near-field effects
1. Check free surface displacement components (u_x from F_z) at the surface
1. Compare SEM radiation pattern with analytic P-SV radiation coefficients

## Summary

| Item | Module | Priority | Effort |
|------|--------|----------|--------|
| SLS attenuation | preprocess + forward/viscoelastic | High | Large |
| Compress integration | forward/ | Low | Tiny |
| Compression benchmark | compress | Low | Small |
| HIP/SYCL backends | forward/elastic/ | Low | Medium |
| Force normalization | forward/share + postprocess | High | Medium |
