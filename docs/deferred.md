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

**Status:** Root cause identified — Cartesian hex mesh anisotropy.
Diagonal components correct (1.01–1.03×); P-SV coupling bias
(~0.5–2×) is a symptom of direction-dependent wave propagation from
a point force on a rectangular GLL grid. Resolution options remain
open.

Full debug analysis: [`superpowers/plans/2026-07-16-pysv-coupling-debug.md`](superpowers/plans/2026-07-16-pysv-coupling-debug.md)

### Resolved

1. **PyFK force unit** (`8e4cd8e`): `FORCE_AMPLITUDE` 1.0→1e5 (dyne→Newton).
   `_sync_trace_to_time` already converts cm→m.
1. **Shared-edge source inflation** (`e1ac709`): source at 4-element
   shared edge caused ~3× diagonal inflation. Moved source to element
   interior (center) → diagonal 1.01–1.03×.
1. **Convolution truncation** (`49d3455`): `mode='same'`→`mode='full'`
   to preserve trailing STF energy.

### Verified correct

1. GLL Lagrange weights normalized to sum=1 across shared elements ✓
1. Mass = ρ·J·w_i·w_j·w_k, density applied in `cli.py:448` ✓
1. Newmark-β (β=¼, γ=½) in solver.cpp:57–68 ✓
1. Total force = stf_val × Σ(weights) = 1.0 N ✓
1. Element residual: isotropic elastic stress correct ✓

### Root cause: Cartesian mesh anisotropy

**Critical finding (`ecc0d6d`):** 4 surface vertices at identical distance
R=481m from the source show **1.77× East-West asymmetry** in displacement.
For a vertical force in a halfspace, displacement should be axisymmetric.

| Vertex | Position | F_z→u_z | F_x→u_x |
|--------|----------|---------|----------|
| NE (5556,5556,0) | East (x=5556) | 1.20e-14 | 9.32e-15 |
| SE (5556,5000,0) | East (x=5556) | 1.20e-14 | 9.32e-15 |
| NW (5000,5556,0) | West (x=5000) | 6.79e-15 | 5.20e-15 |
| SW (5000,5000,0) | West (x=5000) | 6.79e-15 | 5.20e-15 |

Source verified at exact element center. Mesh symmetric (±278m to
both x-faces). PML symmetric (3 elements each side). The asymmetry
originates from the Cartesian GLL grid: a point force is distributed
to nodes on a rectangular grid, producing direction-dependent wave
propagation.

**Impact on P-SV coupling:**

| Component | SEM/ref (raw vertex, no interp) |
|-----------|--------------------------------|
| F_x→u_x, F_y→u_y, F_z→u_z (diagonals) | 1.01–1.03 ✓ |
| F_x→u_y, F_y→u_x (in-plane off-diag) | 0.94 ✓ |
| F_z→u_x, F_z→u_y (z→horizontal) | **0.51** ❌ |
| F_x→u_z, F_y→u_z (horizontal→z) | **1.84** ❌ |

The P-SV coupling errors are symptoms of the mesh anisotropy — not
free-surface bugs or stiffness matrix errors.

### Resolution options

1. **Finer mesh** — more elements → smaller anisotropy per element
1. **Higher polynomial order** — more GLL points → smoother force distribution
1. **Radial/unstructured mesh** around source — isotropic by construction
1. **Moment-tensor source correction** — apply direction-dependent corrections
1. **Accept and calibrate** — diagonal & in-plane components are within 6%;
   P-SV coupling ~2× error may be acceptable for many applications

### GPU Newmark corrector: hardcoded β=0

The GPU `cuda_newmark_correct` and its kernels hardcode β=0 in the
displacement update formula (`d_disp += dt*v + 0.5*dt²*a_old`)
instead of using `dt² * ((0.5-β)*a_old + β*a_new)`. This is a latent
bug — currently masked because the solver defaults to β=0, but any
non-zero β would produce incorrect results on GPU.

**Fix needed:** Pass β through `cuda_newmark_correct` to the kernels,
update both `newmark_correct_kernel` and `newmark_correct_rank_kernel`
to use the correct formula.

## Summary

| Item | Module | Priority | Effort |
|------|--------|----------|--------|
| SLS attenuation | preprocess + forward/viscoelastic | High | Large |
| Compress integration | forward/ | Low | Tiny |
| Compression benchmark | compress | Low | Small |
| HIP/SYCL backends | forward/elastic/ | Low | Medium |
| Force normalization | forward/share + postprocess | High | Medium |
