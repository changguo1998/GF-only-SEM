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
  /field/cell/tau_sigma[n_cell, NGLL, NGLL, NGLL, n_sls]
  /field/cell/tau_epsilon[n_cell, NGLL, NGLL, NGLL, n_sls]
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

**Status: RESOLVED.** The P-SV coupling "bias" (~0.5–2×) documented here
was misdiagnosed as Cartesian mesh anisotropy. Investigation on 2026-07-19
found it was a **Green tensor index convention mismatch** (transpose bug)
in the postprocess. After the fix, all 9 Green tensor components match the
Lamb analytic reference within 0.94–1.03× at raw vertices.

### Two separate issues, both now fixed

**Issue A — East-West wavefield asymmetry (1.77×):**
Four surface vertices at identical distance R=481 m from a centered source
showed 1.77× E/W asymmetry. This was NOT mesh anisotropy — it was a
CG-SEM element-interface assembly bug. **Fixed by the global GLL node
numbering repair** (`cff2cd1`, 2026-07-17): cross-element stiffness
coupling was broken, producing directional wave propagation errors.
After the fix, E/W ratio = 1.00 for all components.

**Issue B — P-SV coupling component "bias" (0.51× / 1.84×):**
The off-diagonal z-coupling components (F_z→u_x, F_x→u_z) appeared
biased because the postprocess assembled `displacement_tensor` as
`[force_dir, disp_comp]` while the documented convention (and
`greens_tensor`, and the analytic reference, and `compare.py`) is
`[disp_comp, force_dir]`. `compare.py` compared a transposed SEM tensor
to the analytic reference, so the non-symmetric z-coupling components
swapped. **Fixed 2026-07-19** (`postprocess/cpp/main.cpp`): displacement,
velocity, and acceleration assembly transposed to `[disp, force]`, now
mirroring `greens_tensor [strain, force]`.

### Verified after both fixes (raw vertex, no interpolation)

Source (5278, 5278, 278) → vertex (5556, 5556, 0), R=482 m:

| Component | SEM/Lamb |
|-----------|----------|
| u_x(F_x), u_y(F_y), u_z(F_z) (diagonals) | 1.02–1.03 ✓ |
| u_x(F_y), u_y(F_x) (in-plane off-diag) | 0.94 ✓ |
| u_x(F_z), u_y(F_z) (horizontal←vertical) | 1.00 ✓ |
| u_z(F_x), u_z(F_y) (vertical←horizontal) | 0.95 ✓ |

rel_l2 (full waveform, raw vertex) = 0.21. All peak components within 6%.

### Remaining: trilinear interpolation degradation

The example `compare.sh` queries a receiver point that is NOT at a mesh
vertex, so the library trilinearly interpolates the 3×3 Green tensor over
8 corner vertices. Off-diagonal components vary strongly with azimuth and
distance, so interpolating them across vertices with different geometries
introduces error. This degrades the interpolated rel_l2 to ~0.58 (halfspace)
and ~0.88 (layer). **Mitigation:** query at recorded vertices (no
interpolation) for accurate comparison, or implement GLL-basis
interpolation. This is a query-accuracy limitation, not a solver bug.

### Resolved (historical)

1. **PyFK force unit** (`8e4cd8e`): `FORCE_AMPLITUDE` 1.0→1e5 (dyne→Newton).
1. **Shared-edge source inflation** (`e1ac709`): source at 4-element
   shared edge caused ~3× diagonal inflation. Moved source to element
   interior (center) → diagonal 1.01–1.03×.
1. **Convolution truncation** (`49d3455`): `mode='same'`→`mode='full'`.
1. **E-W asymmetry** (`cff2cd1`): global node numbering fixed interface coupling.
1. **Green tensor convention** (2026-07-19): postprocess transpose to `[disp, force]`.

### Verified correct (solver physics)

1. GLL Lagrange weights normalized to sum=1 across shared elements ✓
1. Mass = ρ·J·w_i·w_j·w_k, density applied in `cli.py` ✓
1. Newmark explicit central difference (β=0, γ=½) in solver.cpp ✓
1. Total force = stf_val × Σ(weights) = 1.0 N ✓
1. Element residual: isotropic elastic stress correct ✓
1. E-W axisymmetry: centered source gives identical E/W displacement ✓

### GPU Newmark corrector: hardcoded β=0 — FIXED

**Fixed in `a54e320`:** The GPU `cuda_newmark_correct` and its kernels now accept
`beta` as a parameter and use the correct formula `dt² * ((0.5-β)*a_old + β*a_new)`
instead of hardcoding `0.5*dt²*a_old`. The solver.cpp GPU paths pass `beta`.
Backward-compatible: with β=0 both formulas are identical.

## Summary

| Item | Module | Priority | Effort |
|------|--------|----------|--------|
| SLS attenuation | preprocess + forward/viscoelastic | High | Large |
| Compress integration | forward/ | Low | Tiny |
| Compression benchmark | compress | Low | Small |
| HIP/SYCL backends | forward/elastic/ | Low | Medium |
| Cartesian mesh anisotropy | forward + preprocess | Medium | Medium |
