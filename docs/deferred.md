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

## 2. Compress Module (Placeholder)

**Status:** Placeholder - not in current scope. The `compress/` module (header-only
HDF5 compression/chunking/precision utilities) has been removed. Record files are
written uncompressed. If compression is needed in the future, re-implement from
[`design/compress.md`](design/compress.md) (kept as historical design reference).

______________________________________________________________________

## 3. Full C-PML Implementation (Strain Correction)

**Status:** PARTIALLY IMPLEMENTED. Displacement-based C-PML (acceleration
correction, 3 memory variables/node) is implemented in the preprocessor and
forward solver. The remaining strain-based correction (A₆…A₂₃ coefficients,
18+ memory variables/node, element kernel modification) is deferred.

### Implemented (displacement-based C-PML)

- [`docs/design/cpml.md`](design/cpml.md): Full design document.
- `preprocess/pml_cpml.py`: C-PML profile computation (K, d, α per direction)
  and convolution coefficients (α/β 9 each, Ā₁…Ā₅, A₆…A₂₃).
- `forward/`: C-PML data structures in `types.hpp`, I/O in `io.cpp`,
  memory variable update + accel contribution in `pml.hpp/cpp`.
- `solver.cpp`: Old `v -= d·v` replaced with C-PML accel correction.
- Backward compatible: falls back to old damping when C-PML data absent.

### Remaining (strain-based correction)

- Element kernel modification (PML stress via A₆…A₂₃ convolution):
  `element_cpu.cpp`, `element_cuda.cu`.
- CUDA C-PML memory variable update.
- Restart I/O for C-PML memory state.
- Absorption quality validation (compare with old linear ramp).

### Preprocess

Needed:

- C-PML precompute already implemented in `preprocess/pml_cpml.py`.

### Forward

Current: displacement-based C-PML (3 memory vars/node, accel correction).
Needed: strain-based C-PML (18+ memory vars/node, stress modification).

```
d_axis = -(NPOWER + 1) * vp * ln(R_coef) / (2 * pml_width) * dist^(1.2 * NPOWER)
K_axis = K_MIN + (K_MAX - 1) * dist
α_axis = α_MAX * (1 - dist)

# Convolution coefficients (second-order, Xie et al. 2014)
coef0 = exp(-b*dt)
coef1 = (1 - exp(-b*dt/2)) / b
coef2 = coef1 * exp(-b*dt/2)

# Accel-update coefficients (l_parameter_computation)
Ā₁..Ā₅ from K, d, α, CPML_region

# Strain-update coefficients (lijk_parameter_computation)
A₆..A₂₃ from K, d, α, CPML_region
```

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

**Status: RESOLVED.** Two issues were found and fixed:

- **E-W wavefield asymmetry (1.77×)**: CG-SEM element-interface assembly bug.
  Fixed by global GLL node numbering (`cff2cd1`, 2026-07-17).
- **P-SV coupling "bias" (0.5–2×)**: misdiagnosed as Cartesian mesh anisotropy;
  actual cause was a Green tensor index convention mismatch (transpose bug) in
  the postprocess. Fixed 2026-07-19 (`postprocess/cpp/main.cpp`).

After both fixes, all 9 Green tensor components match the Lamb analytic
reference within 0.94–1.03× at raw vertices (rel_l2 ≈ 0.21).

### Remaining: trilinear interpolation degradation

The example `compare.sh` queries a receiver point that is NOT at a mesh
vertex, so the library trilinearly interpolates the 3×3 Green tensor over
8 corner vertices. Off-diagonal components vary strongly with azimuth and
distance, so interpolating them across vertices with different geometries
introduces error. This degrades the interpolated rel_l2 to ~0.58 (halfspace)
and ~0.88 (layer). **Mitigation:** query at recorded vertices (no
interpolation) for accurate comparison, or implement GLL-basis
interpolation. This is a query-accuracy limitation, not a solver bug.

### Verified correct (solver physics)

1. GLL Lagrange weights normalized to sum=1 across shared elements ✓
1. Mass = ρ·J·w_i·w_j·w_k, density applied in `cli.py` ✓
1. Newmark explicit central difference (β=0, γ=½) in solver.cpp ✓
1. Total force = stf_val × Σ(weights) = 1.0 N ✓
1. Element residual: isotropic elastic stress correct ✓
1. E-W axisymmetry: centered source gives identical E/W displacement ✓

## Summary

| Item | Module | Priority | Effort |
|------|--------|----------|--------|
| SLS attenuation | preprocess + forward/viscoelastic | High | Large |
C-PML (strain correction) | forward + preprocess | Medium | Large (displacement-based done) |
| Compress module | - | - | Placeholder (removed, see §2 above) |
| HIP/SYCL backends | forward/elastic/ | Low | Medium |
| ~~Cartesian mesh anisotropy~~ | forward + preprocess | - | RESOLVED (misdiagnosis, see §6 above) |
