# Deferred Designs and Plans

This file lists known deferred work. When work resumes, use the linked design or plan.

______________________________________________________________________

## 1. SLS Viscoelastic Attenuation

**Status: COMPLETE.** Implemented via 7 commits (Jul 2026):

- `c90268e`: SLS namespace, constants, `precompute_sls_coefficients()`, RankData fields
- `7ec4d18`: Preprocess τ computation (`preprocess/attenuation.py`), I/O + restart
- `a265668`: Viscoelastic solver skeleton (`forward/viscoelastic/`)
- `a8b083a`: CPU viscoelastic element kernel with SLS memory update
- `b0749ac`: Shared solver SLS integration + viscoelastic runner
- `f493e5e`: CUDA viscoelastic element kernel + SLS runtime
- `590d774`: Unit tests (Python + C++), 204 tests pass

**Architecture:**

- Independent solver binary `gf_solver_viscoelastic_mpi` reuses `libgf_shared`
  and the 5 shared `kernel_helpers`; only the stress computation differs
- SLS memory update inline in element kernel (R = a·R + b·Δσ)
- Space-varying Q per GLL node, n_sls = 3 (compile-time)
- `has_attenuation` flag gates all SLS code paths

**Preprocess:** `preprocess/attenuation.py`

- `compute_tau_from_q()`: τ-method with log-spaced τ_σ, least-squares fit for τ_ε
- `write_attenuation_to_model()`: writes tau_sigma/tau_epsilon to model.h5

**Forward solver executables:**

- `gf_solver_viscoelastic_mpi` (CPU + MPI) — verified: 16 ranks, 1000 steps
- `gf_solver_viscoelastic_cuda` (CUDA, no MPI)
- `gf_solver_viscoelastic_mpi_cuda` (CUDA + MPI)

**Spec:** [`docs/_archive/specs/2026-07-21-sls-viscoelastic-design.md`](../_archive/specs/2026-07-21-sls-viscoelastic-design.md)
**Plan:** [`docs/_archive/plans/2026-07-21-sls-viscoelastic.md`](../_archive/plans/2026-07-21-sls-viscoelastic.md)

______________________________________________________________________

## 2. Compress Module (Placeholder)

**Status:** Placeholder - not in current scope. The `compress/` module (header-only
HDF5 compression/chunking/precision utilities) has been removed. Record files are
written uncompressed. If compression is needed in the future, re-implement from
[`design/compress.md`](design/compress.md) (kept as historical design reference).

______________________________________________________________________

## 3. Full C-PML Implementation (Strain Correction)

**Status: COMPLETE.** Displacement-based C-PML (acceleration correction, 3 memory
variables/node) and strain-based correction (A₆…A₂₃, 18 memory variables/node)
are both implemented. Key commits:

- `18b89ca`: CUDA C-PML strain correction (element kernel + runtime)
- `71aac4a`: C-PML strain correction unit tests (7 new, 1443 assertions)

### Implemented

- [`docs/design/cpml.md`](design/cpml.md): Full design document.
- `preprocess/pml_cpml.py`: C-PML profile computation (K, d, α per direction)
  and convolution coefficients (α/β 9 each, Ā₁…Ā₅, A₆…A₂₃).
- `forward/`: C-PML data structures in `types.hpp`, I/O in `io.cpp`,
  memory variable update + accel contribution + strain correction in `pml.hpp/cpp`.
- `solver.cpp`: C-PML accel correction + strain memory update integrated.
- Backward compatible: falls back to old damping when C-PML data absent.
- Strain-based correction: element kernel (CPU + CUDA) modifies physical gradient
  via A₆…A₂₃ convolution, restart I/O for memory state.
- Unit tests: 7 Catch2 tests covering all CpmlStrain helper functions.

### Remaining: absorption quality validation

Comparing strain-based C-PML absorption with the old linear-ramp damping is
pending. The solver runs and passes the existing validation, but a dedicated
absorption quality benchmark has not yet been written.

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
|| ~~SLS attenuation~~ | preprocess + forward/viscoelastic | High | Large | **IMPLEMENTED** |
|| ~~C-PML (strain correction)~~ | forward + preprocess | Medium | Large | **IMPLEMENTED** |
| Compress module | - | - | Placeholder (removed, see §2 above) |
| HIP/SYCL backends | forward/elastic/ | Low | Medium |
| ~~Cartesian mesh anisotropy~~ | forward + preprocess | - | RESOLVED (misdiagnosis, see §6 above) |
