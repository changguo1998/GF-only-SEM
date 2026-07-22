# Known Bugs (2026-07-22)

> Discovered during elastic solver correctness verification (halfspace example,
> CPU + MPI, 16 ranks, direction x/y/z).

______________________________________________________________________

## Bug 1: C-PML causes numerical divergence (CRITICAL)

**Status:** Not fixed
**Severity:** Critical - blocks all correctness verification
**Affected:** `gf_solver_elastic_mpi`, `gf_solver_elastic_cuda`,
`gf_solver_viscoelastic_mpi`, `gf_solver_viscoelastic_cuda`

### Symptom

With C-PML enabled (default), the solver diverges to ±inf at step ~100
(t = 0.5 s), before the source fires at t₀ = 1.0 s. By step 500, all
fields are nan.

| Step | Time | Displacement | Velocity | Acceleration |
|------|-------|--------------|----------|--------------|
| 50 | 0.25s | ~1e-7 | ~1e-7 | ~1e-7 |
| 100 | 0.50s | ±inf | ±inf | ±inf |
| 500 | 2.50s | nan | nan | nan |

### Root Cause

The partial-fraction decomposition in `_l_parameter()` and
`_lijk_parameter()` (in `preprocess/pml_cpml.py`) produces astronomically
large coefficients when PML direction alphas are equal or near-equal.

**Mechanism:** The A₃/A₄/A₅ formulas for multi-direction PML regions
(CPML_XYZ, CPML_XY_ONLY, etc.) contain denominators like
`(α_y - α_x)`, `(α_z - α_x)`. When α_x ≈ α_y, these denominators → 0,
and the coefficients blow up.

**When it happens:** Alpha values are equal when:

- Two PML faces are at the same normalized distance `dist` (e.g. a node
  on the edge of two PML layers with equal widths)
- Alpha = 0 at the outer boundary (`dist = 1`, all directions)

**Evidence (partition_0.h5, halfspace example):**

```
Region 7 (XYZ corner), node 62:
  alpha = [0.0, 0.0, 1.15]     ← alpha_x == alpha_y == 0
  d     = [51.8, 51.8, 33.4]
  abar  = [137, 6111, 0, 0, -8.58e4]   ← A5 = -85828 (huge!)
  strain max_abs = 8.94e7              ← A-strain coefficients reach 1e8!

Overall:
  pml_coef_abar:   min=-2.11e6  max=1.69e5
  pml_coef_strain: min=-2.68e9  max=7.20e3   ← 1e9 magnitude!
```

These coefficients enter the acceleration update as
`accel_pml = scale * (A1*v + A2*u + A3*mem + ...)`, so a coefficient of
1e6-1e9 instantly amplifies tiny numerical noise to inf.

### Current Code Behavior

`preprocess/pml_cpml.py:_l_parameter()` uses `np.where` to clamp
denominators to `MIN_DISTANCE = 1e-6`:

```python
dxy = np.where(np.abs(ax - ay) < MIN_DISTANCE, MIN_DISTANCE, ax - ay)
A3 = A0 * ax**2 * (bx - ax) * (by - ax) * (bz - ax) / (dyx * dzx)
```

This prevents division-by-zero but does NOT prevent the explosion: the
numerator is still computed with the full formula, and dividing by 1e-6
produces values of order 1e6-1e9.

### SPECFEM3D Reference Behavior

SPECFEM3D (`pml_compute_accel_contribution.f90:485`) uses a hard `if`
check and calls `stop 'Error'` when alphas are too close:

```fortran
if (abs(alpha_x - alpha_y) >= min_distance .and.
    abs(alpha_x - alpha_z) >= min_distance .and.
    abs(alpha_y - alpha_z) >= min_distance) then
    ! ... compute A3/A4/A5 ...
else
    stop 'Error occured in l_parameter_computation in CPML_XYZ region'
endif
```

SPECFEM3D avoids this in practice because the anisotropic alpha_max
multipliers (0.9, 1.0, 1.1) ensure alpha_x ≠ alpha_y ≠ alpha_z for
interior PML nodes (dist < 1). At dist = 1 (outer boundary), all alphas
= 0, but SPECFEM3D's mesh rarely places a GLL node exactly at the
boundary.

### Verification

Disabling C-PML (zeroing `pml_coef_abar`, `pml_coef_strain`,
`pml_coef_alpha`, `pml_coef_beta` in partition files, falling back to
legacy `damping`) eliminates the divergence: solver runs stably for 1000
steps with no inf/nan.

**Fix plan:** See
[`docs/superpowers/plans/2026-07-22-cpml-divergence-fix.md`](superpowers/plans/2026-07-22-cpml-divergence-fix.md)

______________________________________________________________________

## Bug 2: Postprocess velocity/acceleration tensors are all zero

**Status:** Not fixed
**Severity:** Medium - prevents velocity/acceleration Green function
validation, but displacement still works

### Symptom

Green function tiles contain zero velocity and acceleration tensors:

```
field/displacement_tensor:  norm=6.37e-01  (non-zero, correct)
field/velocity_tensor:      norm=0.0       (ALL ZERO)
field/acceleration_tensor:  norm=0.0       (ALL ZERO)
```

But the raw record files (from the solver) DO contain non-zero
velocity/acceleration:

```
record_10_100.h5:
  velocity:     min=-3.87  max=4.56     (non-zero)
  acceleration: min=-118   max=159      (non-zero)
```

### Root Cause (preliminary)

The postprocess step (`gf_postprocess`) fails to extract velocity and
acceleration from the record files into the Green function tiles. The
displacement extraction works, but velocity/acceleration are left as
zeros.

**Suspected location:** `postprocess/` - the tile assembly logic likely
reads only displacement and leaves velocity/acceleration tensors
uninitialized.

### Impact

- `compare.py` reports `velocity rel_l2 = 1.0` and
  `acceleration rel_l2 = 1.0` (100% error) because SEM values are zero.
- Displacement comparison is unaffected.

### Fix Approach

Investigate the postprocess tile assembly code to find where
velocity/acceleration are dropped. Likely a missing read or write step
in the record-to-tile pipeline.

______________________________________________________________________

## Issue 3: Displacement amplitude mismatch (~1e9×)

**Status:** Not investigated - may be a normalization convention issue,
not a bug
**Severity:** Low (if convention) / High (if real bug)

### Symptom

Even with C-PML disabled (stable solver), the SEM displacement amplitude
is ~1e9× smaller than the Lamb analytical reference:

```
                SEM norm      Reference norm    Ratio
displacement    2.26e-3       6.95e6           3.1e9 (SEM smaller)
velocity        0.0           9.78e7           (Bug 2)
acceleration    0.0           1.64e9           (Bug 2)

best-fit SEM scale = 2.23e9
```

### Analysis

- **Waveform shape:** Cross-correlation (normalized) = 0.69-0.83, meaning
  the waveform shape has moderate similarity but is not a perfect match.
- **Arrival time:** SEM argmax ≈ 126, reference argmax ≈ 122 (4-step
  difference = 0.02 s, close).
- **Amplitude:** SEM is ~1e9× too small. With source amplitude 1e20 N,
  the SEM displacement should be much larger than a 1 N reference, not
  smaller.

### Possible Causes

1. **Source normalization / Green function convention:** The Green
   function library may define G = u / F (displacement per unit force),
   while the reference uses a different convention.
1. **Reciprocity convention:** The postprocess reciprocity mapping
   (source ↔ receiver swap) may introduce a scaling error.
1. **Interpolation:** `compare.py` reports `interpolated SEM: True`,
   meaning the receiver is not at a mesh vertex. Trilinear interpolation
   of off-diagonal components degrades accuracy.
1. **Residual C-PML instability:** Even with C-PML coefficients zeroed,
   the legacy damping may still affect the wavefield near boundaries.

### Dependency

This issue cannot be properly evaluated until Bug 1 (C-PML divergence)
and Bug 2 (velocity/acceleration = 0) are fixed. The waveform shape
similarity (0.69-0.83) suggests the solver physics is approximately
correct, but amplitude and shape need investigation after the blocking
bugs are resolved.

______________________________________________________________________

## Summary

| # | Bug | Severity | Status | Blocks |
|---|-----|----------|--------|--------|
| 1 | C-PML numerical divergence | Critical | Not fixed | All verification |
| 2 | Velocity/acceleration = 0 in postprocess | Medium | Not fixed | Vel/acc validation |
| 3 | Displacement amplitude ~1e9× mismatch | TBD | Not investigated | Final validation |
