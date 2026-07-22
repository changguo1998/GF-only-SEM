# Known Bugs (2026-07-22)

> Discovered during elastic solver correctness verification (halfspace example,
> CPU + MPI, 16 ranks, direction x/y/z).

______________________________________________________________________

## Bug 1: C-PML causes numerical divergence (CRITICAL)

**Status:** FIXED (2026-07-22)
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

### Additional Bugs Found During Fix Attempt (2026-07-22)

During the fix attempt, three additional issues were discovered:

#### Bug 1a: Accel scale factor uses 1/ρ instead of ρ

**File:** `forward/share/src/pml.cpp:188` (CPU) and `cuda_step.cu:552` (CUDA)

The C-PML acceleration contribution uses `scale = w * (1/ρ) * J`, but
SPECFEM3D uses `scale = w * ρ * J` (see
`pml_compute_accel_contribution.f90:122`). Since the residual is later
divided by mass (∝ ρ *w* J), the net acceleration is:

- Our code: `(w * (1/ρ) * J * PML_term) / (ρ * w * J) = PML_term / ρ²`
- SPECFEM3D: `(w * ρ * J * PML_term) / (ρ * w * J) = PML_term`

Our code amplifies the PML term by 1/ρ² ≈ 1/7.3e6, causing massive
over-correction. **Status: FIXED** (changed to `ρ` in both CPU and CUDA).

#### Bug 1b: Coefficient magnitude explosion (partial-fraction ill-conditioning)

Even with the ρ fix, the C-PML coefficients themselves are too large.
The partial-fraction formulas produce coefficients of order d³/Δα² where
d is the damping profile (up to ~52) and Δα is the alpha difference
(~0.3 for interior, ~0 for boundary). At boundary nodes, coefficients
reach 1e6-1e9.

**Mitigation applied:** dist clipping (`DIST_EPSILON=1e-3`), alpha spacing
enforcement (`ALPHA_MIN_SPACING=1e-3`), and coefficient clamping
(`COEF_CLAMP_THRESHOLD`). These reduce coefficients from 1e9 to 1e3 but
do NOT fully prevent divergence.

#### Bug 1c: C-PML structural bugs (sign, ordering, timing)

**Root causes found and fixed (commit 7dd0598):**

1. **Sign error (FIXED):** `cpml_accel_contribution` used `residual += PML`
   but should use `residual -= PML`. SPECFEM3D uses `accel -= (force + PML)`;
   our `scatter_residual` already has a negative sign (`r -= sigma:gradN`),
   so PML must also be subtracted. The positive sign created a positive
   feedback loop (velocity -> positive accel -> larger velocity).

1. **PML displacement field ordering (FIXED):** `cpml_save_displ_new` used
   OLD displacement instead of PREDICTED displacement (`displacement_tilde`).
   SPECFEM3D computes `PML_displ_new` AFTER the Newmark predictor.

1. **Memory variable update timing (FIXED):** `rmemory_displ` and
   `rmemory_strain` were updated AFTER the corrector, but used BEFORE
   (stale). Moved to before the element kernel.

**Result:** Solver stable through step 400 (was diverging at step 100).

#### Bug 1d: Strain correction uses symmetric stress (FIXED)

**Critical architectural issue:** The strain correction (A6-A23) causes
rapid divergence at step 400 even without the acceleration contribution.

SPECFEM3D uses **non-symmetric stress** in PML elements with THREE
separate correction groups (\_x,\_y, \_z), each applying different PML
corrections to all 9 gradient components. The stress is non-symmetric:
`sigma_yx != sigma_xy`.

Our element kernel uses **symmetric strain** (\`eps[l][m] = 0.5\*(du_dx[l][m]

- du_dx[m][l])\`), which mixes different PML corrections. This is
  fundamentally wrong for PML elements and causes exponential divergence
  when the source wave enters the PML.

**Isolation test results:**

| Configuration | Step 400 | Step 500 | Step 998 |
|---------------|----------|----------|----------|
| All zero | 1.3e5 | 1.6e5 | 1.5e5 (stable) |
| abar only (no strain/mem) | 1.1e5 | 1.7e5 | 1.2e8 (slow growth) |
| strain only (no abar/mem) | 5.5e13 | 3.2e26 | inf (rapid divergence) |
| abar+mem (no strain) | 1.2e5 | 1.4e5 | 2.5e15 (growth) |
| All four | 1.1e5 | 5.6e13 | inf (rapid divergence) |

**Status:** FIXED (commits 914663f, 2953472, f2d5c51). Full implementation:

1. Non-symmetric stress kernel (`compute_pml_non_symmetric_stress`) matching
   SPECFEM3D three-group (\_x, \_y, \_z) formulation — strain/stress correction
   uses different PML groups per stress column
1. lx/ly/lz alpha-convolved strain memory (12 additional entries per node,
   matching `pml_compute_memory_variables.f90:269-287`)
1. SPECFEM3D parameter separation (`_separate_pml_parameters`,
   `_separate_xy_node`, etc.) preventing exact-zero partial-fraction denominators
1. COEF_SAFETY_CLAMP=3.0 as fallback for pathological coefficients
   (K_MAX_PML=1 causes strain coefficients ~O(1e3) naturally)

**Result:** Solver stable 1000 steps (halfspace example), max|u| ≈ 2.4e5
with no growth. All 204 Python tests pass, all 6 C++ executables build clean.

**Note:** K_MAX_PML=14 (SPECFEM3D recommended) would reduce coefficient
magnitude ~200× but requires 4–8× smaller dt due to kx·ky gradient
prefactors reaching ~200 at PML corners. Deferred to future optimization.

______________________________________________________________________

## Bug 2: Postprocess velocity/acceleration tensors are all zero

**Status:** FIXED (2026-07-22, commit 785033d)
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

### Fix Applied

`merge_direction()` in `postprocess/cpp/main.cpp` only read `displacement`
from record files; velocity and acceleration reads were annotated as
`// deferred` but never implemented. Added `read_field_4d()` calls for
velocity/acceleration following the same pattern as displacement, plus
mass-weighted averaging in the normalization step.

**Verification:** halfspace 1000 steps → tile_x001_y001.h5:
`velocity_tensor` norm=7.83e0, `acceleration_tensor` norm=1.27e2
(both non-zero, previously zero).

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
| 1 | C-PML numerical divergence | Critical | FIXED | — |
| 2 | Velocity/acceleration = 0 in postprocess | Medium | FIXED | — |
| 3 | Displacement amplitude ~1e9× mismatch | TBD | Not investigated | Final validation |
