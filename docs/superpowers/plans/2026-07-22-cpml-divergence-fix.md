# Fix Plan: C-PML Numerical Divergence (Bug 1)

> **Bug:** [`docs/bugs.md`](../../bugs.md) §1
> **Created:** 2026-07-22
> **Goal:** Eliminate the C-PML coefficient explosion that causes the solver
> to diverge at step ~100, restoring stable wave propagation.

______________________________________________________________________

## 1. Problem Statement

The C-PML partial-fraction decomposition in
`preprocess/pml_cpml.py` (`_l_parameter`, `_lijk_parameter`,
`_lx/ly/lz_parameter`) produces coefficients of magnitude 1e6-1e9 when
PML direction alphas are equal or near-equal. These coefficients enter
the acceleration update as `accel_pml = scale * (A2*u + A3*mem + ...)`,
instantly amplifying numerical noise to ±inf.

**Root cause:** The formulas for A₃/A₄/A₅ (multi-direction PML regions)
contain denominators `(α_y - α_x)`, `(α_z - α_x)`, etc. When
α_x ≈ α_y, the denominator -> 0 and the coefficient explodes.

**Our code** clamps denominators to `MIN_DISTANCE = 1e-6` via `np.where`,
which prevents division-by-zero but does NOT prevent the explosion
(numerator is still full-size, divided by 1e-6 = huge).

**SPECFEM3D** uses a hard `if` check and calls `stop 'Error'` when
alphas are too close, relying on the anisotropic alpha_max multipliers
(0.9, 1.0, 1.1) to ensure alphas differ at interior PML nodes.

______________________________________________________________________

## 2. Fix Strategy

Two-pronged approach:

### Fix A: Degenerate-alpha guard (primary fix)

When direction alphas are equal or near-equal (within `MIN_DISTANCE`),
use the **limit form** of the partial-fraction decomposition (collapsed
formula for repeated roots) instead of clamping the denominator.

**Mathematical basis:** When α_x -> α_y, the three-term partial fraction

```
f(s) / [(s+α_x)(s+α_y)(s+α_z)]  →  f(s) / [(s+α_x)²(s+α_z)]
```

collapses to a two-term form with a repeated root. The coefficient of
the repeated root is the derivative w.r.t. that root, not a ratio.

**Implementation:** Replace the `np.where` clamp with explicit branching:

```python
if abs(ax - ay) < MIN_DISTANCE:
    # Collapsed limit: alpha_x == alpha_y
    # Use the 2-direction (XY collapsed) formula
    ...
elif abs(ax - az) < MIN_DISTANCE:
    # Collapsed limit: alpha_x == alpha_z
    ...
elif abs(ay - az) < MIN_DISTANCE:
    # Collapsed limit: alpha_y == alpha_z
    ...
else:
    # Full 3-direction formula (original)
    ...
```

For the fully degenerate case (all three alphas equal, e.g. at the outer
boundary where dist=1 and all alphas=0), collapse to the single-direction
formula.

### Fix B: Alpha-shift at boundary (safety net)

Ensure alpha is never exactly zero at the outer boundary by clipping
`dist` to `[0, 1 - epsilon]` (e.g. `epsilon = 1e-3`), so
`alpha = alpha_max * (1 - dist) >= alpha_max * epsilon > 0`.

This prevents the all-zero-alpha case at boundary GLL nodes, which is
the most common trigger for the degeneracy.

**Side effect:** The outermost PML nodes will have slightly non-zero
alpha (less absorption shift), but this is a negligible physical change
that does not affect absorption quality.

______________________________________________________________________

## 3. Implementation Tasks

### Task 1: Add degenerate-alpha limit forms to `_l_parameter()`

**File:** `preprocess/pml_cpml.py`, function `_l_parameter()`

For each multi-direction region (CPML_XYZ, CPML_XY_ONLY, CPML_XZ_ONLY,
CPML_YZ_ONLY), replace the `np.where`-clamped division with explicit
limit-form branches:

- **CPML_XYZ** (3 directions):

  - If all three alphas distinct: original A3/A4/A5 formula
  - If α_x ≈ α_y: collapsed 2-direction formula (treat XY as one
    direction, Z as the other)
  - If α_x ≈ α_z: collapsed (XZ as one, Y as the other)
  - If α_y ≈ α_z: collapsed (YZ as one, X as the other)
  - If all three ≈ equal: single-direction formula

- **CPML_XY_ONLY** (2 directions):

  - If α_x ≈ α_y: single-direction formula with α = α_x, d = d_x + d_y
  - Else: original A3/A4 formula

- Same pattern for CPML_XZ_ONLY, CPML_YZ_ONLY.

**Reference:** The collapsed formulas are derived by taking the limit
of the partial fraction as two roots merge. For the 2-direction case
(α_x -> α_y), A3 + A4 collapses to:

```
A_collapsed = A0 * α_x² * (β_x - α_x) * d/dα_x[(β_x-α_x)(β_y-α_x)] / 1
```

(derivative of the numerator w.r.t. the repeated root, per standard
partial-fraction theory for repeated roots).

**Verification:** Unit test - compute coefficients with α_x = α_y and
check that the result is finite and O(1) magnitude, not 1e6+.

### Task 2: Add degenerate-alpha guard to `_lijk_parameter()`

**File:** `preprocess/pml_cpml.py`, function `_lijk_parameter()`

Same limit-form branching as Task 1, but for the strain-update
coefficients (A₆-A₁₇). The `_lijk_parameter` function has the same
partial-fraction structure with direction permutations.

### Task 3: Clip dist to avoid alpha=0 at boundary

**File:** `preprocess/pml_cpml.py`, function `compute_pml_profiles()`

Change:

```python
dist = np.clip(dist, 0.0, 1.0)
```

to:

```python
dist = np.clip(dist, 0.0, 1.0 - DIST_EPSILON)  # DIST_EPSILON = 1e-3
```

This ensures `alpha = alpha_max * (1 - dist) >= alpha_max * 1e-3 > 0`
at the outer boundary, preventing the all-zero-alpha degeneracy.

**Constant:** Add `DIST_EPSILON = 1e-3` to the module constants.

### Task 4: Add coefficient-range validation

**File:** `preprocess/pml_cpml.py`, function
`compute_abar_coefficients()` and `compute_strain_coefficients()`

After computing coefficients, add a validation check that warns (or
raises) if any coefficient exceeds a sanity threshold (e.g. 1e4):

```python
max_abar = np.max(np.abs(coef_abar))
if max_abar > 1e4:
    import warnings
    warnings.warn(
        f"C-PML abar coefficient max={max_abar:.2e} exceeds 1e4, "
        f"possible degenerate-alpha issue"
    )
```

This catches future regressions early.

### Task 5: Unit tests for degenerate alpha cases

**File:** `tests/preprocess/test_pml_cpml.py` (new or extend)

Test cases:

1. **All alphas equal (boundary node):** α_x = α_y = α_z = 0, verify
   coefficients are finite and < 1e3.
1. **Two alphas equal (edge node):** α_x = α_y ≠ α_z, verify
   coefficients are finite.
1. **All alphas distinct (interior node):** original formula, verify
   unchanged behavior.
1. **Single-direction regions:** CPML_X_ONLY etc., verify unaffected.
1. **Coefficient range:** verify no coefficient exceeds 1e4 for
   realistic halfspace parameters.

### Task 6: Integration test - halfspace solver stability

**File:** `examples/halfspace/run.sh` (verification, not permanent test)

Run the halfspace example with C-PML enabled (default) for 1000 steps
and verify:

- No inf/nan in any step
- Displacement at step 100 (t=0.5s, before source) is < 1e-3 (noise
  level, not growing)
- Displacement at step 200 (t=1.0s, source fires) shows physical
  wave propagation

**Command:**

```bash
cd examples/halfspace
mpirun --oversubscribe -n 16 gf_solver_elastic_mpi --direction x
# Check record files for inf/nan
python -c "import h5py, numpy as np; ..."
```

______________________________________________________________________

## 4. Verification Checklist

- [ ] Task 1: `_l_parameter()` limit forms implemented
- [ ] Task 2: `_lijk_parameter()` limit forms implemented
- [ ] Task 3: `dist` clipping to `1 - DIST_EPSILON`
- [ ] Task 4: Coefficient range validation + warning
- [ ] Task 5: Unit tests pass (degenerate alpha cases)
- [ ] Task 6: Halfspace solver runs 1000 steps without inf/nan
- [ ] Coefficient max in partition files < 1e3 (was 1e6-1e9)
- [ ] All 204 existing tests still pass
- [ ] `bash format.sh` clean

______________________________________________________________________

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Limit-form formulas incorrect | Medium | High (wrong PML absorption) | Unit test against known values; compare with non-degenerate case in the limit |
| dist clipping changes absorption | Low | Low (epsilon=1e-3 negligible) | Verify absorption quality unchanged |
| Existing tests break | Low | Low | Run all 204 tests after each task |
| CUDA backend needs same fix | N/A | N/A | Fix is in preprocessor (Python), CUDA reads precomputed coefficients - no CUDA change needed |

______________________________________________________________________

## 6. Out of Scope

- Bug 2 (velocity/acceleration = 0 in postprocess) - separate fix
- Issue 3 (displacement amplitude mismatch) - investigate after Bug 1
  and Bug 2 are fixed
- C-PML absorption quality validation (plane wave reflection test) -
  future work after stability is restored
- CUDA-specific C-PML kernel changes - not needed, coefficients are
  precomputed in Python
