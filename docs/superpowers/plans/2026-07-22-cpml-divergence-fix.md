# Fix Plan: C-PML Numerical Divergence (Bug 1)

> **Bug:** [`docs/bugs.md`](../../bugs.md) §1
> **Created:** 2026-07-22
> **Updated:** 2026-07-22 (status update after deeper investigation)
> **Goal:** Eliminate C-PML divergence causing inf/nan at step ~100.
>
> **Status:** Partially fixed. Three C++ bugs found and fixed (commit `7dd0598`);
> one deep architectural issue remains (Bug 1d, strain correction).
> See [`docs/bugs.md`](../../bugs.md) for full diagnostics.

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

Three-pronged approach. The core insight is that the partial-fraction
formulas are mathematically correct only when direction alphas are
distinct; when they coincide (repeated roots), individual terms diverge
even though their sum converges. Rather than deriving limit forms for
every degenerate case (complex, error-prone), we ensure alphas are never
degenerate via small perturbations.

### Fix A: Alpha spacing enforcement (primary fix)

After computing alpha profiles, enforce a minimum spacing between
direction alphas at each GLL node. When two direction alphas are closer
than `ALPHA_MIN_SPACING`, perturb them apart by a tiny amount.

**Why this works:** The C-PML coefficients depend continuously on alpha.
A perturbation of order 1e-4 in alpha produces a negligible change in
the physical absorption (alpha controls the frequency shift, and 1e-4
\<< the alpha_max values of ~5-7). But it prevents the partial-fraction
denominators from collapsing to the clamp threshold, keeping coefficients
bounded.

**Implementation** (in `compute_pml_profiles()`, after alpha computation):

```python
# For each PML node, ensure direction alphas are spaced apart
for axis_pair in [(0,1), (0,2), (1,2)]:
    a, b = axis_pair
    diff = alpha_store[..., a] - alpha_store[..., b]
    too_close = np.abs(diff) < ALPHA_MIN_SPACING
    # Perturb: push b away from a by ALPHA_MIN_SPACING
    sign = np.sign(diff + ALPHA_MIN_SPACING)  # default + if diff==0
    alpha_store[..., b] = np.where(
        too_close,
        alpha_store[..., a] + sign * ALPHA_MIN_SPACING,
        alpha_store[..., b]
    )
```

**Constants:** `ALPHA_MIN_SPACING = 1e-3` (10x larger than MIN_DISTANCE
= 1e-6, so the np.where clamp in_l_parameter never triggers).

### Fix B: dist clipping at boundary (safety net)

Ensure alpha is never exactly zero at the outer boundary by clipping
`dist` to `[0, 1 - DIST_EPSILON]`, so
`alpha = alpha_max * (1 - dist) >= alpha_max * DIST_EPSILON > 0`.

This prevents the all-zero-alpha case at boundary GLL nodes, which is
the most common trigger for the degeneracy (e.g. node 62 in the
diagnostics: alpha = [0, 0, 1.15]).

**Side effect:** The outermost PML nodes will have slightly non-zero
alpha (less absorption shift), but this is a negligible physical change
that does not affect absorption quality.

### Fix C: Coefficient range validation (regression guard)

After computing coefficients, warn if any exceeds a sanity threshold
(e.g. 1e4). This catches future regressions where the alpha spacing or
dist clip might not be sufficient.

______________________________________________________________________

## 3. Implementation Tasks

### Task 1: Add alpha-spacing enforcement to `compute_pml_profiles()`

**File:** `preprocess/pml_cpml.py`, function `compute_pml_profiles()`

After computing `alpha_store`, add a post-processing step that enforces
minimum spacing between direction alphas at each GLL node. This is the
**primary fix** - it prevents the partial-fraction denominators from
collapsing.

**Implementation:**

```python
# Enforce minimum alpha spacing between directions (Fix A)
for a, b in [(0, 1), (0, 2), (1, 2)]:
    diff = alpha_store[..., a] - alpha_store[..., b]
    too_close = np.abs(diff) < ALPHA_MIN_SPACING
    sign = np.where(diff >= 0, 1.0, -1.0)
    alpha_store[..., b] = np.where(
        too_close,
        alpha_store[..., a] + sign * ALPHA_MIN_SPACING,
        alpha_store[..., b]
    )
```

**Constants:** Add `ALPHA_MIN_SPACING = 1e-3` to module constants. This
is 1000x larger than `MIN_DISTANCE = 1e-6`, so the `np.where` clamp in
`_l_parameter()` / `_lijk_parameter()` never triggers.

**Why not modify `_l_parameter()` / `_lijk_parameter()`:** Those
functions receive alpha as input. If the input alphas are already
well-spaced (ensured by this task), their existing `np.where` clamp
never activates and the formulas work correctly. No changes needed to
the coefficient functions.

### Task 2: Clip dist to avoid alpha=0 at boundary

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

**Constant:** Add `DIST_EPSILON = 1e-3` to module constants.

### Task 4: Unit tests for degenerate alpha cases

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

### Task 5: Integration test - halfspace solver stability

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

- [ ] Task 1: Alpha-spacing enforcement in `compute_pml_profiles()`
- [ ] Task 2: `dist` clipping to `1 - DIST_EPSILON`
- [ ] Task 3: Coefficient range validation + warning
- [ ] Task 4: Unit tests pass (degenerate alpha cases)
- [ ] Task 5: Halfspace solver runs 1000 steps without inf/nan
- [ ] Coefficient max in partition files < 1e3 (was 1e6-1e9)
- [ ] All 204 existing tests still pass
- [ ] `bash format.sh` clean

______________________________________________________________________

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Alpha perturbation changes absorption | Low | Low (perturbation ~1e-3 \<< alpha_max ~5-7) | Verify absorption quality unchanged; coefficient range check |
| dist clipping changes absorption | Low | Low (epsilon=1e-3 negligible) | Verify absorption quality unchanged |
| Existing tests break | Low | Low | Run all 204 tests after each task |
| CUDA backend needs same fix | N/A | N/A | Fix is in preprocessor (Python), CUDA reads precomputed coefficients - no CUDA change needed |

______________________________________________________________________

## 7. Actual Progress (2026-07-22)

### Completed

**C++ Solver Bugs (commits `b9f3c74`, `7dd0598`):**

- [x] **Sign error:** `residual += PML` → `residual -= PML` (root cause)
- [x] **PML displ field ordering:** Split into `cpml_save_displ_old` (before predictor) and `cpml_save_displ_new` (after predictor, using `displacement_tilde` + predicted velocity)
- [x] **Memory variable timing:** Moved before element kernel (step 3b)
- [x] **Accel contribution uses predicted fields:** Uses `displacement_tilde` and `v + dt/2*a`

**Python Preprocessor Mitigations (`b9f3c74`):**

- [x] Task 1: Alpha spacing enforcement
- [x] Task 2: `dist` clipping to `1 - DIST_EPSILON`
- [x] Task 3: Coefficient range validation + warning

### Not Done

- [ ] Task 4: Unit tests for degenerate alpha cases
- [ ] Task 5: Integration test (blocked by Bug 1d)

### New Discoveries (beyond original scope)

#### Bug 1a (FIXED): Accel scale factor uses 1/ρ instead of ρ

`pml.cpp:188` and `cuda_step.cu:552` used `scale = w * (1/ρ) * J`.
SPECFEM3D uses `scale = w * ρ * J`. Fixed in both CPU and CUDA.

#### Bug 1c (FIXED): Sign, ordering, timing errors

Three root-cause C++ bugs caused energy injection instead of absorption.
See commit `7dd0598` for details.

#### Bug 1d (NOT FIXED): Strain correction uses symmetric stress

**Critical architectural issue** — SPECFEM3D uses non-symmetric stress
with three correction groups. Our kernel uses symmetric strain, mixing
different PML corrections. Isolation tests show strain correction alone
causes rapid divergence at step 400.

**Resolution:** Requires separate PML element kernel with non-symmetric
stress. Major architectural change — deferred.

### Next Steps

1. Defer Bug 1d (strain correction) — major architectural change
1. Run with `pml_coef_strain` zeroed for stability
1. Proceed to Lamb verification with C-PML (accel contrib only)
1. Fix Bug 2 (postprocess velocity/acceleration = 0)
