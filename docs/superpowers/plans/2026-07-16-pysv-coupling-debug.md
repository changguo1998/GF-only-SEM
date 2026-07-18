# Debug: F_z→u_x Radiation Pattern — P-SV Coupling Issue

## Status

**CORRECTION (2026-07-19):** The conclusion below ("Cartesian mesh
anisotropy") was WRONG. The E-W asymmetry was a CG-SEM interface assembly
bug (fixed by global node numbering `cff2cd1`), and the P-SV coupling
"bias" was a Green tensor index convention mismatch (transpose bug) in
the postprocess (fixed 2026-07-19). After both fixes, all 9 components
match the Lamb reference within 0.94–1.03× at raw vertices. See
[`docs/deferred.md`](../../deferred.md) §6 for the corrected analysis. The
body below is preserved as historical record.
Diagonal components verified perfect (1.01-1.03×).
P-SV coupling bias (~0.5-2×) is a symptom, not a bug.

## Timeline

| Commit | Finding |
|--------|---------|
| `49d3455` | Fixed `mode='same'` convolution bug in both reference.py files |
| `2e0cbff` | Updated docs: example pipeline status |
| `8e4cd8e` | Fixed PyFK force unit: `FORCE_AMPLITUDE` 1.0→1e5 (dyne→Newton) |
| `e1ac709` | Moved source to element interior: fixed diagonal ~3× discrepancy |
| `ecc0d6d` | Root cause: Cartesian mesh E-W asymmetry (1.77×) — causes P-SV coupling bias |
| `a54e320` | Fixed GPU Newmark corrector: pass beta parameter (was hardcoded to 0) |
| Latest | Docs cleaned up: stale references removed, deferred.md §6 updated |

## Verified Correct

1. **SEM force normalization** — GLL Lagrange weights sum to 1; normalization
   across shared elements correct (`source_locator.py:364`).
1. **Mass matrix** — M = ρ·J·w_i·w_j·w_k; density applied in `cli.py:448`.
1. **Newmark integration** — explicit central difference β=0, γ=½ (`solver.cpp:57-69`).
1. **Element residual** — stress computation (`element_cpu.cpp:83-109`) correct;
   includes P-SV coupling via σ_xz = 2μ·ε_xz → x-component residual.
1. **Source weights** — computed from Lagrange basis at source natural coordinates
   (`source_locator.py:140-162`); partition of unity holds.
1. **PyFK unit convention** — output in cm (docstring); force_amplitude in dyne;
   1e-15 scaling converts to internal m0; already converted cm→m in
   `_sync_trace_to_time` (`reference.py:96`).

## Key Data: Like-for-Like Comparison at Vertex (5556,5556,0)

Source: (5278, 5278, 278) — center of element (9,9,0), interior
Vertex: (5556, 5556, 0) — surface corner, azimuth 45°, R=481m

| Component | SEM | Lamb Reference | SEM/ref |
|-----------|-----|----------------|---------|
| F_x→u_x (diag) | 9.32e-15 | 9.07e-15 | **1.03** |
| F_y→u_y (diag) | 9.32e-15 | 9.07e-15 | **1.03** |
| F_z→u_z (diag) | 1.20e-14 | 1.18e-14 | **1.01** |
| F_x→u_y (in-plane off-diag) | 3.73e-15 | 3.99e-15 | 0.94 |
| F_y→u_x (in-plane off-diag) | 3.73e-15 | 3.99e-15 | 0.94 |
| **F_z→u_x (z-coupling)** | **3.00e-15** | **5.87e-15** | **0.51** ❌ |
| **F_z→u_y (z-coupling)** | **3.00e-15** | **5.87e-15** | **0.51** ❌ |
| **F_x→u_z (z-coupling)** | **5.83e-15** | **3.16e-15** | **1.84** ❌ |
| **F_y→u_z (z-coupling)** | **5.82e-15** | **3.16e-15** | **1.84** ❌ |

### Summary

- Diagonal components: perfect (unique vertex, no interpolation)
- In-plane off-diagonal (F_x→u_y, F_y→u_x): excellent (0.94)
- z-axis coupling components: **systematically biased**
  - F_z→u_x and F_z→u_y: ~2× too SMALL (0.51×)
  - F_x→u_z and F_y→u_z: ~1.8× too LARGE (1.84×)
- Triliear interpolation DEGRADES accuracy: F_z→u_x goes from 0.51× (raw
  vertex) to 0.09× (interpolated). Interpolation mixes vertices with different
  azimuth and distance, causing significant off-diagonal errors.

## Critical Finding: East-West Wavefield Asymmetry

**All 4 vertices at same distance R=481m from source show systematic asymmetry:**

| Vertex | Position | F_z→u_z | F_x→u_x |
|--------|----------|---------|----------|
| NE (5556,5556,0) | East side | 1.20e-14 | 9.32e-15 |
| SE (5556,5000,0) | East side | 1.20e-14 | 9.32e-15 |
| NW (5000,5556,0) | **West side** | **6.79e-15** | **5.20e-15** |
| SW (5000,5000,0) | **West side** | **6.79e-15** | **5.20e-15** |

**East/West ratio = 1.77×** — same for all components and force directions.
For a vertical force in a halfspace, displacement should be axisymmetric.
This violates fundamental physics.

Source verified at exact element center (xi=0.0008, eta=0.0008).
Element size symmetric (556m both sides). Only 1 source element.
PML symmetric (3 elements each side).

### Root cause hypothesis

The SEM applies force on a Cartesian GLL grid (5×5 nodes in horizontal
plane). Even with symmetric force distribution in reference coordinates,
the wave propagation on this rectangular grid is inherently anisotropic.

The GLL nodes cluster near element boundaries (xi=±1). The force at nodes
near the East and West faces (xi≈±1) has different numerical coupling to
the free surface and volume than nodes near the center (xi≈0).

### Impact

This mesh anisotropy explains ALL observed discrepancies:

1. F_z→u_x at 0.51× — the East-West asymmetry distorts the P-SV coupling
   2\. F_x→u_z at 1.84× — same root cause, different manifestation
   3\. 1.77× u_z variation at same R — direct evidence of asymmetry

### Resolution options

1. **Finer mesh** — more elements → smaller anisotropy per element
1. **Higher polynomial order** — more GLL points → smoother force distribution
1. **Radial/unstructured mesh** around source — isotropic by construction
1. **Moment-tensor source correction** — apply a correction factor for
   direction-dependent radiation
1. **Accept and calibrate** — the Green tensor is "only" 2× off for
   P-SV coupling; diagonal & in-plane components are within 6%.
   For many applications (strain-based Green's functions at larger
   distances), this might be acceptable.

## Previous Analysis: z-axis Coupling Bias

## Trilinear Interpolation Issue

The GreenFunctionLibrary uses trilinear interpolation over element corner
vertices. For the query point (5778,5278,0), the 4 interpolation vertices
are at:

- (5556,5000,0): azimuth ~135°, R=393m horizontal + 278m vertical
- (6111,5000,0): azimuth ~108°, R=876m horizontal + 278m vertical
- (5556,5556,0): azimuth ~45°, R=393m horizontal + 278m vertical
- (6111,5556,0): azimuth ~71°, R=876m horizontal + 278m vertical

These vertices have DIFFERENT distances and azimuths from the source.
Trilinear interpolation fails for off-diagonal displacement components
because u_x/u_z ratio varies strongly with azimuth and distance.

**Recommendation**: Use query points at recorded vertices (no
interpolation) for accurate comparison. Or implement higher-order
interpolation using the GLL basis functions.
