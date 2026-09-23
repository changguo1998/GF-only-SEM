# C-PML Design (Recursive Convolution Perfectly Matched Layer)

**Status:** COMPLETE (2026-07-22) — verified against SPECFEM3D, 99.1% scaled waveform
correlation with Lamb reference. The former ~3× factor was a postprocess count bug and is fixed.
**Reference:** Wang et al. (2006), Xie et al. (2014), SPECFEM3D implementation

**Latest stability validation (2026-09-22):** the XYZ acceleration residue `A5`
uses the SPECFEM denominator `(alpha_y-alpha_z)*(alpha_x-alpha_z)`. After
regenerating `model.h5` and all partitions with this sign, the half-space
configuration (`22x22x11`, `N=4`, `dt=5 ms`, all seven PML region types)
completed 5 s on the CUDA solver;
strain, displacement, velocity, and acceleration remained finite. Existing
generated HDF5 files must be regenerated after changing CPML coefficient code;
otherwise they retain stale coefficients and can reproduce the old corner
overflow.

## 1. Overview

Replace the current simple linear-ramp velocity damping (`v -= d·v`) with full
recursive-convolution C-PML (Wang et al. 2006, θ=1/8). The C-PML modifies
both the stress computation (strain derivatives) and the inertia term
(acceleration correction) inside PML elements.

### Current vs. C-PML

| Aspect | Current (linear ramp) | C-PML |
|--------|----------------------|-------|
| Damping | `v -= d·v` on velocity | Recursive convolution on strain + accel correction |
| Memory vars | 0 | 39 strain + 9 displ = 48 per GLL node (PML elements only) |
| Element kernel | Unchanged | Modified for PML elements |
| Absorption quality | Poor (reflections at boundary) | Near-perfect (matched layer) |
| Profiles | Single `d` per node | K, d, α per direction per node |

## 2. Mathematical Formulation

### 2.1 Damping Profiles (per direction, per GLL node)

For each PML direction (x, y, z), compute at each GLL node:

```
dist = |coord - pml_start| / pml_width    ∈ [0, 1]

K_axis = K_MIN + (K_MAX - K_MIN) * dist  (K_MIN = K_MAX = 1, SPECFEM3D reference)

d_axis = -(NPOWER + 1) * vp * ln(R_coef) / (2 * pml_width) * dist^(1.2 * NPOWER)
         (NPOWER = 2, R_coef = 1e-5 typical)

α_axis = α_MAX_axis * (1 - dist)           (α_MAX = π * f0 * {0.9, 1.0, 1.1})

β_axis = α_axis + d_axis / K_axis          (used for β convolution coefficients)
```

Where:

- `vp` = global maximum P-wave velocity (the SPECFEM3D convention)
- `f0` = dominant frequency of the source (Ricker peak frequency)
- `R_coef` = target reflection coefficient (typically 1e-5)
- `NPOWER` = polynomial grading exponent (typically 2)

Non-PML elements: K=1, d=0, α=0 (no effect).

### 2.2 PML Region Classification

Each PML element is classified by which faces it touches:

| Region | Code | Description |
|--------|------|-------------|
| X_ONLY | 1 | PML on one x-face |
| Y_ONLY | 2 | PML on one y-face |
| Z_ONLY | 3 | PML on one z-face |
| XY_ONLY | 4 | PML on x and y faces (edge) |
| XZ_ONLY | 5 | PML on x and z faces (edge) |
| YZ_ONLY | 6 | PML on y and z faces (edge) |
| XYZ | 7 | PML on all three faces (corner) |

For each region, only the active directions have non-zero K/d/α.
E.g., X_ONLY: only K_x, d_x, α_x are non-zero; K_y=d_y=α_y=0, etc.

### 2.3 Convolution Coefficients (precomputed, per GLL node)

#### 2.3.1 α and β convolution coefficients (9 each)

For each direction axis ∈ {x, y, z}:

```
temp = exp(-0.5 * b * dt)           where b = α_axis or β_axis

coef0 = temp * temp                  (= exp(-b * dt))
coef1 = (1 - temp) / b              (if |b| >= min_distance)
coef2 = coef1 * temp
```

For small |b| < min_distance (Taylor expansion):

```
coef1 = dt/2 + (-1/8 * dt² * b + 1/48 * dt³ * b² - 1/384 * dt⁴ * b³)
coef2 = dt/2 + (-3/8 * dt² * b + 7/48 * dt³ * b² - 5/128 * dt⁴ * b³)
```

Stored as:

- `pml_convolution_coef_alpha[9]`: 3 directions × {coef0, coef1, coef2}
- `pml_convolution_coef_beta[9]`: same for β

#### 2.3.2 Accel-update coefficients Ā₁…Ā₅ (Xie et al. 2014, Appendix A1)

Computed by `l_parameter_computation(K, d, α, region)`.

For **CPML_XYZ** (all three directions active, α_x ≠ α_y ≠ α_z):

```
β_x = α_x + d_x/K_x,  β_y = α_y + d_y/K_y,  β_z = α_z + d_z/K_z

Ā₀ = K_x * K_y * K_z
Ā₁ = Ā₀ * (β_x + β_y + β_z - α_x - α_y - α_z)
Ā₂ = Ā₀ * [(β_x-α_x)(β_y-α_y-α_x) + (β_y-α_y)(β_z-α_z-α_y) + (β_z-α_z)(β_x-α_x-α_z)]
Ā₃ = Ā₀ * α_x² * (β_x-α_x)(β_y-α_x)(β_z-α_x) / [(α_y-α_x)(α_z-α_x)]
Ā₄ = Ā₀ * α_y² * (β_x-α_y)(β_y-α_y)(β_z-α_y) / [(α_x-α_y)(α_z-α_y)]
Ā₅ = Ā₀ * α_z² * (β_x-α_z)(β_y-α_z)(β_z-α_z) / [(α_y-α_z)(α_x-α_z)]
```

For **single-direction** (e.g., CPML_X_ONLY):

```
Ā₀ = K_x
Ā₁ = Ā₀ * (β_x - α_x)
Ā₂ = -Ā₀ * α_x * (β_x - α_x)
Ā₃ = Ā₀ * α_x² * (β_x - α_x)
Ā₄ = 0
Ā₅ = 0
```

Similar (simpler) formulas for XY_ONLY, XZ_ONLY, YZ_ONLY.

#### 2.3.3 Strain-update coefficients A₆…A₂₃

Computed by `lijk_parameter_computation` (A₆-A₁₇), `lx/ly/lz_parameter_computation` (A₁₈-A₂₃).

These are used to modify the strain derivatives inside PML elements. The
formulas are similar to Ā₁…Ā₅ but with different variable permutations
(see SPECFEM3D `pml_compute_memory_variables.f90`).

Stored as `pml_convolution_coef_strain[18]`.

### 2.4 PML Displacement Fields (for second-order convolution)

```
PML_displ_old = u + (1-2θ)/2 * dt * v + (1-θ)/2 * dt² * a    (previous step)
PML_displ_new = u + (1-2θ)/2 * dt * v                          (current step)
```

Where θ = 1/8 (Wang et al. 2006).

### 2.5 Memory Variable Update

#### 2.5.1 Displacement memory (3 per node, for accel correction)

```
rmemory_displ[d] = coef0_α[d] * rmemory_displ[d]
                 + coef1_α[d] * PML_displ_new
                 + coef2_α[d] * PML_displ_old
```

Where d ∈ {x, y, z} and coef0/1/2 are the α convolution coefficients.

#### 2.5.2 Strain memory (18 per node, for stress correction)

For each displacement gradient ∂u_i/∂x_j, a memory variable is updated
using the α or β convolution coefficients for the corresponding direction.

See SPECFEM3D `pml_compute_memory_variables_elastic` for the full
mapping of which coefficients apply to which gradient.

### 2.6 Acceleration Correction

Added to the residual (force) for PML elements:

```
accel_pml = w_i * w_j * w_k * (1/ρ) * J * (
    Ā₁ * v + Ā₂ * u
    + Ā₃ * rmemory_displ[x]
    + Ā₄ * rmemory_displ[y]
    + Ā₅ * rmemory_displ[z]
)
```

### 2.7 Stress Modification (inside element kernel, PML elements only)

For PML elements, the displacement gradients are modified by the C-PML
convolution before computing stress:

```
∂u/∂x_pml = A₆ * ∂u/∂x + A₇ * rmemory_dux_dxl[x] + A₈ * rmemory_dux_dxl[y] + A₉ * rmemory_dux_dxl[z]
```

(Similar for all 9 gradient components, using A₆-A₁₇ coefficients.)

The regular stress σ = λ·tr(ε)·I + 2μ·ε is then computed from the modified
gradients, and the force is computed from the modified stress divergence.

## 3. Data Layout

### 3.1 Precomputed Data (in partition files)

All C-PML data is per-element (stored for PML elements only, indexed by
`is_pml` flag):

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `/field/cell/pml_K` | [n_pml_cell, NGLL³, 3] | float64 | K per direction |
| `/field/cell/pml_d` | [n_pml_cell, NGLL³, 3] | float64 | d per direction |
| `/field/cell/pml_alpha` | [n_pml_cell, NGLL³, 3] | float64 | α per direction |
| `/field/cell/pml_region` | [n_pml_cell] | int32 | PML region code (1-7) |
| `/field/cell/pml_coef_alpha` | [n_pml_cell, NGLL³, 9] | float64 | α conv coefs |
| `/field/cell/pml_coef_beta` | [n_pml_cell, NGLL³, 9] | float64 | β conv coefs |
| `/field/cell/pml_coef_abar` | [n_pml_cell, NGLL³, 5] | float64 | Ā₁…Ā₅ |
| `/field/cell/pml_coef_strain` | [n_pml_cell, NGLL³, 18] | float64 | A₆…A₂₃ |

Backward compatibility: if these datasets are absent, solver falls back to
the old linear-ramp damping using `/field/cell/damping`.

### 3.2 Runtime State (in memory, not persisted except restart)

| Array | Shape | Description |
|-------|-------|-------------|
| `pml_displ_old` | [n_pml_cell, NGLL³, 3] | Displacement at previous step |
| `pml_displ_new` | [n_pml_cell, NGLL³, 3] | Displacement at current step |
| `rmemory_displ` | [n_pml_cell, NGLL³, 3, 3] | Displacement memory (3 dirs × 3 comps) |
| `rmemory_strain` | [n_pml_cell, NGLL³, 39] | Strain memory: 27 lijk (β-conv) + 12 lx/ly/lz (α-conv) |

Total memory per PML GLL node: 3 + 3 + 9 + 39 = 54 doubles = 432 bytes.
For a typical mesh with ~32000 PML cells × 125 nodes = 4M PML nodes:
~1.3 GB additional memory.

### 3.3 Restart I/O

C-PML memory state must be saved/restored in restart files:

- `/restart/pml_displ_old`
- `/restart/pml_displ_new`
- `/restart/rmemory_displ`
- `/restart/rmemory_strain`

## 4. Implementation Summary

All phases complete (Jul 2026, 8+ commits). Key files:

| Layer | File | Status |
|-------|------|--------|
| Preprocess | `preprocess/pml_cpml.py` | COMPLETE — profiles, coefficients, SPECFEM3D parameter separation |
| Preprocess C++ | `preprocess/cpp/cpml.cpp` | COMPLETE — matched profiles, separation, and coefficients |
| Preprocess | `preprocess/model_writer.py` | COMPLETE — C-PML datasets in partition files |
| Forward types | `forward/share/include/gf/types.hpp` | COMPLETE — `CpmlData` struct, memory arrays |
| Forward I/O | `forward/share/src/io.cpp` | COMPLETE — read from partition, backward compat |
| PML module | `forward/share/include/gf/pml.hpp` | COMPLETE — displacement + strain memory update, accel contribution |
| PML module | `forward/share/src/pml.cpp` | COMPLETE — CPU implementations |
| PML CUDA | `forward/share/src/cuda_step.cu` | COMPLETE — CUDA implementations |
| Element CPU | `forward/elastic/src/element_cpu.cpp` | COMPLETE — `compute_pml_non_symmetric_stress()` |
| Element CUDA | `forward/elastic/src/element_cuda.cu` | COMPLETE — same for CUDA |
| Non-sym stress | `forward/share/include/gf/kernel_helpers.hpp` | COMPLETE — three-group formulation |
| Non-sym stress CUDA | `forward/share/include/gf/kernel_helpers.cuh` | COMPLETE — CUDA device version |
| Solver | `forward/share/src/solver.cpp` | COMPLETE — PML steps integrated |
| Restart | `forward/share/src/restart.cpp` | COMPLETE — memory state save/restore |
| Tests | `tests/test_pml.cpp` | COMPLETE — 7 Catch2 tests, 1443 assertions |
| Python tests | `tests/` | COMPLETE — 204 passing |

### Non-Symmetric Stress (key design decision)

SPECFEM3D uses **non-symmetric stress** in PML elements (three groups: \_x, \_y, \_z).
Our `compute_pml_non_symmetric_stress()` implements this exactly:

- `sigma[0][0]` uses \_x group, `sigma[0][1]` uses \_y group, `sigma[0][2]` uses \_z group
- Each group applies different PML corrections to all 9 gradient components
- Stored column-major, transposed on scatter (matches SPECFEM3D layout)

### lx/ly/lz Alpha-Convolved Memory

12 additional strain memory entries per node (4 lx + 4 ly + 4 lz) using alpha
coefficients for time convolution, matching SPECFEM3D
`pml_compute_memory_variables.f90:269-287`. Total strain memory: 39 entries/node
(27 lijk β-conv + 12 alpha-conv).

### SPECFEM3D Parameter Separation

1. **Parameter separation** (`_separate_pml_parameters`) matches SPECFEM3D's
   `pml_set_local_dampingcoeff.f90:1378-1833` — adjusts α/beta values to prevent
   near-zero partial-fraction denominators, recomputes d to preserve absorption.
1. **Coefficient handling** — coefficients are not silently clipped. As in
   SPECFEM3D, parameter separation is applied before the partial-fraction
   formulas; non-finite coefficients abort preprocessing explicitly.
   The physical PML boundary is evaluated at `dist=1`, so `alpha=0` there.
1. **K_MAX_PML=K_MIN_PML=1.0** — this matches the local SPECFEM3D reference
   implementation used by the project; no coordinate-stretching CFL factor is
   applied.

## 5. Key Constants

```python
THETA = 1.0 / 8.0          # Wang et al. (2006)
K_MIN_PML = 1.0             # SPECFEM3D default
K_MAX_PML = 1.0             # SPECFEM3D reference implementation
NPOWER = 2                  # Polynomial grading
R_COEF = 1e-5               # Target reflection coefficient
ALPHA_MAX_X = pi * f0 * 0.9 # Slightly different per direction
ALPHA_MAX_Y = pi * f0 * 1.0
ALPHA_MAX_Z = pi * f0 * 1.1
MIN_DISTANCE = 1e-6         # Singularity avoidance threshold
```

## 6. Verification

| Check | Result |
|-------|--------|
| All 204 Python tests pass | ✅ |
| All 6 C++ executables build clean | ✅ |
| Halfspace 1000-step stability (max|u|≈2.4e5, no inf/nan) | ✅ |
| 4 SPECFEM3D-alignment bugs fixed (see `docs/bugs.md`) | ✅ |
| Lamb reference waveform correlation: 99.1% (scaled, best-aligned) | ✅ |
| Backward compatibility (old damping fallback) | ✅ |

### Known Limitations

1. **CFL**: K_MAX_PML=1 in the SPECFEM3D reference implementation, so the
   ordinary SEM CFL limit is used.
1. **Coefficient validity**: no safety clamp is applied. If parameter separation
   fails and produces a non-finite coefficient, preprocessing stops explicitly.
1. **PML&SVD mutually exclusive**: Viscoelastic kernels skip SLS memory for PML nodes.
