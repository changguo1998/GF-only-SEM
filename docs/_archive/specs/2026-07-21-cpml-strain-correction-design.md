# C-PML Strain Correction — Implementation Spec

**Date:** 2026-07-21
**Status:** In progress
**Parent design:** [`docs/design/cpml.md`](../../design/cpml.md)
**Context:** Completing the second half of C-PML — strain-based correction using
A₆…A₂₃ coefficients inside the element kernel. Displacement-based correction
(Ā₁…Ā₅ accel contribution) is already implemented and verified.

## 1. Scope

This spec covers ONLY the strain-based C-PML correction. The displacement-based
correction (accel contribution via Ā₁…Ā₅) is complete:

- Preprocess computes all coefficients (K, d, α, α/β conv, Ā₁…Ā₅, A₆…A₂₃)
- Forward I/O reads all C-PML datasets
- `cpml_initialize`, `cpml_update_displ_fields`, `cpml_update_displ_memory`,
  `cpml_accel_contribution` are implemented and integrated in solver

## 2. Data Model

### 2.1 Nine Physical Gradient Components

The element kernel computes 3 displacement components × 3 spatial derivative
directions = 9 physical gradient values per GLL node:

```
dux_dx, dux_dy, dux_dz
duy_dx, duy_dy, duy_dz
duz_dx, duz_dy, duz_dz
```

These are stored in `du_dx[displacement_component][derivative_direction]` in the
existing kernel code (C-order: `du_dx[comp][dir]`).

### 2.2 Strain Memory Variables (`rmemory_strain`)

For each of the 9 gradient components, 3 convolution memory variables track the
convolution along x, y, z directions. Total: **27 doubles per GLL node**.

Flat storage: `rmemory_strain` per node uses `CpmlStrain::MEMORY_PER_NODE` (= 27) doubles, organized as:

```
for each node:
    for each gradient_component (0..8):
        rmemory_conv_x       // convolution along x-direction
        rmemory_conv_y       // convolution along y-direction
        rmemory_conv_z       // convolution along z-direction
```

Gradient component order:

| Index | Name | Meaning |
|-------|------|---------|
| 0 | `dux_dx` | ∂u_x / ∂x |
| 1 | `dux_dy` | ∂u_x / ∂y |
| 2 | `dux_dz` | ∂u_x / ∂z |
| 3 | `duy_dx` | ∂u_y / ∂x |
| 4 | `duy_dy` | ∂u_y / ∂y |
| 5 | `duy_dz` | ∂u_y / ∂z |
| 6 | `duz_dx` | ∂u_z / ∂x |
| 7 | `duz_dy` | ∂u_z / ∂y |
| 8 | `duz_dz` | ∂u_z / ∂z |

Access pattern (C++ helper):

```cpp
inline size_t strain_memory_offset(size_t node, int gradient, int conv_dir) {
    return node * CpmlStrain::MEMORY_PER_NODE
           + gradient * CpmlStrain::MEMORY_PER_GRADIENT
           + conv_dir;
}
// Usage:
//   rmemory_strain[strain_memory_offset(node, DUY_DX, CONV_Z)]
//   → memory for ∂u_y/∂x convolved in z-direction
```

### 2.3 Strain Correction Coefficients (`pml_coef_strain`)

18 doubles per GLL node, grouped into 6 structs by purpose.

Flat storage: `pml_coef_strain` per node uses `CpmlStrain::COEFS_PER_NODE` (= 18) doubles.

**Off-diagonal gradients** use `_lijk_parameter` (4 values each, 3 groups):

```cpp
struct OffDiagonalCorrection {
    double gradient_prefactor;       // multiplies the original physical gradient
    double memory_coef_conv_dir0;    // multiplies mem for this gradient in dir_0
    double memory_coef_conv_dir1;    // multiplies mem for this gradient in dir_1
    double memory_coef_conv_dir2;    // multiplies mem for this gradient in dir_2
};
```

| Group | lijk perm | Applies to | dir_0 | dir_1 | dir_2 |
|-------|-----------|------------|-------|-------|-------|
| `grad_wrt_x` | (z, y, x) | `duy_dx`, `duz_dx` | z | y | x |
| `grad_wrt_y` | (x, z, y) | `dux_dy`, `duz_dy` | x | z | y |
| `grad_wrt_z` | (x, y, z) | `dux_dz`, `duy_dz` | x | y | z |

**Diagonal gradients** use `_lx/_ly/_lz_parameter` (2 values each, 3 groups):

```cpp
struct DiagonalCorrection {
    double gradient_prefactor;       // multiplies the original physical gradient
    double memory_coef_local_dir;    // multiplies mem for this gradient in its own direction
};
```

| Group | Param | Applies to | local_dir |
|-------|-------|------------|-----------|
| `dux_dx_correction` | lx | `dux_dx` | x |
| `duy_dy_correction` | ly | `duy_dy` | y |
| `duz_dz_correction` | lz | `duz_dz` | z |

**Complete coefficient struct:**

```cpp
struct StrainCoefficients {
    OffDiagonalCorrection grad_wrt_x;   // index 0..3  (lijk z,y,x)
    OffDiagonalCorrection grad_wrt_y;   // index 4..7  (lijk x,z,y)
    OffDiagonalCorrection grad_wrt_z;   // index 8..11 (lijk x,y,z)
    DiagonalCorrection     dux_dx;      // index 12..13 (lx)
    DiagonalCorrection     duy_dy;      // index 14..15 (ly)
    DiagonalCorrection     duz_dz;      // index 16..17 (lz)
};
```

**Flat storage layout** (18 doubles per node, written by Python `compute_strain_coefficients`):

| Offset | SPECFEM3D name | Python source | C++ struct field |
|--------|---------------|---------------|------------------|
| 0 | A₆ | `_lijk_parameter(z,y,x)` → A₀ | `grad_wrt_x.gradient_prefactor` |
| 1 | A₇ | `_lijk_parameter(z,y,x)` → A₁ | `grad_wrt_x.memory_coef_conv_dir0` (z) |
| 2 | A₈ | `_lijk_parameter(z,y,x)` → A₂ | `grad_wrt_x.memory_coef_conv_dir1` (y) |
| 3 | A₉ | `_lijk_parameter(z,y,x)` → A₃ | `grad_wrt_x.memory_coef_conv_dir2` (x) |
| 4 | A₁₀ | `_lijk_parameter(x,z,y)` → A₀ | `grad_wrt_y.gradient_prefactor` |
| 5 | A₁₁ | `_lijk_parameter(x,z,y)` → A₁ | `grad_wrt_y.memory_coef_conv_dir0` (x) |
| 6 | A₁₂ | `_lijk_parameter(x,z,y)` → A₂ | `grad_wrt_y.memory_coef_conv_dir1` (z) |
| 7 | A₁₃ | `_lijk_parameter(x,z,y)` → A₃ | `grad_wrt_y.memory_coef_conv_dir2` (y) |
| 8 | A₁₄ | `_lijk_parameter(x,y,z)` → A₀ | `grad_wrt_z.gradient_prefactor` |
| 9 | A₁₅ | `_lijk_parameter(x,y,z)` → A₁ | `grad_wrt_z.memory_coef_conv_dir0` (x) |
| 10 | A₁₆ | `_lijk_parameter(x,y,z)` → A₂ | `grad_wrt_z.memory_coef_conv_dir1` (y) |
| 11 | A₁₇ | `_lijk_parameter(x,y,z)` → A₃ | `grad_wrt_z.memory_coef_conv_dir2` (z) |
| 12 | A₁₈ | `_lx_parameter(x)` → A₀ | `dux_dx.gradient_prefactor` |
| 13 | A₁₉ | `_lx_parameter(x)` → A₁ | `dux_dx.memory_coef_local_dir` (x) |
| 14 | A₂₀ | `_ly_parameter(y)` → A₀ | `duy_dy.gradient_prefactor` |
| 15 | A₂₁ | `_ly_parameter(y)` → A₁ | `duy_dy.memory_coef_local_dir` (y) |
| 16 | A₂₂ | `_lz_parameter(z)` → A₀ | `duz_dz.gradient_prefactor` |
| 17 | A₂₃ | `_lz_parameter(z)` → A₁ | `duz_dz.memory_coef_local_dir` (z) |

**Named constants** (defined in `pml.hpp` under `namespace CpmlStrain`).
Values derived from `NDIM` unless noted otherwise:

```cpp
namespace CpmlStrain {
    // --- Spatial dimension (root constant) ---
    constexpr int NDIM = 3;  // 3D — all count/stride constants below derive from this

    // --- Counts (derived from NDIM) ---
    constexpr int NUM_DISPLACEMENT_COMPS = NDIM;          // ux, uy, uz
    constexpr int NUM_DERIVATIVE_DIRS    = NDIM;          // d/dx, d/dy, d/dz
    constexpr int NUM_CONV_DIRECTIONS    = NDIM;          // convolution along x, y, z
    constexpr int NUM_GRADIENT_COMPS     = NDIM * NDIM;   // 9
    constexpr int NUM_OFF_DIAG_GROUPS    = NDIM;          // lijk: grad_wrt_x/y/z
    constexpr int NUM_DIAG_GROUPS        = NDIM;          // lx, ly, lz

    // --- Coefficient strides ---
    // OffDiagonalCorrection: 1 prefactor + NDIM memory directions = 4
    constexpr int LINGK_COEFS_PER_GROUP  = 1 + NDIM;
    // DiagonalCorrection: 1 prefactor + 1 local direction (independent of NDIM) = 2
    constexpr int DIAG_COEFS_PER_GROUP   = 2;  // NOT derived from NDIM
    constexpr int COEFS_PER_NODE         = NUM_OFF_DIAG_GROUPS * LINGK_COEFS_PER_GROUP
                                         + NUM_DIAG_GROUPS * DIAG_COEFS_PER_GROUP;

    // --- Strain memory strides (derived from NDIM) ---
    constexpr int MEMORY_PER_GRADIENT    = NDIM;
    constexpr int MEMORY_PER_NODE        = NUM_GRADIENT_COMPS * MEMORY_PER_GRADIENT;

    // --- β convolution coefficient strides (pml_coef_beta array) ---
    // 3 values per direction: coef0, coef1, coef2 (fixed by 2nd-order RK scheme, NOT from NDIM)
    constexpr int BETA_COEFS_PER_DIR     = 3;  // NOT derived from NDIM
    constexpr int BETA_COEFS_PER_NODE    = NDIM * BETA_COEFS_PER_DIR;
    constexpr int BETA_COEF0             = 0;
    constexpr int BETA_COEF1             = 1;
    constexpr int BETA_COEF2             = 2;

    // --- Displacement memory strides (rmemory_displ array, derived from NDIM) ---
    constexpr int DISPL_MEM_PER_NODE     = NDIM * NDIM;

    // --- Displacement field stride (pml_displ_old/new arrays, derived from NDIM) ---
    constexpr int DISPL_FIELD_PER_NODE   = NDIM;

    // --- Offsets into pml_coef_strain (derived from group sizes) ---
    constexpr int OFFSET_GRAD_WRT_X = 0;
    constexpr int OFFSET_GRAD_WRT_Y = OFFSET_GRAD_WRT_X + LINGK_COEFS_PER_GROUP;
    constexpr int OFFSET_GRAD_WRT_Z = OFFSET_GRAD_WRT_Y + LINGK_COEFS_PER_GROUP;
    constexpr int OFFSET_DUX_DX      = OFFSET_GRAD_WRT_Z + LINGK_COEFS_PER_GROUP;
    constexpr int OFFSET_DUY_DY      = OFFSET_DUX_DX      + DIAG_COEFS_PER_GROUP;
    constexpr int OFFSET_DUZ_DZ      = OFFSET_DUY_DY      + DIAG_COEFS_PER_GROUP;

    // --- Enums (values derived from NDIM-ordering, but expressed explicitly for clarity) ---
    enum Gradient : int {
        DUX_DX = 0, DUX_DY = 1, DUX_DZ = 2,
        DUY_DX = 3, DUY_DY = 4, DUY_DZ = 5,
        DUZ_DX = 6, DUZ_DY = 7, DUZ_DZ = 8
    };
    enum Component : int { DUX = 0, DUY = 1, DUZ = 2 };
    enum Direction : int { DX = 0, DY = 1, DZ = 2 };
    enum ConvDir : int { CONV_X = 0, CONV_Y = 1, CONV_Z = 2 };

    // --- Map (component, direction) to gradient index ---
    constexpr int gradient_of(int comp, int dir) { return comp * NUM_DERIVATIVE_DIRS + dir; }
}
```

Only 3 constants are NOT derived from `NDIM`:

- `DIAG_COEFS_PER_GROUP = 2` — the lx/ly/lz parameters each consist of 1 prefactor + 1 memory coefficient
- `BETA_COEFS_PER_DIR = 3` — the 2nd-order recursive convolution uses exactly 3 coefficients (coef0, coef1, coef2)
- `BETA_COEF0/1/2` — indices into the 3-tuple

Access helper:

```cpp
StrainCoefficients load_strain_coefficients(const double* flat, size_t node) {
    const double* base = flat + node * CpmlStrain::COEFS_PER_NODE;
    StrainCoefficients c;
    c.grad_wrt_x = {base[OFFSET_GRAD_WRT_X + 0], base[OFFSET_GRAD_WRT_X + 1],
                    base[OFFSET_GRAD_WRT_X + 2], base[OFFSET_GRAD_WRT_X + 3]};
    c.grad_wrt_y = {base[OFFSET_GRAD_WRT_Y + 0], base[OFFSET_GRAD_WRT_Y + 1],
                    base[OFFSET_GRAD_WRT_Y + 2], base[OFFSET_GRAD_WRT_Y + 3]};
    c.grad_wrt_z = {base[OFFSET_GRAD_WRT_Z + 0], base[OFFSET_GRAD_WRT_Z + 1],
                    base[OFFSET_GRAD_WRT_Z + 2], base[OFFSET_GRAD_WRT_Z + 3]};
    c.dux_dx     = {base[OFFSET_DUX_DX + 0], base[OFFSET_DUX_DX + 1]};
    c.duy_dy     = {base[OFFSET_DUY_DY + 0], base[OFFSET_DUY_DY + 1]};
    c.duz_dz     = {base[OFFSET_DUZ_DZ + 0], base[OFFSET_DUZ_DZ + 1]};
    return c;
}
```

## 3. Remaining Work

### 3.1 Strain Memory Variable Update (`pml.hpp/cpp`)

```cpp
void cpml_update_strain_memory(RankData& part, int n_node);
```

For each PML element and GLL node:

1. Compute the 9 reference-space gradients of `pml_displ_new` and `pml_displ_old`
   using the GLL derivative matrix D (same pattern as the element kernel).
1. Transform reference-space gradients to physical gradients using inverse
   Jacobian `dxi_dx`.
1. Update 27 strain memory variables using β convolution coefficients:

```cpp
// For each of the 9 gradient components...
for (int gradient = 0; gradient < CpmlStrain::NUM_GRADIENT_COMPS; ++gradient) {
    double new_grad_value = pml_displ_new_physical_gradient[gradient];
    double old_grad_value = pml_displ_old_physical_gradient[gradient];

    // ...update 3 convolution directions
    for (int conv_dir = 0; conv_dir < CpmlStrain::NUM_CONV_DIRECTIONS; ++conv_dir) {
        size_t offset = strain_memory_offset(node, gradient, conv_dir);
        int coef_base = conv_dir * CpmlStrain::BETA_COEFS_PER_DIR;  // offset into β coefficients

        part.rmemory_strain[offset] =
            part.pml_coef_beta[node * CpmlStrain::BETA_COEFS_PER_NODE
                                + coef_base + CpmlStrain::BETA_COEF0] * part.rmemory_strain[offset] +
            part.pml_coef_beta[node * CpmlStrain::BETA_COEFS_PER_NODE
                                + coef_base + CpmlStrain::BETA_COEF1] * new_grad_value +
            part.pml_coef_beta[node * CpmlStrain::BETA_COEFS_PER_NODE
                                + coef_base + CpmlStrain::BETA_COEF2] * old_grad_value;
    }
}
```

### 3.2 Element Kernel Signature (`element.hpp`)

```cpp
template <typename Backend>
void compute_element_residual(
    int n_elem,
    const double* dxi_dx, const double* jacobian,
    const double* lambda_, const double* mu_,
    const double* D, const double* weights, int NGLL,
    const double* displacement, double* residual,
    // C-PML strain correction (nullptr → skip PML branch)
    const int32_t* pml_region = nullptr,
    const double* pml_coef_strain = nullptr,
    const double* rmemory_strain = nullptr);
```

**Decision:** Modify existing signature. One kernel handles interior + PML
elements.

### 3.3 CPU Element Kernel (`element_cpu.cpp`)

After computing the 9 physical gradients `du_dx[comp][deriv]`, insert the
C-PML strain correction for PML elements:

```cpp
if (pml_region && pml_region[elem] != 0) {
    StrainCoefficients coef = load_strain_coefficients(pml_coef_strain,
                                                        elem * n_node + node);

    // --- Off-diagonal gradients (use OffDiagonalCorrection) ---
    // duy/dx and duz/dx — corrected by grad_wrt_x (lijk z,y,x)
    for (int comp : {DUY, DUZ}) {
        double grad = du_dx[comp][DX];
        double mem_z = rmemory_strain[strain_memory_offset(node, gradient_of(comp, DX), CONV_Z)];
        double mem_y = rmemory_strain[strain_memory_offset(node, gradient_of(comp, DX), CONV_Y)];
        double mem_x = rmemory_strain[strain_memory_offset(node, gradient_of(comp, DX), CONV_X)];
        du_dx[comp][DX] = coef.grad_wrt_x.gradient_prefactor * grad
                        + coef.grad_wrt_x.memory_coef_conv_dir0 * mem_z   // dir_0 = z
                        + coef.grad_wrt_x.memory_coef_conv_dir1 * mem_y   // dir_1 = y
                        + coef.grad_wrt_x.memory_coef_conv_dir2 * mem_x;  // dir_2 = x
    }

    // dux/dy and duz/dy — corrected by grad_wrt_y (lijk x,z,y)
    for (int comp : {DUX, DUZ}) {
        double grad = du_dx[comp][DY];
        double mem_x = rmemory_strain[strain_memory_offset(node, gradient_of(comp, DY), CONV_X)];
        double mem_z = rmemory_strain[strain_memory_offset(node, gradient_of(comp, DY), CONV_Z)];
        double mem_y = rmemory_strain[strain_memory_offset(node, gradient_of(comp, DY), CONV_Y)];
        du_dx[comp][DY] = coef.grad_wrt_y.gradient_prefactor * grad
                        + coef.grad_wrt_y.memory_coef_conv_dir0 * mem_x   // dir_0 = x
                        + coef.grad_wrt_y.memory_coef_conv_dir1 * mem_z   // dir_1 = z
                        + coef.grad_wrt_y.memory_coef_conv_dir2 * mem_y;  // dir_2 = y
    }

    // dux/dz and duy/dz — corrected by grad_wrt_z (lijk x,y,z)
    for (int comp : {DUX, DUY}) {
        double grad = du_dx[comp][DZ];
        double mem_x = rmemory_strain[strain_memory_offset(node, gradient_of(comp, DZ), CONV_X)];
        double mem_y = rmemory_strain[strain_memory_offset(node, gradient_of(comp, DZ), CONV_Y)];
        double mem_z = rmemory_strain[strain_memory_offset(node, gradient_of(comp, DZ), CONV_Z)];
        du_dx[comp][DZ] = coef.grad_wrt_z.gradient_prefactor * grad
                        + coef.grad_wrt_z.memory_coef_conv_dir0 * mem_x   // dir_0 = x
                        + coef.grad_wrt_z.memory_coef_conv_dir1 * mem_y   // dir_1 = y
                        + coef.grad_wrt_z.memory_coef_conv_dir2 * mem_z;  // dir_2 = z
    }

    // --- Diagonal gradients (use DiagonalCorrection) ---
    // dux/dx — corrected by dux_dx (lx)
    {
        double grad = du_dx[DUX][DX];
        double mem_x = rmemory_strain[strain_memory_offset(node, DUX_DX, CONV_X)];
        du_dx[DUX][DX] = coef.dux_dx.gradient_prefactor * grad
                       + coef.dux_dx.memory_coef_local_dir * mem_x;
    }
    // duy/dy — corrected by duy_dy (ly)
    {
        double grad = du_dx[DUY][DY];
        double mem_y = rmemory_strain[strain_memory_offset(node, DUY_DY, CONV_Y)];
        du_dx[DUY][DY] = coef.duy_dy.gradient_prefactor * grad
                       + coef.duy_dy.memory_coef_local_dir * mem_y;
    }
    // duz/dz — corrected by duz_dz (lz)
    {
        double grad = du_dx[DUZ][DZ];
        double mem_z = rmemory_strain[strain_memory_offset(node, DUZ_DZ, CONV_Z)];
        du_dx[DUZ][DZ] = coef.duz_dz.gradient_prefactor * grad
                       + coef.duz_dz.memory_coef_local_dir * mem_z;
    }
}
```

The rest of the kernel (strain → stress → residual scatter) is **unchanged**.

### 3.4 CUDA Element Kernel (`element_cuda.cu`)

Same logic as CPU, ported to CUDA. Each thread already handles one GLL node.
Add the same if-branch with the named struct-based correction.

### 3.5 CUDA C-PML Runtime (`cuda_step.cu/hpp`)

Add GPU-side C-PML functions:

- `cuda_cpml_update_displ_fields()` — PML_displ field update kernel
- `cuda_cpml_update_displ_memory()` — displacement memory update kernel
- `cuda_cpml_update_strain_memory()` — strain memory update kernel
- `cuda_cpml_accel_contribution()` — accel correction kernel

Add to `CudaDeviceState`:

- `d_pml_region`, `d_pml_coef_alpha`, `d_pml_coef_beta`, `d_pml_coef_abar`,
  `d_pml_coef_strain`
- `d_pml_displ_old`, `d_pml_displ_new`
- `d_rmemory_displ`, `d_rmemory_strain`

Replace current `cuda_pml_damping()` (legacy linear ramp) with full C-PML.

### 3.6 Restart I/O (`record.cpp`)

Save/restore 4 C-PML memory arrays:

| Dataset | Shape | Constant | Description |
|---------|-------|----------|-------------|
| `/restart/pml_displ_old` | \[n_pml_node × `DISPL_FIELD_PER_NODE`\] | = 3 | Previous step displacement field (ux, uy, uz) |
| `/restart/pml_displ_new` | \[n_pml_node × `DISPL_FIELD_PER_NODE`\] | = 3 | Current step displacement field (ux, uy, uz) |
| `/restart/rmemory_displ` | \[n_pml_node × `DISPL_MEM_PER_NODE`\] | = 9 | Displacement memory (3 comps × 3 conv dirs) |
| `/restart/rmemory_strain` | \[n_pml_node × `MEMORY_PER_NODE`\] | = 27 | Strain memory (9 grads × 3 conv dirs) |

Only written/read when `part.has_cpml` is true. Reader checks dataset
existence for backward compatibility with pre-C-PML restart files.

### 3.7 Solver Integration (`solver.cpp`)

Add strain memory update after displacement memory update:

```cpp
if (part.has_cpml) {
    cpml_update_displ_memory(part, n_node);
    cpml_update_strain_memory(part, n_node);  // NEW
}
```

Pass C-PML parameters to the element kernel:

```cpp
compute_element_residual<Backend>(
    n_local_cell, dxi_dx_ptr, jacobian_ptr, lambda_ptr, mu_ptr,
    D_mat.data(), gll_weights.data(), ngll,
    displacement_ptr, element_residual_ptr,
    part.pml_region.empty()       ? nullptr : part.pml_region.data(),
    part.pml_coef_strain.empty()  ? nullptr : part.pml_coef_strain.data(),
    part.rmemory_strain.empty()   ? nullptr : part.rmemory_strain.data());
```

### 3.8 CUDA Solver Integration

Replace `cuda_pml_damping(gpu_state)` with full C-PML pipeline in the CUDA
timestep. Allocate and manage C-PML GPU buffers in `CudaDeviceState`.

### 3.9 Validation

- **Unit test:** `cpml_update_strain_memory` produces non-zero memory for PML
  elements, zero for interior
- **Integration:** Run halfspace example with C-PML enabled
  - Absorption quality improved vs old linear ramp
  - Interior solution unchanged
  - No energy blow-up for long simulation
- **Regression:** All 221 existing tests pass

## 4. Implementation Order (Bottom-Up)

| Step | Task | Files |
|------|------|-------|
| 1 | Named structs + accessors + `cpml_update_strain_memory()` | `pml.hpp`, `pml.cpp` |
| 2 | CPU element kernel C-PML branch | `element.hpp`, `element_cpu.cpp` |
| 3 | Solver integration + strain memory call | `solver.cpp` |
| 4 | Restart I/O | `record.cpp` |
| 5 | CUDA element kernel C-PML branch | `element_cuda.cu` |
| 6 | CUDA C-PML runtime (all functions) | `cuda_step.cu`, `cuda_step.hpp` |
| 7 | Validation: absorption quality + regression | `tests/` |

## 5. Risk Mitigation

1. **Backward compatible:** `nullptr` defaults on new kernel params → old code
   skips PML branch
1. **Incremental testing:** Each step testable independently
1. **CPU first, then CUDA:** Validate on CPU before GPU port
1. **Existing tests must not break:** Full test suite after every step
