# C-PML Strain Correction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete C-PML strain-based correction: add strain memory update, modify element
kernel to apply A₆…A₂₃ correction to physical gradients for PML elements, wire into solver
timestep, add restart I/O, and port everything to CUDA.

**Architecture:** Bottom-up — add the `CpmlStrain` namespace with named constants and
structs first, then the strain memory update function, then modify the element kernel
signature and CPU implementation, integrate into solver, add restart I/O, then port to
CUDA element kernel and CUDA runtime. Each task builds on the previous and is
independently testable.

**Tech Stack:** C++17, Catch2, CUDA, HDF5

## Global Constraints

- Backward compatible: `nullptr` defaults on new kernel params — old code skips PML branch
- CPU first, then CUDA — validate on CPU before GPU port
- All 221 existing tests must continue to pass after each task
- Full project spec: [`docs/superpowers/specs/2026-07-21-cpml-strain-correction-design.md`](../../superpowers/specs/2026-07-21-cpml-strain-correction-design.md)
- Parent design: [`docs/design/cpml.md`](../../design/cpml.md)

______________________________________________________________________

### Task 1: CpmlStrain Namespace + Strain Memory Update

**Files:**

- Modify: `forward/share/include/gf/pml.hpp`
- Modify: `forward/share/src/pml.cpp`

**Interfaces:**

- Consumes: `RankData` (from `types.hpp`), GLL derivative matrix D (passed as param)

- Produces: `namespace CpmlStrain` (constants, structs, enums), `cpml_update_strain_memory()`

- [ ] **Step 1: Add `CpmlStrain` namespace to `pml.hpp`**

Add after the existing `#include` block, before the function declarations:

```cpp
// ============================================================================
// C-PML Strain Correction — named constants, structs, and enums
// All dimension-dependent values derived from NDIM = 3.
// See docs/superpowers/specs/2026-07-21-cpml-strain-correction-design.md §2.
// ============================================================================
namespace CpmlStrain {

    // --- Spatial dimension (root constant) ---
    constexpr int NDIM = 3;

    // --- Counts (derived from NDIM) ---
    constexpr int NUM_DISPLACEMENT_COMPS = NDIM;
    constexpr int NUM_DERIVATIVE_DIRS    = NDIM;
    constexpr int NUM_CONV_DIRECTIONS    = NDIM;
    constexpr int NUM_GRADIENT_COMPS     = NDIM * NDIM;
    constexpr int NUM_OFF_DIAG_GROUPS    = NDIM;
    constexpr int NUM_DIAG_GROUPS        = NDIM;

    // --- Coefficient strides ---
    constexpr int LINGK_COEFS_PER_GROUP  = 1 + NDIM;  // prefactor + NDIM memory dirs
    constexpr int DIAG_COEFS_PER_GROUP   = 2;          // NOT derived from NDIM
    constexpr int COEFS_PER_NODE         = NUM_OFF_DIAG_GROUPS * LINGK_COEFS_PER_GROUP
                                         + NUM_DIAG_GROUPS * DIAG_COEFS_PER_GROUP;

    // --- Strain memory strides ---
    constexpr int MEMORY_PER_GRADIENT    = NDIM;
    constexpr int MEMORY_PER_NODE        = NUM_GRADIENT_COMPS * MEMORY_PER_GRADIENT;

    // --- β convolution coefficient strides ---
    constexpr int BETA_COEFS_PER_DIR     = 3;  // NOT derived from NDIM
    constexpr int BETA_COEFS_PER_NODE    = NDIM * BETA_COEFS_PER_DIR;
    constexpr int BETA_COEF0             = 0;
    constexpr int BETA_COEF1             = 1;
    constexpr int BETA_COEF2             = 2;

    // --- Displacement memory strides ---
    constexpr int DISPL_MEM_PER_NODE     = NDIM * NDIM;

    // --- Displacement field stride ---
    constexpr int DISPL_FIELD_PER_NODE   = NDIM;

    // --- Offsets into pml_coef_strain (derived from group sizes) ---
    constexpr int OFFSET_GRAD_WRT_X = 0;
    constexpr int OFFSET_GRAD_WRT_Y = OFFSET_GRAD_WRT_X + LINGK_COEFS_PER_GROUP;
    constexpr int OFFSET_GRAD_WRT_Z = OFFSET_GRAD_WRT_Y + LINGK_COEFS_PER_GROUP;
    constexpr int OFFSET_DUX_DX      = OFFSET_GRAD_WRT_Z + LINGK_COEFS_PER_GROUP;
    constexpr int OFFSET_DUY_DY      = OFFSET_DUX_DX      + DIAG_COEFS_PER_GROUP;
    constexpr int OFFSET_DUZ_DZ      = OFFSET_DUY_DY      + DIAG_COEFS_PER_GROUP;

    // --- Structs ---
    struct OffDiagonalCorrection {
        double gradient_prefactor;
        double memory_coef_conv_dir0;
        double memory_coef_conv_dir1;
        double memory_coef_conv_dir2;
    };

    struct DiagonalCorrection {
        double gradient_prefactor;
        double memory_coef_local_dir;
    };

    struct StrainCoefficients {
        OffDiagonalCorrection grad_wrt_x;
        OffDiagonalCorrection grad_wrt_y;
        OffDiagonalCorrection grad_wrt_z;
        DiagonalCorrection     dux_dx;
        DiagonalCorrection     duy_dy;
        DiagonalCorrection     duz_dz;
    };

    // --- Enums ---
    enum Gradient : int {
        DUX_DX = 0, DUX_DY = 1, DUX_DZ = 2,
        DUY_DX = 3, DUY_DY = 4, DUY_DZ = 5,
        DUZ_DX = 6, DUZ_DY = 7, DUZ_DZ = 8
    };
    enum Component : int { DUX = 0, DUY = 1, DUZ = 2 };
    enum Direction : int { DX = 0, DY = 1, DZ = 2 };
    enum ConvDir : int { CONV_X = 0, CONV_Y = 1, CONV_Z = 2 };

    // --- Helpers ---
    constexpr int gradient_of(int comp, int dir) { return comp * NUM_DERIVATIVE_DIRS + dir; }

    inline size_t strain_memory_offset(size_t node, int gradient, int conv_dir) {
        return node * MEMORY_PER_NODE + gradient * MEMORY_PER_GRADIENT + conv_dir;
    }

    inline StrainCoefficients load_strain_coefficients(const double* flat, size_t node) {
        const double* base = flat + node * COEFS_PER_NODE;
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

}  // namespace CpmlStrain
```

- [ ] **Step 2: Declare `cpml_update_strain_memory` in `pml.hpp`**

Add after the existing `cpml_update_displ_memory` declaration:

```cpp
/// Update C-PML strain memory variables via β convolution.
///
/// For each PML element and GLL node:
///   1. Compute 9 reference-space gradients of pml_displ_new and pml_displ_old
///      using the GLL derivative matrix D.
///   2. Transform to physical gradients using the inverse Jacobian dxi_dx.
///   3. Update rmemory_strain using β convolution coefficients.
///
/// @param[in,out] part     RankData with memory variables and coefficients
/// @param[in]     D        GLL derivative matrix [NGLL * NGLL]
/// @param[in]     weights  GLL quadrature weights [NGLL]
/// @param[in]     NGLL     Number of GLL points per axis
void cpml_update_strain_memory(RankData& part,
                               const double* D,
                               const double* weights,
                               int NGLL);
```

- [ ] **Step 3: Implement `cpml_update_strain_memory` in `pml.cpp`**

```cpp
void cpml_update_strain_memory(RankData& part,
                               const double* D,
                               const double* weights,
                               int NGLL) {
    if (!part.has_cpml) return;

    const int n_local_cell = part.n_local_cell;
    const int n_node = NGLL * NGLL * NGLL;

    for (int e = 0; e < n_local_cell; ++e) {
        int region = (e < static_cast<int>(part.pml_region.size()))
                         ? part.pml_region[e] : 0;
        if (region == 0) continue;

        const int elem_off = e * n_node;

        for (int n = 0; n < n_node; ++n) {
            const int i = n / (NGLL * NGLL);
            const int j = (n / NGLL) % NGLL;
            const int k = n % NGLL;

            // --- 1. Compute reference-space gradients of PML displ fields ---
            // PML_displ_new gradients
            double new_dudxi[3] = {0, 0, 0};
            double new_dudeta[3] = {0, 0, 0};
            double new_dudzeta[3] = {0, 0, 0};
            double old_dudxi[3] = {0, 0, 0};
            double old_dudeta[3] = {0, 0, 0};
            double old_dudzeta[3] = {0, 0, 0};

            for (int s = 0; s < NGLL; ++s) {
                const double Di_s = D[i * NGLL + s];
                const double Dj_s = D[j * NGLL + s];
                const double Dk_s = D[k * NGLL + s];

                const int n_sjk = (s * NGLL + j) * NGLL + k;
                const int n_isk = (i * NGLL + s) * NGLL + k;
                const int n_ijs = (i * NGLL + j) * NGLL + s;

                const int new_sjk = (elem_off + n_sjk) * 3;
                const int new_isk = (elem_off + n_isk) * 3;
                const int new_ijs = (elem_off + n_ijs) * 3;
                const int old_sjk = (elem_off + n_sjk) * 3;
                const int old_isk = (elem_off + n_isk) * 3;
                const int old_ijs = (elem_off + n_ijs) * 3;

                for (int dir = 0; dir < 3; ++dir) {
                    new_dudxi[dir]   += Di_s * part.pml_displ_new[new_sjk + dir];
                    new_dudeta[dir]  += Dj_s * part.pml_displ_new[new_isk + dir];
                    new_dudzeta[dir] += Dk_s * part.pml_displ_new[new_ijs + dir];
                    old_dudxi[dir]   += Di_s * part.pml_displ_old[old_sjk + dir];
                    old_dudeta[dir]  += Dj_s * part.pml_displ_old[old_isk + dir];
                    old_dudzeta[dir] += Dk_s * part.pml_displ_old[old_ijs + dir];
                }
            }

            // --- 2. Transform to physical gradients ---
            const double* dd = &part.dxi_dx[(elem_off + n) * 9];
            // dd[0]=dξ/dx, dd[1]=dη/dx, dd[2]=dζ/dx
            // dd[3]=dξ/dy, dd[4]=dη/dy, dd[5]=dζ/dy
            // dd[6]=dξ/dz, dd[7]=dη/dz, dd[8]=dζ/dz

            double new_phys_grad[9];
            double old_phys_grad[9];
            for (int comp = 0; comp < 3; ++comp) {
                new_phys_grad[comp * 3 + 0] = new_dudxi[comp] * dd[0] + new_dudeta[comp] * dd[1] + new_dudzeta[comp] * dd[2];
                new_phys_grad[comp * 3 + 1] = new_dudxi[comp] * dd[3] + new_dudeta[comp] * dd[4] + new_dudzeta[comp] * dd[5];
                new_phys_grad[comp * 3 + 2] = new_dudxi[comp] * dd[6] + new_dudeta[comp] * dd[7] + new_dudzeta[comp] * dd[8];
                old_phys_grad[comp * 3 + 0] = old_dudxi[comp] * dd[0] + old_dudeta[comp] * dd[1] + old_dudzeta[comp] * dd[2];
                old_phys_grad[comp * 3 + 1] = old_dudxi[comp] * dd[3] + old_dudeta[comp] * dd[4] + old_dudzeta[comp] * dd[5];
                old_phys_grad[comp * 3 + 2] = old_dudxi[comp] * dd[6] + old_dudeta[comp] * dd[7] + old_dudzeta[comp] * dd[8];
            }

            // --- 3. Update strain memory with β convolution ---
            using namespace CpmlStrain;
            for (int grad = 0; grad < NUM_GRADIENT_COMPS; ++grad) {
                double new_grad = new_phys_grad[grad];
                double old_grad = old_phys_grad[grad];
                for (int conv_dir = 0; conv_dir < NUM_CONV_DIRECTIONS; ++conv_dir) {
                    size_t mem_off = strain_memory_offset(elem_off + n, grad, conv_dir);
                    int beta_off = (elem_off + n) * BETA_COEFS_PER_NODE + conv_dir * BETA_COEFS_PER_DIR;
                    part.rmemory_strain[mem_off] =
                        part.pml_coef_beta[beta_off + BETA_COEF0] * part.rmemory_strain[mem_off] +
                        part.pml_coef_beta[beta_off + BETA_COEF1] * new_grad +
                        part.pml_coef_beta[beta_off + BETA_COEF2] * old_grad;
                }
            }
        }
    }
}
```

- [ ] **Step 4: Build and verify compilation**

```bash
cd forward && cmake -B build -DGF_DEVICE_BACKEND=CPU && cmake --build build
```

Expected: clean build, no errors.

- [ ] **Step 5: Run existing PML tests**

```bash
cd forward/build && ctest -R pml
```

Expected: existing PML tests pass.

- [ ] **Step 6: Commit**

```bash
git add forward/share/include/gf/pml.hpp forward/share/src/pml.cpp
git commit -m "feat: add CpmlStrain namespace and cpml_update_strain_memory"
```

______________________________________________________________________

### Task 2: Element Kernel Signature + CPU C-PML Branch

**Files:**

- Modify: `forward/share/include/gf/element.hpp`
- Modify: `forward/elastic/src/element_cpu.cpp`

**Interfaces:**

- Consumes: `CpmlStrain::StrainCoefficients`, `load_strain_coefficients`, `strain_memory_offset`, enums

- Produces: Updated `compute_element_residual` signature with C-PML params

- [ ] **Step 1: Update template declaration in `element.hpp`**

Replace the existing `compute_element_residual` declaration with:

```cpp
template <typename Backend>
void compute_element_residual(int n_elem, const double* dxi_dx, const double* jacobian,
                              const double* lambda_, const double* mu_, const double* D,
                              const double* weights, int NGLL, const double* u, double* r,
                              const int32_t* pml_region = nullptr,
                              const double* pml_coef_strain = nullptr,
                              const double* rmemory_strain = nullptr);
```

Update the extern template declarations (two blocks — `#ifndef GF_ELEMENT_CPU_SOURCE` and `#ifdef GF_WITH_CUDA`):

```cpp
#ifndef GF_ELEMENT_CPU_SOURCE
extern template void compute_element_residual<BackendCPU>(
    int, const double*, const double*, const double*, const double*,
    const double*, const double*, int, const double*, double*,
    const int32_t*, const double*, const double*);
#endif

#ifdef GF_WITH_CUDA
#ifndef GF_ELEMENT_CUDA_SOURCE
extern template void compute_element_residual<BackendCUDA>(
    int, const double*, const double*, const double*, const double*,
    const double*, const double*, int, const double*, double*,
    const int32_t*, const double*, const double*);
#endif
#endif
```

- [ ] **Step 2: Add C-PML includes and using declarations in `element_cpu.cpp`**

After `#include "gf/gll.hpp"`, add:

```cpp
#include "gf/pml.hpp"
using namespace CpmlStrain;
```

- [ ] **Step 3: Add C-PML strain correction in element kernel**

In `element_cpu.cpp`, after the physical gradient computation block (after the `du_dx[comp][0/1/2]` assignments, before the `// --- Symmetric strain tensor ---` comment), insert:

```cpp
                    // --- C-PML strain correction (modifies physical gradients) ---
                    if (pml_region && pml_region[elem] != 0) {
                        StrainCoefficients coef = load_strain_coefficients(
                            pml_coef_strain, elem * n_node + n);

                        // Off-diagonal: duy/dx and duz/dx — corrected by grad_wrt_x (lijk z,y,x)
                        for (int comp : {DUY, DUZ}) {
                            double grad = du_dx[comp][DX];
                            size_t m_base = strain_memory_offset(elem * n_node + n,
                                gradient_of(comp, DX), 0);
                            double mem_z = rmemory_strain[m_base + CONV_Z];
                            double mem_y = rmemory_strain[m_base + CONV_Y];
                            double mem_x = rmemory_strain[m_base + CONV_X];
                            du_dx[comp][DX] = coef.grad_wrt_x.gradient_prefactor * grad
                                + coef.grad_wrt_x.memory_coef_conv_dir0 * mem_z
                                + coef.grad_wrt_x.memory_coef_conv_dir1 * mem_y
                                + coef.grad_wrt_x.memory_coef_conv_dir2 * mem_x;
                        }

                        // Off-diagonal: dux/dy and duz/dy — corrected by grad_wrt_y (lijk x,z,y)
                        for (int comp : {DUX, DUZ}) {
                            double grad = du_dx[comp][DY];
                            size_t m_base = strain_memory_offset(elem * n_node + n,
                                gradient_of(comp, DY), 0);
                            double mem_x = rmemory_strain[m_base + CONV_X];
                            double mem_z = rmemory_strain[m_base + CONV_Z];
                            double mem_y = rmemory_strain[m_base + CONV_Y];
                            du_dx[comp][DY] = coef.grad_wrt_y.gradient_prefactor * grad
                                + coef.grad_wrt_y.memory_coef_conv_dir0 * mem_x
                                + coef.grad_wrt_y.memory_coef_conv_dir1 * mem_z
                                + coef.grad_wrt_y.memory_coef_conv_dir2 * mem_y;
                        }

                        // Off-diagonal: dux/dz and duy/dz — corrected by grad_wrt_z (lijk x,y,z)
                        for (int comp : {DUX, DUY}) {
                            double grad = du_dx[comp][DZ];
                            size_t m_base = strain_memory_offset(elem * n_node + n,
                                gradient_of(comp, DZ), 0);
                            double mem_x = rmemory_strain[m_base + CONV_X];
                            double mem_y = rmemory_strain[m_base + CONV_Y];
                            double mem_z = rmemory_strain[m_base + CONV_Z];
                            du_dx[comp][DZ] = coef.grad_wrt_z.gradient_prefactor * grad
                                + coef.grad_wrt_z.memory_coef_conv_dir0 * mem_x
                                + coef.grad_wrt_z.memory_coef_conv_dir1 * mem_y
                                + coef.grad_wrt_z.memory_coef_conv_dir2 * mem_z;
                        }

                        // Diagonal: dux/dx — corrected by dux_dx (lx)
                        {
                            double grad = du_dx[DUX][DX];
                            size_t m_off = strain_memory_offset(elem * n_node + n, DUX_DX, CONV_X);
                            double mem_x = rmemory_strain[m_off];
                            du_dx[DUX][DX] = coef.dux_dx.gradient_prefactor * grad
                                + coef.dux_dx.memory_coef_local_dir * mem_x;
                        }
                        // Diagonal: duy/dy — corrected by duy_dy (ly)
                        {
                            double grad = du_dx[DUY][DY];
                            size_t m_off = strain_memory_offset(elem * n_node + n, DUY_DY, CONV_Y);
                            double mem_y = rmemory_strain[m_off];
                            du_dx[DUY][DY] = coef.duy_dy.gradient_prefactor * grad
                                + coef.duy_dy.memory_coef_local_dir * mem_y;
                        }
                        // Diagonal: duz/dz — corrected by duz_dz (lz)
                        {
                            double grad = du_dx[DUZ][DZ];
                            size_t m_off = strain_memory_offset(elem * n_node + n, DUZ_DZ, CONV_Z);
                            double mem_z = rmemory_strain[m_off];
                            du_dx[DUZ][DZ] = coef.duz_dz.gradient_prefactor * grad
                                + coef.duz_dz.memory_coef_local_dir * mem_z;
                        }
                    }
```

Note: the `comp : {DUY, DUZ}` range-for syntax requires initializer_list support.
Alternative if compiler complains — use explicit `int comp = DUY;` then `comp = DUZ;` loops.

- [ ] **Step 4: Build CPU and verify compilation**

```bash
cd forward && cmake -B build -DGF_DEVICE_BACKEND=CPU && cmake --build build
```

Expected: clean build.

- [ ] **Step 5: Run all tests**

```bash
cd forward/build && ctest --output-on-failure
```

Expected: all existing tests pass (C-PML path not triggered since no test sets up C-PML data).

- [ ] **Step 6: Commit**

```bash
git add forward/share/include/gf/element.hpp forward/elastic/src/element_cpu.cpp
git commit -m "feat: add C-PML strain correction branch to CPU element kernel"
```

______________________________________________________________________

### Task 3: Solver Integration

**Files:**

- Modify: `forward/share/src/solver.cpp`

**Interfaces:**

- Consumes: `cpml_update_strain_memory`, updated `compute_element_residual` signature

- Produces: Integrated C-PML strain correction in solver timestep

- [ ] **Step 1: Add `#include "gf/pml.hpp"` if not already present**

Check `solver.cpp` includes. The existing C-PML functions (`cpml_initialize`, etc.) are already used, so `#include "gf/pml.hpp"` should already be present. If not, add it.

- [ ] **Step 2: Pass C-PML params to element kernel**

In `solver.cpp`, locate the `compute_element_residual<gf::ActiveBackend>(` call (around line 552). Change:

```cpp
                compute_element_residual<gf::ActiveBackend>(
                    n_local_cell, part.dxi_dx.data(), part.jacobian.data(), part.lambda_.data(),
                    part.mu_.data(), D_mat.data(), gll_wts.data(), ngll,
                    local_cell_displacement.data(), local_cell_residual.data());
```

To:

```cpp
                compute_element_residual<gf::ActiveBackend>(
                    n_local_cell, part.dxi_dx.data(), part.jacobian.data(), part.lambda_.data(),
                    part.mu_.data(), D_mat.data(), gll_wts.data(), ngll,
                    local_cell_displacement.data(), local_cell_residual.data(),
                    part.pml_region.empty()      ? nullptr : part.pml_region.data(),
                    part.pml_coef_strain.empty() ? nullptr : part.pml_coef_strain.data(),
                    part.rmemory_strain.empty()  ? nullptr : part.rmemory_strain.data());
```

- [ ] **Step 3: Add strain memory update after displacement memory update**

Locate the block (around line 619):

```cpp
                // 8. C-PML: Update displacement memory variables (after corrector)
                if (part.has_cpml) {
                    cpml_update_displ_memory(part, n_node);
                }
```

Replace with:

```cpp
                // 8. C-PML: Update displacement and strain memory variables (after corrector)
                if (part.has_cpml) {
                    cpml_update_displ_memory(part, n_node);
                    cpml_update_strain_memory(part, D_mat.data(), gll_wts.data(), ngll);
                }
```

- [ ] **Step 4: Build and verify**

```bash
cd forward && cmake --build build
```

Expected: clean build.

- [ ] **Step 5: Run full test suite**

```bash
cd forward/build && ctest --output-on-failure
```

Expected: all 221 tests pass.

- [ ] **Step 6: Commit**

```bash
git add forward/share/src/solver.cpp
git commit -m "feat: integrate C-PML strain memory update and kernel params into solver"
```

______________________________________________________________________

### Task 4: Restart I/O for C-PML Memory State

**Files:**

- Modify: `forward/share/include/gf/restart.hpp`
- Modify: `forward/share/src/restart.cpp` (or wherever RestartWriter::write is implemented)

**Interfaces:**

- Consumes: `RankData::has_cpml`, `pml_displ_old/new`, `rmemory_displ`, `rmemory_strain`

- Produces: Extended `RestartWriter::write` and `RestartReader` with C-PML datasets

- [ ] **Step 1: Find RestartWriter::write implementation**

```bash
grep -rn "RestartWriter::write" forward/share/src/
```

- [ ] **Step 2: Modify `RestartWriter::write` signature**

Add a `const RankData&` parameter (or individual C-PML vectors). Minimal approach — add `const RankData& part`:

In `restart.hpp`, change:

```cpp
    void write(int step, double time_s, const std::vector<double>& displacement,
               const std::vector<double>& velocity, const std::vector<double>& acceleration,
               const std::vector<double>& pml_damping);
```

To:

```cpp
    void write(int step, double time_s, const std::vector<double>& displacement,
               const std::vector<double>& velocity, const std::vector<double>& acceleration,
               const std::vector<double>& pml_damping,
               const RankData* part = nullptr);
```

- [ ] **Step 3: Add C-PML dataset writes**

In the `write` implementation, after writing `pml_damping`, add:

```cpp
    // --- C-PML runtime state (only when present) ---
    if (part && part->has_cpml) {
        write_dataset(file_id_, "/restart/pml_displ_old", part->pml_displ_old);
        write_dataset(file_id_, "/restart/pml_displ_new", part->pml_displ_new);
        write_dataset(file_id_, "/restart/rmemory_displ", part->rmemory_displ);
        write_dataset(file_id_, "/restart/rmemory_strain", part->rmemory_strain);
    }
```

(Exact HDF5 write helper name may vary — check existing `write_dataset` pattern in the file.)

- [ ] **Step 4: Update solver restart calls**

In `solver.cpp`, both restart write call sites, change:

```cpp
                restart_writer.write(step, step * solver_dt, displacement, velocity, acceleration,
                                     part.pml_damping);
```

To:

```cpp
                restart_writer.write(step, step * solver_dt, displacement, velocity, acceleration,
                                     part.pml_damping, &part);
```

- [ ] **Step 5: Build and verify**

```bash
cd forward && cmake --build build
```

Expected: clean build.

- [ ] **Step 6: Run full test suite**

```bash
cd forward/build && ctest --output-on-failure
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add forward/share/include/gf/restart.hpp forward/share/src/restart.cpp forward/share/src/solver.cpp
git commit -m "feat: add C-PML memory state to restart I/O"
```

______________________________________________________________________

### Task 5: CUDA Element Kernel C-PML Branch

**Files:**

- Modify: `forward/elastic/src/element_cuda.cu`

**Interfaces:**

- Consumes: `CpmlStrain` namespace (from `pml.hpp`), updated kernel signature

- Produces: CUDA kernel with C-PML strain correction

- [ ] **Step 1: Add `#include "gf/pml.hpp"` to `element_cuda.cu`**

After the existing includes.

- [ ] **Step 2: Add `using namespace CpmlStrain;`**

After the `namespace gf {` block opening.

- [ ] **Step 3: Add C-PML strain correction to `element_residual_kernel`**

The CUDA kernel is in `__global__ void element_residual_kernel(...)`.
After the physical gradient computation and before the strain tensor computation, add the same C-PML branch as the CPU version. The key differences:

- Device memory pointers are already in global memory
- Use the same struct-based access pattern
- Index calculations use `elem_offset` instead of `elem * n_node`

Insert after the physical gradient transform block:

```cpp
    // --- C-PML strain correction ---
    if (pml_region && pml_region[e] != 0) {
        int node_flat = elem_offset + n;
        StrainCoefficients coef = load_strain_coefficients(pml_coef_strain, node_flat);

        // duy/dx and duz/dx
        if (comp_filter_duy_duz) { // see note below
            // ... same as CPU but with device pointers
        }
        // ... etc.
    }
```

*Implementation note:* CUDA's `__device__` code doesn't support `std::initializer_list`
(range-for `{DUY, DUZ}`). Use explicit if-blocks or a helper `__device__` function:

```cpp
        // Off-diagonal: duy/dx
        {
            constexpr int comp = DUY;
            double grad = du_dx[comp][DX];
            size_t m_base = strain_memory_offset(node_flat, gradient_of(comp, DX), 0);
            double mem_z = rmemory_strain[m_base + CONV_Z];
            double mem_y = rmemory_strain[m_base + CONV_Y];
            double mem_x = rmemory_strain[m_base + CONV_X];
            du_dx[comp][DX] = coef.grad_wrt_x.gradient_prefactor * grad
                + coef.grad_wrt_x.memory_coef_conv_dir0 * mem_z
                + coef.grad_wrt_x.memory_coef_conv_dir1 * mem_y
                + coef.grad_wrt_x.memory_coef_conv_dir2 * mem_x;
        }
        // Off-diagonal: duz/dx
        {
            constexpr int comp = DUZ;
            double grad = du_dx[comp][DX];
            size_t m_base = strain_memory_offset(node_flat, gradient_of(comp, DX), 0);
            double mem_z = rmemory_strain[m_base + CONV_Z];
            double mem_y = rmemory_strain[m_base + CONV_Y];
            double mem_x = rmemory_strain[m_base + CONV_X];
            du_dx[comp][DX] = coef.grad_wrt_x.gradient_prefactor * grad
                + coef.grad_wrt_x.memory_coef_conv_dir0 * mem_z
                + coef.grad_wrt_x.memory_coef_conv_dir1 * mem_y
                + coef.grad_wrt_x.memory_coef_conv_dir2 * mem_x;
        }
        // ... repeat for the other 7 gradient corrections
```

- [ ] **Step 4: Update `cuda_launch_element_residual` to pass C-PML pointers**

The host-side wrapper passes pointers to the kernel. Add C-PML params after the
existing arguments:

```cpp
    element_residual_kernel<<<grid, block>>>(
        state.d_dxi_dx, state.d_jacobian, state.d_lambda_, state.d_mu_,
        state.d_D, state.d_weights, ngll, d_input, d_output,
        state.d_pml_region, state.d_pml_coef_strain, state.d_rmemory_strain);
```

Also update the `compute_element_residual<BackendCUDA>` specialization and the
`cuda_launch_element_residual` function to handle the new parameters (pass
`nullptr` when C-PML not available).

- [ ] **Step 5: Build CUDA**

```bash
cd forward && cmake -B build -DGF_DEVICE_BACKEND=CUDA && cmake --build build
```

Expected: clean CUDA build.

- [ ] **Step 6: Commit**

```bash
git add forward/elastic/src/element_cuda.cu
git commit -m "feat: add C-PML strain correction branch to CUDA element kernel"
```

______________________________________________________________________

### Task 6: CUDA C-PML Runtime

**Files:**

- Modify: `forward/share/include/gf/cuda_step.hpp`
- Modify: `forward/share/src/cuda_step.cu`

**Interfaces:**

- Consumes: `CpmlStrain` constants, C-PML data arrays in `RankData`

- Produces: GPU C-PML functions + updated `CudaDeviceState`

- [ ] **Step 1: Add C-PML fields to `CudaDeviceState` in `cuda_step.hpp`**

After `d_pml` field, add:

```cpp
    // --- C-PML device buffers (persistent) ---
    int32_t* d_pml_region = nullptr;          // [n_local_cell]
    double* d_pml_coef_alpha = nullptr;       // [n_total_nodes * 9]
    double* d_pml_coef_beta = nullptr;        // [n_total_nodes * 9]
    double* d_pml_coef_abar = nullptr;        // [n_total_nodes * 5]
    double* d_pml_coef_strain = nullptr;      // [n_total_nodes * 18]
    double* d_pml_displ_old = nullptr;        // [n_total_nodes * 3]
    double* d_pml_displ_new = nullptr;        // [n_total_nodes * 3]
    double* d_rmemory_displ = nullptr;        // [n_total_nodes * 9]
    double* d_rmemory_strain = nullptr;       // [n_total_nodes * 27]
    bool has_cpml = false;
```

- [ ] **Step 2: Add C-PML GPU function declarations to `cuda_step.hpp`**

Replace `cuda_pml_damping` declaration with:

```cpp
/// C-PML: update PML displacement fields on device.
void cuda_cpml_update_displ_fields(CudaDeviceState& state, double dt, int n_node);

/// C-PML: update displacement memory variables on device.
void cuda_cpml_update_displ_memory(CudaDeviceState& state, int n_node);

/// C-PML: update strain memory variables on device.
void cuda_cpml_update_strain_memory(CudaDeviceState& state, int ngll, int n_node);

/// C-PML: add acceleration correction to element-local residual on device.
void cuda_cpml_accel_contribution(CudaDeviceState& state, int ngll, int n_node);
```

- [ ] **Step 3: Implement CUDA kernels and host wrappers in `cuda_step.cu`**

Implement GPU kernels for each C-PML function:

- `cpml_update_displ_fields_kernel` — per-GLL-node, same logic as CPU
- `cpml_update_displ_memory_kernel` — per-GLL-node
- `cpml_update_strain_memory_kernel` — per-GLL-node, computes gradients + β convolution
- `cpml_accel_contribution_kernel` — per-GLL-node, adds Ā₁…Ā₅ correction

Grid strategy: 1D grid of blocks, each thread handles one GLL node.
`grid = (n_pml_nodes + block_size - 1) / block_size`.

- [ ] **Step 4: Update `cuda_allocate_state` to upload C-PML data**

Add C-PML data upload in the allocation function when `RankData::has_cpml` is true.

- [ ] **Step 5: Update `cuda_free_state` to free C-PML buffers**

- [ ] **Step 6: Build CUDA**

```bash
cd forward && cmake --build build
```

Expected: clean CUDA build.

- [ ] **Step 7: Commit**

```bash
git add forward/share/include/gf/cuda_step.hpp forward/share/src/cuda_step.cu
git commit -m "feat: add CUDA C-PML runtime (displ fields, memory, accel, strain)"
```

______________________________________________________________________

### Task 7: Unit Tests + Validation

**Files:**

- Modify: `tests/test_pml.cpp`

- [ ] **Step 1: Add `cpml_update_strain_memory` unit test**

Add to `tests/test_pml.cpp`:

```cpp
#include "gf/pml.hpp"
#include "gf/types.hpp"
using namespace CpmlStrain;

TEST_CASE("cpml_update_strain_memory initializes and runs", "[pml][cpml]") {
    // Setup: create minimal RankData with 1 PML element
    RankData part;
    int ngll = 4;
    int n_node = ngll * ngll * ngll;  // 64
    part.n_local_cell = 1;
    part.ngll = ngll;
    part.has_cpml = true;

    // Allocate flat arrays
    int n_total = n_node;  // 1 elem × 64 nodes

    part.pml_region = {1};  // CPML_X_ONLY
    part.pml_coef_alpha.assign(n_total * 9, 0.0);
    part.pml_coef_beta.assign(n_total * 9, 0.0);
    part.pml_dxi_dx.assign(n_total * 9, 0.0);
    // Set identity Jacobian (dξ/dx=1, others=0 — interior node, no PML effect on gradient)
    for (int n = 0; n < n_total; ++n) {
        part.dxi_dx[n * 9 + 0] = 1.0;   // dξ/dx
        part.dxi_dx[n * 9 + 4] = 1.0;   // dη/dy
        part.dxi_dx[n * 9 + 8] = 1.0;   // dζ/dz
    }

    // Non-zero β coefficients so memory updates
    for (int n = 0; n < n_total; ++n) {
        for (int d = 0; d < 3; ++d) {
            part.pml_coef_beta[n * 9 + d * 3 + 0] = 0.5;  // coef0
            part.pml_coef_beta[n * 9 + d * 3 + 1] = 0.3;  // coef1
            part.pml_coef_beta[n * 9 + d * 3 + 2] = 0.2;  // coef2
        }
    }

    // Initialize memory
    part.pml_displ_old.assign(n_total * 3, 0.1);
    part.pml_displ_new.assign(n_total * 3, 0.2);
    part.rmemory_displ.assign(n_total * 9, 0.0);
    part.rmemory_strain.assign(n_total * 27, 0.0);

    // GLL derivative matrix (simplified: identity for test)
    std::vector<double> D(ngll * ngll, 0.0);
    std::vector<double> weights(ngll, 1.0);

    // Call the function
    cpml_update_strain_memory(part, D.data(), weights.data(), ngll);

    // Verify: strain memory is no longer all zero
    bool has_nonzero = false;
    for (double v : part.rmemory_strain) {
        if (std::abs(v) > 1e-15) { has_nonzero = true; break; }
    }
    REQUIRE(has_nonzero);
}

TEST_CASE("Interior elements have no strain memory update", "[pml][cpml]") {
    RankData part;
    int ngll = 4;
    int n_node = ngll * ngll * ngll;
    part.n_local_cell = 1;
    part.ngll = ngll;
    part.has_cpml = true;

    part.pml_region = {0};  // interior — NOT PML
    part.pml_coef_beta.assign(n_node * 9, 0.5);
    part.pml_displ_old.assign(n_node * 3, 0.1);
    part.pml_displ_new.assign(n_node * 3, 0.2);
    part.rmemory_strain.assign(n_node * 27, 0.0);
    part.dxi_dx.assign(n_node * 9, 0.0);
    for (int n = 0; n < n_node; ++n) {
        part.dxi_dx[n * 9 + 0] = 1.0;
        part.dxi_dx[n * 9 + 4] = 1.0;
        part.dxi_dx[n * 9 + 8] = 1.0;
    }

    std::vector<double> D(ngll * ngll, 0.0);
    std::vector<double> weights(ngll, 1.0);
    cpml_update_strain_memory(part, D.data(), weights.data(), ngll);

    // Interior element: strain memory should remain zero
    for (double v : part.rmemory_strain) {
        REQUIRE(std::abs(v) < 1e-15);
    }
}
```

- [ ] **Step 2: Build and run new tests**

```bash
cd forward && cmake -B build -DGF_DEVICE_BACKEND=CPU && cmake --build build
cd build && ctest -R "pml" --output-on-failure
```

Expected: 2 new tests + existing tests all pass.

- [ ] **Step 3: Run full test suite**

```bash
cd forward/build && ctest --output-on-failure
```

Expected: all 223 tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pml.cpp
git commit -m "test: add C-PML strain memory update unit tests"
```

______________________________________________________________________

### Task 8: End-to-End Validation

- [ ] **Step 1: Run halfspace example with C-PML enabled**

```bash
cd examples/halfspace
# Ensure config.py has C-PML enabled (check preprocess runs pml_cpml.py)
bash run.sh
```

- [ ] **Step 2: Check output for stability**

Verify solver completes without energy blow-up. Check displacement magnitudes
are physically reasonable.

- [ ] **Step 3: Compare absorption quality**

Compare wavefield snapshots at PML boundary against previous results
(linear ramp damping). C-PML should show reduced boundary reflections.

- [ ] **Step 4: Commit any config adjustments**

```bash
git add examples/ && git commit -m "chore: update halfspace example for C-PML validation"
```
