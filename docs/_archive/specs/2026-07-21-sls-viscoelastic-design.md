# SLS Viscoelastic Attenuation — Implementation Spec

**Date:** 2026-07-21
**Status:** Draft
**Parent design:** [`docs/deferred.md`](../../deferred.md) §1
**Context:** Adding standard linear solid (SLS) frequency-independent Q attenuation
to the SEM forward solver. Elastic solver and C-PML are complete.

## 1. Scope

This spec covers the full SLS attenuation pipeline:

- **Preprocess**: per-GLL-node τ_σ, τ_ε, unrelaxed moduli
- **Forward**: SLS memory variable arrays, update function, element kernel stress
  modification, CUDA port, restart I/O
- **Integration**: independent solver binary under `forward/viscoelastic/`,
  reusing `libgf_shared` and the five `kernel_helpers` extracted in commit `fbad6b7`

Out of scope:

- Depth-dependent Q profiles beyond per-node Q_kappa/Q_mu from material model
- Alternative attenuation models (Maxwell, Burgers, etc.)
- Q-model parameter estimation

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| n_sls | 3 (compile-time constant) | Matches SPECFEM3D default; covers 0.01–10 Hz band |
| Q model | Space-varying (per-GLL-node Q_kappa, Q_mu) | Follows λ/μ pattern; ready for complex geology |
| Architecture | Independent solver binary + shared kernel_helpers | Zero elastic code modification; clean separation |
| τ computation | τ-method with log-spaced τ_σ | Proven in SPECFEM3D; stable least-squares fit |
| Stress model | Subtract memory tensor from elastic stress | Minimal kernel change; σ = σ_elastic − ΣR_l |

## 3. Mathematical Summary

The standard linear solid introduces n_sls relaxation mechanisms per GLL node.
For mechanism *l*:

- Stress relaxation time: τ_σ^l
- Strain relaxation time: τ_ε^l (τ_ε^l > τ_σ^l)

The viscoelastic stress:

```
σ_ij(t) = σ_ij^elastic(t) − Σ_{l=1}^{n_sls} R_ij^l(t)
```

where the memory stress tensor R_ij^l evolves via a first-order recursive
convolution:

```
R_ij^l(t+Δt) = a_l · R_ij^l(t) + b_l · Δσ_ij^elastic
```

with:

```
a_l = exp(−Δt / τ_σ^l)
b_l = (τ_ε^l / τ_σ^l − 1) · (1 − a_l)
Δσ_ij^elastic = σ_ij^elastic(t+Δt) − σ_ij^elastic(t)
```

The preprocess computes τ_σ^l and τ_ε^l per GLL node from Q_kappa and Q_mu
using the τ-method (logarithmically spaced τ_σ, linear least-squares fit for
τ_ε). 3 mechanisms × 2 moduli (κ, μ) = 6 relaxation times per node.
For isotropic elasticity the same τ values are used for all stress components
(κ and μ share the mechanism frequencies, but each has its own τ_ε/τ_σ ratio).

Simplified: preprocess computes one set of τ_σ^l and τ_ε^l per node (shared by
all 6 stress components) from the harmonic mean of Q_kappa and Q_mu, or
separate sets if needed.

**Final decision**: one set of τ_σ^l, τ_ε^l per node, derived from Q_mu only
(follows SPECFEM3D practice — shear attenuation dominates seismic waveforms).
Q_kappa is read but deferred for a future "bulk attenuation" flag.

## 4. Data Model

### 4.1 Compile-Time Constants

```cpp
namespace SLS {

constexpr int N_SLS              = 3;   // number of relaxation mechanisms
constexpr int VOIGT_COMPONENTS   = 6;   // independent stress/strain components (3D sym)
constexpr int MEMORY_PER_NODE    = N_SLS * VOIGT_COMPONENTS;  // = 18 doubles
constexpr int TAU_PER_NODE       = N_SLS * 2;                 // τ_σ + τ_ε × 3

}  // namespace SLS
```

### 4.2 Voigt Index Convention

| Voigt Index | Tensor Index | Name |
|-------------|-------------|------|
| 0 | (0,0) | xx |
| 1 | (1,1) | yy |
| 2 | (2,2) | zz |
| 3 | (0,1) = (1,0) | xy (shear) |
| 4 | (0,2) = (2,0) | xz (shear) |
| 5 | (1,2) = (2,1) | yz (shear) |

Helper: `inline int voigt(int l, int m) { return (l == m) ? l : 3 + l + m; }`
(valid only for l,m in {0,1,2}).

### 4.3 Memory Variable Array: `rmemory_sls`

Flat storage per element: `n_node × SLS::MEMORY_PER_NODE` doubles.

```
for each GLL node:
    for each SLS mechanism (0..N_SLS-1):
        R_xx, R_yy, R_zz, R_xy, R_xz, R_yz   (6 Voigt components)
```

Access:

```cpp
inline size_t sls_memory_offset(size_t node, int mechanism, int voigt_idx) {
    return node * SLS::MEMORY_PER_NODE + mechanism * SLS::VOIGT_COMPONENTS + voigt_idx;
}
```

### 4.4 Relaxation Times: `tau_sigma`, `tau_epsilon`

Flat storage per element: `n_node × SLS::TAU_PER_NODE` doubles each.

```
for each GLL node:
    τ_σ[0], τ_σ[1], τ_σ[2], τ_ε[0], τ_ε[1], τ_ε[2]
```

### 4.5 Coefficient Arrays: `sls_coef_a`, `sls_coef_b`

Derived from τ_σ^l and solver_dt. Precomputed once at simulation start.

```
for each GLL node × N_SLS:
    a_l = exp(−solver_dt / τ_σ^l)
    b_l = (τ_ε^l / τ_σ^l − 1) · (1 − a_l)
```

### 4.6 HDF5 Layout (model.h5)

```
/field/cell/tau_sigma     [n_cell, NGLL³, N_SLS]   float64
/field/cell/tau_epsilon   [n_cell, NGLL³, N_SLS]   float64
/field/cell/q_kappa       [n_cell, NGLL³]          float64  (read from material)
/field/cell/q_mu          [n_cell, NGLL³]          float64  (read from material)
```

PML datasets remain unchanged (C-PML applies equally to viscoelastic kernel).

### 5.6 Restart I/O

```
/restart/rmemory_sls    [n_rank_node × MEMORY_PER_NODE]  float64
```

Only written/read when `has_attenuation` is true.

## 5. Forward Implementation

### 5.1 SLS Memory Variable Update

Located in a new header `forward/share/include/gf/attenuation.hpp` (or added
to `pml.hpp` as a sibling namespace). Called once per timestep, after
Newmark corrector produces new displacement but before the next residual
evaluation.

```cpp
namespace SLS {

/// Update SLS memory stress tensors for all PML + interior nodes.
/// Must be called AFTER displacement is updated (Newmark corrector)
/// and BEFORE the next element residual computation.
///
/// For each node and SLS mechanism l:
///   R_l(t+dt) = a_l * R_l(t) + b_l * (sigma_elastic(t+dt) - sigma_elastic(t))
void update_sls_memory(RankData& part,
                       const double* displacement,
                       const double* solver_dt_ptr,
                       int n_node);

}  // namespace SLS
```

Internal logic:

```
for each GLL node (global index):
    // 1. Compute elastic stress from current displacement
    //    (reuse transform_to_physical + compute_strain + elastic stress)
    //    OR: the element kernel already computes sigma_elastic;
    //        store it between kernel call and memory update.

    // 2. For each SLS mechanism:
    for l in 0..N_SLS-1:
        a = sls_coef_a[node * N_SLS + l]
        b = sls_coef_b[node * N_SLS + l]
        for each Voigt component v in 0..5:
            delta_sigma = sigma_elastic_v - sigma_old_v
            R = rmemory_sls[offset(node, l, v)]
            R = a * R + b * delta_sigma
            rmemory_sls[offset(node, l, v)] = R

    // 3. Save sigma_elastic → sigma_old for next timestep
```

**Design note**: `sigma_old` (6 Voigt components per node) requires an
additional scratch array `n_node × 6` doubles. This is allocated per-rank
and stored in `RankData`. Alternative: store strain instead of stress
(reduces dependency on elastic moduli at memory-update time).

### 5.2 Element Kernel Structure

`viscoelastic/src/element_cpu.cpp` — identical flow to the elastic kernel
with one difference at the stress computation step:

```
compute_element_residual<BackendCPU>:
  for each element:
    for each GLL node (i,j,k):
      compute_reference_gradient()        // [1] shared helper
      transform_to_physical()             // [2] shared helper
      apply_cpml_strain_correction()      // [3] shared helper
      compute_strain_tensor()             // [4] shared helper
      ───────────────────────────────────
      // Elastic trial stress
      eps_kk = eps[0][0] + eps[1][1] + eps[2][2]
      for l,m in 0..2:
          sigma[l][m] = 2.0 * mu_unrelaxed * eps[l][m]
      sigma[l][l] += lambda_unrelaxed * eps_kk

      // Subtract SLS memory (viscoelastic-only)
      for sls in 0..N_SLS-1:
          const double* R = &rmemory_sls[n * MEMORY_PER_NODE + sls * 6]
          sigma[0][0] -= R[0]; sigma[1][1] -= R[1]; sigma[2][2] -= R[2]
          sigma[0][1] -= R[3]; sigma[0][2] -= R[4]; sigma[1][2] -= R[5]
          sigma[1][0] = sigma[0][1]  // symmetrize
          sigma[2][0] = sigma[0][2]
          sigma[2][1] = sigma[1][2]
      ───────────────────────────────────
      scatter_residual()                  // [5] shared helper
```

**Unrelaxed moduli**: The elastic λ and μ stored in `model.h5` are the
UNRELAXED moduli (λ_u, μ_u). The relaxed moduli are lower:
λ_r = λ_u / (1 + τ_κ), μ_r = μ_u / (1 + τ_μ) where τ_κ, τ_μ are the
total anelastic factors summed over all SLS mechanisms. The element kernel
uses the unrelaxed moduli directly; the SLS memory subtraction accounts for
the relaxation.

**Kernel signature**: same as elastic — `compute_element_residual<BackendCPU>`
with the same 10 parameters plus `rmemory_sls` passed through the existing
C-PML parameter slots (reinterpreted) or via a new parameter.

**Decision**: add a new `const double* rmemory_sls` parameter (with `nullptr`
default) to the template in `element.hpp`. Both elastic and viscoelastic
kernels share the same signature but viscoelastic passes a non-null pointer
and uses it in the stress section.

### 5.3 Solver Integration

`viscoelastic/src/solver.cpp` follows the same structure as
`forward/share/src/solver.cpp`:

```
time loop (step 0..nsteps-1):
    newmark_predict()
    gather_from_rank()            // if global DOF
    compute_element_residual()     // elastic + SLS subtraction
    scatter_to_rank()              // if global DOF
    exchange_halo()
    apply_source()
    newmark_correct()

    if (has_attenuation):
        compute_sls_coefficients()  // once at step 0: a_l, b_l from tau, dt
        update_sls_memory()         // every step: R update + sigma_old save

    if (has_cpml):
        cpml_update_displ_memory()
        cpml_update_strain_memory()

    snapshot_output()
    restart_output()
```

### 5.4 Types Additions (`types.hpp`)

```cpp
struct RankData {
    // ... existing fields ...

    // SLS attenuation
    bool     has_attenuation = false;
    int32_t  n_sls = 0;                // from config (compile-time constant N_SLS)
    std::vector<double> tau_sigma;     // [n_node × N_SLS]
    std::vector<double> tau_epsilon;   // [n_node × N_SLS]
    std::vector<double> sls_coef_a;    // [n_node × N_SLS]  precomputed a_l
    std::vector<double> sls_coef_b;    // [n_node × N_SLS]  precomputed b_l
    std::vector<double> rmemory_sls;   // [n_node × MEMORY_PER_NODE]
    std::vector<double> sigma_old;     // [n_node × 6]  previous elastic stress
};
```

### 5.5 Coefficient Precomputation

At simulation start (step 0), after reading tau_sigma/tau_epsilon from HDF5:

```cpp
for each node:
    for l in 0..N_SLS-1:
        double tau_s = tau_sigma[node * N_SLS + l];
        double tau_e = tau_epsilon[node * N_SLS + l];
        double a = std::exp(-solver_dt / tau_s);
        double b = (tau_e / tau_s - 1.0) * (1.0 - a);
        sls_coef_a[node * N_SLS + l] = a;
        sls_coef_b[node * N_SLS + l] = b;
```

## 6. CUDA Implementation

### 6.1 GPU Element Kernel

`viscoelastic/src/element_cuda.cu` — same structure as the CUDA elastic
kernel, one thread per GLL node, with the viscoelastic stress modification
from §5.2. Calls the five `__device__` helpers from `kernel_helpers.cuh`.

```
element_residual_kernel<<<n_elem, NGLL³>>>():
    // one thread per (elem, i, j, k)
    compute_reference_gradient()       // [1] __device__ helper
    transform_to_physical()            // [2] __device__ helper
    apply_cpml_strain_correction()     // [3] __device__ helper
    compute_strain_tensor()            // [4] __device__ helper
    // --- viscoelastic stress (identical to CPU §5.2) ---
    scatter_residual()                 // [5] __device__ helper (atomicAdd)
```

### 6.2 GPU SLS Memory Update Kernel

A new `__global__` kernel in `cuda_step.cu`:

```cpp
__global__ void sls_memory_kernel(
    int n_node,
    const double* __restrict__ sls_coef_a,
    const double* __restrict__ sls_coef_b,
    const double* __restrict__ sigma_current,   // newly computed elastic stress
    double* __restrict__ sigma_old,             // previous-step elastic stress
    double* __restrict__ rmemory_sls);
```

One thread per GLL node. For each SLS mechanism, computes
R = a*R + b*(σ_current - σ_old) for all 6 Voigt components, then copies
σ_current → σ_old.

### 6.3 CudaDeviceState Additions

```cpp
struct CudaDeviceState {
    // ... existing fields ...
    bool     has_attenuation = false;
    double*  d_tau_sigma = nullptr;
    double*  d_tau_epsilon = nullptr;
    double*  d_sls_coef_a = nullptr;
    double*  d_sls_coef_b = nullptr;
    double*  d_rmemory_sls = nullptr;
    double*  d_sigma_old = nullptr;
};
```

New host functions:

- `cuda_upload_sls_data(CudaDeviceState&, const RankData&, int n_node)`
- `cuda_update_sls_memory(CudaDeviceState&, const double* d_displacement, int n_node)`
- `cuda_free_sls_data(CudaDeviceState&)`

## 7. Implementation Order

| Task | Description | Files |
|------|-------------|-------|
| 1 | SLS namespace + constants + `update_sls_memory()` | `pml.hpp/cpp` or new `attenuation.hpp/cpp` |
| 2 | Preprocess: read Q, compute τ, write HDF5 | `preprocess/pml_cpml.py` or new `preprocess/attenuation.py` |
| 3 | I/O: read tau + rmemory_sls, restart read/write | `io.cpp/hpp`, `restart.cpp/hpp` |
| 4 | `viscoelastic/` skeleton: main + CMakeLists | `viscoelastic/src/main.cpp`, `viscoelastic/CMakeLists.txt` |
| 5 | CPU viscoelastic element kernel | `viscoelastic/src/element_cpu.cpp` |
| 6 | Solver integration (memory update + kernel call) | `viscoelastic/src/solver.cpp` (or share solver with flag) |
| 7 | CUDA viscoelastic element kernel | `viscoelastic/src/element_cuda.cu` |
| 8 | CUDA runtime (SLS memory upload/update kernels) | `cuda_step.cu/hpp` |
| 9 | Unit tests + halfspace validation | `tests/`, `examples/halfspace/` |

## 8. Risk Mitigation

1. **Elastic solver untouched**: viscoelastic/ is a separate directory and binary
1. **kernel_helpers already shared**: commit `fbad6b7` extracted 5 helpers —
   viscoelastic kernel calls them identically, only stress changes
1. **CPU first, CUDA second**: validate physics on CPU before GPU port
1. **has_attenuation flag**: all SLS code paths gated behind runtime flag;
   elastic path unchanged
1. **Backward-compatible restart**: reader checks dataset existence, old
   restart files load without SLS memory (zeros-initialized)
