# Forward Solver — Green's Function SEM Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build libgf, the C++17 physics library powering the elastic spectral-element forward solver, and gf_solver, the MPI-parallel executable that drives it.

> **Design**: Technical decisions (data types, component architecture, C-PML formulas, L2 strain smoothing, 39-index memory variable mapping) are documented in [`docs/superpowers/design/forward.md`](../../docs/superpowers/design/forward.md). This file contains only the implementation plan.

---

## Run Config

3 independent forward runs per source location (one per orthogonal force direction: x, y, z). Postprocess assembles the 6×3 strain Green's tensor from these 3 record sets.

---

---

- [ ] 11. Checkpoint Writer with L2 Strain Smoothing

  **Files:**
  - Create: `forward/include/gf/checkpoint.hpp`
  - Create: `forward/src/checkpoint.cpp`

  Extendible HDF5 strain writer using the compress module. One file per MPI rank at `wavefields/{direction}/record_{r}.h5`.

  **L2 strain smoothing** is performed before every checkpoint write (not every time step).
  Two-phase algorithm:

  ```
  Phase A: Compute discontinuous element strain from corrected u
    For each local element e:
      For each GLL node (i,j,k):
        ∇u = GLL_derivatives × dxi_dx[e][i][j][k]
        ε_elem = ½(∇u + ∇uᵀ)   // 6-vector per GLL node

  Phase B: L2 projection onto C⁰-continuous basis
    // Assemble weighted strain into global array
    s_global[6, n_global_nodes] = 0
    For each local element e:
      For each GLL node (i,j,k):
        iglob = gll_to_global[e][i][j][k]
        weight = jacobian[e][i][j][k] × w_gll[i] × w_gll[j] × w_gll[k]
        s_global[0..5][iglob] += weight × ε_elem[i][j][k][0..5]

    // MPI exchange: sum shared ghost-node contributions
    // (same face-pair lists as residual assembly)

    // Apply lumped mass inverse
    ε_smooth[e][i][j][k][0..5] = s_global[0..5][iglob] / mass[e][i][j][k]
  ```

  - [ ] Step 1: Implement `compute_element_strain()` — compute ε_elem from corrected displacement u
  - [ ] Step 2: Implement `l2_project_strain()` — Phase B assembly + lumped mass inverse
  - [ ] Step 3: Implement record writer — first call creates file + `local_element_ids` + `strain` extendible dataset (`[UNLIMITED, n_elem_local, NGLL, NGLL, NGLL, 6]`). Subsequent calls extend dim 0 by 1. Overwrite `/restart/` with (u, v, a).
  - [ ] Step 4: Write Catch2 tests — verify L2 projection of constant strain (should be identity), verify round-trip of extendible writes
  - [ ] Step 5: Commit

  **Commit**: YES
  - Message: `feat(forward): checkpoint writer with L2 strain smoothing`

---

- [ ] 12. Solver Driver

  **Files:**
  - Create: `forward/include/gf/solver.hpp`
  - Create: `forward/src/solver.cpp`

  `run_forward()` orchestrates the full simulation loop.

  - [ ] Step 1: Implement runtime loop matching design:

  ```
  for step in 0..nsteps-1:
      1. Newmark predict: ũ, ṽ
      2. Compute element residual (matrix-free K·u for local + ghost):
         Zero global residual r[NDIM, n_global_nodes] = 0
         For each element e:
           For each GLL node (i,j,k):
             ∇ũ via GLL derivatives × dxi_dx
             ε = ½(∇ũ + ∇ũᵀ), σ = C:ε (elastic)
             Accumulate: r[:, iglob] -= Bᵀ·σ·detJ
      3. C-PML memory variable update (CPML elements only):
         For each CPML element:
           pml_compute_memory_variables(e, ispec_CPML, rmemory, ...)
           pml_compute_accel_contribution(e, ispec_CPML, PML_displ_old, PML_displ_new, ...)
           Accumulate C-PML correction into r
      4. Source injection (if step < STF length):
         For each source element in precomputed list:
           r[:, iglob] += STF[step] × w_ijk × direction_vector[d]
      5. MPI halo exchange:
         Pack/send face GLL node values → recv ghost → unpack + sum
      6. Newmark correct:
         a_new = M⁻¹·r  (lumped mass, pointwise division)
         v = ṽ + dt·γ·a_new
         u = ũ + dt²·β·a_new
      7. If step % checkpoint_interval == 0:
         compute_element_strain()     // ε_elem from corrected u
         l2_project_strain()          // C⁰-continuous ε_smooth
         write_record(ε_smooth)       // append to HDF5
  ```

  - [ ] Step 2: Write integration test — small forward run (N=3, single element, few steps), verify no crash, verify record file created with correct shape
  - [ ] Step 3: Commit

  **Commit**: YES
  - Message: `feat(forward): solver driver with full C-PML + Newmark + L2 checkpoint`

---

## Final File Layout

```
forward/
├── CMakeLists.txt
├── include/gf/
│   ├── types.hpp             — Vec3, Mat33, RankData, GLLQuad, NewmarkParams
│   ├── gll.hpp               — header-only GLL points/weights/derivatives/Lagrange
│   ├── element.hpp           — matrix-free element residual
│   ├── assembly.hpp          — global residual assembly
│   ├── viscoelastic.hpp      — SLS — deferred
│   ├── pml.hpp               — C-PML memory variables + accel contribution
│   ├── newmark.hpp           — Newmark predictor/corrector
│   ├── source.hpp            — source injection
│   ├── exchange.hpp          — MPI halo exchange
│   ├── io.hpp                — partition_{r}.h5 + config.h5 reader
│   ├── checkpoint.hpp        — L2 strain smoothing + HDF5 writer
│   └── solver.hpp            — run_forward() orchestrator
├── src/
│   ├── main.cpp              — MPI init, CLI parse, run forward
│   ├── element.cpp
│   ├── assembly.cpp
│   ├── viscoelastic.cpp      — placeholder
│   ├── pml.cpp
│   ├── newmark.cpp
│   ├── source.cpp
│   ├── exchange.cpp
│   ├── io.cpp
│   ├── checkpoint.cpp
│   └── solver.cpp
└── tests/
    ├── test_element.cpp
    ├── test_assembly.cpp
    ├── test_pml.cpp
    ├── test_newmark.cpp
    ├── test_source.cpp
    ├── test_io.cpp
    ├── test_checkpoint.cpp
    └── test_integration.cpp
```

```
Wave 1 (foundation):
├── Task 1: Project scaffolding + types.hpp
├── Task 2: GLL quadrature header
├── Task 3: Matrix-free element residual
└── Task 4: Assembly

Wave 2 (core physics, MAX PARALLEL):
├── Task 5: Viscoelastic SLS — DEFERRED
├── Task 6: C-PML recursive convolution (depends: 1)
├── Task 7: Newmark time integration (depends: 1)
├── Task 8: Source injection (depends: 1)
├── Task 9: MPI halo exchange (depends: 1)
└── Task 10: I/O reader (depends: 1)

Wave 3 (integration):
├── Task 11: Checkpoint writer with L2 strain smoothing (depends: 2, 3)
└── Task 12: Solver driver + integration test (depends: 4, 6, 7, 8, 9, 10, 11)
```

---

## TODOs

- [ ] 1. Common Types and Project Scaffolding

  **Files:**
  - Create: `forward/CMakeLists.txt`
  - Create: `forward/include/gf/types.hpp`

  Build the minimal CMake scaffold and define every shared type the rest of the library needs.

  **Steps:**
  - [ ] Step 1: Write CMakeLists.txt with libgf target + source files (element.cpp, assembly.cpp, pml.cpp, newmark.cpp, source.cpp, exchange.cpp, io.cpp, solver.cpp)
  - [ ] Step 2: Write types.hpp with design-aligned data structures

  ```cpp
  namespace gf {
  using Vec3  = Eigen::Vector3d;
  using Mat33 = Eigen::Matrix3d;
  using Mat93 = Eigen::Matrix<double, 9, 3>;

  struct GLLQuad {
      int N;
      std::vector<double> points, weights, derivatives;
  };

  struct RankData {
      int n_local_elem, n_ghost_elem, n_total_elem;
      int ngll;
      std::vector<int64_t> local_element_ids, ghost_element_ids;
      std::vector<int32_t> ghost_owners;
      std::vector<double> coords, jacobian, dxi_dx, mass;
      std::vector<double> vp, vs, density;
      std::vector<double> pml_damping;   // 0=interior
      std::vector<int32_t> neighbors;
  };

  struct NewmarkParams { double beta = 0.0, gamma = 0.5, dt = 0.0; };
  }
  ```

  - [ ] Step 3: Create placeholder source files so libgf compiles
  - [ ] Step 4: Write placeholder main.cpp with MPI init/finalize
  - [ ] Step 5: Verify `cmake -B build -S . && cmake --build build` passes
  - [ ] Step 6: Commit

  **Commit**: YES
  - Message: `feat(forward): project scaffolding with types.hpp and CMakeLists.txt`

---

- [ ] 2. GLL Quadrature and Lagrange Basis

  **Files:**
  - Create: `forward/include/gf/gll.hpp`

  Header-only GLL utilities: compute GLL points/weights for order N (up to N=5), derivative matrix, Lagrange basis evaluation.

  - [ ] Step 1: Implement `gll_points(N)` and `gll_weights(N)` — standard Legendre-polynomial approach
  - [ ] Step 2: Implement `gll_derivative_matrix(N)` — `D[i][j] = derivative of ℓ_j at ξ_i`
  - [ ] Step 3: Implement `lagrange_basis(xi, nodes)` — evaluate all N+1 basis functions at arbitrary ξ ∈ [-1,1]
  - [ ] Step 4: Write Catch2 tests — verify D·1 = 0 (derivative of constant), verify D·ξ = 1 (derivative of identity)
  - [ ] Step 5: Commit

  **Commit**: YES
  - Message: `feat(forward): GLL quadrature header-only (nodes, weights, derivatives, Lagrange basis)`

---

- [ ] 3. Matrix-Free Element Residual

  **Files:**
  - Create: `forward/include/gf/element.hpp`
  - Create: `forward/src/element.cpp`

  Compute K_e·u element-by-element using precomputed dξ/dx and detJ. No global matrix.

  - [ ] Step 1: Implement `compute_element_residual()` — for each GLL node, ε = ½(∇u + ∇uᵀ), σ = C:ε, accumulate `r -= Bᵀ·σ·detJ·w`
  - [ ] Step 2: Write Catch2 tests — residual = 0 for rigid-body motion, matches analytical for uniform strain
  - [ ] Step 3: Commit

  **Commit**: YES
  - Message: `feat(forward): matrix-free element residual`

---

- [ ] 4. Assembly

  **Files:**
  - Create: `forward/include/gf/assembly.hpp`
  - Create: `forward/src/assembly.cpp`

  `assemble_residual()` — accumulates element contributions into global residual.

  - [ ] Step 1: Implement assembly — element-local GLL → global residual via gll_to_global
  - [ ] Step 2: Write Catch2 tests — verify global residual shape, element contribution scatter
  - [ ] Step 3: Commit

  **Commit**: YES
  - Message: `feat(forward): assembly of global residual`

---

- [ ] 5. Viscoelastic SLS Update — DEFERRED

  SLS attenuation deferred to future work. Elastic-only for initial milestone.

  **Files:**
  - Create: `forward/include/gf/viscoelastic.hpp`
  - Create: `forward/src/viscoelastic.cpp`

  - [ ] (Future) Implement SLS memory variable update
  - [ ] (Future) Write Catch2 tests for viscoelastic stress relaxation

  **Commit**: NO (placeholder only, not compiled)

---

- [ ] 6. C-PML Recursive Convolution

  **Files:**
  - Create: `forward/include/gf/pml.hpp`
  - Create: `forward/src/pml.cpp`

  Implements the second-order recursive convolution C-PML scheme
  (Wang et al. 2006, Xie et al. 2014) matching SPECFEM3D.

  All profiles (d_x/K_x/α_x) and convolution coefficients (α_c, β_c, ā, strain)
  are precomputed by the preprocessor and read from `partition_{r}.h5`.

  Two runtime arrays per CPML element, in addition to the 39-index `rmemory`:

  ```
  rmemory[n_elem_local, NGLL, NGLL, NGLL, 39]    // 39 scalars per GLL node per CPML element
  PML_displ_old[NDIM, NGLL, NGLL, NGLL, n_elem_local]  // PML displ at t_{n-1}
  PML_displ_new[NDIM, NGLL, NGLL, NGLL, n_elem_local]  // PML displ at t_n
  ```

  > **See design doc** `docs/superpowers/design/forward.md` for:
  > - Exact damping profile formulas (d_x = -((NPOWER+1)·vp_max·log(Rcoef) / (2·width)) · dist^(1.2) etc.)
  > - 39-index flat mapping: indices 0-26 (9 arrays × 3 time levels) and 27-38 (12 arrays × 1 time level)
  > - Active directions per element type (face/edge/corner)
  > - Convolution coefficient formulas (compute_convolution_coef, l_parameter_computation)
  > - Contains block (end of file) with helper subroutines

  - [ ] Step 1: Declare CPML runtime arrays in `pml.hpp` — `rmemory`, `PML_displ_old`, `PML_displ_new`, `cpml_type` (face/edge/corner tag per element, from partition data)
  - [ ] Step 2: Implement `pml_compute_memory_variables(ispec, ispec_CPML, ...)`:
    Reads precomputed conv_coef_alpha[9], conv_coef_beta[9], conv_coef_strain[18]
    per GLL node. Computes modified displacement gradients using second-order
    recursive convolution. Only active directions per cpml_type are computed.
  - [ ] Step 3: Implement `pml_compute_accel_contribution(ispec, ispec_CPML, ...)`:
    Reads conv_coef_abar[5] and PML_displ_old/PML_displ_new.
    Computes C-PML correction to acceleration.
  - [ ] Step 4: Write Catch2 tests — synthetic CPML element with known profile,
    run one convolution step, verify memory variables against hand-calculated values.
    Test face/edge/corner element types.
  - [ ] Step 5: Commit

  **Commit**: YES
  - Message: `feat(forward): C-PML second-order recursive convolution with 39 memory variables`

---

- [ ] 7. Newmark Explicit Time Integration

  **Files:**
  - Create: `forward/include/gf/newmark.hpp`
  - Create: `forward/src/newmark.cpp`

  Second-order explicit Newmark predictor-corrector (β=0, γ=½ — central difference).

  - [ ] Step 1: Implement `NewmarkPredictor` — `ũ = u + dt·v + (dt²/2)·a`, `ṽ = v + dt·(1-γ)·a`
  - [ ] Step 2: Implement `NewmarkCorrector` — `a_new = M⁻¹·r` (lumped mass), `v = ṽ + dt·γ·a_new`, `u = ũ`
  - [ ] Step 3: Write Catch2 tests — predictor for constant acceleration, corrector with lumped mass, energy-conserving for undamped free vibration
  - [ ] Step 4: Commit

  **Commit**: YES
  - Message: `feat(forward): Newmark explicit predictor-corrector (β=0, γ=½)`

---

- [ ] 8. Source Injection

  **Files:**
  - Create: `forward/include/gf/source.hpp`
  - Create: `forward/src/source.cpp`

  Single point force source. Position, direction, and STF(t) array from config.h5.

  - [ ] Step 1: Implement `PointForceSource` — precomputed element list + Lagrange weights from config.h5, distribute STF[t] × w_ijk via gll_to_global. No runtime element search.
  - [ ] Step 2: Write Catch2 tests — source at GLL node (delta), source at element center (symmetric), verify total force conserved
  - [ ] Step 3: Commit

  **Commit**: YES
  - Message: `feat(forward): source injection with precomputed weights`

---

- [ ] 9. MPI Halo Exchange

  **Files:**
  - Create: `forward/include/gf/exchange.hpp`
  - Create: `forward/src/exchange.cpp`

  MPI halo exchange using precomputed face-pair patterns from partition_{r}.h5 `/partition/exchange/`.

  - [ ] Step 1: Implement exchange — pack face GLL node values → MPI send → recv ghost values → unpack (matching SPECFEM3D exchange pattern)
  - [ ] Step 2: Write Catch2 test (multi-rank) — create 2-rank partition, verify ghost node values match owner's values after exchange
  - [ ] Step 3: Commit

  **Commit**: YES
  - Message: `feat(forward): MPI halo exchange with precomputed face pairs`

---

- [ ] 10. I/O — partition_{r}.h5 and config.h5 Reader

  **Files:**
  - Create: `forward/include/gf/io.hpp`
  - Create: `forward/src/io.cpp`

  Read partition_{r}.h5 (topology + field/element + partition/exchange) and config.h5 (simulation + domain + source). Each rank opens `partitions/partition_{R}.h5`.

  - [ ] Step 1: Implement `read_partition()` — read `/topology/*`, `/field/element/*`, `/partition/*` → populate `RankData`
  - [ ] Step 2: Implement `read_config()` — read `/simulation/`, `/domain/`, `/source/` from config.h5. Direction from CLI `--direction` flag.
  - [ ] Step 3: Write Catch2 tests — synthetic partition_{r}.h5 + config.h5, verify round-trip
  - [ ] Step 4: Commit

  **Commit**: YES
  - Message: `feat(forward): partition_{r}.h5 and config.h5 reader`

---