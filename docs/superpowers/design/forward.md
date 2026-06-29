# Forward Solver — Technical Design

> Parent: [docs/design-decisions.md](../../design-decisions.md)
> Implementation plan: [docs/superpowers/plans/2026-06-08-forward.md](../plans/2026-06-08-forward.md)

## Goal

`libgf` C++ physics library plus MPI `gf_solver` for elastic SEM forward modeling.

## Data Flow

```
config.h5 (single, rank-invariant: simulation + domain + source)
partitions/partition_{r}.h5 (local subset per rank: topology + field/element + cpml + partition metadata)
          │
          ▼
    gf_solver --direction {x,y,z}  (MPI-parallel)
    ├── parse --direction CLI flag
    ├── Each rank reads partitions/partition_{R}.h5 where R = MPI_Comm_rank()
    ├── All ranks read config.h5 (same file, rank-invariant)
    ├── allocate runtime arrays per rank
    │       global residual r[NDIM, n_global_nodes]    — CG-SEM assembly target
    │       C-PML memory variables (see CPML section for exact layout)
    ├── Newmark time loop
    │   ├── NEWMARK PREDICT: ũ = u + solver_dt·v + (solver_dt²/2)·(1-2β)·a
    │   ├── Zero global residual: r[:, :] = 0
    │   ├── Element residual (matrix-free, accumulate into r via gll_to_global)
    │   ├── C-PML memory variable update + acceleration correction
    │   ├── Source injection (distribute STF[t] × w_ijk via precomputed weights)
    │   ├── MPI halo exchange (precomputed face-pair lists)
    │   ├── NEWMARK CORRECT: a_new = M⁻¹·r, v, u update
    │   ├── L2 strain smoothing — compute ε_elem from ∇u_new, project via M⁻¹Σ_e∫N·ε_elem dΩ
    │   ├── Compute ε_smooth from L2 projection (second element pass)
    │   ├── Write shallow mesh-vertex strain record when step % snapshot_stride == 0
    │   └── Overwrite full-volume restart when step % restart_stride == 0
    │
    ├── wavefields/{direction}/record_{r}.h5  (extendible shallow mesh-vertex strain)
    └── restart/{direction}/restart_{r}.h5    (latest-only full-volume restart)
```

### CLI

```
mpirun -np N gf_solver --direction {x,y,z}
```

All paths are fixed relative to CWD:

- Input: `config.h5`, `partitions/partition_{r}.h5`
- Strain output: `wavefields/{direction}/record_{r}.h5`
- Restart output: `restart/{direction}/restart_{r}.h5`

| Arg | Description |
|-----|-------------|
| `--direction {x,y,z}` | force direction (x, y, or z) |
Caller creates directories.

## Architecture

`libgf` is a static physics library linked into MPI executable `gf_solver`.

**Matrix-free assembly**: no global matrix. Elements add `r = Σ Bᵀ_e · σ_e` into `r[NDIM, n_global_nodes]` through `gll_to_global`. Shared nodes sum by using the same global ID.

Preprocess writes all mesh data to per-rank partitions. Rank `R` reads `partitions/partition_{R}.h5`. No global model file. No geometry recompute.

**Source injection**: read source elements, weights, and `STF[n]` from `config.h5`. Distribute to GLL nodes. No runtime search.

**C-PML**: read precomputed damping, stretch, and convolution coefficients from `partition_{r}.h5`. Keep all CPML memory variables. Use Wang et al. (2006), θ=1/8.

**Partition discovery**: rank `R` opens `partitions/partition_{R}.h5`. All ranks read same `config.h5`.

**Force direction**: pass `--direction {x,y,z}`. It is not in `config.h5`. Three jobs share one config.

**Parallelism**: pure MPI, one rank per core. GPU element residual (CUDA) works alongside MPI — only the element kernel runs on GPU; residual is copied back to CPU for `exchange_halo`. See [`gpu.md`](gpu.md). No OpenMP.

## Technology

- C++17, CMake
- MPI (OpenMPI/MPICH)
- Eigen3 — small matrices (3×3 for vectors, up to NGLL×NGLL for derivative matrices)
- HDF5 (C API) — read partition\_{r}.h5 + config.h5, write strain record and restart files
- Catch2 — testing

## Input Files

### partition\_{r}.h5

Per-rank file from preprocess. Each rank reads only its file. Contains local element data and partition metadata.

**Local element data** (layout `[n_elem_local, NGLL, NGLL, NGLL, ...]`, NGLL = N+1):

| Group | Content |
|-------|---------|
| `/topology/` | Element connectivity for local + ghost elements (X2Y, 1-based, signed direction) |
| `/field/element/coords` | GLL node (x,y,z) — physical domain, local elements only |
| `/field/element/jacobian` | det(J) — integration factor |
| `/field/element/dxi_dx` | ∂ξ_i/∂x_j — stiffness computation + strain |
| `/field/element/mass` | Lumped mass diagonal — Newmark solve |
| `/field/element/vp, vs, density` | Elastic constants per GLL node |
| `/field/element/cpml/*` | All C-PML arrays: cpml_type, d_x/y/z, K_x/y/z, alpha_x/y/z, convolution coefficients (conv_coef_alpha, beta, abar), element type tags (face/edge/corner) |
| `/field/surface/boundary_tag` | 0=interior, 1=free surface, 2=absorbing |
**Partition metadata** (`/partition/`):

| Group | Content |
|-------|---------|
| `/partition/n_ranks` | attr int32 — total number of MPI ranks |
| `/partition/element_to_rank` | int64[n_elem_total] — rank assignment for every element |
| `/partition/local_element_ids` | int64[n_elem_local] — owned element IDs (1-based) |
| `/partition/ghost_element_ids` | int64[n_ghost_elem] — halo element IDs (1-based) |
| `/partition/ghost_owners` | int32[n_ghost_elem] — source rank for each ghost element |
| `/partition/gll_to_global` | int64[n_elem_local, NGLL, NGLL, NGLL] — global GLL node ID per local element (1-based, 0=null) |
| `/recording/vertex_ids` | int64[n_record_vertices] — global mesh vertex IDs to write from this rank |
| `/recording/source_element_local_index` | int32[n_record_vertices] — local source element for each recorded vertex |
| `/recording/source_corner_index` | int32[n_record_vertices] — SEM corner index for each recorded vertex |
| `/partition/rank_{r}/exchange/` | Per-rank exchange patterns (precomputed face-pair lists for MPI halo) |
Note: forward solver reads from `partition_{r}.h5`, not a global model file. Each MPI rank opens and reads only its own `partition_{r}.h5` at startup. The preprocessor generates these per-rank files from the global mesh.

### config.h5

Read from [`preprocess.md`](preprocess.md):

| Group | Used By Forward |
|-------|----------------|
| `/simulation/` | solver_dt, output_dt_s, snapshot_stride, restart_dt_s, restart_stride, nsteps, cfl_safety, snapshot_precision, record_depth_max_m, record_depth_actual_m, nx_elements, ny_elements, nz_elements, pml_{x,y,z}{min,max}, tilex_elements, tiley_elements |
| `/domain/` | Bounds, pml_thickness per face |
| `/source/` | Position (x,y,z), stf[nsteps] (precomputed time series), precomputed element list + Lagrange weights |
No `/attenuation/` — elastic-only, attenuation deferred. No `direction` — passed via CLI `--direction` flag.

## Physics Components

| Component | Responsibility |
|-----------|---------------|
| **gll** | GLL points/weights, Lagrange basis, derivative matrix (header-only, N-dependent) |
| **element** | Matrix-free K_e·u: stiffness × displacement using precomputed dξ/dx and detJ. Accumulates into global residual via gll_to_global |
| **assembly** | `assemble_residual()` zeros global r, calls element loop, handles within-rank accumulation |
| **cpml** | C-PML memory variable update + acceleration correction. Second-order recursive convolution (Wang et al. 2006, θ=1/8). 39 scalar memory values per GLL node per CPML element (see CPML Memory Variables) |
| **newmark** | NewmarkPredictor, NewmarkCorrector (2nd order explicit, β=0, γ=½) |
| **source** | Reads precomputed element list + Lagrange weights from config.h5. Distributes STF(t) × w_ijk to global residual |
| **exchange** | MPI halo exchange using precomputed face-pair lists from /partition/rank\_{r}/exchange/ |
| **record/snapshot** | L2 strain smoothing (global projection, C⁰ continuous ε_smooth) + writer: append shallow mesh-vertex strain to extendible HDF5 dataset at snapshot steps |
| **solver** | `run_forward()` main time loop; shallow strain output + latest-only restart/resume |

## Core Types

```
using Vec3  = Eigen::Vector3d;
using Mat33 = Eigen::Matrix3d;
using Mat93 = Eigen::Matrix<double, 9, 3>;  // ∂x/∂ξ (9 partials as 9×3)

struct GLLQuad {
    int N;
    std::vector<double> points;      // N+1
    std::vector<double> weights;     // N+1
    std::vector<double> derivatives; // (N+1)×(N+1) flattened
};

// Per-rank data (subset of partition_{r}.h5)
struct RankData {
    int n_local_elem, n_ghost_elem, n_total_elem;
    int64_t n_global_nodes;                   // unique GLL nodes on this rank
    std::vector<int64_t> local_element_ids;   // owned
    std::vector<int64_t> ghost_element_ids;   // halo
    std::vector<int32_t> ghost_owners;        // which rank owns each ghost

    // Global GLL node numbering: [n_elem_total, NGLL, NGLL, NGLL]
    // gll_to_global[e][i][j][k] = global node ID (1-based, 0=null)
    std::vector<int64_t> gll_to_global;

    // Precomputed fields at GLL nodes (coords, jacobian, dxi_dx, mass, material)
    // C-PML arrays: cpml_type, d_x/y/z, K_x/y/z, alpha_x/y/z, conv coefficients
    // Precomputed exchange patterns
    // ...
};
```

## Material at GLL Nodes

Material is stored directly at GLL nodes in partition\_{r}.h5 — no runtime interpolation needed. Forward reads `[n_elem, NGLL, NGLL, NGLL]` arrays for vp, vs, density, etc. directly.

## Source Injection

Single point force source on the free surface (z = z_min, top of domain). Precomputed by the preprocessor:

- Element list + natural coordinates (ξ_s, η_s, ζ_s) + Lagrange weights w_ijk stored in config.h5 `/source/elements/`
- Forward solver reads these at startup — no runtime Newton iteration or element search
- At each timestep, source injects into the global residual: `r(iglob) += STF(t) × w_ijk × direction_vector`
  where `direction_vector` = (1,0,0), (0,1,0), or (0,0,1) depending on the `--direction` CLI flag

If the source lies on a shared face/edge/vertex on the free surface, all sharing elements are included
in the precomputed element list — the preprocessor handles this during source location.

## CPML Memory Variables

The C-PML implementation follows the second-order recursive convolution scheme
of Wang et al. (2006), equation (21), with parameter θ = 1/8.

### Damping Profile Formulas (from SPECFEM3D)

Three damping-related quantities are computed per GLL node per CPML element,
precomputed by the preprocessor and stored in `partition_{r}.h5`.

**Normalized distance** (for a GLL node in the x-direction PML layer):

```
abscissa_in_PML = |x_node - x_interface|    (distance from PML/interior interface)
dist = abscissa_in_PML / CPML_width_x       (normalized to [0, 1])
```

where `x_interface` is the boundary between interior and PML region
(not the domain boundary). `CPML_width_x` is the maximum physical PML
thickness across all MPI ranks.

**Damping profile** `d_x` (from SPECFEM3D `pml_damping_profile_l`):

```
d_x = -((NPOWER + 1) · vp_max · log(CPML_Rcoef) / (2 · CPML_width_x)) · dist^(1.2 · NPOWER)
```

where `NPOWER = 1`, `CPML_Rcoef = 0.001`. The P-wave speed `vp_max` is the
maximum vp across all CPML elements (constant for all nodes). The same formula
applies for `d_y` and `d_z` with their respective widths.

Reference: INRIA research report RR-3471 section 6.1,
<http://hal.inria.fr/docs/00/07/32/19/PDF/RR-3471.pdf>

**Stretched-coordinate factor** `K_x`:

```
K_x = K_MIN_PML + (K_MAX_PML - K_MIN_PML) · dist
```

with `K_MIN_PML = K_MAX_PML = 1` (constant K = 1 everywhere — no stretching).

**Shifted frequency-domain pole** `α_x`:

```
α_x = ALPHA_MAX_PML_x · (1 - dist)
ALPHA_MAX_PML_x = π · f0_FOR_PML · 0.9
ALPHA_MAX_PML_y = π · f0_FOR_PML · 1.0
ALPHA_MAX_PML_z = π · f0_FOR_PML · 1.1
```

where `f0_FOR_PML` is the dominant source frequency
(derived from the STF's central frequency). The asymmetry
`α_x < α_y < α_z` helps avoid singularities in the PML parameter
separation (Festa & Vilotte 2005).

### Convolution Coefficients (from SPECFEM3D)

The second-order recursive convolution scheme (Xie et al. 2014, eq. 60, D6a-D6b)
requires precomputed coefficients. Three sets are precomputed per GLL node:

**β (auxiliary decay rate)** per direction:

```
β_x = α_x + d_x / K_x
β_y = α_y + d_y / K_y
β_z = α_z + d_z / K_z
```

**Recursive convolution coefficients** `compute_convolution_coef(b) → (coef0, coef1, coef2)`:

```
temp = exp(-½ · b · Δt)

coef0 = temp · temp     (= exp(-b · Δt))

Second-order scheme:
  coef1 = (1 - temp) / b       if |b| ≥ ε
  coef2 = coef1 · temp

  Small-b approximation (|b| < ε, Taylor to 3rd order):
  coef1 = Δt/2 + (-1/8 · Δt² · b + 1/48 · Δt³ · b² - 1/384 · Δt⁴ · b³)
  coef2 = Δt/2 + (-3/8 · Δt² · b + 7/48 · Δt³ · b² - 5/128 · Δt⁴ · b³)
```

These coefficients are computed for both `α_x,y,z` and `β_x,y,z`, producing
9 values each for `pml_convolution_coef_alpha` and `pml_convolution_coef_beta`.

**Accel-update coefficients** `Ā` (from `l_parameter_computation`):
See Appendix A1 of Xie et al. (2014). Yields 5 coefficients `Ā₁…Ā₅`
stored in `pml_convolution_coef_abar[5, NGLL, NGLL, NGLL, NSPEC_CPML]`.

**Strain-update coefficients** A₆…A₁₇ (from `lijk_parameter_computation`):
Permuted for 3 index orderings (123, 132, 231), yielding 12 coefficients
(4 per ordering) stored in `pml_convolution_coef_strain[18, NGLL, NGLL, NGLL, NSPEC_CPML]`.
The 18 slots: indices 1-4 = ordering 231, 5-8 = ordering 132, 9-12 = ordering 123,
13-14 = s_x, 15 = s_y, 16-17 = s_z, 18 = η (see SPECFEM3D `prepare_timerun.F90`).

### Precomputed CPML Arrays in partition\_{r}.h5

| Array | Shape | Description |
|-------|-------|-------------|
| cpml_type | int8[n_elem_local] | 1=face, 2=edge, 3=corner |
| d_x, d_y, d_z | float64[n_elem_local, NGLL, NGLL, NGLL] | Damping profiles |
| K_x, K_y, K_z | float64[n_elem_local, NGLL, NGLL, NGLL] | K=1 everywhere (no stretch) |
| alpha_x, alpha_y, alpha_z | float64[n_elem_local, NGLL, NGLL, NGLL] | Frequency-shift profiles |
| conv_coef_alpha | float64[9, n_elem_local, NGLL, NGLL, NGLL] | coef0,1,2 for α_x, α_y, α_z |
| conv_coef_beta | float64[9, n_elem_local, NGLL, NGLL, NGLL] | coef0,1,2 for β_x, β_y, β_z |
| conv_coef_abar | float64[5, n_elem_local, NGLL, NGLL, NGLL] | Ā₁…Ā₅ for accel update |
| conv_coef_strain | float64[18, n_elem_local, NGLL, NGLL, NGLL] | A₆…A₁₇ for strain update |
**Memory variable layout:** 21 rmemory arrays total per CPML element,
organized as 3 PML directions (x, y, z) × 7 arrays:

| Direction | Arrays | Description |
|-----------|--------|-------------|
| x | 7 | Memory variables for ∂/∂x displacement gradient |
| y | 7 | Memory variables for ∂/∂y displacement gradient |
| z | 7 | Memory variables for ∂/∂z displacement gradient |
**Time-level storage:** 9 of the 21 arrays require 3 time levels
(second-order convolution requires the previous two timesteps),
while the remaining 12 arrays require only 1 time level.

**Effective storage:** 39 scalar values per GLL node per CPML element:

- 9 arrays × 3 time levels = 27 scalars
- 12 arrays × 1 time level = 12 scalars
- Total = 39 scalars

**Implementation recommendation:** flatten all rmemory data into a single
5D array (matching SPECFEM3D indexing):

```
rmemory[NSPEC_CPML, NGLL, NGLL, NGLL, 39]
```

**Flat index layout (index 0–38), derived from SPECFEM3D allocation order:**

**Indices 0–26: arrays requiring 3 time levels** (9 arrays × 3 = 27 scalars).
These are the arrays where the derivative's primary direction matches the PML direction:

| Base | Array | PML Dir | Time level offsets |
|------|-------|---------|--------------------|
| 0 | rmemory_dux_dxl | x | 0=n-1, 1=n, 2=n+1 |
| 3 | rmemory_dux_dyl | x | 0=n-1, 1=n, 2=n+1 |
| 6 | rmemory_dux_dzl | x | 0=n-1, 1=n, 2=n+1 |
| 9 | rmemory_duy_dxl | y | 0=n-1, 1=n, 2=n+1 |
| 12 | rmemory_duy_dyl | y | 0=n-1, 1=n, 2=n+1 |
| 15 | rmemory_duy_dzl | y | 0=n-1, 1=n, 2=n+1 |
| 18 | rmemory_duz_dxl | z | 0=n-1, 1=n, 2=n+1 |
| 21 | rmemory_duz_dyl | z | 0=n-1, 1=n, 2=n+1 |
| 24 | rmemory_duz_dzl | z | 0=n-1, 1=n, 2=n+1 |
Access pattern: `rmemory[e][i][j][k][base + t]` where `t ∈ {0,1,2}`.

**Indices 27–38: arrays requiring 1 time level** (12 arrays × 1 = 12 scalars):

| Index | Array | PML Dir |
|-------|-------|---------|
| 27 | rmemory_duy_dxl | x |
| 28 | rmemory_duy_dyl | x |
| 29 | rmemory_duz_dxl | x |
| 30 | rmemory_duz_dzl | x |
| 31 | rmemory_dux_dxl | y |
| 32 | rmemory_dux_dyl | y |
| 33 | rmemory_duz_dyl | y |
| 34 | rmemory_duz_dzl | y |
| 35 | rmemory_dux_dxl | z |
| 36 | rmemory_dux_dzl | z |
| 37 | rmemory_duy_dyl | z |
| 38 | rmemory_duy_dzl | z |
**Active directions per element type:**

| cpml_type | Active PML directions | Active memory variables |
|-----------|----------------------|------------------------|
| 1 (face) | 1 direction | 7 arrays × (3 or 1 levels) for that direction only |
| 2 (edge) | 2 directions | 14 arrays across 2 directions (face count × 2) |
| 3 (corner) | 3 directions | All 21 arrays (full 39 scalars) |
For face/edge elements, the inactive direction's memory variables are simply
never updated (remain zero), using `cpml_type` to skip computation per element.

**Additional PML state arrays (stored separately, NOT in rmemory):**

```
PML_displ_old[NDIM, NGLL, NGLL, NGLL, NSPEC_CPML]
PML_displ_new[NDIM, NGLL, NGLL, NGLL, NSPEC_CPML]
```

**Additional PML state arrays:**

```
PML_displ_old[NDIM, NGLL, NGLL, NGLL, NSPEC_CPML]
PML_displ_new[NDIM, NGLL, NGLL, NGLL, NSPEC_CPML]
```

These hold displacement fields specific to PML elements at the old and new
time levels for the convolution update.

**Precomputed coefficients in partition\_{r}.h5 (`/field/element/cpml/`):**

| Array | Description |
|-------|-------------|
| cpml_type | Element classification (0=interior, 1=face-x, 2=face-y, 3=face-z, 4=edge, 5=corner) |
| d_x, d_y, d_z | Damping profiles per GLL node |
| K_x, K_y, K_z | Stretched-coordinate metric factors |
| alpha_x, alpha_y, alpha_z | Shifted frequency-domain pole |
| conv_coef_alpha | Convolution coefficient α for recursive update |
| beta | Convolution coefficient β |
| abar | Convolution coefficient a̅ (reduced) |

## Runtime Loop

````
for step in 0..nsteps-1:
    1. Newmark predict:
       ũ = u + solver_dt·v + (solver_dt²/2)·(1-2β)·a   (β=0 for explicit central difference)
       ṽ = v + solver_dt·(1-γ)·a                  (γ=½)

    2. Zero global residual:
       r[1..NDIM, 1..n_global_nodes] = 0

    3. Element stiffness (matrix-free K·u for local + ghost elements):
       For each element e:
         For each GLL node (i,j,k):
           iglob = gll_to_global[e][i][j][k]       ← global node ID
           Compute ∇ũ via GLL derivatives × dξ/dx
           ε = ½(∇ũ + ∇ũᵀ)
           σ = C:ε  (elastic)
           Accumulate: r(:, iglob) -= Bᵀ·σ·detJ   ← element contribution to global residual
         End
       End
       // Within-rank shared nodes are implicitly summed via accumulation into r

    4. C-PML update:
       For each CPML element:
         Update memory variables (exact layout below)
         Compute C-PML acceleration correction
         Accumulate correction into global residual r at CPML element GLL nodes

    5. Source injection:
       For each source element e in precomputed list:
         For each GLL node (i,j,k):
           iglob = gll_to_global[e][i][j][k]
           r(d, iglob) += STF(t) × w_ijk × direction_vector[d]   ← d=1,2,3

    6. MPI halo exchange:
       For each neighbor rank:
         Pack r values at interface GLL nodes → send
         Recv r values for ghost GLL nodes → unpack
       Sum shared GLL node contributions (standard CG-SEM across ranks)

    7. Newmark correct:
       a_new = M⁻¹·r              (lumped mass, pointwise division per GLL node)
       v = ṽ + solver_dt·γ·a_new
       u = ũ + solver_dt²·β·a_new

    8. L2 strain smoothing (snapshot timesteps only):
       Compute ∇u from corrected u (element pass):
         For each local element e, for each GLL node (i,j,k):
           ∇u = GLL derivatives[e] × dxi_dx[e][i][j][k]
           ε_elem = ½(∇u + ∇uᵀ)   (6-vector per GLL node per element)
       Global L2 projection onto C⁰-continuous basis.

       **L2 projection algorithm (element-loop form):**

       The projection assembles a weighted strain vector `s_global` at global
       GLL nodes, then applies the lumped mass inverse:

       ```
       // Phase A: Assemble weighted strain
       s_global[6, n_global_nodes] = 0              // 6 strain components

       for each local element e:
         iglob = gll_to_global[e][i][j][k]          // for each GLL node (i,j,k)
         detJ   = jacobian_store[e][i][j][k]
         weight = detJ * w_GLL[i] * w_GLL[j] * w_GLL[k]   // GLL quadrature weight

         for comp = 0..5:
           s_global[comp][iglob] += weight * ε_elem[e][i][j][k][comp]

       // Phase B: Apply lumped mass inverse
       for each local element e:
         for each GLL node (i,j,k):
           iglob = gll_to_global[e][i][j][k]

           for comp = 0..5:
             ε_smooth[comp][i][j][k] = s_global[comp][iglob] / mass[e][i][j][k]
       ```

       The operation `∫ N_α · ε_elem dΩ` simplifies at GLL nodes because
       the Lagrange basis satisfies `N_α(ξ_i, η_j, ζ_k) = δ_αi · δ_αj · δ_αk`
       — the quadrature weight at node (i,j,k) directly scales the
       element-wise strain value. The assembly loop accumulates weighted
       strain into the shared global node. Since within-rank shared nodes
       are implicitly summed (standard CG-SEM exchange already handles
       contributions from adjacent elements sharing the same GLL node),
       the lumped mass division produces the C⁰-continuous projection.

       MPI ghost nodes in `s_global`: after Phase A, the MPI halo exchange
       (same face-pair lists used for residual assembly) sums s_global
       contributions at shared GLL nodes across rank boundaries. This
       ensures ε_smooth is continuous across the entire mesh, not just
       within a rank.

    9. Strain snapshot (when `step % snapshot_stride == 0`):
       Use ε_smooth from the L2 projection (not raw ε_elem).
       Sample only preprocessing-selected mesh vertices (`/recording/` map):
       non-PML, depth <= record_depth_actual_m, mesh corners only.
       Fast path: only compute strain at recorded GLL corner nodes,
       skipping the full NGLL³ interior for elements with no recorded vertices.
       Append to wavefields/{direction}/record_{r}.h5
       (6 components: εxx, εyy, εzz, εxy, εxz, εyz)

    10. Restart overwrite (when `step % restart_stride == 0`):
        Write all full-volume state required for exact resume.
````

## Snapshot Output

One strain record per rank per run. Append only at `step % snapshot_stride == 0`. Output is shallow mesh vertices, not full GLL:

```
wavefields/{direction}/record_{r}.h5
├── attrs:
│   ├── rank                    : int32
│   ├── source_direction        : string
│   ├── basis                   : "mesh_vertices"
│   ├── record_depth_max_m      : float64
│   ├── record_depth_actual_m   : float64
│   └── excludes_pml            : bool
├── vertex_ids                  : int64[n_record_vertices]
└── strain                      : float32[n_snapshots, n_record_vertices, 6]
```

Values are L2-smoothed strain at selected SEM corner nodes. Interior GLL points are not recorded.

## Restart Output

Restart is separate and latest-only:

```
restart/{direction}/restart_{r}.h5
├── attrs:
│   ├── rank                    : int32
│   ├── source_direction        : string
│   ├── step                    : int32
│   ├── time_s                  : float64
│   └── ngll                    : int32
├── displacement                : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
├── velocity                    : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
├── acceleration                : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
└── pml_memory_*                : float64[...]  # all active C-PML memory arrays
```

With `--resume`, `gf_solver` restores state and continues at `step + 1`.

## Discretization Parameters

- Polynomial order: N=3 (testing), N=5 (production)
- NGLL = N+1 = 4 (test) / 6 (prod)
- Time integration: Newmark explicit (β=0, γ=½ — central difference)
- CFL: conditional stability, standard SEM constraint
- `solver_dt`: from preprocess CFL logic. Snapshots use `snapshot_stride`/`output_dt_s`. Restarts use `restart_stride`/`restart_dt_s`.

## Run Config (3 Simulations per Source)

Green extraction uses 3 runs per source: force x, y, z. Each run calls `gf_solver --direction ...` and writes 6 strain components.

## Namespace

`gf`

## File Layout

```
forward/
├── CMakeLists.txt
├── include/gf/
│   ├── types.hpp              — Vec3, Mat33, GLLQuad, RankData
│   ├── gll.hpp                — GLL quadrature (header-only)
│   ├── element.hpp            — backend-templated matrix-free stiffness × displacement
│   ├── backend.hpp            — BackendCPU, BackendCUDA, ActiveBackend tags
│   ├── cuda_check.h           — GF_CUDA_CHECK() error macro
│   ├── cuda_device_manager.hpp— persistent CUDA device buffer manager
│   ├── assembly.hpp           — assemble_residual(), add_source_to_rhs()
│   ├── pml.hpp                — PML damping application
│   ├── newmark.hpp            — NewmarkPredictor, NewmarkCorrector
│   ├── source.hpp             — PointForceSource (Lagrange interpolation)
│   ├── exchange.hpp           — MPI halo exchange (precomputed face lists)
│   ├── record.hpp             — shallow-vertex strain HDF5 writer
│   ├── restart.hpp            — restart/resume full-volume HDF5 writer/reader
│   ├── io.hpp                 — partition_{r}.h5 + config.h5 reader
│   └── solver.hpp             — run_forward() loop, --resume support
├── src/
│   ├── element_cpu.cpp, element_cuda.cu, assembly.cpp
│   ├── pml.cpp, newmark.cpp, source.cpp
│   ├── exchange.cpp, record.cpp, restart.cpp, io.cpp
│   ├── solver.cpp
│   └── main.cpp               — gf_solver entry point
└── tests/
    ├── CMakeLists.txt
    ├── test_gll.cpp, test_element.cpp, test_element_cuda.cu
    ├── test_assembly.cpp
    ├── test_pml.cpp, test_newmark.cpp
    ├── test_source.cpp, test_record.cpp, test_integration.cpp
```
