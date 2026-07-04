# Forward Solver — Technical Design

> Parent: [../design-decisions.md](../design-decisions.md)

## Goal

`libgf` C++ physics library plus 3 solver binaries (`gf_solver_mpi`, `gf_solver_cuda`, `gf_solver_mpi_cuda`) for elastic SEM forward modeling.

## Data Flow

```
config.h5 (single, rank-invariant: simulation + domain + source)
partitions/partition_{r}.h5 (local subset per rank: topology + field/element + cpml + partition metadata)
          │
          ▼
    gf_solver_{mpi,cuda,mpi_cuda} --direction {x,y,z}  (CPU/GPU, MPI optional)
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
    d40|    │   ├── PML damping (linear ramp applied to velocity)
    │   ├── Source injection (distribute STF[t] × w_ijk via precomputed weights)
    │   ├── MPI halo exchange (precomputed face-pair lists)
    │   ├── NEWMARK CORRECT: a_new = M⁻¹·r, v, u update
    1ec|    │   ├── Per-vertex strain — compute ε from ∇u_new via derivative matrix + chain rule
    4f8|    │   ├── Compute per-corner strain at recorded mesh vertices
    │   ├── Write shallow mesh-vertex strain record when step % snapshot_stride == 0
    │   └── Overwrite full-volume restart when step % restart_stride == 0
    │
    ├── wavefields/{direction}/record_{r}.h5  (extendible shallow mesh-vertex strain)
    └── restart/{direction}/restart_{r}.h5    (latest-only full-volume restart)
```

### CLI

```
mpirun -np N bin/gf_solver_mpi --direction {x,y,z}        # CPU + MPI
bin/gf_solver_cuda --direction x                          # single GPU (no MPI)
mpirun -np N bin/gf_solver_mpi_cuda --direction x         # multi-GPU
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

`libgf` is a static physics library linked into solver executables. Three variants:

| Binary | Backend | MPI | Build definition |
|--------|---------|-----|-----------------|
| `gf_solver_mpi` | CPU | yes | default |
| `gf_solver_cuda` | CUDA | no | `GF_WITH_CUDA` + `GF_NO_MPI` |
| `gf_solver_mpi_cuda` | CUDA | yes | `GF_WITH_CUDA` |

GPU auto-detects via `cudaGetDeviceCount()` and assigns `cudaSetDevice(rank % n_devices)`.
When MPI ranks exceed GPUs on a node, excess ranks exit and remaining ranks
redistribute partitions via `read_partition_range()`. Single-rank mode (1 MPI rank or
non-MPI) reads all partitions via `read_partition_all()` — no repartitioning needed.

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
| `/field/element/vp, vs, density, lambda, mu` | Elastic constants per GLL node (vp, vs, density from config; lambda, mu precomputed) |
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
| `/partition/exchange/neighbor_{N}/` | Per-rank exchange patterns (precomputed face-pair lists for MPI halo) |
Note: forward solver reads from `partition_{r}.h5`, not a global model file. Each MPI rank opens and reads only its own `partition_{r}.h5` at startup. The preprocessor generates these per-rank files from the global mesh.

### config.h5

Read from [`preprocess.md`](preprocess.md):

| Group | Used By Forward |
|-------|----------------|
| `/simulation/` | solver_dt, output_dt_s, snapshot_stride, restart_dt_s, restart_stride, log_stride, nsteps, cfl_safety, snapshot_precision, record_depth_max_m, record_depth_actual_m, nx_elements, ny_elements, nz_elements, pml\_{x,y,z}{min,max}, tilex_elements, tiley_elements |
| `/domain/` | Bounds, pml_thickness per face |
| `/source/` | Position (x,y,z), stf[nsteps] (precomputed time series), precomputed element list + Lagrange weights |
No `/attenuation/` — elastic-only, attenuation deferred. No `direction` — passed via CLI `--direction` flag.

## Physics Components

| Component | Responsibility |
|-----------|---------------|
| **gll** | GLL points/weights, Lagrange basis, derivative matrix (header-only, N-dependent) |
| **element** | Matrix-free K_e·u: stiffness × displacement using precomputed dξ/dx and detJ. Accumulates into global residual via gll_to_global |
| **assembly** | `assemble_residual()` zeros global r, calls element loop, handles within-rank accumulation |
| **pml** | Simple linear-ramp PML damping: v ← v - d(node)·v. Full recursive-convolution C-PML (Wang et al. 2006, 39 scalar memory values) is deferred. |
| **newmark** | NewmarkPredictor, NewmarkCorrector (2nd order explicit, β=0, γ=½) |
| **source** | Reads precomputed element list + Lagrange weights from config.h5. Distributes STF(t) × w_ijk to global residual |
| **exchange** | MPI halo exchange using precomputed face-pair lists from /partition/exchange/neighbor\_{N}/ |
| **record/snapshot** | Per-vertex strain at recorded mesh corners (direct gradient from displacement) + writer: append shallow mesh-vertex strain to extendible HDF5 dataset at snapshot steps |
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

Material is stored directly at GLL nodes in partition\_{r}.h5 — no runtime interpolation needed. Forward reads `[n_elem, NGLL, NGLL, NGLL]` arrays for lambda, mu, etc. directly.

## Source Injection

Single point force source on the free surface (z = z_min, top of domain). Precomputed by the preprocessor:

- Element list + natural coordinates (ξ_s, η_s, ζ_s) + Lagrange weights w_ijk stored in config.h5 `/source/elements/`
- Forward solver reads these at startup — no runtime Newton iteration or element search
- At each timestep, source injects into the global residual: `r(iglob) += STF(t) × w_ijk × direction_vector`
  where `direction_vector` = (1,0,0), (0,1,0), or (0,0,1) depending on the `--direction` CLI flag

If the source lies on a shared face/edge/vertex on the free surface, all sharing elements are included
in the precomputed element list — the preprocessor handles this during source location.

## PML Damping

The current implementation uses a simple linear-ramp damping profile applied to the
velocity field. Precomputed by the preprocessor and stored in `partition_{r}.h5`
as a single per-GLL-node damping array:

```
/field/element/damping   float64[n_elem_total, NGLL, NGLL, NGLL]
```

### Update Formula

At each timestep, PML damping is applied to the velocity field after the element
residual computation:

```
v(d, iglob) -= damping[e][i][j][k] * v(d, iglob)
```

All 3 displacement components at a node share the same damping coefficient.
Non-PML elements have damping = 0 everywhere (no effect).

### Damping Profile

For each PML element, the damping coefficient ramps linearly from 0 at the
PML-entry interface (interior edge of the PML layer) to 1 at the physical
domain boundary:

```
ramp = clamp((coord - pml_start) / pml_width, 0.0, 1.0)
damping = ramp
```

### Element Classification

PML elements are tagged by `is_pml` flag (int8) per element, computed during
preprocessing. Layer expansion uses element grid position `(i,j,k)` for structured
hex meshes; unstructured meshes fall back to 1-layer surface detection.

### Deferred: Full C-PML

Full recursive-convolution C-PML (Wang et al. 2006, θ=1/8) with 39 memory
variables per GLL node — matching SPECFEM3D — is documented in the `docs/math.md`
formulation, but not yet implemented. The deferred design includes:

- d/K/α damping profiles per direction
- Second-order convolution coefficients (α_x,y,z, β_x,y,z)
- 21 memory arrays, 39 scalars per GLL node
- Accel-update coefficients Ā₁…Ā₅ (Xie et al. 2014)
- Strain-update coefficients A₆…A₁₇

See [`docs/deferred.md`](../deferred.md) for status.

## Runtime Loop

```
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

    4. PML damping:
       v(d, iglob) -= damping[e][i][j][k] * v(d, iglob)
       Applied to velocity at each GLL node using precomputed damping profile.
       Interior nodes (damping = 0) are unaffected.

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

    8. Strain snapshot (when step % snapshot_stride == 0):
       For each recorded vertex in the recording map:
         iglob = gll_to_global[elem][i][j][k]
         Compute ∇u at that corner via GLL derivative matrix × dxi_dx
         ε = ½(∇u + ∇uᵀ)   (6-component Voigt)
       Append to wavefields/{direction}/record_{r}.h5
       (6 components: εxx, εyy, εzz, εxy, εxz, εyz)
       (6 components: εxx, εyy, εzz, εxy, εxz, εyz)

    10. Restart overwrite (when `step % restart_stride == 0`):
        Write all full-volume state required for exact resume.
```

## Snapshot Output

One record file per rank per run. Append only at `step % snapshot_stride == 0`. Output is shallow mesh vertices, not full GLL:

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
├── strain                      : float32[n_snapshots, n_record_vertices, 6]
├── displacement                : float32[n_snapshots, n_record_vertices, 3]
├── velocity                    : float32[n_snapshots, n_record_vertices, 3]
└── acceleration                : float32[n_snapshots, n_record_vertices, 3]
```

Values are per-vertex (direct gradient from displacement at corrected corner nodes). Interior GLL points are not recorded. Displacement, velocity, acceleration are extracted from the same recorded-vertex set using the recording map.

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
│   └── main.cpp               — solvers entry point
└── tests/
    ├── CMakeLists.txt
    ├── test_gll.cpp, test_element.cpp, test_element_cuda.cu
    ├── test_assembly.cpp
    ├── test_pml.cpp, test_newmark.cpp
    ├── test_source.cpp, test_record.cpp, test_integration.cpp
```
