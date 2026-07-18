# Forward Solver — Technical Design

> Parent: [../design-decisions.md](../design-decisions.md)

## Goal

`libgf` C++ physics library plus 3 solver binaries (`gf_solver_elastic_mpi`, `gf_solver_elastic_cuda`, `gf_solver_elastic_mpi_cuda`) for elastic SEM forward modeling.

## Data Flow

```
config.h5 (single, rank-invariant: simulation + domain + source)
partitions/partition_{r}.h5 (local subset per rank: topology + field/element + PML damping + partition metadata)
          │
          ▼
    gf_solver_{mpi,cuda,mpi_cuda} --direction {x,y,z}  (CPU/GPU, MPI optional)
    ├── parse --direction CLI flag
    ├── Each rank reads partitions/partition_{R}.h5 where R = MPI_Comm_rank()
    ├── All ranks read config.h5 (same file, rank-invariant)
    ├── allocate runtime arrays per rank
    │       two DOF modes (use_global_dof flag):
    │       · global DOF (CG-SEM): rank_node_*[n_rank_node * 3] + elem-local temps
    │       · element-local (legacy): elem_dof[n_elem * n_node * 3]
    │       PML damping array (linear-ramp profile)
    ├── Newmark time loop (global DOF path)
    │   ├── NEWMARK PREDICT: ũ = u + dt·v + (dt²/2)·(1-2β)·a
    │   ├── ũ SYNC (multi-rank): exchange + average at shared interface nodes
    │   ├── GATHER: ũ[rank_node] → elem_ũ[n_elem, n_node, 3] via local_cell2rank_node
    │   ├── Element residual (matrix-free): K_e·elem_ũ → elem_r
    │   ├── PML damping on global velocity
    │   ├── Source injection into elem_r
    │   ├── SCATTER: elem_r → residual[rank_node*3 + d] via local_cell2rank_node
    │   ├── MPI halo exchange on residual (precomputed face-pair lists)
    │   ├── NEWMARK CORRECT: a_new = (r_j+r_k) / (m_j+m_k),
    │   │                    u += dt·v + dt²·(½-β)·a_old + dt²·β·a_new,
    │   │                    v += dt·((1-γ)·a_old + γ·a_new)
    │   ├── Per-vertex strain — compute ε from ∇u via GATHER + derivative matrix
    │   ├── Write shallow mesh-vertex strain record when step % snapshot_stride == 0
    │   └── Overwrite restart when step % restart_stride == 0 (use_global_dof flag)
    │
    ├── wavefields/{direction}/record_{r}_{step}.h5  (one per snapshot)
    └── restart/{direction}/restart_{r}.h5    (latest-only full-volume restart)
```

| `/field/cell/local_cell2global_node` | int64[n_local_cell × NGLL³] — global GLL node IDs (0-based). Written by preprocessor, used by `read_partition_all` for single-rank/GPU merge. Same size as `local_cell2rank_node`. |

### CLI

```
mpirun -np N bin/gf_solver_elastic_mpi --direction {x,y,z}        # CPU + MPI
bin/gf_solver_elastic_cuda --direction x                          # single GPU (no MPI)
mpirun -np N bin/gf_solver_elastic_mpi_cuda --direction x         # multi-GPU
```

All paths are fixed relative to CWD:

- Input: `config.h5`, `partitions/partition_{r}.h5`
- Strain output: `wavefields/{direction}/record_{r}_{step}.h5`
- Restart output: `restart/{direction}/restart_{r}.h5`

| Arg | Description |
|-----|-------------|
| `--direction {x,y,z}` | force direction (x, y, or z) |
Caller creates directories.

## Architecture

`libgf` is a static physics library linked into solver executables. Three variants:

| Binary | Backend | MPI | Build definition |
|--------|---------|-----|-----------------|
| `gf_solver_elastic_mpi` | CPU | yes | default |
| `gf_solver_elastic_cuda` | CUDA | no | `GF_WITH_CUDA` + `GF_NO_MPI` |
| `gf_solver_elastic_mpi_cuda` | CUDA | yes | `GF_WITH_CUDA` |

GPU auto-detects via `cudaGetDeviceCount()` and assigns `cudaSetDevice(rank % n_devices)`.
When MPI ranks exceed GPUs on a node, excess ranks exit and remaining ranks
redistribute partitions via `read_partition_range()`. Single-rank mode (1 MPI rank or
non-MPI) reads all partitions via `read_partition_all()` — no repartitioning needed.

**Matrix-free assembly**: no global matrix. Elements add `r = Σ Bᵀ_e · σ_e` into `r[NDIM, n_global_nodes]` through `local_cell2rank_node`. Shared nodes sum by using the same global ID.

Preprocess writes all mesh data to per-rank partitions. Rank `R` reads `partitions/partition_{R}.h5`. No global model file. No geometry recompute.

**Source injection**: read source elements, weights, and `STF[n]` from `config.h5`. Distribute to GLL nodes. No runtime search.

**PML damping**: simple linear-ramp profile applied to velocity. Profile precomputed by preprocessor, read from `partition_{r}.h5`. Full recursive-convolution C-PML is deferred.

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

**Local element data** (layout `[n_local_cell, NGLL, NGLL, NGLL, ...]`, NGLL = N+1):

| Group | Content |
|-------|---------|
| `/topology/` | Element connectivity for local + ghost elements (X2Y, 1-based, signed direction) |
| `/field/cell/coords` | GLL node (x,y,z) — physical domain, local elements only |
| `/field/cell/jacobian` | det(J) — integration factor |
| `/field/cell/dxi_dx` | ∂ξ_i/∂x_j — stiffness computation + strain |
| `/field/cell/mass` | Lumped mass diagonal — Newmark solve |
| `/field/cell/vp, vs, density, lambda, mu` | Elastic constants per GLL node (vp, vs, density from config; lambda, mu precomputed) |
| `/field/cell/damping` | PML damping profile (linear ramp) |
| `/field/surface/boundary_tag` | 0=interior, 1=free surface, 2=absorbing |
**Partition metadata** (`/partition/`):

| Group | Content |
|-------|---------|
| `/partition/n_ranks` | attr int32 — total number of MPI ranks |
| `/partition/n_local_cell` | attr int32 — number of local elements on this rank |
| `/partition/n_rank_node` | attr int32 — unique GLL nodes on this rank (ibool range) |
| `/partition/use_global_dof` | attr int8 — 1=CG-SEM global DOF, 0=legacy element-local (absent→0) |
| `/partition/local_cell_ids` | int64[n_local_cell] — owned element IDs (1-based) |
| `/partition/ghost_cell_ids` | int64[n_ghost_cellent] — halo element IDs (1-based, absent if none) |
| `/partition/ghost_owners` | int32[n_ghost_cellent] — source rank for each ghost (absent if none) |
| `/field/cell/local_cell2rank_node` | int64[n_local_cell × NGLL³] — flat ibool: per-rank GLL→node map (absent→legacy) |
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
| **element** | Matrix-free K_e·ũ: stiffness × displacement using precomputed dξ/dx and detJ. Reads/writes element-local temp arrays (gathered/scattered by assembly). |
| **assembly** | `gather_from_rank()` / `scatter_to_rank()` via `local_cell2rank_node`. Connects element-local temp arrays to rank-level global state vectors. |
| **pml** | Simple linear-ramp PML damping: v ← v - d(node)·v. Full recursive-convolution C-PML deferred |
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
    int n_local_cell, n_ghost_cell, n_total_cell;
    int64_t n_global_nodes;                   // unique GLL nodes on this rank
    std::vector<int64_t> local_cell_ids;   // owned
    std::vector<int64_t> ghost_cell_ids;   // halo
    std::vector<int32_t> ghost_owners;        // which rank owns each ghost

    // Global GLL node numbering: [n_elem_total, NGLL, NGLL, NGLL]
    // local_cell2rank_node[e][i][j][k] = global node ID (1-based, 0=null)
    std::vector<int64_t> local_cell2rank_node;

    // Precomputed fields at GLL nodes (coords, jacobian, dxi_dx, mass, material)
    // PML damping profile
    // Precomputed exchange patterns
    // ...
};
```

## Material at GLL Nodes

Material is stored directly at GLL nodes in partition\_{r}.h5 — no runtime interpolation needed. Forward reads `[n_elem, NGLL, NGLL, NGLL]` arrays for lambda, mu, etc. directly.

## Source Injection

Single point force source on the free surface (z = z_min, top of domain). Precomputed by the preprocessor:

- Element list + natural coordinates (ξ_s, η_s, ζ_s) + Lagrange weights w_ijk stored in config.h5 `/source/cells/`
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
/field/cell/damping   float64[n_elem_total, NGLL, NGLL, NGLL]
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

Two DOF modes (controlled by `use_global_dof` flag in partition file). The global DOF (CG-SEM) path is shown below. When the flag is false (legacy partition files), the old element-local path applies.

```
for step in 0..nsteps-1:
    1. Newmark predictor (global arrays):
       ũ[d, iglob] = u[d, iglob] + dt·v[d, iglob] + dt²·(½-β)·a[d, iglob]
       (β=0: ũ = u + dt·v + dt²/2·a)

    2. ũ sync (multi-rank only):
       Exchange ũ at shared interface nodes via MPI, then average:
         ũ[recv_dof] = 0.5 × (ũ_local + ũ_neighbor)
       Both ranks use consistent displacement → correct element kernel forces.

    3. Gather: rank-level → element-local
       For each element e:
         For each GLL node (i,j,k):
           iglob = local_cell2rank_node[e][i][j][k]
           elem_ũ[e][i][j][k][d] = ũ[d, iglob]

    4. Element residual (matrix-free K·u):
       For each element e:
         For each GLL node (i,j,k):
           Compute ∇ũ via GLL derivatives × dξ/dx
           ε = ½(∇ũ + ∇ũᵀ)
           σ = C:ε  (elastic isotropic)
           elem_r[e][i][j][k][d] -= (Bᵀ·σ·detJ)[d]

    5. PML damping on global velocity:
       v[d, iglob] -= damping[e][i][j][k] × v[d, iglob]
       Applied to global velocity (no gather needed).

    6. Source injection into elem_r:
       For each source element e in precomputed list:
         For each GLL node (i,j,k):
           elem_r[e][i][j][k][d] += STF[t] × w_ijk × direction_vector[d]

    7. Scatter: element-local → rank-level global residual
       For each element e:
         For each GLL node (i,j,k):
           iglob = local_cell2rank_node[e][i][j][k]
           r[d, iglob] += elem_r[e][i][j][k][d]
       (GPU: atomicAdd at shared nodes)

    8. MPI halo exchange on residual:
       For each neighbor rank:
         Pack r at send DOFs → MPI_Send
         Recv r at recv DOFs → MPI_Recv
         r[recv_dof] += recv_buf  (additive accumulation)

    9. Mass exchange (multi-rank):
       Exchange rank_node_mass at shared nodes:
         m[iglob] = m_local[iglob] + m_neighbor[iglob]
       Corrects a = (r_j+r_k) / (m_j+m_k) vs a = (r_j+r_k) / m_j.

    10. Newmark corrector (global arrays, exchanged mass):
        a_new[d, iglob] = r[d, iglob] / m[iglob]   (skip ghost-only: m ≤ 0)
        u[d, iglob]  += dt·v + dt²·(½-β)·a_old + dt²·β·a_new
        v[d, iglob]  += dt·((1-γ)·a_old + γ·a_new)
        a[d, iglob]   = a_new

    11. Strain snapshot (when step % snapshot_stride == 0):
        GATHER displacement to owning element → compute ∇u at corner
        via GLL derivative matrix × dxi_dx
        ε = ½(∇u + ∇uᵀ)   (6-component Voigt)
        Write wavefields/{direction}/record_{r}_{step}.h5

    12. Restart overwrite (when step % restart_stride == 0):
         Format controlled by use_global_dof:
         · Global DOF: flat float64[n_rank_node * 3] per field
         · Element-local: float64[n_elem, NGLL, NGLL, NGLL, 3] per field
```

## Snapshot Output

One record file per rank per snapshot (at `step % snapshot_stride == 0`). Output is shallow mesh vertices, not full GLL:

```
wavefields/{direction}/record_{r}_{step}.h5
├── attrs:
│   ├── rank                    : int32
│   ├── source_direction        : string
│   ├── basis                   : "mesh_vertices"
│   ├── record_depth_max_m      : float64
│   ├── record_depth_actual_m   : float64
│   └── excludes_pml            : bool
├── vertex_ids                  : int64[n_record_vertices]
├── strain                      : float32[1, n_record_vertices, 6]
├── displacement                : float32[1, n_record_vertices, 3]
├── velocity                    : float32[1, n_record_vertices, 3]
└── acceleration                : float32[1, n_record_vertices, 3]
```

Values are per-vertex (direct gradient from displacement at corrected corner nodes). Interior GLL points are not recorded. Displacement, velocity, acceleration are extracted from the same recorded-vertex set using the recording map.

## Restart Output

Restart is separate and latest-only. Format depends on `use_global_dof`:

```
restart/{direction}/restart_{r}.h5
├── attrs:
│   ├── rank                    : int32
│   ├── source_direction        : string
│   ├── step                    : int32
│   ├── time_s                  : float64
│   ├── ngll                    : int32
│   ├── use_global_dof          : int8    — 1=flat 1D arrays, 0=5D element-local (absent→0)
│   ├── n_rank_node             : int32   — present only when use_global_dof=1
│   ├── n_local_cell         : int32   — present only when use_global_dof=1
│
│   Global DOF mode (use_global_dof=1):
│   ├── displacement            : float64[n_rank_node × 3]    — flat 1D array
│   ├── velocity                : float64[n_rank_node × 3]
│   ├── acceleration            : float64[n_rank_node × 3]
│
│   Legacy element-local mode (use_global_dof=0/absent):
│   ├── displacement            : float64[n_local_cell, NGLL, NGLL, NGLL, 3]
│   ├── velocity                : float64[n_local_cell, NGLL, NGLL, NGLL, 3]
│   ├── acceleration            : float64[n_local_cell, NGLL, NGLL, NGLL, 3]
│
└── pml_damping                : float64[...]  # PML damping array for exact resume
```

Reader auto-detects format via `use_global_dof` attribute.
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
