# Workflow Comparison: gf-calculation vs SPECFEM3D

> Generated: 2026-07-11
>
> Compares key design decisions and implementation details of the current
> gf-calculation forward solver against the SPECFEM3D Cartesian reference
> implementation (`external_reference_codes/specfem3d/`).

______________________________________________________________________

## Table of Contents

1. [DOF Numbering & CG-SEM Assembly](#1-dof-numbering--cg-sem-assembly)
1. [Newmark Time Scheme](#2-newmark-time-scheme)
1. [Mass Matrix](#3-mass-matrix)
1. [Element Residual (Stiffness) Computation](#4-element-residual-stiffness-computation)
1. [Source Injection](#5-source-injection)
1. [PML Absorbing Boundaries](#6-pml-absorbing-boundaries)
1. [MPI Exchange & Halo Communication](#7-mpi-exchange--halo-communication)
1. [Strain Recording & Output](#8-strain-recording--output)
1. [Post-processing (Green's Function Extraction)](#9-post-processing-greens-function-extraction)
1. [Configuration & Preprocessing](#10-configuration--preprocessing)
1. [Summary](#11-summary)

______________________________________________________________________

## 1. DOF Numbering & CG-SEM Assembly

| Aspect | SPECFEM3D | gf-calculation (current) | Impact |
|--------|-----------|--------------------------|--------|
| **Node numbering** | Global `ibool(i,j,k,ispec) → iglob` mapping. Each physical GLL node has one unique global ID (`NGLOB_AB`) | Element-local linear index: `dof = (e * n_node + n) * 3 + d`. Shared nodes are **duplicated** across elements | **CRITICAL**: gf's duplicated nodes mean each element evolves independently; wave cannot propagate between elements on the same rank |
| **Residual assembly** | `accel(:,iglob) += element_contribution` — accumulates into global array at shared nodes | `assemble_residual()` does **trivial 1:1 copy**: `global_r[i] = elem_r[i]`. No accumulation at shared nodes | gf lacks CG assembly entirely within a rank |
| **Within-rank shared nodes** | Implicitly assembled via `ibool` — multiple elements writing to same `iglob` are summed | Duplicated — each element has its own independent copy. No summation or averaging | Elements on the same rank are de-coupled (DG-like but with incorrect physics) |
| **Cross-rank shared nodes** | MPI `assemble_MPI_vector` sums at partition boundaries (send buffer → receive → accumulate) | `exchange_halo()` exists but **exchange patterns are empty** (preprocessor skips `r1==r2` cases but also fails to generate cross-rank patterns) | Even MPI exchange is non-functional, leaving all ranks isolated |
| **State vector size** | `NDIM × NGLOB_AB` — one value per unique global node | `n_elem × NGLL³ × 3` — one value per element per GLL node | gf uses ~(number of elements sharing a node)× more memory (6× for interior nodes in hex mesh) |
| **Key files** | `shared/get_global.f90` (global numbering), `specfem3D/assemble_MPI_vector.f90` | `forward/share/src/solver.cpp`, `forward/share/src/assembly.cpp`, `forward/share/src/exchange.cpp` | — |

### Detail: ibool mapping (SPECFEM)

```fortran
! SPECFEM: each physical point has one iglob
! accel is [NDIM, NGLOB_AB]
do ispec = 1, NSPEC_AB
  do k = 1, NGLLZ
    do j = 1, NGLLY
      do i = 1, NGLLX
        iglob = ibool(i,j,k,ispec)
        ! All elements sharing iglob accumulate here:
        accel(:,iglob) = accel(:,iglob) - sigma_B_N
      enddo
    enddo
  enddo
enddo
! Then MPI exchanges cross-rank contributions:
call assemble_MPI_vector(NPROC, NGLOB_AB, accel, ...)
```

### Detail: element-local numbering (gf-calculation)

```cpp
// gf: each element has its own DOF range
// residual is [n_elem * n_node * 3], no sharing
int n_local_dof = n_local * n_node * 3;

// Element residual kernel writes to element's private range via atomicAdd:
element_residual_kernel<<<grid, block>>>(..., d_displacement_tilde, d_residual);
// d_residual[e * n_node * 3 + ...] — no cross-element accumulation

// "Assembly" (= trivial copy):
rank_node_residual[global_offset + d] = local_cell_residual[local_offset + d];

// MPI exchange_halo — empty patterns, so no-op
exchange_halo(exchange_patterns, residual, 3);
```

______________________________________________________________________

## 2. Newmark Time Scheme

| Aspect | SPECFEM3D | gf-calculation (current) | Impact |
|--------|-----------|--------------------------|--------|
| **Predictor** | `u += Δt·v + 0.5Δt²·a` ; `v += 0.5Δt·a` ; `a = 0` (zero for accumulation) | CUDA: stores predictor in separate `d_displacement_tilde`, keeps `d_displacement` intact. CPU: same as SPECFEM but uses `displacement_tilde` vector | Equivalent for β=0. gf's CUDA predictor does not zero acceleration (corrector sets it from residual). Superficially equivalent |
| **Force computation** | Element residual accumulated into `accel` (global array): `accel(:,iglob) += -K_e * u_e + f_ext` | Element residual stored in element-local array (no global accumulation) | **CRITICAL**: Without global accumulation, elements don't couple |
| **Corrector (velocity)** | `v += 0.5Δt·a` — separate subroutine called after MPI assembly | Inline in `newmark_correct`: `v += 0.5·Δt·(a_old + a_new)` ; `u += Δt·v + 0.5Δt²·a_old` (for β=0, u is already at ũ) | Equivalent computation, different code organization |
| **Acceleration update** | Set during force computation (`accel = 0` before, then accumulated into) | Set in corrector: `a_new = residual / mass` | In SPECFEM, `accel` is the output of `M⁻¹(-K·u + f)`. In gf, residual = `-K·u + f` and correction divides by mass. Same result |
| **Key files** | `specfem3D/update_displacement_scheme.f90`, `specfem3D/iterate_time.F90` | `forward/share/src/newmark.cpp`, `forward/share/src/cuda_step.cu`, `forward/share/src/solver.cpp` | — |

### SPECFEM time step (elastic, CPU):

```
update_displ_Newmark():
  displ += dt * veloc + 0.5*dt² * accel    ! predictor
  veloc += 0.5*dt * accel                   ! predictor
  accel = 0                                  ! zero for accumulation

compute_forces_viscoelastic_calling():
  accel(:,iglob) += -K_e * u_e + f_ext      ! accumulate into global
  
assemble_MPI_vector(accel)                   ! sum across MPI ranks

update_veloc_elastic():
  veloc += 0.5*dt * accel                    ! corrector (completes velocity)
```

### gf-calculation time step (CPU):

```
newmark_predict:
  u_tilde = u + Δt·v + 0.5Δt²·a
  v_tilde = v + 0.5Δt·a

compute_element_residual:
  residual = -K·u_tilde + f_ext              ! element-local

exchange_halo(residual)                       ! MPI (no-op — empty patterns)

newmark_correct:
  a_new = residual / mass
  u += Δt·v + 0.5Δt²·a_old                   ! (= ũ for β=0)
  v += 0.5Δt·(a_old + a_new)                 ! = v_tilde + 0.5Δt·a_new
```

______________________________________________________________________

## 3. Mass Matrix

| Aspect | SPECFEM3D | gf-calculation (current) | Impact |
|--------|-----------|--------------------------|--------|
| **Storage** | 1-D array `rmass[NGLOB_AB]` — assembled global lumped mass | 4-D array `mass[n_cell, NGLL, NGLL, NGLL]` — per-element | SPECFEM stores one value per unique node; gf stores per-element duplicates |
| **Assembly** | `rmass(iglob) += ρ·detJ·w_i·w_j·w_k` — accumulated at shared nodes via `ibool` | Each element's mass is stored independently. For shared nodes, each element has its own value | In SPECFEM: `a = r / m` uses globally assembled values. In gf: `a_new = residual[i] / mass[node]` uses per-element values |
| **How used** | `accel(:,i) = accel(:,i) / rmass(i)` after assembly | `a_new = residual[i] / mass[i / 3]` (element-local) | gf's division is per-element, but without assembled residual, acceleration is wrong at shared nodes |
| **Size check** | `nglob ≈ 0.7 * n_elem * NGLL³` (non-corner nodes shared by 2-8 elements) | `n_dof = n_elem * NGLL³ * 3` (all nodes duplicated) | gf uses ~1.4× more memory for mass storage |
| **Key files** | `shared/define_mass_matrices.f90` | `preprocess/gll_geometry.py`, `preprocess/cli.py` | — |

______________________________________________________________________

## 4. Element Residual (Stiffness) Computation

| Aspect | SPECFEM3D | gf-calculation (current) | Impact |
|--------|-----------|--------------------------|--------|
| **Physics** | Isotropic elastic: `σ = λ·tr(ε)·I + 2μ·ε` | Same formulation | Equivalent |
| **Gradient computation** | Derivative matrix `hprime` (GLL) + `xix/xigll` transform | Same: `D_mat` (GLL derivative matrix) + `dxi_dx` (inverse Jacobian) | Equivalent |
| **Integration** | `factor = jacobianl * wxgll(i) * wygll(j) * wzgll(k)` | `factor = jacobian[n] * weights[i] * weights[j] * weights[k]` | Equivalent |
| **Scatter target** | `accel(:,iglob)` — **global** array, multiple elements add to same node | `r[e * n_node * 3 + n * 3 + d]` — element-local, no sharing | **CRITICAL**: SPECFEM assembles via ibool; gf writes to non-shared element ranges |
| **CUDA path** | Separate CUDA version (`compute_forces_viscoelastic_cuda`) | `element_residual_kernel` in `element_cuda.cu` | Both compute the same physics but target different indexing |
| **Execution** | `compute_forces_viscoelastic(iphase, ...)` called with inner/outer element phases | One kernel call for all elements | SPECFEM uses two-phase (inner/outer) for cache efficiency; gf uses single pass |
| **Key files** | `specfem3D/compute_forces_viscoelastic.F90` | `forward/elastic/src/element_cuda.cu`, `forward/share/src/assembly.cpp` | — |

______________________________________________________________________

## 5. Source Injection

| Aspect | SPECFEM3D | gf-calculation (current) | Impact |
|--------|-----------|--------------------------|--------|
| **Source type** | Moment tensor + point force; CMT, force, and explosion | Point force only (3 directions, one per solver run) | gf runs 3 separate forward simulations (x/y/z) instead of storing all 3 components in one run |
| **Source time function** | Ricker, Gaussian, Heaviside, user-defined via `source_time_function` | Ricker hardcoded in `config.py`; STF stored in `config.h5` as array | Equivalent flexibility |
| **Spatial distribution** | `sourcearrays` — precomputed interpolation at source location to GLL nodes, stored for all source elements | `src_weights` — interpolation weights computed by preprocessor, stored per source element per GLL node | Equivalent |
| **Injection into rhs** | `call compute_add_sources_viscoelastic(accel)` — adds to global assembled `accel` array | `cuda_source_injection(gpu_state, dir, stf_val, ...)` — adds to element-local `d_residual` | Both add `stf_val × weight` to the residual at source GLL nodes |
| **Multiple sources** | NSOURCES (arbitrary count), with MPI rank ownership | Single source point; multiple elements per rank own parts of the source | SPECFEM supports arbitrary source counts; gf hardcodes single source |
| **Direction** | Point force can be in any direction in one run | One Cartesian direction per run; 3 runs needed for full Green's tensor | Same physics, different run strategy |
| **Key files** | `specfem3D/compute_add_sources_viscoelastic.F90` | `forward/share/src/source.cpp`, `forward/share/src/cuda_step.cu` | — |

______________________________________________________________________

## 6. PML Absorbing Boundaries

| Aspect | SPECFEM3D | gf-calculation (current) | Impact |
|--------|-----------|--------------------------|--------|
| **Formulation** | C-PML (convolutional PML) with auxiliary memory variables | C-PML (same formulation) | Likely equivalent, based on `docs/design-decisions.md` |
| **Implementation** | Separate PML arrays (`PML_displ_old/new`, `PML_potential_*`), stored per CPML element | Inline PML damping in the solver: `apply_pml_damping()` modifies velocity/residual; stored in `part.pml_damping` | Different coding pattern, same underlying physics |
| **PML state storage** | Full auxiliary fields stored per PML element (6 memory variables per GLL node) | `restart_writer` saves PML state for restart; in-memory storage via `pml_damping` vector | SPECFEM stores more comprehensive PML state for adjoint/kernel simulations |
| **Where applied** | In `compute_forces_*` subroutines: modifies the residual with PML contributions | Separate kernel `cuda_pml_damping` or CPU `apply_pml_damping` | Equivalent, but gf applies after element residual rather than within it |
| **Key files** | `specfem3D/compute_forces_viscoelastic.F90` (PML sections), `pml_par/` | `forward/share/src/pml.cpp`, `forward/share/src/cuda_step.cu` | — |

______________________________________________________________________

## 7. MPI Exchange & Halo Communication

| Aspect | SPECFEM3D | gf-calculation (current) | Impact |
|--------|-----------|--------------------------|--------|
| **Send/recv indexing** | `ibool_interfaces_ext_mesh` — global node indices at MPI interfaces | Precomputed `send_dof` / `recv_dof` — flat DOF indices in element-local array | SPECFEM uses global numbering; gf uses element-local flat indices |
| **Exchange pattern generation** | Done by `decompose_mesh` / `get_MPI.f90` — identifies shared faces between partitions | `partition.py` — computes face-pair exchange for elements on **different** ranks only (`if r1 == r2: continue`) | **CRITICAL**: gf skips within-rank interfaces entirely. But with element-local numbering, within-rank shared nodes also need assembly |
| **Pattern content** | `ibool_interfaces_ext_mesh` is `[max_nibool, num_interfaces]` — maps to `NGLOB_AB` | `send_dof` / `recv_dof` are flat arrays of element-local DOF indices | Different indexing conventions, same conceptual approach |
| **MPI API** | Non-blocking `MPI_Isend`/`MPI_Irecv` + `MPI_Waitall`, with async/sync variants | `MPI_Isend`/`MPI_Irecv` + `MPI_Waitall` | Equivalent MPI pattern |
| **Accumulation** | Received values are **added** to local array (CG assembly) | Received values are **added** to local array | Same CG assembly convention |
| **Empty patterns (single rank)** | No MPI calls when `NPROC == 1` | No MPI calls (no-op stub) | Both handle single-rank via no-op. But SPECFEM uses global numbering so within-rank assembly works without MPI; gf's element-local numbering fails |
| **Key files** | `specfem3D/assemble_MPI_vector.f90`, `shared/get_MPI.f90` | `forward/share/src/exchange.cpp`, `preprocess/partition.py` | — |

______________________________________________________________________

## 8. Strain Recording & Output

| Aspect | SPECFEM3D | gf-calculation (current) | Impact |
|--------|-----------|--------------------------|--------|
| **What is recorded** | Seismograms at receiver locations; full wavefield snapshots (optional) | Strain at shallow mesh vertices (no receiver search); 3 directions x 3 separate runs | Different philosophy: gf post-processes full Green's tensors from recorded strain |
| **Recording depth** | Arbitrary receiver positions (interpolated) | Vertices within `record_depth_max_m` of free surface, at GLL corner nodes only | gf records at mesh vertices only, no interpolation during forward run |
| **Output format** | ASDF, SAC, or SU seismogram files (`write_seismograms()`) | Per-step per-rank HDF5 files (`record_{r}_{step}.h5`), later merged by postprocessor | Different I/O strategy. gf's approach is suited for Green's function extraction |
| **Strain computation** | In `compute_forces_viscoelastic` during stiffness evaluation; deviatoric strain stored for attenuation | `cuda_compute_strain` kernel called after corrector; copies from GPU to host at snapshot stride | Equivalent physics, different timing |
| **Displacement recording** | Written as seismograms at receiver locations | Extracted from element-local displacement at corner nodes (recorded vertices) | gf's recording is tied to element-local indexing (no global node IDs) |
| **Snapshot strategy** | Users specify `NTSTEP_BETWEEN_OUTPUT_SEISMO`; interpolation at receiver positions | Snapshot stride from config (`snapshot_stride = output_dt_s / solver_dt`); recorded at all shallow vertices | Different output philosophy |
| **Key files** | `specfem3D/write_seismograms.f90`, `specfem3D/compute_interpolated_dva.f90` | `forward/share/src/record.cpp`, `forward/share/src/cuda_step.cu` (strain kernel) | — |

______________________________________________________________________

## 9. Post-processing (Green's Function Extraction)

| Aspect | SPECFEM3D | gf-calculation (current) | Impact |
|--------|-----------|--------------------------|--------|
| **Green's function extraction** | Not a built-in feature. Adjoint/kernel approach used for sensitivity kernels | Dedicated `gf_postprocess` binary merges 3 directional runs → assembles 6×3 strain Green's tensor + 3×3 displacement tensor → tiles | Unique feature of gf-calculation |
| **Tile-based storage** | N/A — seismograms are time series at discrete receivers | `tile_x{i}_y{j}.h5` files with spatial tiles of recorded vertices; indexed by `greenfun/` library | Novel approach enabling reciprocity queries |
| **Reciprocity queries** | N/A — requires separate kernel simulation | `GreenFunctionLibrary.query(source, receiver)` uses reciprocity to compute strain/displacement at arbitrary source locations | Built-in reciprocity support |
| **Key files** | N/A | `postprocess/cpp/main.cpp`, `greenfun/library.py`, `greenfun/source_run.py` | — |

______________________________________________________________________

## 10. Configuration & Preprocessing

| Aspect | SPECFEM3D | gf-calculation (current) | Impact |
|--------|-----------|--------------------------|--------|
| **Mesh format** | Mesh files (CUBIT, Gmsh, internal) → `xstore/ystore/zstore` arrays + `ibool` | Gmsh → `model.h5` with topology + element-to-vertex connectivity | Different formats, same information |
| **Config format** | Fortran `Par_file` (text), read via `shared_parameters` | Python `config.py` (imported at runtime), compiled to `config.h5` | Python config is more flexible but creates runtime dependency |
| **Partitioning** | `decompose_mesh` (Scotch or METIS) → separate mesh database per slice | METIS (Python) + `partition.py` → `partition_{r}.h5` per rank | SPECFEM creates separate mesh files; gf creates partition files referencing the same `model.h5` |
| **Global numbering** | `get_global` subroutine: coordinate sorting → unique `iglob` mapping | **Not implemented** — no global node numbering exists in the pipeline | **CRITICAL**: Without `iglob`, gf cannot perform CG-SEM assembly |
| **Exchange pattern storage** | Stored as arrays in mesh database (e.g., `ibool_interfaces_ext_mesh`) | Stored as datasets in partition HDF5 (`/partition/exchange/neighbor_N/send_dof`) | Equivalent concept, different format |
| **Recording map** | Receiver interpolation coefficients stored separately | `recording_map` built during preprocessing: vertex_id → (element, corner) mapping | gf's approach is simpler (records at corners only) |
| **Key files** | `decompose_mesh/`, `generate_databases/` | `preprocess/partition.py`, `preprocess/cli.py`, `preprocess/model_writer.py` | — |

______________________________________________________________________

## 11. Summary

### Critical Issues (Blocking Correct Results)

| # | Issue | Category | Root Cause | Impact |
|---|-------|----------|------------|--------|
| 1 | **No global DOF numbering** | Preprocessing | `get_global` / `ibool` equivalent not implemented. All nodes are element-local duplicates | Without unique node IDs, CG-SEM assembly is impossible |
| 2 | **No within-rank CG assembly** | Solver | `assemble_residual()` does trivial copy, not accumulation. No summation at shared nodes | Elements on the same rank are de-coupled; waves cannot propagate |
| 3 | **Exchange patterns empty** | Preprocessing/Solver | `partition.py` skips `r1 == r2` (correct for cross-rank only), but also `exchange` group is missing from partition files | MPI `exchange_halo` is a no-op even in multi-rank runs |
| 4 | **CUDA path has no assembly at all** | Solver | CUDA path uses `cuda_newmark_correct` directly on element-local arrays, no exchange, no within-rank sum | All elements isolated; wave trapped in source elements |

### Non-Critical Differences

| # | Difference | Notes |
|---|-----------|-------|
| 1 | 3 separate runs (x/y/z) vs 1 run with all components | Functionally equivalent for Green's function extraction |
| 2 | Python config vs Fortran Par_file | Preference, not correctness |
| 3 | HDF5 tile storage vs seismogram files | Different output philosophy |
| 4 | Element-local strain kernel vs inline strain in force computation | Same physics, different code organization |
| 5 | Inner/outer element phases missing | Performance, not correctness |

### Recommended Fix Order

1. **Implement global DOF numbering in preprocessor** — equivalent to SPECFEM's `get_global` + `ibool`. Assign a unique global ID to each physical GLL node.
1. **Rewrite residual assembly** — accumulate element contributions at shared global nodes (both within-rank and cross-rank).
1. **Generate exchange patterns for all shared interfaces** — both within-rank (for future use) and cross-rank.
1. **Update solver** — use global indexing for state vectors (`displacement`, `velocity`, `acceleration`, `residual`) sized by `nglob` instead of `n_elem * n_node`.
1. **Adapt CUDA path** — match the new global indexing scheme.
