# CG-SEM Assembly Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the forward solver's CG-SEM assembly to enable correct wave propagation through element interfaces, matching the SPECFEM3D reference implementation.

**Root cause:** The solver uses element-local DOF numbering (`n_dof = n_elem × NGLL³ × 3`) without any mechanism to enforce continuity at shared GLL nodes. This means each element evolves independently — waves cannot propagate between elements on the same rank. MPI exchange patterns exist for cross-rank interfaces, but within-rank shared nodes are never assembled.

**Approach:** Introduce global DOF numbering (`ibool` mapping from SPECFEM3D) throughout the preprocessing → solver pipeline. State vectors become globally-sized (`nglob × 3`), element contributions are scattered to and accumulated at shared global nodes, and MPI exchange handles cross-rank accumulation on the global array.

**Architecture:** Five phases:

1. **Preprocessor** — compute and store `ibool` mapping + convert exchange patterns to global indices
1. **Solver infrastructure** — add global arrays, scatter/gather routines, assemble global mass/damping
1. **CPU solver loop** — rewrite Newmark and time loop for global numbering
1. **CUDA solver loop** — parallel implementation with atomic scatter
1. **I/O and testing** — adapt recording, restart, exchange; validate end-to-end
   **Tech Stack:** C++17, CUDA, HDF5, METIS, MPI

______________________________________________________________________

## Phase 0: Preprocessor — Global DOF Numbering

### Task 0.1: Implement ibool Computation in Preprocessor

**Files:**

- Create: (none new)
- Modify: `preprocess/partition.py`
- Test: (new Python tests in `tests/test_partition.py`)

**Interfaces:**

- Consumes: `TopologyData` (from `topology_reader.py`) with GLL coordinates for all elements
- Produces: `ibool` array `[n_cell, NGLL, NGLL, NGLL]` mapping `int32` — each entry is a unique per-rank global node ID (0-based for C++)
- Produces: `nglob` scalar — count of unique global nodes on this rank
- Note: ibool is computed for all elements (local + ghost) in the preprocessor so that shared interface nodes receive the same iglob within a rank. Only the `[0:n_local_elem]` slice is stored in the partition file — ghost elements have no geometry/material data in partitions (they exist only in exchange patterns). The solver runs the element kernel on local elements exclusively.

**Algorithm (following SPECFEM's `get_global`):**

```python
def compute_ibool(topology, gll_coords):
    """
    Assign unique global IDs to physical GLL nodes.
    
    Two GLL nodes at the same (x,y,z) belong to different elements
    but represent the same physical point → same iglob.
    
    Uses coordinate sorting (SPECFEM's approach) or a hash map.
    """
    n_cell, NGLL, _, _, _ = gll_coords.shape
    n_node = NGLL ** 3
    
    # Flatten to list of (x, y, z, element_id, gll_idx)
    points = []
    for e in range(n_cell):
        for i in range(NGLL):
            for j in range(NGLL):
                for k in range(NGLL):
                    x = gll_coords[e, i, j, k, 0]
                    y = gll_coords[e, i, j, k, 1]
                    z = gll_coords[e, i, j, k, 2]
                    points.append((x, y, z, e, (i * NGLL + j) * NGLL + k))
    
    # Sort by (x, y, z) — points at same coordinate cluster together
    # Use a tolerance based on mesh extent (SPECFEM: SMALLVALTOL)
    domain_extent = max(x_max - x_min, y_max - y_min, z_max - z_min)
    tol = 1e-12 * domain_extent
    
    # Assign iglob: same coordinate → same iglob
    ibool = np.zeros((n_cell, NGLL, NGLL, NGLL), dtype=np.int32)
    iglob = 0  # 0-based for C++
    prev_x = prev_y = prev_z = None
    for (x, y, z, e, n) in sorted(points, key=lambda p: (p[0], p[1], p[2])):
        if (prev_x is None or abs(x - prev_x) > tol or
            abs(y - prev_y) > tol or abs(z - prev_z) > tol):
            iglob += 1
        i, j, k = n // (NGLL * NGLL), (n // NGLL) % NGLL, n % NGLL
        ibool[e, i, j, k] = iglob - 1  # 0-based for C++
        prev_x, prev_y, prev_z = x, y, z
    
    return ibool, iglob - 1  # nglob
```

- [ ] **Step 1: Write `compute_ibool` function in `preprocess/partition.py`**
- [ ] **Step 2: Write unit test** — 2-element mesh sharing a face → verify shared nodes have same iglob, interior nodes have unique iglob
- [ ] **Step 3: Run test to verify**
- [ ] **Step 4: Commit**

### Task 0.2: Write ibool to Partition Files

**Files:**

- Modify: `preprocess/model_writer.py`
- Modify: `preprocess/cli.py` (pass ibool/nglob through pipeline)

**Interfaces:**

- Produces: `/field/ibool` dataset in each partition file — flat `[n_local * n_node]`, dtype int32 (maps (elem, node) → per-rank global node ID)

- Produces: `/field/nglob` attribute in each partition file — scalar int32
  **Flattening:** The preprocessor computes ibool as a 4D array `[n_cell, NGLL, NGLL, NGLL]`. Before writing, slice to local elements and flatten: `ibool_local = ibool[:n_local_elem].reshape(-1)` → 1D array of shape `[n_local_elem * n_node]`.

- [ ] **Step 1: Add `ibool` and `nglob` fields to partition data dict in `partition.py`**

- [ ] **Step 2: Add writing code in `model_writer.py`** — write ibool under `/field/`, nglob as attr

- [ ] **Step 3: Run halfspace preprocessor** and verify datasets exist with `h5ls`

- [ ] **Step 4: Commit**

### Task 0.3: Generate Exchange Patterns with Global DOF Indices

**Files:**

- Modify: `preprocess/partition.py`

**Current behavior:** Exchange patterns (`send_dof`, `recv_dof`) use element-local DOF indices (flat: `(elem * n_node + node) * 3 + dir`). These must be converted to per-rank global DOF indices (`iglob * 3 + dir`) so `exchange_halo` can operate on the global-sized residual array.

**Key insight:** Within-rank assembly at shared nodes is handled by `scatter_to_global` — two elements sharing a physical node get the same iglob, so scatter naturally accumulates both contributions. **No special within-rank exchange patterns are needed.** This task focuses solely on converting the existing cross-rank exchange patterns to global indexing.

**Approach A (in preprocessor — REQUIRED):** Before writing exchange patterns to partition files, compute `global_dof = ibool[e * n_node + node] * 3 + dir` for each DOF and write the global indices. The preprocessor has access to the full ibool for all elements (local + ghost), so both send and recv DOFs can be converted correctly.

**Why Approach B (solver-side conversion) does NOT work:** The solver stores only `n_local_elem` ibool entries in its partition file. Ghost element data (coordinates, material params, ibool) is NOT stored in partition files (verified: io.cpp reads only `[n_local_elem, ...]` shaped datasets). Converting recv_dof indices that reference ghost elements would require `part.ibool[ghost_elem * n_node + node]` — an out-of-bounds access. Only the preprocessor, which has the complete ibool for all elements, can perform this conversion.

**Implementation sketch (in `partition.py`):**

```python
# After ibool is computed for all elements, before writing exchange patterns:
for rank_data in all_rank_data:
    ibool_flat = rank_data['ibool'].reshape(-1)  # [n_cell * n_node]
    for neighbor_rank, send_list, recv_list in rank_data['exchange_dof']:
        for i in range(len(send_list)):
            elem_idx = send_list[i] // (n_node * 3)
            local_dof = send_list[i] % (n_node * 3)
            node = local_dof // 3
            direction = local_dof % 3
            iglob = ibool_flat[elem_idx * n_node + node]
            send_list[i] = iglob * 3 + direction
        for i in range(len(recv_list)):
            elem_idx = recv_list[i] // (n_node * 3)
            local_dof = recv_list[i] % (n_node * 3)
            node = local_dof // 3
            direction = local_dof % 3
            iglob = ibool_flat[elem_idx * n_node + node]
            recv_list[i] = iglob * 3 + direction
```

````

- [ ] **Step 1: Implement the DOF index conversion in `partition.py`** (Approach A — preprocessor-side, see implementation sketch above)
- [ ] **Step 2: Verify exchange patterns use valid ibool indices (0 ≤ idx < nglob * 3) for both send and recv**
- [ ] **Step 3: Commit**

---

## Phase 1: Solver Data Structures

### Task 1.1: Add ibool and Global Array Support to RankData

**Files:**
- Modify: `forward/share/include/gf/types.hpp`
- Modify: `forward/share/src/io.cpp`

**Changes to `RankData`:**

```cpp
struct RankData {
    // ... existing fields ...
    
    // NEW: Global DOF numbering
    std::vector<int32_t> ibool;     // [n_local_elem * n_node] — maps (elem, node) → global node ID (0-based)
    int nglob = 0;                  // total unique global nodes on this rank (subset of global)
    
    // NEW: Reverse mapping for recording/strain
    // Element-local DOF → global DOF for each element's nodes
    // Derived from ibool: global_dof = ibool[e * n_node + n] * 3 + d
};
````

**Reading ibool from partition file (io.cpp):**

```cpp
// In read_partition():
// After reading existing fields, read ibool:
data.ibool = read_dataset_int32(fid, "/field/ibool");
// Read as flat [n_local * n_node] for C++ convenience

hid_t attr = H5Aopen(fid, "/field/nglob", H5P_DEFAULT);
if (attr >= 0) {
    H5Aread(attr, H5T_NATIVE_INT32, &data.nglob);
    H5Aclose(attr);
}
```

**Memory estimate for state vectors:**

- Old: `n_dof = n_local * n_node * 3` — typically ~68k for halfspace per rank
- New: `nglob_rank * 3` — `nglob_rank` is the number of unique global nodes on this rank (≈60-70% of old value due to shared nodes)

We still need element-local arrays for the element residual kernel (it operates element-by-element), but the main state vectors (displacement, velocity, acceleration, residual) should be global-sized.

- [ ] **Step 1: Add `ibool` and `nglob` fields to `RankData` struct**
- [ ] **Step 2: Add reading code in `io.cpp` `read_partition()`**
- [ ] **Step 3: Build and verify** — compile forward solver, run with new partition files, check ibool is loaded
- [ ] **Step 4: Commit**

### Task 1.2: Add Global State Vector Allocation

**Files:**

- Modify: `forward/share/src/solver.cpp`

**Changes in `run_forward()`:**

```cpp
// Replace:
int n_local_dof = n_local * n_node * 3;

// With:
int n_global_dof = part.nglob * 3;

// Keep element-local arrays for element kernel only:
int n_elem_dof = n_local * n_node * 3;
std::vector<double> elem_displacement(n_elem_dof, 0.0);  // for element kernel

// Global state vectors (CG-SEM):
std::vector<double> displacement(n_global_dof, 0.0);
std::vector<double> velocity(n_global_dof, 0.0);
std::vector<double> acceleration(n_global_dof, 0.0);
std::vector<double> residual(n_global_dof, 0.0);
```

- [ ] **Step 1: Modify solver.cpp to allocate global-sized state vectors**
- [ ] **Step 2: Build and verify compilation**
- [ ] **Step 3: Commit**

### Task 1.3: Add Scatter/Gather Routines

**Files:**

- Create: `forward/share/include/gf/assembly.hpp` (rewrite)
- Modify: `forward/share/src/assembly.cpp`

**New functions:**

```cpp
namespace gf {

/// Scatter element-local residual to global residual, accumulating at shared nodes.
/// After this, global_residual[3*iglob + d] = Σ_e (element_residual from element e at node iglob)
void scatter_to_global(
    const std::vector<double>& elem_residual,  // [n_local * n_node * 3]
    const std::vector<int32_t>& ibool,         // [n_local * n_node]
    int n_local,
    int n_node,
    std::vector<double>& global_residual       // [nglob * 3], accumulated
);

/// Gather global displacement to element-local array for element kernel.
/// elem_disp[e * n_node * 3 + n * 3 + d] = global_disp[ibool[e * n_node + n] * 3 + d]
void gather_from_global(
    const std::vector<double>& global_field,   // [nglob * 3]
    const std::vector<int32_t>& ibool,         // [n_local * n_node]
    int n_local,
    int n_node,
    std::vector<double>& elem_field            // [n_local * n_node * 3]
);

}  // namespace gf
```

```cpp
// assembly.cpp implementation sketch:
void scatter_to_global(
    const std::vector<double>& elem_residual,
    const std::vector<int32_t>& ibool,
    int n_local,
    int n_node,
    std::vector<double>& global_residual)
{
    std::fill(global_residual.begin(), global_residual.end(), 0.0);
    for (int e = 0; e < n_local; ++e) {
        for (int n = 0; n < n_node; ++n) {
            int iglob = ibool[e * n_node + n];
            int elem_base = (e * n_node + n) * 3;
            int glob_base = iglob * 3;
            for (int d = 0; d < 3; ++d) {
                global_residual[glob_base + d] += elem_residual[elem_base + d];
            }
        }
    }
}
```

- [ ] **Step 1: Rewrite `assembly.hpp` with new function declarations**
- [ ] **Step 2: Implement `scatter_to_global` and `gather_from_global` in `assembly.cpp`**
- [ ] **Step 3: Write unit test** — 2 elements sharing one node → verify residual is summed at shared node
- [ ] **Step 4: Commit**

### Task 1.4: Assemble Global Mass and Damping Arrays

**Files:**

- Modify: `forward/share/src/solver.cpp` (startup routine)

**Purpose:** Before entering the time loop, assemble element-local `mass` and `pml_damping` into global-sized arrays using ibool. This is a one-time operation at solver startup.

```cpp
// In solver.cpp, after loading partition and ibool:
std::vector<double> global_mass(part.nglob, 0.0);
std::vector<double> global_damping(part.nglob, 0.0);
for (int e = 0; e < n_local; ++e) {
    for (int n = 0; n < n_node; ++n) {
        int iglob = part.ibool[e * n_node + n];
        global_mass[iglob] += part.mass[e * n_node + n];
        // Assignment (not accumulation): all elements sharing the same
        // physical node have the same damping value.
        global_damping[iglob] = part.pml_damping[e * n_node + n];
    }
}
```

- [ ] **Step 1: Add global mass and global damping assembly code**
- [ ] **Step 2: Update `newmark_correct` to use `global_mass` instead of `part.mass`**
- [ ] **Step 3: Build and verify compilation**
- [ ] **Step 4: Commit**

______________________________________________________________________

## Phase 2: CPU Solver Loop

### Task 2.1: Refactor Inline Newmark Functions for Global Arrays

**Files:**

- Modify: `forward/share/src/solver.cpp` (inline functions in anonymous namespace, lines 40-64)

**Important:** The solver uses LOCAL inline `newmark_predict` and `newmark_correct` functions defined at the top of `solver.cpp` (anonymous namespace), NOT the library functions in `newmark.cpp` (which are only used by `tests/test_newmark.cpp`). All changes below apply to the solver-local inline versions.

**Changes to the inline `newmark_predict`:**

```cpp
// Current signature (solver.cpp:40):
inline void newmark_predict(double solver_dt, double beta,
    const std::vector<double>& displacement,
    const std::vector<double>& velocity,
    const std::vector<double>& acceleration,
    std::vector<double>& displacement_tilde);

// Modified: arrays are global-sized [nglob * 3], logic unchanged.
// With beta=0, displacement_tilde[i] = displacement[i] + dt * velocity[i] + 0.5 * dt^2 * acceleration[i].
// Note: displacement_tilde is a SEPARATE output — displacement is NOT overwritten.
```

**Changes to the inline `newmark_correct`:**

```cpp
// Current signature (solver.cpp:52):
inline void newmark_correct(double solver_dt, double beta, double gamma,
    const std::vector<double>& mass,          // [nglob] — node-sized (already mass[i/3])
    std::vector<double>& displacement,         // [nglob * 3]
    std::vector<double>& velocity,             // [nglob * 3]
    std::vector<double>& acceleration,         // [nglob * 3]
    std::vector<double>& residual);            // [nglob * 3]

// The corrector updates all three state vectors in one call:
//   a_new[i] = residual[i] / mass[i/3]
//   displacement[i] += dt * v[i] + dt^2 * ((0.5-beta)*a_old + beta*a_new)
//   velocity[i] += dt * ((1.0-gamma)*a_old + gamma*a_new)
//   acceleration[i] = a_new
//
// With beta=0: displacement update equals predictor output (redundant but correct).
// Key constraint: displacement[i] must hold the OLD value (before predictor),
// NOT u_tilde. The predictor writes to a separate displacement_tilde array.
```

**Mass size change:** Currently the solver calls `newmark_correct(..., part.mass, ...)` where `part.mass` is element-local `[n_local * n_node]`. After assembly (Task 1.4), pass `global_mass` (assembled `[nglob]`) instead. The corrector already uses `mass[i/3]` (node-sized access) — no loop structure change needed, just resize the mass array.

- [ ] **Step 1: Update the inline `newmark_predict` signature in `solver.cpp`** to accept global-sized arrays
- [ ] **Step 2: Update the inline `newmark_correct` call site** to pass `global_mass` instead of `part.mass`
- [ ] **Step 3: Add `#include "gf/assembly.hpp"` to `solver.cpp`** (for scatter/gather)
- [ ] **Step 4: Build and verify compilation**
- [ ] **Step 5: Commit**

### Task 2.2: Rewrite CPU Solver Loop

**Files:**

- Modify: `forward/share/src/solver.cpp`

**New CPU time step:**

```cpp
for (int step = start_step; step < cfg.nsteps; ++step) {
    // === CPU path ===
    
    // 1. Newmark predictor (global arrays, no gather needed beforehand)
    newmark_predict(solver_dt, beta, displacement, velocity, acceleration,
                    displacement_tilde);
    
    // 2. Gather predicted displacement to element-local for element kernel
    gather_from_global(displacement_tilde, part.ibool, n_local, n_node, elem_displacement_tilde);
    
    // 3. Zero element-local residual
    std::vector<double> elem_residual(n_elem_dof, 0.0);
    
    // 4. Element residual (element kernel, computes r_e = K_e * u_tilde)
    compute_element_residual<gf::ActiveBackend>(
        n_local, part.dxi_dx.data(), part.jacobian.data(),
        part.lambda_.data(), part.mu_.data(), D_mat.data(),
        gll_wts.data(), ngll, elem_displacement_tilde.data(),
        elem_residual.data());
    
    // 5. PML damping on global velocity (direct — no gather/scatter)
    for (int iglob = 0; iglob < part.nglob; ++iglob) {
        double d = global_damping[iglob];
        if (d > 0.0) {
            int base = iglob * 3;
            velocity[base + 0] -= d * velocity[base + 0];
            velocity[base + 1] -= d * velocity[base + 1];
            velocity[base + 2] -= d * velocity[base + 2];
        }
    }
    
    // 6. Source injection into element-local residual (unchanged logic)
    {
        int dir = (direction == "x") ? 0 : ((direction == "y") ? 1 : 2);
        double stf_val = 0.0;
        if (step < static_cast<int>(cfg.stf_t.size())) stf_val = cfg.stf_values[step];
        if (stf_val != 0.0) {
            for (int si = 0; si < cfg.n_src_elements; ++si) {
                int elem_idx = src_elem_to_local[si];
                if (elem_idx < 0) continue;
                int weight_off = si * n_node;
                int dof_base_elem = elem_idx * n_node * 3;
                for (int n = 0; n < n_node; ++n) {
                    double w = cfg.src_weights[weight_off + n];
                    if (w == 0.0) continue;
                    elem_residual[dof_base_elem + n * 3 + dir] += stf_val * w;
                }
            }
        }
    }
    
    // 7. Scatter element-local residual → global (accumulates at shared nodes)
    scatter_to_global(elem_residual, part.ibool, n_local, n_node, residual);
    
    // 8. MPI halo exchange on global residual
    exchange_halo(exchange_patterns, residual, 3);
    
    // 9. Newmark corrector (global arrays)
    newmark_correct(solver_dt, beta, gamma, global_mass, displacement, velocity,
                    acceleration, residual);
    
    // 10. Write snapshots (adapted in Task 4.1)
}
```

- [ ] **Step 1: Rewrite the CPU branch of the time loop in `solver.cpp`**
- [ ] **Step 2: Build and verify compilation**
- [ ] **Step 3: Commit**

### Task 2.3: Fix Source Injection for Global Indexing

**Files:**

- Modify: `forward/share/src/source.cpp`

The source injection uses element-local indexing. It needs to target the global DOF array, but the source weights are per-element-per-GLL-node. The approach: inject into the element-local `elem_residual` first (unchanged), then let `scatter_to_global` handle the accumulation.

Actually, since the source injection happens BEFORE scatter_to_global, and it already uses `residual[dof_base + dir] += stf_val * w` where `dof_base` is element-local, we need to either:

1. Inject into elem_residual (the element-local temp array), then scatter_to_global
1. Or compute the global DOF and inject directly into global residual

Option 1 is simpler — no changes needed to source.cpp, just let scatter_to_global handle it.

- [ ] **Step 1: Verify source injection works with element-local temp array**
- [ ] **Step 2: Update if needed — compute global DOF index for direct injection into global residual**
- [ ] **Step 3: Commit**

### Task 2.4: Apply PML Damping Directly to Global Velocity

**Files:**

- Modify: `forward/share/src/solver.cpp` (inline in time loop, no changes to pml.cpp needed)

**Current behavior:** `apply_pml_damping` modifies velocity in-place: `v[i] -= damping[node] * v[i]`. With global arrays, each physical node has exactly one velocity value in the global array. Therefore PML damping can be applied **directly** to global velocity — no gather/scatter involved.

**Why gather/scatter is wrong for PML:** If we gather global velocity to element-local, apply PML, then scatter back with accumulation, shared nodes would receive duplicate damping (each element sharing the node contributes, scatter sums them).

**Implementation:** Replace `apply_pml_damping(part.pml_damping, ..., velocity, ...)` with an inlined loop using the pre-assembled `global_damping` from Task 1.4:

```cpp
for (int iglob = 0; iglob < part.nglob; ++iglob) {
    double d = global_damping[iglob];
    if (d > 0.0) {
        int base = iglob * 3;
        velocity[base + 0] -= d * velocity[base + 0];
        velocity[base + 1] -= d * velocity[base + 1];
        velocity[base + 2] -= d * velocity[base + 2];
    }
}
```

- [ ] **Step 1: Remove the `apply_pml_damping` call; inline global damping loop in the time loop**
- [ ] **Step 2: Build and verify**
- [ ] **Step 3: Commit**

______________________________________________________________________

## Phase 3: CUDA Solver Loop

### Task 3.1: Adapt CUDA Element Residual Kernel

**Files:**

- Modify: `forward/elastic/src/element_cuda.cu`
- Modify: `forward/share/src/cuda_step.cu`
- Modify: `forward/share/include/gf/cuda_step.hpp`

**CudaDeviceState changes:**

```cpp
struct CudaDeviceState {
    // ... existing fields ...
    
    // NEW: Global arrays (GPU)
    double* d_global_displacement = nullptr;        // [nglob * 3] — old displacement (preserved for corrector)
    double* d_global_displacement_tilde = nullptr;  // [nglob * 3] — predicted displacement (predictor output)
    double* d_global_velocity = nullptr;            // [nglob * 3]
    double* d_global_acceleration = nullptr;        // [nglob * 3]
    double* d_global_residual = nullptr;            // [nglob * 3]
    double* d_global_mass = nullptr;                // [nglob]
    double* d_global_damping = nullptr;             // [nglob]
    int* d_ibool = nullptr;                         // [n_local * n_node]
    int n_global_dof = 0;
    
    // Keep these for element kernel:
    double* d_elem_displacement = nullptr;          // [n_local * n_node * 3] — gathered u_tilde for kernel
    double* d_elem_residual = nullptr;              // [n_local * n_node * 3]
};
```

**Why separate `_tilde` array is needed:** The inline `newmark_correct` uses the OLD displacement value: `displacement[i] += dt * v[i] + ...`. If the predictor overwrites `d_global_displacement` in-place, the old value is lost. Two distinct arrays preserve both old (for corrector) and predicted (for kernel).

````

**New CUDA kernels:**
```cuda
// Scatter element-local residual → global (accumulate)
__global__ void scatter_to_global_kernel(
    const double* elem_residual,    // [n_local * n_node * 3]
    const int* ibool,               // [n_local * n_node]
    double* global_residual,        // [nglob * 3]
    int n_local, int n_node);

// Gather global displacement → element-local
__global__ void gather_from_global_kernel(
    const double* global_field,     // [nglob * 3]
    const int* ibool,               // [n_local * n_node]
    double* elem_field,             // [n_local * n_node * 3]
    int n_local, int n_node);
````

**Additional CUDA changes:**

- **`d_global_damping`**: One more global array for PML (allocate in `cuda_allocate_state`)

- **PML kernel**: Replace `pml_damping_kernel` (element-local) with `pml_damping_global_kernel` that operates on `d_global_velocity` and `d_global_damping` directly, matching the CPU approach

- **`cuda_compute_strain` / `recorded_strain_kernel`**: Currently reads `state.d_displacement` with element-local indexing `(elem * n_node + corner_node) * 3`. After the change, `state.d_displacement` becomes `d_global_displacement`. The kernel must be updated to use ibool:

  ```cuda
  int iglob = d_ibool[elem * n_node + corner_node];
  const double* disp_ptr = &d_global_displacement[iglob * 3];
  ```

  This is one extra device memory read per vertex — negligible overhead.

- [ ] **Step 1: Add global array pointers (`d_global_*`, `d_global_damping`, `d_ibool`) to `CudaDeviceState`**

- [ ] **Step 2: Implement `scatter_to_global_kernel`** — MUST use `atomicAdd` because multiple elements sharing a node write to the same global DOF

- [ ] **Step 3: Implement `gather_from_global_kernel`** — one-to-one mapping, no atomics needed

- [ ] **Step 4: Implement `pml_damping_global_kernel`** — operates directly on `d_global_velocity`

- [ ] **Step 5: Update `cuda_allocate_state` to allocate global arrays and upload ibool / global_damping**

- [ ] **Step 6: Update `recorded_strain_kernel`** to use ibool for displacement lookup

- [ ] **Step 7: Commit**

### Task 3.2: Rewrite CUDA Solver Loop

**Files:**

- Modify: `forward/share/src/solver.cpp`

**New CUDA time step:**

```cpp
#ifdef GF_WITH_CUDA
// 1. Newmark predictor (on global arrays, no gather needed beforehand)
cuda_newmark_predict_global(gpu_state, solver_dt, beta);

// 2. Gather predicted global displacement → element-local for kernel
gather_from_global_kernel<<<...>>>(gpu_state.d_global_displacement_tilde,
    gpu_state.d_ibool, gpu_state.d_elem_displacement, n_local, n_node);

// 3. Zero element-local residual
cudaMemset(gpu_state.d_elem_residual, 0, n_elem_dof * sizeof(double));

// 4. Element residual kernel → writes to d_elem_residual (unchanged)
cuda_launch_element_residual(gpu_state, ngll, n_local);

// 5. PML damping on global velocity (direct — no gather/scatter)
cuda_pml_damping_global(gpu_state);

// 6. Source injection into element-local residual (unchanged)
cuda_source_injection(gpu_state, ...);

// 7. Scatter element-local residual → global (atomicAdd to accumulate at shared nodes)
scatter_to_global_kernel<<<...>>>(gpu_state.d_elem_residual,
    gpu_state.d_ibool, gpu_state.d_global_residual, n_local, n_node);

// 8. CUDA Newmark corrector (global arrays)
cuda_newmark_correct_global(gpu_state, solver_dt, gamma);
#endif
```

**Critical note:** `scatter_to_global_kernel` MUST use `atomicAdd` when writing to `d_global_residual`, because multiple elements sharing a physical node (same iglob) will write concurrently to the same destination. The CPU version uses simple `+=` but is single-threaded; the GPU version needs atomics to be correct.

- [ ] **Step 1: Rewrite the CUDA branch of the time loop** with corrected ordering and global PML
- [ ] **Step 2: Implement `cuda_newmark_predict_global`** — same logic as `cuda_newmark_predict` but on `d_global_*` arrays
- [ ] **Step 3: Implement `cuda_newmark_correct_global`** — operates on `d_global_*` arrays
- [ ] **Step 4: Implement `cuda_pml_damping_global`** — direct damping on `d_global_velocity`
- [ ] **Step 5: Build and verify compilation**
- [ ] **Step 6: Commit**

______________________________________________________________________

## Phase 4: I/O and Recording

### Task 4.1: Fix Strain Computation and Recording

**Files:**

- Modify: `forward/share/src/record.cpp`
- Modify: `forward/share/src/cuda_step.cu` (strain kernel)

**Issue:** The recording map uses `src_elem_local` (element-local index) and `src_corner` to extract displacement/strain at recorded mesh vertices. The displacement is now a global array, not element-local.

**Fix:** The strain computation kernel already operates on the element-local displacement (it accesses `disp_ptr` from a specific element's displacement). So we need to:

1. Gather global displacement → element-local before strain computation
1. The recorded vertex extraction stays element-local (it indexes into element-local arrays)

```cpp
// In solver.cpp, snapshot writing:
if (has_recording) {
    // Gather global → element-local for strain computation
    gather_from_global(displacement, part.ibool, n_local, n_node, elem_displacement);
    cuda_compute_strain(gpu_state, D_mat.data(), ngll, part.dxi_dx);
    cuda_copy_strain_to_host(gpu_state, rec_strain.data());
    
    // Extract recorded displacement from global array using ibool
    for (size_t vertex_idx = 0; vertex_idx < n_vertices; ++vertex_idx) {
        int elem = part.recording.src_elem_local[vertex_idx];
        int corner = part.recording.src_corner[vertex_idx];

        // Decode corner index to GLL node (existing pattern from solver.cpp:312-322)
        int corner_i = (corner & 1) ? (ngll - 1) : 0;
        int corner_j = (corner & 2) ? (ngll - 1) : 0;
        int corner_k = (corner & 4) ? (ngll - 1) : 0;
        int corner_node = (corner_i * ngll + corner_j) * ngll + corner_k;

        int iglob = part.ibool[elem * n_node + corner_node];
        for (int d = 0; d < 3; ++d) {
            rec_displacement[vertex_idx * 3 + d] = displacement[iglob * 3 + d];
        }
    }
}
```

- [ ] **Step 1: Update snapshot writing to gather global→element-local before strain computation**
- [ ] **Step 2: Update recorded displacement extraction to use ibool** (with corner node decoding)
- [ ] **Step 3: Build and verify**
- [ ] **Step 4: Commit**

### Task 4.2: Verify Exchange Patterns with Global DOF Indices

**Files:**

- Verify: `forward/share/src/exchange.cpp`
- Verify: `forward/share/src/exchange_noop.cpp`

**Status after Task 0.3:** Exchange patterns already contain per-rank global DOF indices (converted in preprocessor). The `exchange_halo` function operates on the global residual array `[nglob * 3]`. The send/recv accumulation logic is unchanged — it adds received values into the target DOFs.

**Verification needed:** After Task 0.3 conversion, send_dof and recv_dof within each pattern reference the same per-rank global DOF (shared physical node → same iglob on this rank). This is the CG-SEM accumulation pattern: rank A sends at iglob=K, rank B receives into iglob=K.

- [ ] **Step 1: Verify exchange patterns contain global DOF indices (converted in Task 0.3 preprocessor step)**
- [ ] **Step 2: Confirm `exchange_halo` accumulation logic (`buffer[d] += recv_buf[d]`) is correct for global arrays**
- [ ] **Step 3: Commit**

### Task 4.3: Update Restart Writer

**Files:**

- Modify: `forward/share/src/restart.cpp`

Restart currently saves/loads element-local state vectors. Update to save/load global arrays.

- [ ] **Step 1: Update restart format to global-sized arrays**
- [ ] **Step 2: Build and verify**
- [ ] **Step 3: Commit**

______________________________________________________________________

## Phase 5: Testing and Validation

### Task 5.1: Unit Tests for ibool

**Files:**

- Modify: `tests/test_partition.py` (Python)

Test cases:

1. Two elements sharing a face → verify GLL nodes on shared face have same iglob
1. Two elements not sharing a face → all iglob values are unique
1. Element at domain corner → 4 elements meeting at a corner node → all share the same iglob
1. Verify nglob = number of unique physical GLL nodes

- [ ] **Step 1: Write test cases**
- [ ] **Step 2: Run tests and verify**
- [ ] **Step 3: Commit**

### Task 5.2: Unit Tests for Scatter/Gather

**Files:**

- Modify: `tests/test_assembly.cpp` (C++ Catch2)

Test cases:

1. Two elements sharing one node → scatter adds residual contributions at shared node
1. Gather from global → element-local produces correct values
1. Round-trip: gather → modify → scatter → values preserved at non-shared nodes, summed at shared nodes

- [ ] **Step 1: Write test cases**
- [ ] **Step 2: Build and run tests**
- [ ] **Step 3: Commit**

### Task 5.3: Integration Test — Halfspace Example

- [ ] **Step 1: Run full halfspace pipeline** — preprocess → forward (MPI, CPU) → postprocess
- [ ] **Step 2: Check wavefield values are physically reasonable** — displacement magnitude at source ≈ 1e-7 to 1e-4 m (not 1e-11)
- [ ] **Step 3: Check strain values are non-zero and propagate through element interfaces**
- [ ] **Step 4: Run comparison with Lamb reference solution** — `bash compare.sh`
- [ ] **Step 5: Verify rel_l2 error is reasonable** (should be ≪ 1.0)
- [ ] **Step 6: Commit**

### Task 5.4: CUDA Integration Test

- [ ] **Step 1: Run halfspace with CUDA solver** — verify same results as MPI CPU
- [ ] **Step 2: Compare CUDA and MPI output record files** — should match within machine precision
- [ ] **Step 3: Commit**

### Task 5.5: Verify Existing Test Suite

- [ ] **Step 1: Run full pytest suite**: `python -m pytest tests/ -q`
- [ ] **Step 2: Fix any regressions in existing tests**
- [ ] **Step 3: All 182+ tests pass**
- [ ] **Step 4: Commit**

______________________________________________________________________

## Execution Order Summary

| Phase | Tasks | Depends On | Risk |
|-------|-------|-----------|------|
| 0: Preprocessor | 0.1-0.3 | None | Medium — ibool algorithm correctness |
| 1: Solver infra | 1.1-1.4 | Phase 0 | Low — pure additions, no behavioral change |
| 2: CPU loop | 2.1-2.4 | Phase 1 | **High** — fundamental solver rewrite |
| 3: CUDA loop | 3.1-3.2 | Phase 2 | Medium — follows CPU pattern, atomics required |
| 4: I/O | 4.1-4.3 | Phase 2 | Medium — recording needs global→element mapping |
| 5: Testing | 5.1-5.5 | All above | Low — incremental validation |

**Rollback strategy:** Each task is independently revertible. The solver still builds after every task (though it may not produce correct results until later tasks). The order ensures that the preprocessor changes land first, then solver infrastructure, then the actual solver behavior change.

## Key Design Decisions (Post-Review)

1. **ibool is per-rank**: Each rank computes ibool from its local+ghost elements alone. Cross-rank exchange uses rank-local ibool indices. This avoids the complexity of a globally-unique numbering scheme.
1. **Within-rank assembly via scatter_to_global**: No special exchange patterns needed for same-rank shared nodes — scatter naturally accumulates both elements' contributions at the same iglob.
1. **PML damping on global velocity directly**: Each physical node has exactly one velocity value in the global array. No gather/scatter overhead. No risk of double-counting.
1. **CUDA scatter uses atomicAdd**: Multiple GPU threads (from different elements sharing a node) write to the same iglob simultaneously. Atomics guarantee correct accumulation.
1. **Strain kernel reads global displacement via ibool**: One extra device memory read per recorded vertex instead of an extra kernel launch for gather.
