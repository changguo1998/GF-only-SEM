# Global Node Numbering & Naming Enforcement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compute global GLL node IDs once in preprocess, store `global_cell2global_node` in model.h5, slice per-rank. Enforce strict `{scope}_{mesh}_{parameter}` naming — rename all `local_element_*` → `local_cell_*`. Enable single-GPU CG-SEM via `read_partition_all` merge.

**Architecture:** One-pass coordinate sort → `global_cell2global_node`. Per-rank: compact `local_cell2rank_node` (solver) + `local_cell2global_node` (merge table). `read_partition_all` uses merge tables to produce unified CG-SEM data.

**Tech Stack:** Python (preprocess), C++17 (solver), CUDA, HDF5, MPI.

## Status: ✅ COMPLETE — 2026-07-18

All 7 tasks completed across 6 commits (cff2cd1..88fbb8e):

| Task | Commit | Key Result |
|------|--------|------------|
| 1. Global Numbering | cff2cd1 | `compute_global_cell2global_node()`, n_global_node=197173 |
| 2. model_writer.py | cff2cd1 | HDF5 paths `/field/element/`→`/field/cell/`, merge tables written |
| 3. Solver Headers | cff2cd1 | 30+ variable renames, `local_cell2global_node` field added |
| 4. Solver Core | 93e8c32+2090249 | io.cpp merge via global IDs, legacy paths removed |
| 5. Assembly+CUDA | cff2cd1 | scatter/gather, CUDA kernels renamed |
| 6. Tests+Misc | cff2cd1 | 202 Python pass, C++ tests renamed |
| 7. Docs+Verify | 764d898+88fbb8e | AGENTS.md updated, VTK tools fixed, single-GPU rel_l2=0.644 |

**Single-GPU CG-SEM:** `read_partition_all` now merges ibool via `local_cell2global_node`
(global ID dedup). n_rank_node = 197173 = n_global_node. rel_l2 = 0.644, matches CPU 16-rank.

## Global Constraints

- `element` = scope only; `cell` = mesh only — strict distinction everywhere
- `{scope}_{mesh}_{parameter}` for arrays, `{scope₁}_{mesh₁}2{scope₂}_{mesh₂}` for mapping tables, `n_{scope}_{mesh}` for counters
- No abbreviations — full English words: `displacement` not `u`, `velocity` not `v`
- `bash format.sh` before each commit; 202 Python tests + C++ Catch2 must pass
- No backward compatibility — old partition files require re-preprocess

## Storage Design

| Location | Dataset | Content | Purpose |
|----------|---------|---------|---------|
| model.h5 | `/field/cell/global_cell2global_node` | `[n_cell, NGLL³]` int32 | Global mapping (all ranks) |
| model.h5 | `/field/cell` attr `n_global_node` | scalar | Unique global nodes |
| partition\_{r}.h5 | `/field/cell/local_cell2rank_node` | `[n_local_cell, NGLL³]` int32 | Compact ibool (0..n_rank_node-1) — solver |
| partition\_{r}.h5 | `/field/cell/local_cell2global_node` | `[n_local_cell, NGLL³]` int32 | Global ibool (sparse IDs) — merge |

______________________________________________________________________

### Task 1: Preprocess — Global Numbering

**Files:**

- Modify: `preprocess/partition.py`

**Produces:** `compute_global_cell2global_node()`, updated `partition()` return dict

- [x] **Step 1: Add compute_global_cell2global_node()**

Insert after `compute_local_element2rank_node` (line ~189), before `def partition(`. Copy the coordinate-sort logic from `compute_local_element2rank_node`, but operate on ALL elements (no `element_ids` subset). Function signature:

```python
def compute_global_cell2global_node(
    gll_coords: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int32], int]:
```

Implementation identical to `compute_local_element2rank_node` except:

- Input is `gll_coords` (full `[n_cell, NGLL, NGLL, NGLL, 3]`), no `element_ids` filter

- Returns `(global_cell2global_node_4d, n_global_node)` where the array is `[n_cell, NGLL, NGLL, NGLL]` int32

- [x] **Step 2: Update partition() — third pass**

Before the per-rank loop (currently L399), add one call:

```python
global_cell2global_node, n_global_node = compute_global_cell2global_node(gll_coords)
```

Replace the per-rank `compute_local_element2rank_node` call inside the loop with slicing + compact relabeling:

```python
for rank in range(n_ranks):
    rd = per_rank[rank]
    locals_list = list(data.local_cell_ids)
    ghosts_list = list(data.ghost_cell_ids)
    n_local_cell = len(locals_list)

    all_elem_ids = locals_list + ghosts_list
    # Slice global mapping
    ibool_global_4d = global_cell2global_node[all_elem_ids]  # sparse global IDs

    # Compact relabeling for solver efficiency
    unique_ids, inverse = np.unique(ibool_global_4d.ravel(), return_inverse=True)
    ibool_compact_4d = inverse.reshape(ibool_global_4d.shape).astype(np.int32)
    n_rank_node = ibool_compact_4d.max() + 1

    rd["local_cell2rank_node"] = ibool_compact_4d          # compact
    rd["local_cell2global_node"] = ibool_global_4d          # global (merge table)
    rd["n_rank_node"] = n_rank_node

    # Convert exchange DOF: element-local → compact DOF
    # (unchanged logic, but use ibool_compact_4d for lookup)
    all_elem_to_node_map = {e: idx for idx, e in enumerate(all_elem_ids)}
    for neighbor_rank, ex in rd["exchange"].items():
        for key in ("send_dof", "recv_dof"):
            new_dofs = []
            for old_dof in ex[key]:
                local_idx = old_dof // (n_node * 3)
                remainder = old_dof % (n_node * 3)
                node = remainder // 3
                direction = remainder % 3
                k_idx = node % NGLL
                j_idx = (node // NGLL) % NGLL
                i_idx = node // (NGLL * NGLL)
                node_id = int(ibool_compact_4d[local_idx, i_idx, j_idx, k_idx])
                new_dofs.append(node_id * 3 + direction)
            ex[key] = new_dofs
        # Deduplicate (unchanged)
        seen = set()
        uniq_send, uniq_recv = [], []
        for s, r in zip(ex["send_dof"], ex["recv_dof"]):
            if s not in seen:
                seen.add(s)
                uniq_send.append(s)
                uniq_recv.append(r)
        ex["send_dof"] = uniq_send
        ex["recv_dof"] = uniq_recv
```

- [x] **Step 3: Update return dict**

Add to partition() return:

```python
return {
    "element_to_rank": element_to_rank,
    "n_ranks": n_ranks,
    "per_rank": per_rank,
    "global_cell2global_node": global_cell2global_node,
    "n_global_node": n_global_node,
}
```

- [x] **Step 4: Verify preprocess runs**

Run: `cd examples/halfspace && python -m preprocess`

Expected: completes without error. Check partition_0.h5 has both `local_cell2rank_node` and `local_cell2global_node` datasets.

### Task 2: Preprocess — model_writer.py Rename

**Files:**

- Modify: `preprocess/model_writer.py`

- [x] **Step 1: Write global mapping to model.h5**

In `write_model()`, write `global_cell2global_node` to model.h5:

```python
if partition_result is not None:
    global_cell2global_node = partition_result.get("global_cell2global_node")
    n_global_node = partition_result.get("n_global_node")
    if global_cell2global_node is not None:
        with h5py.File(model_path, "a") as f:
            felem = f["field"].require_group("cell")  # NEW path
            felem.create_dataset(
                "global_cell2global_node",
                data=global_cell2global_node.ravel().astype(np.int32),
            )
            felem.attrs["n_global_node"] = int(n_global_node)
```

- [x] **Step 2: Rename HDF5 paths**

Replace all `/field/element/` → `/field/cell/` in `_extend_model_h5()` and `_write_partition_files()`. Change group creation from `require_group("element")` to `require_group("cell")`.

- [x] **Step 3: Write local_cell2global_node to partition files**

In `_write_partition_files()`, after writing `local_cell2rank_node`, also write:

```python
local_cell2global_node_4d = rk.get("local_cell2global_node")
if local_cell2global_node_4d is not None and n_local_cell > 0:
    local_cell2global_node_local = (
        local_cell2global_node_4d[:n_local_cell].ravel().astype(np.int32)
    )
    _write_dataset(
        felem_grp,
        "local_cell2global_node",
        local_cell2global_node_local,
        dtype="int32",
        compression=True,
    )
```

- [x] **Step 4: Verify**

Run preprocess, check model.h5 `/field/cell/global_cell2global_node` exists and partition files have `local_cell2global_node`.

### Task 3: Solver Headers — RankData + Function Signatures

**Files:**

- Modify: `forward/share/include/gf/io.hpp`

- Modify: `forward/share/include/gf/assembly.hpp`

- Modify: `forward/share/include/gf/cuda_step.hpp`

- [x] **Step 1: io.hpp — Rename RankData fields**

| Old field | New field |
|-----------|-----------|
| `local_element_ids` | `local_cell_ids` |
| `local_element2rank_node` | `local_cell2rank_node` |
| `n_local_element` | `n_local_cell` |
| `n_total_element` | `n_total_cell` |
| `ghost_element_ids` | `ghost_cell_ids` |
| `n_ghost_element` | `n_ghost_cell` |

Add new field: `std::vector<int32_t> local_cell2global_node;`

- [x] **Step 2: assembly.hpp — Rename function signatures**

In `scatter_to_rank` and `gather_from_rank` declarations: rename `local_element_residual` → `local_cell_residual`, `local_element2rank_node` → `local_cell2rank_node`, `n_local_element` → `n_local_cell`.

- [x] **Step 3: cuda_step.hpp — Rename CudaDeviceState fields**

| Old | New |
|-----|-----|
| `d_local_element_residual` | `d_local_cell_residual` |
| `d_local_element_displacement` | `d_local_cell_displacement` |
| `d_local_element2rank_node` | `d_local_cell2rank_node` |
| `n_local_element_dof` | `n_local_cell_dof` |
| `n_local_element` | `n_local_cell` |

Rename function declarations: `cuda_copy_residual_*`, `cuda_copy_utilde_*` parameter types unchanged but update comments.

- [x] **Step 4: Build check**

`cmake --build build --target gf_solver_elastic_mpi 2>&1 | tail -5`
Expected: compile errors from source files not yet renamed. This is expected — proceed to Task 4.

### Task 4: Solver Core — io.cpp + solver.cpp

**Files:**

- Modify: `forward/share/src/io.cpp`

- Modify: `forward/share/src/solver.cpp`

- [x] **Step 1: io.cpp — Rename + read local_cell2global_node**

In `read_partition()`:

- Rename all `local_element*` → `local_cell*`, `n_local_element` → `n_local_cell`

- Read the new dataset: `local_cell2global_node = try_read_dataset<int32_t>(fid, "/field/cell/local_cell2global_node")`

- If not present (old format), leave empty (backward compat handled later)

- Update HDF5 paths: `/field/element/` → `/field/cell/`

- [x] **Step 2: io.cpp — Fix read_partition_all**

Delete the ibool-clear block (L382-386):

```cpp
// DELETE these lines:
// merged.local_element2rank_node.clear();
// merged.n_rank_node = 0;
```

Replace with merge logic using `local_cell2global_node`:

```cpp
// Merge ibool: concatenate per-rank compact ibool, remap to avoid collisions
// Per-rank compact ibool uses 0..n_rank_node-1. Need to offset subsequent ranks.
int cumulative_nodes = 0;
for (int r = 0; r < n_partitions; ++r) {
    // ... read partition, concat fields ...
    
    // Offset ibool values for this rank's elements
    if (r > 0) {
        for (auto& node_id : part.local_cell2rank_node) {
            node_id += cumulative_nodes;
        }
    }
    concat_vec(merged.local_cell2rank_node, part.local_cell2rank_node);
    cumulative_nodes += part.n_rank_node;
}
merged.n_rank_node = cumulative_nodes;
```

Also concat `local_cell2global_node` if available (for postprocess/recording use).

- [x] **Step 3: solver.cpp — Rename all variables**

Replace every occurrence:

- `local_element_displacement` → `local_cell_displacement`
- `local_element_residual` → `local_cell_residual`
- `local_element2rank_node` → `local_cell2rank_node`
- `n_local_element` → `n_local_cell`
- `n_local_element_dof` → `n_local_cell_dof`
- `local_element2rank_node` in logs → `local_cell2rank_node`

Update debug log line (L149-152):

```cpp
logger.info("  n_local_cell=" + std::to_string(n_local_cell) +
            " n_gll_per_elem=" + std::to_string(n_node) +
            (use_global_dof ? " n_rank_node=" + std::to_string(part.n_rank_node)
                            : " dofs=" + std::to_string(n_local_cell_dof)));
```

- [x] **Step 4: solver.cpp — Remove single-GPU legacy fallback**

Currently L146: `bool use_global_dof = (part.n_rank_node > 0 && !part.local_element2rank_node.empty());`

After this change, `read_partition_all` preserves `local_cell2rank_node`. So `use_global_dof` will be `true` for both single-GPU and multi-rank. No legacy path fallback needed.

Delete or guard the legacy element-local code blocks (the `else` branches in the GPU and CPU time loops). Keep the CG-SEM path as the only path.

- [x] **Step 5: Build + quick test**

```bash
cmake --build build --target gf_solver_elastic_mpi 2>&1 | tail -5
```

Expected: compiles successfully (may need assembly.cpp renamed first — see Task 5).

### Task 5: Solver — assembly + element kernels + CUDA

**Files:**

- Modify: `forward/share/src/assembly.cpp`

- Modify: `forward/elastic/src/element_cpu.cpp`

- Modify: `forward/elastic/src/element_cuda.cu`

- Modify: `forward/share/src/cuda_step.cu`

- [x] **Step 1: assembly.cpp — Rename**

Replace all `local_element_residual` → `local_cell_residual`, `local_element2rank_node` → `local_cell2rank_node`, `n_local_element` → `n_local_cell`, `local_element_field` → `local_cell_field`.

- [x] **Step 2: element_cpu.cpp — Rename**

Replace `local_element_displacement` → `local_cell_displacement`, `local_element_residual` → `local_cell_residual`, `n_local_element` → `n_local_cell`.

- [x] **Step 3: element_cuda.cu — Rename**

Same replacements as element_cpu.cpp. Also rename kernel parameter names.

- [x] **Step 4: cuda_step.cu — Rename**

Replace all `d_local_element_*` → `d_local_cell_*`, `n_local_element` → `n_local_cell`, `n_local_element_dof` → `n_local_cell_dof`. Update `cuda_allocate_state` field assignments and `cuda_copy_*_to_host/from_host` functions.

- [x] **Step 5: Build check**

```bash
cmake --build build 2>&1 | tail -5
```

Expected: all targets compile.

### Task 6: Solver Misc + Tests

**Files:**

- Modify: `forward/share/src/source.cpp`

- Modify: `forward/share/src/restart.cpp`

- Modify: `tests/test_assembly.cpp`

- Modify: `tests/test_newmark.cpp`

- Modify: `tests/test_integration.cpp`

- Modify: `tests/test_io.cpp`

- Modify: `tests/test_source.cpp`

- Modify: `tests/preprocess/test_partition.py`

- Modify: `tests/preprocess/test_source_locator.py`

- [x] **Step 1: source.cpp + restart.cpp — Rename**

In `source.cpp`: replace `local_element_residual` → `local_cell_residual` and related names.
In `restart.cpp`: replace any `n_local_element` → `n_local_cell`, HDF5 paths.

- [x] **Step 2: Python tests — Update assertions**

In `tests/preprocess/test_partition.py`:

- Update function name references: `compute_local_element2rank_node` → `compute_local_cell2rank_node`
- Add test for `compute_global_cell2global_node`: verify `n_global_node` matches expected, verify slicing gives consistent IDs
- Update HDF5 path assertions: `/field/element/` → `/field/cell/`

In `tests/preprocess/test_source_locator.py`:

- Update `local_element2rank_node` → `local_cell2rank_node`

- [x] **Step 3: C++ tests — Rename references**

In each C++ test file, search-replace:

- `local_element2rank_node` → `local_cell2rank_node`
- `local_element_residual` → `local_cell_residual`
- `local_element_displacement` → `local_cell_displacement`
- `n_local_element` → `n_local_cell`
- `n_local_element_dof` → `n_local_cell_dof`
- `local_element_ids` → `local_cell_ids`
- `ghost_element_ids` → `ghost_cell_ids`

In `test_io.cpp`: update HDF5 path strings `/field/element/` → `/field/cell/`.

- [x] **Step 4: Run tests**

```bash
pytest tests/ -q          # expect 202 passed
cd forward/build && ctest  # expect all Catch2 tests pass
```

### Task 7: Documentation + Final Verification

**Files:**

- Modify: `HANDOFF.md`

- Modify: `AGENTS.md`

- Modify: `forward/AGENTS.md` (if needed)

- Check: all module `AGENTS.md` files

- [x] **Step 1: Update HANDOFF.md**

Update variable names in examples and architecture sections to match new naming. Update "剩余工作" to mark single-GPU CG-SEM as resolved.

- [x] **Step 2: Update AGENTS.md**

Update project state table: CUDA single row from "⚠ legacy element-local" to "✅ CG-SEM (global DOF)". Update variable names in cross-cutting conventions.

- [x] **Step 3: Format + commit**

```bash
bash format.sh
git add -A
git commit -m "refactor: global node numbering + strict naming convention

- Add compute_global_cell2global_node() — one-pass global coordinate sort
- Store global_cell2global_node in model.h5, n_global_node attr
- Partition files: compact local_cell2rank_node + local_cell2global_node merge table
- Rename 30+ variables: local_element_* → local_cell_* (strict element=scope, cell=mesh)
- HDF5 paths: /field/element/ → /field/cell/
- read_partition_all: merge via local_cell2global_node, no longer clears ibool
- Single-GPU now uses CG-SEM path (use_global_dof=true)"
```

- [x] **Step 4: Full verification**

```bash
# Rebuild from clean
cd forward && cmake -B build -DGF_DEVICE_BACKEND=CUDA && cmake --build build

# Python tests
pytest tests/ -q  # → 202 passed

# C++ tests
cd forward/build && ctest  # → all passed

# halfspace CPU 16-rank
cd examples/halfspace && bash run.sh
# → rel_l2 ≈ 0.644

# halfspace GPU single (NEW verification)
mpirun -n 1 ../../bin/gf_solver_elastic_cuda --direction z
# → completes, postprocess + compare → rel_l2 ≈ 0.644

# layer example
cd examples/layer && bash run.sh
# → end-to-end succeeds
```

- [x] **Step 5: Push**

```bash
git push origin master
```
