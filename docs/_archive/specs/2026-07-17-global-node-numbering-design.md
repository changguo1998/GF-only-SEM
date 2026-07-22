# Global Node Numbering & Naming Convention Enforcement — Design Spec

> Created: 2026-07-17
> Parent: [../design-decisions.md](../../design-decisions.md)
> Convention: [../design/naming-convention.md](../../design/naming-convention.md)

## Goal

1. Compute true global GLL node IDs once in the preprocessor (one-pass coordinate sort over all elements), store as `global_cell2global_node` in model.h5, and slice per-rank into partition files so all ranks share the same node-ID space.
1. Enforce the strict `{scope}_{mesh}_{parameter}` naming convention — rename all existing variables where `element` was incorrectly used as a mesh term, replacing it with `cell`.

End result: `read_partition_all` no longer needs to clear ibool, single-GPU CUDA runs the CG-SEM path, and all naming is consistent.

## Architecture

```
Preprocess (one-time):

  model.h5 coords [n_cell, NGLL, NGLL, NGLL, 3]
       │
       ▼
  compute_global_cell2global_node()          ← NEW: global coordinate sort
       │
       ├──▶ model.h5 /field/cell/global_cell2global_node   ← NEW dataset
       │    n_global_node                                  ← NEW attr
       │
       └──▶ per rank: slice by local_cell_ids
            ├── partition_{r}.h5 /field/cell/local_cell2rank_node  ← global IDs
            └── exchange DOFs → global DOF (global_node_id * 3 + dir)

Solver (each run):

  read_partition(path, r)      → CG-SEM (unchanged, IDs now global)
  read_partition_all(dir)      → CG-SEM (NO LONGER clears ibool!)
  use_global_dof = true        → single-GPU CG-SEM path ✓
```

## Rename Map

Every variable where `element` was used as mesh → renamed to `cell`.

### Mapping Tables

| Old | New |
|-----|-----|
| `local_element2rank_node` | `local_cell2rank_node` |
| `global_element2global_node` | `global_cell2global_node` |

### Counters

| Old | New |
|-----|-----|
| `n_local_element` | `n_local_cell` |
| `n_local_element_dof` | `n_local_cell_dof` |

### Element-Kernel Working Buffers

| Old | New |
|-----|-----|
| `local_element_displacement` | `local_cell_displacement` |
| `local_element_residual` | `local_cell_residual` |

### HDF5 Paths

| Old | New |
|-----|-----|
| `/field/element/` | `/field/cell/` |

### CUDA Pointers

| Old | New |
|-----|-----|
| `d_local_element_residual` | `d_local_cell_residual` |
| `d_local_element_displacement` | `d_local_cell_displacement` |
| `d_local_element2rank_node` | `d_local_cell2rank_node` |
| `n_local_element_dof` (as grid param) | `n_local_cell_dof` |

### RankData Struct Fields

| Old | New |
|-----|-----|
| `local_element_ids` | `local_cell_ids` |
| `local_element2rank_node` | `local_cell2rank_node` |
| `n_local_element` | `n_local_cell` |
| `local_element_displacement` | `local_cell_displacement` |
| `local_element_residual` | `local_cell_residual` |

### Ghost-Related

| Old | New |
|-----|-----|
| `ghost_element_ids` | `ghost_cell_ids` |
| `ghost_owners` | Unchanged (no element/cell in name) |

### Source Elements

| Old | New |
|-----|-----|
| `n_src_elements` | `n_src_cell` |
| `src_element_ids` | `src_cell_ids` |

### Source Elements (scope unchanged)

| Variable | Change? |
|----------|---------|
| `n_src_elements` | No — counter for "source elements" (entity, not mesh dimension) |
| `src_element_ids` | No — element IDs, `element` is the entity |

### Function Names

| Old | New |
|-----|-----|
| `compute_local_element2rank_node` | `compute_local_cell2rank_node` |
| `compute_global_element2global_node` | `compute_global_cell2global_node` |

### Variables That Stay Unchanged

Mesh = `node`, never affected: `rank_node_displacement`, `rank_node_velocity`, `rank_node_acceleration`, `rank_node_residual`, `rank_node_displacement_tilde`, `rank_node_mass`, `rank_node_damping`, `n_rank_node`, `n_rank_dof`, `n_global_node`, `n_node`, `node_share_count`.

Exchange: `exchange_patterns` (unrelated naming category), `send_dof`, `recv_dof`.

## Files Affected

### Preprocess (Python)

| File | Change |
|------|--------|
| `preprocess/partition.py` | New `compute_global_cell2global_node()`. Replace per-rank loop. Rename all `local_element*` → `local_cell*`. Exchange DOF conversion uses global IDs. |
| `preprocess/model_writer.py` | Write `global_cell2global_node` to model.h5. Rename HDF5 paths `/field/element/` → `/field/cell/`. |
| `preprocess/config_writer.py` | Minimal — write `n_global_node` attr. |
| `preprocess/topology_reader.py` | No changes needed (topology uses `cell` already). |

### Solver (C++)

| File | Change |
|------|--------|
| `forward/share/src/io.cpp` | `read_partition_all`: delete ibool-clear logic. Rename `local_element*` → `local_cell*`. Update HDF5 paths. |
| `forward/share/src/solver.cpp` | Rename all `local_element*` → `local_cell*`. Single-GPU path uses CG-SEM (no legacy fallback). |
| `forward/share/src/assembly.cpp` | Rename `local_element*` → `local_cell*` in scatter/gather. |
| `forward/share/src/cuda_step.cu` | Rename `d_local_element*` → `d_local_cell*`. Update `cuda_allocate_state`. |
| `forward/share/include/gf/cuda_step.hpp` | Rename struct fields. |
| `forward/share/include/gf/assembly.hpp` | Rename function signatures. |
| `forward/share/include/gf/io.hpp` | Rename `RankData` fields. |
| `forward/elastic/src/element_cpu.cpp` | Rename `local_element_displacement` → `local_cell_displacement` in kernel. |
| `forward/elastic/src/element_cuda.cu` | Same rename in CUDA kernel. |
| `forward/share/src/exchange.cpp` | Unchanged (exchange_patterns independent). |
| `forward/share/src/newmark.cpp` | Unchanged. |
| `forward/share/src/pml.cpp` | Unchanged. |
| `forward/share/src/source.cpp` | Rename if `local_element*` references exist. |
| `forward/share/src/restart.cpp` | Rename if `local_element*` references exist. |

### Tests

| File | Change |
|------|--------|
| `tests/preprocess/test_partition.py` | Update assertions for renamed functions and new global ibool. |
| `tests/preprocess/test_source_locator.py` | Update `local_element2rank_node` → `local_cell2rank_node`. |
| `tests/test_assembly.cpp` | Rename variable references. |
| `tests/test_newmark.cpp` | Rename references. |
| `tests/test_integration.cpp` | Rename references. |
| `tests/test_io.cpp` | Update HDF5 path assertions. |
| `tests/test_source.cpp` | Rename references. |
| `tests/greenfun/test_source_run.py` | No changes (uses high-level API). |
| `tests/greenfun/test_library.py` | No changes. |

### Examples

| File | Change |
|------|--------|
| `examples/halfspace/config.py` | Unchanged. |
| `examples/halfspace/compare.py` | Unchanged. |
| `examples/layer/config.py` | Unchanged. |
| `examples/*/run.sh` | Unchanged. |

### Documentation

| File | Change |
|------|--------|
| `docs/design/naming-convention.md` | Already updated in this session. |
| `HANDOFF.md` | Update variable names in examples. |
| `AGENTS.md` | Update project state references. |
| Module `AGENTS.md` files | Update if they reference renamed variables. |

## Verification

1. **Python tests**: `pytest tests/ -q` → 202 passed
1. **C++ tests**: All Catch2 tests pass
1. **halfspace CPU 16-rank**: rel_l2 ≈ 0.644 (unchanged from current baseline)
1. **halfspace GPU multi-rank**: rel_l2 ≈ 0.644 (unchanged)
1. **halfspace GPU single-rank**: rel_l2 ≈ 0.644 (NEW — was ~1.0 / uncorrelated)
1. **layer example**: End-to-end succeeds, diagonal peaks correlate with PyFK reference

## Non-Goals

- DG-SEM `cellnode` mesh — not needed for CG-SEM
- C-PML implementation
- I/O format backward compatibility (direct format change, old partitions need re-preprocess)
