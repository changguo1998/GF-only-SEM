# Naming Convention

> Parent: [../design-decisions.md](../design-decisions.md)

## Two Orthogonal Axes

Every variable name encodes two independent dimensions:

```
{scope}_{mesh}_{parameter}
```

| Axis | Question | Controlled by |
|------|----------|---------------|
| **scope** | Which subset of the data? | MPI rank layout, element ownership |
| **mesh** | How many values exist for this parameter? | Intrinsic physics — the parameter's *dimensionality* |

The same parameter at different scopes has the **same mesh** — because mesh describes the parameter's intrinsic size, not how you index it.

```
rank_node_velocity      ← rank scope, node mesh, velocity parameter
global_node_velocity    ← global scope, same node mesh, same velocity parameter
```

## Scope Taxonomy

Scope describes the **index range** — which subset of the physical domain.

| Scope | Prefix | Definition |
|-------|--------|------------|
| `global` | `global_` | Across all MPI ranks — the entire physical domain |
| `rank` | `rank_` | Per-MPI-rank assembled data (unique nodes after coordinate dedup) |
| `local` | `local_` | Elements owned by this rank (excludes ghost copies) |
| `ghost` | `ghost_` | Halo element copies from neighbor ranks |
| `element` | `element_` | A single spectral element |

Scope hierarchy (largest → smallest): `global` ⊃ `rank` ⊃ `local` ⊃ `element`

> `global` and `rank` always describe a rank-assembled, deduplicated view of data. `local` and `ghost` describe per-element duplication. `element` is the atomic unit.

## Mesh Taxonomy

Mesh describes the parameter's **intrinsic dimensionality** — how many values the parameter has, independent of scope.

| Mesh | Entity | Dimension | Meaning |
|------|--------|-----------|---------|
| `node` | point | 0D | One value per unique GLL quadrature point |
| `edge` | line | 1D | One value per edge |
| `face` | surface | 2D | One value per face |
| `cell` | volume | 3D | One value per spectral element |

> `cellnode` (volume × point) exists in DG-SEM where each element keeps its own copy at shared nodes. Not used in CG-SEM.

### How to choose mesh

Ask: *"What is the length of this array, intrinsically?"*

- Velocity has `n_node` values (one per GLL point) → mesh = `node`
- PML damping alpha has `n_cell` values (one per spectral element) → mesh = `cell`
- Element-kernel gather buffers have `n_local_cell × n_node × 3` values — indexed per element, GLL-point layout implicit in flat indexing → mesh = `cell`

| Parameter | Intrinsic count | Mesh | CG-SEM example |
|-----------|----------------|------|----------------|
| displacement | `n_node` | `node` | `rank_node_displacement` |
| velocity | `n_node` | `node` | `rank_node_velocity` |
| acceleration | `n_node` | `node` | `rank_node_acceleration` |
| residual | `n_node` | `node` | `rank_node_residual` |
| mass | `n_node` | `node` | `rank_node_mass` |
| pml_damping | `n_cell` | `cell` | `local_cell_pml_damping` |
| jacobian | `n_cell` | `cell` | `local_cell_jacobian` |
| dxi_dx | `n_cell` | `cell` | `local_cell_dxi_dx` |
| element gather buffer | `n_cell × n_node` | `cell` | `local_cell_displacement` |

### element vs cell — strict distinction

- `element` is a **scope** (single spectral element). Never used as mesh.
- `cell` is a **mesh** term (per-element dimensionality). Never used as scope.

In SEM a spectral element *is* a hexahedral cell — same geometry, different conceptual roles in naming:

- `element_*` = "within a single element" (scope)
- `*_cell_*` = "has one value per element" (mesh)

**Both existing and new code must strictly observe this distinction.**
A variable like `local_element_displacement` — where `element` describes per-element dimensionality — is incorrect. Correct form: `local_cell_displacement`.

## Patterns

### 1. Scalar/Vector Arrays

```
{scope}_{mesh}_{parameter}
```

| Example | Scope | Mesh | Parameter | Meaning |
|---------|-------|------|-----------|---------|
| `rank_node_displacement` | rank | node | displacement | Assembled displacement at each rank-level node |
| `global_node_velocity` | global | node | velocity | Velocity at every unique node in the domain |
| `local_cell_pml_damping` | local | cell | pml_damping | PML damping per local element |
| `local_cell_mass` | local | cell | mass | Lumped mass at each local element GLL node |
| `local_cell_displacement` | local | cell | displacement | Gathered displacement for element kernel |
| `local_cell_residual` | local | cell | residual | Element kernel output residual |

### 2. Mapping Tables (X2Y)

```
{scope₁}_{mesh₁}2{scope₂}_{mesh₂}
```

Describes a lookup table from one indexing scheme to another.

| Example | Meaning |
|---------|---------|
| `local_cell2rank_node` | Local element GLL nodes → rank-level unique node IDs |
| `global_cell2global_node` | All elements → global unique node IDs |
| `global_node2xyz` | Global node ID → physical coordinates (x,y,z) |
| `cell2face` | Cell → bounding face IDs (topology, scopes omitted) |

### 3. Counters

```
n_{scope}_{mesh}
```

| Example | Meaning |
|---------|---------|
| `n_global_node` | Number of unique nodes across all ranks |
| `n_rank_node` | Number of unique nodes on this rank |
| `n_local_cell` | Number of local elements on this rank |
| `n_local_cell_dof` | `n_local_cell × n_node × 3` — element-local DOF count |
| `n_node` | `NGLL³` — GLL points per element (scope omitted when unambiguous) |

## Parameter Naming

Parameters use **full English words**, not single letters or abbreviations.
No `u`, `v`, `a`, `dt`, `ss`, `vid` — even in local scope.

| Correct | Wrong |
|---------|-------|
| `displacement` | `u` |
| `velocity` | `v` |
| `acceleration` | `a` |
| `solver_dt` | `dt` |
| `snapshot_stride` | `ss` |

Exception: pure math indices `i`, `j`, `k` in tight loops.

## SI-Unit Suffixes

Config fields carry SI-unit suffixes: `_m`, `_s`, `_m_s`, `_kg_m3`, etc.

## Exceptions

- **Struct fields**: The struct name provides scope + mesh context; internal field names may omit prefixes (e.g. `RankData::mass` instead of `RankData::local_cell_mass`).
- **Function parameters**: Use full prefixed names in declarations.
- **MPI exchange buffers**: Standard convention (`send_buf`, `recv_buf`).
- **HDF5 dataset/group names**: Follow the same convention. Attributes in a group may shorten names when context is clear.
- **Variable naming in Python**: Match the C++ convention. The `_4d` suffix (e.g. `local_cell2rank_node_4d`) denotes a 4D-tensor view `[n_cell, NGLL, NGLL, NGLL]` before flattening.

## SPECFEM3D → Current

| Old (SPECFEM3D) | New | HDF5 Location |
|---|---|---|
| `ibool` | `local_cell2rank_node` | `/field/cell/local_cell2rank_node` |
| `nglob` | `n_rank_node` | `/field/cell` attr `n_rank_node` |
| `scatter_to_global` | `scatter_to_rank` | — |
| `gather_from_global` | `gather_from_rank` | — |

## Key Variables

### Solver — Rank-Level (node mesh)

| Variable | Size | Description |
|----------|------|-------------|
| `rank_node_displacement` | `[n_rank_dof]` | Displacement at each rank node |
| `rank_node_velocity` | `[n_rank_dof]` | Velocity at each rank node |
| `rank_node_acceleration` | `[n_rank_dof]` | Acceleration at each rank node |
| `rank_node_residual` | `[n_rank_dof]` | Residual (effective force) at each rank node |
| `rank_node_displacement_tilde` | `[n_rank_dof]` | Predicted displacement (Newmark predictor) |
| `rank_node_mass` | `[n_rank_node]` | Assembled diagonal mass at each rank node |
| `rank_node_damping` | `[n_rank_node]` | PML damping coefficient at each rank node |

### Solver — Element-Local (cell mesh, element kernel working buffers)

| Variable | Size | Description |
|----------|------|-------------|
| `local_cell_displacement` | `[n_local_cell_dof]` | Gathered displacement for element kernel |
| `local_cell_residual` | `[n_local_cell_dof]` | Element kernel output residual |

### Counters

| Variable | Meaning |
|----------|---------|
| `n_global_node` | Unique GLL nodes across all ranks |
| `n_rank_node` | Unique GLL nodes on this rank |
| `n_rank_dof` | `n_rank_node × 3` |
| `n_local_cell` | Local elements on this rank |
| `n_node` | `NGLL³` — GLL points per element |
| `n_local_cell_dof` | `n_local_cell × n_node × 3` |

### Mapping Tables

| Variable | Size | Description |
|----------|------|-------------|
| `global_cell2global_node` | `[n_cell × n_node]` | Global element → global unique node ID |
| `local_cell2rank_node` | `[n_local_cell × n_node]` | Local element → rank node ID (global IDs in partition files) |

### CUDA Device Pointers

Host variable `xyz` → GPU pointer `d_xyz`. Same naming convention, `d_` prefix.

## Examples

### Time Loop (CPU)

```cpp
// Predict
newmark_predict(solver_dt, beta,
                rank_node_displacement, rank_node_velocity,
                rank_node_acceleration, rank_node_displacement_tilde);

// Gather rank → local element for kernel
gather_from_rank(rank_node_displacement_tilde, local_cell2rank_node,
                 n_local_cell, n_node, local_cell_displacement);

// Element stiffness kernel
compute_element_residual(n_local_cell, ..., local_cell_displacement,
                         local_cell_residual.data());

// PML damping (operates directly on rank-level velocity)
for (int node_id = 0; node_id < n_rank_node; ++node_id) {
    double d = rank_node_damping[node_id];
    if (d > 0.0) {
        int base = node_id * 3;
        rank_node_velocity[base+0] -= d * rank_node_velocity[base+0];
        rank_node_velocity[base+1] -= d * rank_node_velocity[base+1];
        rank_node_velocity[base+2] -= d * rank_node_velocity[base+2];
    }
}

// Scatter local element → rank (accumulates shared node contributions)
scatter_to_rank(local_cell_residual, local_cell2rank_node,
                n_local_cell, n_node, rank_node_residual);

// MPI exchange on rank-level residual
exchange_halo(exchange_patterns, rank_node_residual, 3);

// Newmark corrector (rank-level arrays)
newmark_correct(solver_dt, beta, gamma, rank_node_mass,
                rank_node_displacement, rank_node_velocity,
                rank_node_acceleration, rank_node_residual);
```

### Python Preprocessor

```python
# Compute global element → global node mapping (one pass, all elements)
global_cell2global_node, n_global_node = compute_global_cell2global_node(
    gll_coords
)

# Slice per-rank
global_slice = global_cell2global_node[all_elem_ids]  # local + ghost
local_slice = global_cell2global_node[local_ids]      # local only

# Write to model.h5
felem_grp.create_dataset("global_cell2global_node",
                         data=global_cell2global_node.ravel(), dtype="int32")
felem_grp.attrs["n_global_node"] = int(n_global_node)

# Write to partition file (global IDs)
local_flat = local_slice.ravel()
felem_grp.create_dataset("local_cell2rank_node",
                         data=local_flat, dtype="int32")
felem_grp.attrs["n_rank_node"] = int(n_rank_node)
```
