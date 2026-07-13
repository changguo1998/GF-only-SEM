# Naming Convention

> Parent: [../design-decisions.md](../design-decisions.md)

## Hierarchy

### Scope (largest to smallest)

| Level | Prefix | Definition | Example |
|-------|--------|------------|---------|
| `global` | `global_` | Across all MPI ranks — the entire physical domain | `global_node_xyz` |
| `rank` | `rank_` | Per-MPI-rank assembled data (unique nodes after coordinate dedup) | `rank_node_displacement` |
| `local` | `local_` | Elements owned by this rank (excludes ghost copies) | `local_element_displacement` |
| `ghost` | `ghost_` | Halo element copies from neighbor ranks | `ghost_element_ids` |
| `element` | `element_` | A single spectral element | `element_jacobian` |
| `node` | `node_` | A single GLL quadrature point within an element | `node_coord` |

### Geometry (smallest to largest)

| Entity | Plural | Definition |
|--------|--------|------------|
| `node` | `nodes` | Mesh vertex or GLL quadrature point |
| `edge` | `edges` | Line segment connecting two nodes |
| `face` | `faces` | Quadrilateral surface bounded by 4 edges |
| `cell` | `cells` | Hexahedral volume bounded by 6 faces |

> In spectral element context, "cell" = "element". Use `element` for spectral elements, `cell` for mesh topology relations.

## Rules

### 1. Mapping Tables (X2Y)

```
{scope}_{source}2{target}
```

| Pattern | Example | Meaning |
|---------|---------|---------|
| `local_element2rank_node` | `local_element2rank_node[e*n_node+n] → node_id` | Local element GLL nodes → rank-level unique node IDs |
| `global_node2xyz` | `global_node2xyz[iglob] → (x,y,z)` | Global node ID → physical coordinates |
| `cell2face` | `cell2face[icell] → face_ids` | Cell → its bounding faces |

### 2. Scalar/Vector Arrays

```
{scope}_{entity}_{field}
```

| Pattern | Example | Meaning |
|---------|---------|---------|
| `rank_node_displacement` | `rank_node_displacement[node_id*3+d]` | Displacement at each rank-level node, 3 DOF |
| `local_element_node_mass` | `local_element_node_mass[e*n_node+n]` | Lumped mass at each local element GLL node |
| `rank_node_mass` | `rank_node_mass[node_id]` | Assembled diagonal mass at each rank node |

### 3. Counters

```
n_{scope}_{entity}
```

| Pattern | Example | Meaning |
|---------|---------|---------|
| `n_rank_node` | `n_rank_node = 26498` | Number of unique rank-level nodes |
| `n_local_element` | `n_local_element = 183` | Number of local elements on this rank |
| `n_node` | `n_node = NGLL³ = 125` | Number of GLL quadrature points per element |

### 4. Exceptions

- **Struct fields**: The struct name provides scope context; internal field names may omit the scope prefix (e.g. `RankData::mass` instead of `RankData::local_element_node_mass`).
- **Function parameters**: Use full prefixed names in declarations for clarity.
- **MPI exchange buffers**: Standard MPI convention (`send_buf`, `recv_buf`).
- **HDF5 dataset/group names**: Follow the same convention. HDF5 attribute names may be shorter where context is clear from the parent group.

## Key Variables

### Preprocessor → Solver Pipeline

| Old (SPECFEM3D) | New | HDF5 Location |
|---|---|---|
| `ibool` | `local_element2rank_node` | `/field/element/local_element2rank_node` |
| `nglob` | `n_rank_node` | `/field/element` attr `n_rank_node` |

### Solver — Rank-Level (assembled)

| Variable | Size | Description |
|----------|------|-------------|
| `n_rank_node` | scalar | Number of unique rank-level nodes |
| `n_rank_dof` | `= n_rank_node × 3` | Total rank-level DOF |
| `rank_node_displacement` | `[n_rank_dof]` | Displacement at each rank node |
| `rank_node_velocity` | `[n_rank_dof]` | Velocity at each rank node |
| `rank_node_acceleration` | `[n_rank_dof]` | Acceleration at each rank node |
| `rank_node_residual` | `[n_rank_dof]` | Residual (effective force) at each rank node |
| `rank_node_displacement_tilde` | `[n_rank_dof]` | Predicted displacement (Newmark predictor) |
| `rank_node_mass` | `[n_rank_node]` | Assembled diagonal mass at each rank node |
| `rank_node_damping` | `[n_rank_node]` | PML damping coefficient at each rank node |

### Solver — Element-Local

| Variable | Size | Description |
|----------|------|-------------|
| `n_local_element` | scalar | Number of local elements on this rank |
| `n_node` | `= NGLL³` | GLL quadrature points per element |
| `n_local_element_dof` | `= n_local_element × n_node × 3` | Total element-local DOF |
| `local_element_displacement` | `[n_local_element_dof]` | Gathered displacement for element kernel |
| `local_element_residual` | `[n_local_element_dof]` | Element kernel output residual |

### Part Data (RankData struct fields)

| Field | Shape | Description |
|-------|-------|-------------|
| `local_element2rank_node` | `[n_local_element × n_node]` | Element → rank node mapping |
| `n_rank_node` | scalar | Unique rank-level nodes |
| `n_local_element` | scalar | Local element count |
| `mass` | `[n_local_element × n_node]` | Lumped mass per element GLL node |
| `dxi_dx` | `[n_local_element × n_node × 9]` | Inverse Jacobian derivative |
| `jacobian` | `[n_local_element × n_node]` | Determinant of Jacobian |
| `coords` | `[n_local_element × n_node × 3]` | GLL node coordinates |
| `pml_damping` | `[n_local_element × n_node]` | PML damping per element GLL node |
| `lambda_` | `[n_local_element × n_node]` | First Lamé parameter |
| `mu_` | `[n_local_element × n_node]` | Second Lamé parameter |
| `vp`, `vs`, `density` | `[n_local_element × n_node]` | Material properties |

### CUDA Device Pointers

| Host Variable | Device Pointer | Size |
|---|---|---|
| `n_rank_node` | `d_rank_node_*` arrays | `n_rank_node` or `n_rank_dof` |
| `n_local_element` | `d_local_element_*` arrays | `n_local_element × n_node × 3` |
| `local_element2rank_node` | `d_local_element2rank_node` | `n_local_element × n_node` |

### Function Names

| Old | New | Description |
|-----|-----|-------------|
| `scatter_to_global` | `scatter_to_rank` | Element-local → rank-level assembly |
| `gather_from_global` | `gather_from_rank` | Rank-level → element-local distribution |
| `cuda_scatter_to_global` | `cuda_scatter_to_rank` | GPU scatter |
| `cuda_gather_from_global` | `cuda_gather_from_rank` | GPU gather |
| `cuda_gather_predicted` | `cuda_gather_predicted` | Keep (descriptive of operation) |
| `scatter_to_global_kernel` | `scatter_to_rank_kernel` | CUDA kernel |

## Examples

### Time Loop (CPU)

```cpp
// Predict
newmark_predict(solver_dt, beta, rank_node_displacement, rank_node_velocity,
                rank_node_acceleration, rank_node_displacement_tilde);

// Gather rank → local element for kernel
gather_from_rank(rank_node_displacement_tilde, local_element2rank_node,
                 n_local_element, n_node, local_element_displacement);

// Element stiffness kernel
compute_element_residual(n_local_element, ..., local_element_displacement,
                          local_element_residual.data());

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
scatter_to_rank(local_element_residual, local_element2rank_node,
                n_local_element, n_node, rank_node_residual);

// MPI exchange on rank-level residual
exchange_halo(exchange_patterns, rank_node_residual, 3);

// Newmark corrector (rank-level arrays)
newmark_correct(solver_dt, beta, gamma, rank_node_mass,
                rank_node_displacement, rank_node_velocity,
                rank_node_acceleration, rank_node_residual);
```

### Python Preprocessor

```python
# Compute per-rank element → rank node mapping
local_element2rank_node_4d, n_rank_node = compute_local_element2rank_node(
    gll_coords, all_elem_ids
)

# Write to partition file
local_element2rank_node_flat = local_element2rank_node_4d[:n_local_element].ravel()
dset = felem_grp.create_dataset("local_element2rank_node",
                                 data=local_element2rank_node_flat, dtype="int32")
felem_grp.attrs["n_rank_node"] = int(n_rank_node)
```