# GLL-Point Tile Output with L2-Projected Strain

**Date:** 2026-07-19
**Status:** Design (awaiting implementation plan)
**Author:** brainstorming session

## 1. Overview

Replace the current mesh-vertex (8 corners per cell) recording pipeline with a
full GLL-point pipeline. Every GLL node in the recording region is recorded,
strain is made continuous via L2 projection in postprocess, and the greenfun
library gains a spectral-accuracy GLL Lagrange interpolator.

**Goal:** tiles store the complete GLL-point wavefield, enabling (a) direct
node-value queries (zero interpolation error) and (b) GLL polynomial
interpolation at arbitrary points (spectral accuracy), replacing the current
trilinear vertex interpolation.

## 2. Background & Motivation

### 2.1 Current limitation

The recording map (`preprocess/recording_map.py`) selects only the 8 mesh
corner vertices per cell (`_get_cell_vertex_ids`). The solver
(`forward/share/src/solver.cpp` lines 424-432) extracts field values at those
corners via `corner_node = (corner_i*ngll + corner_j)*ngll + corner_k` where
`corner_{i,j,k} ∈ {0, ngll-1}`. Postprocess merges by `vertex_id` and the
greenfun library trilinearly interpolates across 8 corner vertices.

This discards 117 of 125 GLL nodes per cell and limits query accuracy to
trilinear (1st-order) interpolation.

### 2.2 SPECFEM3D reference

SPECFEM3D stores strain as element-local `epsilondev(5, NGLLX, NGLLY, NGLLZ)`
per cell (`compute_element_strain.F90`). Strain seismograms are computed from a
single element's strain + GLL interpolation weights at the receiver. SPECFEM
does **not** perform strain nodal averaging.

### 2.3 Strain discontinuity in CG-SEM

CG-SEM displacement is C0-continuous (global DOF, shared nodes), but strain
(displacement gradient) is C1-discontinuous at element boundaries. Even on a
regular Cartesian grid with identical Jacobians (verified: interior Jacobian
determinant std = 2.69e-8), strain differs at shared nodes because the
derivative stencil spans all 125 nodes of each element, and neighboring elements
have different neighbor-node displacements:

```
∂u/∂x = Σ_{a,b,c=1}^{125} u(a,b,c) × ∂N_a/∂ξ(i) × N_b(j) × N_c(k) × (∂ξ/∂x)
```

Two elements sharing node P have the same Jacobian but different neighbor-node
displacements `u(a,b,c)`, so the sum differs. This is the mathematical essence
of C0/C1 in spectral elements.

### 2.4 Why not C1 solving (Hermite/IGA)

True C1 continuity requires Hermite basis functions (value + derivative DOF,
4x DOF) or IGA (B-spline/NURBS). Both rewrite the entire forward solver
(element kernel, mass matrix, assembly, exchange, source, Newmark, restart).
Workload: weeks to months. C1 solving does not materially improve Green's
function accuracy because C0 SEM is already high-order spectral (5th-order
GLL); solving error is dominated by grid discretization and numerical
dispersion, not C0/C1 boundary differences.

**Decision: keep C0 solver, make strain continuous in postprocess via L2
projection.**

## 3. Design Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Storage scope | Recording region GLL nodes (`record_depth_max_m`) | Matches current recording concept; data volume controllable |
| 2 | Data organization | Hybrid: global-node-deduplicated fields + `cell_gll_node_index` mapping | Zero field redundancy + preserves cell structure for GLL interpolation |
| 3 | Strain continuity | Postprocess L2 projection (mass-weighted nodal averaging) | Solver unchanged; output continuous; enables deduplication |
| 4 | Corner tile relationship | Replace (no corner mode) | Simplifies code, no branching |
| 5 | Field quantities | strain + displacement + velocity + acceleration | Maintains current completeness |

## 4. Strain L2 Projection

### 4.1 Principle

Project element-local strain onto the continuous global-node space via
mass-weighted averaging (L2 projection approximation):

```
strain_global[node] = Σ_cell (strain_cell[cell,node] × M[cell,node])
                    / Σ_cell M[cell,node]

where M[cell, i,j,k] = density[cell,i,j,k] × jacobian[cell,i,j,k] × w_i × w_j × w_k
```

`M` is the mass-matrix diagonal entry. This is the standard SEM nodal averaging
used for continuous field output.

### 4.2 Mass weight source

Postprocess reads `/field/cell/mass` directly from `model.h5` - shape
`[n_cell, NGLL, NGLL, NGLL]`, already computed by `gll_geometry.py` as
`mass = density × jacobian × w_i × w_j × w_k`. Verified: `density`, `jacobian`,
and `mass` are all per-cell-node `[n_cell, 5, 5, 5]`. No recomputation needed.

GLL node physical coordinates are also pre-computed in `/field/cell/coords`
`[n_cell, NGLL, NGLL, NGLL, 3]` by `gll_geometry.py` (line 155), so
`recording_map.py` can extract GLL node coords directly.

The recording cell global IDs (from record file `/cell_gll_node_ids`) index
into these arrays.

### 4.3 Displacement deduplication

Displacement/velocity/acceleration are global DOF (C0-continuous). The record
stores them redundantly per cell (125 nodes). Postprocess deduplicates: for each
unique GLL node, take the value from any containing cell (values are identical
because global DOF). No averaging needed.

### 4.4 Cross-rank nodes

GLL nodes on rank boundaries are shared. Each rank records its local cells'
125 nodes. Postprocess merges across ranks by `gll_node_id` (same logic as
current vertex merge). For strain L2 projection, contributions from all ranks
containing a shared node are accumulated before division.

## 5. Architecture

### 5.1 Data flow

```
preprocess/recording_map.py
  → select recording-region cells, build cell_gll_node_index + gll_node_ids
  → write to partition files + config.h5

forward/share/src/solver.cpp
  → every snapshot_stride: extract 125-node strain[element-local] + displacement[redundant]
  → write record_{rank}_{step}.h5 [n_rec_cell, 125, ncomp]

postprocess/cpp/main.cpp
  → read records (cell-level)
  → strain L2 projection → continuous [n_unique_gll, 6]
  → displacement deduplication → [n_unique_gll, 3]
  → assemble Green tensor [nt, n_unique_gll, 6, 3]
  → write tile (basis="gll")

greenfun/library.py + interpolator.py
  → detect tile basis="gll"
  → GLLInterpolator: locate cell → gather 125 nodes → Lagrange basis sum
  → or direct node-value if query point is at a GLL node
```

### 5.2 Layer-by-layer changes

#### Layer 1: `preprocess/recording_map.py`

`_build_rank_recording` currently builds `cell_vertex_map` (cell → 8 corners)
and assigns each vertex to one cell + corner. Replace with GLL-node selection:

```python
# Input: global_cell2global_node [n_cell, 125] (already in model.h5)
# For recording-region cells:
#   cell_gll_node_ids[ci] = global_cell2global_node[cell]  # 125 global node IDs
# Deduplicate across cells:
#   gll_node_ids = sorted(unique(all cell_gll_node_ids))
#   cell_gll_node_index[ci, n] = index into gll_node_ids

# New output dict per rank:
{
    "rec_cell_global_ids": [...],          # [n_rec_cell]
    "rec_cell_local_index": [...],         # [n_rec_cell]
    "cell_gll_node_ids": [[...125...]],    # [n_rec_cell, 125]
    "gll_node_ids": [...],                 # [n_unique_gll]
    "gll_node_coords": [[x,y,z]],          # [n_unique_gll, 3]
    "cell_gll_node_index": [[...125...]],  # [n_rec_cell, 125] → index into gll_node_ids
}
```

GLL node physical coordinates: computed from cell vertices + GLL reference points
(ξ_i, η_j, ζ_k). For regular Cartesian grid:
`coord = cell_origin + (gll_point + 1)/2 × cell_size`. The `gll_geometry.py`
module already has GLL reference points.

#### Layer 2: `forward/share/include/gf/types.hpp`

```cpp
struct RecordingMap {
    bool has_recording = false;
    // Replaces: vertex_ids, src_elem_local, src_corner
    std::vector<int64_t> gll_node_ids;         // [n_unique_gll]
    std::vector<double> gll_node_coords;       // [n_unique_gll * 3]
    std::vector<int32_t> rec_cell_local;       // [n_rec_cell]
    std::vector<int32_t> cell_gll_node_index;  // [n_rec_cell * 125]
};
```

#### Layer 3: `forward/share/src/solver.cpp` (lines 398-443)

Replace corner extraction with full 125-node extraction:

```cpp
size_t n_rec_cell = part.recording.rec_cell_local.size();
int n_node = ngll * ngll * ngll;  // 125
rec_strain.resize(n_rec_cell * n_node * 6, 0.0);
rec_displacement.resize(n_rec_cell * n_node * 3, 0.0);
// velocity, acceleration同理

// GPU: cuda_compute_strain already computes full element_strain [n_local_cell, 125, 6]
// CPU: element kernel already computes full element_strain

for (size_t ci = 0; ci < n_rec_cell; ++ci) {
    int elem = part.recording.rec_cell_local[ci];
    for (int n = 0; n < n_node; ++n) {
        // strain: element-local, direct copy
        for (int c = 0; c < 6; ++c)
            rec_strain[(ci*n_node+n)*6+c] = element_strain[elem*n_node*6 + n*6 + c];
        // displacement: global DOF, map via local_cell2rank_node
        int node_id = part.local_cell2rank_node[elem*n_node + n];
        for (int d = 0; d < 3; ++d) {
            rec_displacement[(ci*n_node+n)*3+d] = displacement[node_id*3+d];
            rec_velocity[(ci*n_node+n)*3+d]     = velocity[node_id*3+d];
            rec_acceleration[(ci*n_node+n)*3+d] = acceleration[node_id*3+d];
        }
    }
}
```

The strain kernel (`cuda_compute_strain` / CPU `compute_element_residual`)
already computes all 125 nodes — no kernel change needed, only the extraction
loop changes from 1 corner to 125 nodes.

#### Layer 4: `forward/share/include/gf/record.hpp` + `src/record.cpp`

`write_step` writes 4D datasets instead of 3D:

```
record_{rank}_{step}.h5:
  /strain        [1, n_rec_cell, 125, 6]   float32 or float64
  /displacement  [1, n_rec_cell, 125, 3]
  /velocity      [1, n_rec_cell, 125, 3]
  /acceleration  [1, n_rec_cell, 125, 3]
  /cell_gll_node_ids [n_rec_cell, 125]      int32 (index into gll_node_ids)
  /gll_node_ids  [n_unique_gll]             int64
  /gll_node_coords [n_unique_gll, 3]        float64
  attrs: basis="gll", record_depth_max_m, record_depth_actual_m, ngll, n_rec_cell, n_unique_gll
```

`RecordWriter` constructor takes `RecordingMap` (new fields) instead of
vertex-based map. `basis_` = `"gll"`.

#### Layer 5: `postprocess/cpp/reader.hh` + `main.cpp` + `writer.hh`

**reader.hh**: read 4D `[n_rec_cell, 125, ncomp]` datasets. Read
`cell_gll_node_ids`, `gll_node_ids`, `gll_node_coords` from record files.

**main.cpp `merge_direction`**:

```cpp
// 1. Read cell-level data: strain[n_steps, n_rec_cell, 125, 6]
//    displacement[n_steps, n_rec_cell, 125, 3], etc.

// 2. Read mass weight from model.h5:
//    jacobian[n_cell, 5,5,5], density[n_cell, 5,5,5]
//    M[ci, n] = density[rec_cell, n] × jacobian[rec_cell, n] × w[i] × w[j] × w[k]
//    (i,j,k = unravel(n, 5,5,5))

// 3. Strain L2 projection:
std::vector<double> strain_weighted(n_unique_gll * 6, 0.0);
std::vector<double> weight_total(n_unique_gll, 0.0);
for (step, ci, n) {
    int global_node = cell_gll_node_index[ci * 125 + n];
    double w = M[ci * 125 + n];
    for (c = 0..5)
        strain_weighted[global_node*6+c] += strain[step, ci, n, c] × w;
    weight_total[global_node] += w;
}
for (global_node)
    strain_global[global_node] = strain_weighted / weight_total;  // continuous

// 4. Displacement deduplication (values identical, take first):
for (ci, n) {
    int global_node = cell_gll_node_index[ci * 125 + n];
    if (!filled[global_node])
        displacement_global[global_node] = displacement[step, ci, n];
}

// 5. Green tensor: [nt, n_unique_gll, 6, 3]
```

**writer.hh** tile schema:

```
tile_{src}_{dir}.h5:
  attrs: basis="gll", source_xyz_m, source_directions, ngll, n_unique_gll, n_rec_cell
  /time/t [nt] float64
  /mesh/gll_node_ids [n_unique_gll] int64
  /mesh/gll_node_coords [n_unique_gll, 3] float64
  /mesh/cell_ids [n_rec_cell] int32
  /mesh/cell_gll_node_index [n_rec_cell, 125] int32
  /source/stf_t [nt] float64
  /source/stf_values [nt] float64
  /field/greens_tensor [nt, n_unique_gll, 6, 3]
  /field/displacement_tensor [nt, n_unique_gll, 3, 3]
  /field/velocity_tensor [nt, n_unique_gll, 3, 3]
  /field/acceleration_tensor [nt, n_unique_gll, 3, 3]
```

#### Layer 6: `greenfun/interpolator.py` + `library.py`

New `GLLInterpolator` class (alongside existing `TrilinearInterpolator`):

```python
class GLLInterpolator:
    """Spectral-accuracy GLL Lagrange interpolation."""

    def __init__(self, gll_node_coords, cell_ids, cell_gll_node_index, ngll=5):
        # Build cell locator: KDTree of cell centers (or grid lookup for Cartesian)
        # Store GLL reference points (ξ_i) and precompute Lagrange polynomials

    def locate_cell(self, point_xyz):
        """Find cell containing the query point."""

    def to_reference(self, point_xyz, cell):
        """Map physical → reference coord (ξ,η,ζ) ∈ [-1,1]^3.
        For Cartesian: ξ = 2(x - origin)/cell_size - 1"""

    def interpolate(self, point_xyz, values):
        """values: [n_unique_gll, ...]. Returns interpolated value.
        1. locate cell
        2. idx = cell_gll_node_index[cell]  → 125 node indices
        3. gather = values[idx]  → [125, ...]
        4. compute Lagrange basis L_i(ξ), L_j(η), L_k(ζ)
        5. result = Σ gather[i,j,k] × L_i(ξ) × L_j(η) × L_k(ζ)
        """

    def query_exact(self, point_xyz, values, tol=1e-6):
        """Direct node value if point matches a GLL node (no interpolation)."""
```

`library.py query`: detect tile `basis` attribute:

- `"gll"` → use `GLLInterpolator`
- `"mesh_vertices"` → use `TrilinearInterpolator` (backward compat for old tiles)

#### Layer 7: config + partition + io + tests

- `config.py`: `record_basis = "gll"` (default; corner mode removed)
- `config_writer.py`: write new recording-map fields to config.h5
- `partition.py`: pass new recording map structure to partition files
- `forward/share/src/io.cpp`: read new recording map format from partition files
- Tests: `test_record.cpp`, `test_io.cpp`, `greenfun/test_library.py`,
  `preprocess/test_recording_map.py`, `postprocess` integration tests

## 6. Record File Schema

```
wavefields/{direction}/record_{rank}_{step}.h5

/strain          [1, n_rec_cell, 125, 6]   float32|float64
/displacement    [1, n_rec_cell, 125, 3]
/velocity        [1, n_rec_cell, 125, 3]
/acceleration    [1, n_rec_cell, 125, 3]
/cell_gll_node_ids [n_rec_cell, 125]       int32
/gll_node_ids    [n_unique_gll]            int64
/gll_node_coords [n_unique_gll, 3]         float64

attrs:
  basis = "gll"
  ngll = 5
  n_rec_cell
  n_unique_gll
  record_depth_max_m
  record_depth_actual_m
  source_direction
  rank
```

## 7. Tile Schema

```
tile_{src}_{dir}.h5

attrs:
  basis = "gll"
  source_xyz_m = [x, y, z]
  source_directions = [list of force directions]
  ngll = 5
  n_unique_gll
  n_rec_cell
  record_depth_max_m
  record_depth_actual_m

/time/t                    [nt]                     float64
/mesh/gll_node_ids         [n_unique_gll]           int64
/mesh/gll_node_coords      [n_unique_gll, 3]        float64
/mesh/cell_ids             [n_rec_cell]             int32
/mesh/cell_gll_node_index  [n_rec_cell, 125]        int32
/source/stf_t              [nt]                     float64
/source/stf_values         [nt]                     float64
/field/greens_tensor       [nt, n_unique_gll, 6, 3] float32|float64
/field/displacement_tensor [nt, n_unique_gll, 3, 3]
/field/velocity_tensor     [nt, n_unique_gll, 3, 3]
/field/acceleration_tensor [nt, n_unique_gll, 3, 3]
```

## 8. GLL Interpolation Math

### 8.1 Reference coordinate mapping

For a regular Cartesian cell with origin `(x0, y0, z0)` and size `(dx, dy, dz)`:

```
ξ = 2(x - x0)/dx - 1
η = 2(y - y0)/dy - 1
ζ = 2(z - z0)/dz - 1
```

`(ξ, η, ζ) ∈ [-1, 1]^3` is the reference coordinate.

### 8.2 GLL Lagrange basis

The 1D GLL Lagrange polynomial of degree N at GLL point `ξ_i`:

```
L_i(ξ) = Π_{j≠i} (ξ - ξ_j) / (ξ_i - ξ_j)
```

3D basis is the tensor product: `L_{ijk}(ξ,η,ζ) = L_i(ξ) × L_j(η) × L_k(ζ)`.

### 8.3 Interpolation

```
u(P) = Σ_{i,j,k=0}^{N} u(node_{ijk}) × L_i(ξ_P) × L_j(η_P) × L_k(ζ_P)
```

This is spectral-accuracy (order N+1) for smooth fields.

### 8.4 Direct node-value mode

If the query point matches a GLL node within tolerance (`|P - node_coord| < tol`),
return the node value directly (zero interpolation error). This is the "pure
solver accuracy" mode.

## 9. Data Volume Analysis

Halfspace example: n_cell=2916, NGLL=5, recording region ~1000 cells (surface
layer within `record_depth_max_m=2000m`).

| Component | Size | Notes |
|-----------|------|-------|
| Recording cells | ~1000 | surface layer |
| Unique GLL nodes | ~40,000-70,000 | after dedup (125/cell, shared ~2-3x) |
| Record file/step | ~1000 × 125 × (6+3+3+3) × 4B = ~7.5 MB | per rank per step |
| Total records | 500 steps × 7.5 MB = ~3.75 GB | per direction |
| Tile (deduped) | 500 × 50000 × (6+3+3+3) × 4B × 3 dirs = ~5.4 GB | after L2 projection |

Compared to current corner tiles (~100 MB total), this is ~50x larger but
enables spectral-accuracy queries. Acceptable for the recording region scope.

## 10. Testing Strategy

### 10.1 Unit tests

- `preprocess/test_recording_map.py`: verify `cell_gll_node_index` correctness,
  `gll_node_ids` deduplication, coordinate computation
- `test_record.cpp`: write + read 4D record format, verify 125-node extraction
- `test_io.cpp`: read new recording map from partition files
- `greenfun/test_library.py`: GLL interpolation accuracy (polynomial exactness),
  direct node-value mode, cell locator

### 10.2 Integration tests

- Halfspace example end-to-end: preprocess → forward → postprocess → greenfun
- Verify: tile `basis="gll"`, `cell_gll_node_index` valid, L2-projected strain
  continuous (shared nodes have identical values from all containing cells)
- Compare GLL-interpolated query vs analytic reference: expect lower rel_l2 than
  current trilinear (target: < 0.3 vs current ~0.58)

### 10.3 Regression

- Green tensor convention `[disp_comp, force_dir]` preserved
- STF storage in tiles preserved
- Precision (`snapshot_precision`) respected

## 11. Compatibility & Migration

### 11.1 No backward compatibility

Following project convention ("No backward compatibility for partition files"):

- Old partition files and record files are incompatible → re-preprocess required
- Old corner tiles (`basis="mesh_vertices"`) still readable by greenfun
  (TrilinearInterpolator kept for backward compat)

### 11.2 Migration path

1. Update preprocess → regenerate model.h5 + partition files
1. Update solver → re-run forward (new record format)
1. Update postprocess → regenerate tiles (`basis="gll"`)
1. greenfun auto-detects tile basis

## 12. Resolved Questions

All design questions were resolved during investigation:

1. **GLL node coordinates**: available in `model.h5` `/field/cell/coords`
   `[n_cell, 5, 5, 5, 3]`, computed by `gll_geometry.py`. No new computation
   needed.

1. **Mass weight for L2 projection**: available in `model.h5`
   `/field/cell/mass` `[n_cell, 5, 5, 5]`, already =
   `density × jacobian × w_i × w_j × w_k`. Postprocess reads directly.

1. **Density format**: per-cell-node `[n_cell, 5, 5, 5]` (not uniform).

1. **Cell locator for non-Cartesian regions**: PML cells may be deformed, but
   `record_depth_max_m` already excludes PML from the recording region. v1
   GLL tile is restricted to non-PML recording region (regular Cartesian).

1. **STF and source unchanged**: STF storage in tiles (`/source/stf_t`,
   `/source/stf_values`) is independent of recording basis. No change needed.

1. **GLL weights**: available in both Python (`gll_geometry.gll_weights`) and
   C++ (`gf/gll.hpp:87`).

## 13. Summary

This design replaces 8-corner vertex recording with full 125-node GLL recording,
makes strain continuous via postprocess L2 projection (solver unchanged), and
adds spectral-accuracy GLL Lagrange interpolation to the greenfun library. The
solver core (element kernel, assembly, Newmark) is untouched — changes are
confined to the recording extraction loop, record/postprocess I/O, and the query
interpolator.
