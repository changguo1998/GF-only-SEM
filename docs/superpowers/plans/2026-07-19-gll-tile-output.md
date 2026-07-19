# GLL-Point Tile Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 8-corner vertex recording with full 125-node GLL recording, make strain continuous via postprocess L2 projection, and add spectral-accuracy GLL Lagrange interpolation to the greenfun library.

**Architecture:** Solver core unchanged. Changes confined to: (1) preprocess recording map selects GLL nodes, (2) solver extracts 125 nodes per recording cell, (3) record files store 4D `[n_rec_cell, 125, ncomp]`, (4) postprocess L2-projects strain to continuous global nodes + deduplicates displacement, (5) greenfun gains GLLInterpolator. Strain L2 projection uses mass weight from `model.h5 /field/cell/mass` (already computed). See [`docs/superpowers/specs/2026-07-19-gll-tile-design.md`](../specs/2026-07-19-gll-tile-design.md) for full design.

**Tech Stack:** C++17 (forward + postprocess), Python (preprocess + greenfun), HDF5, Catch2 (C++ tests), pytest (Python tests)

## Global Constraints

- **Naming:** `{scope}_{mesh}_{parameter}` convention. `element` = scope only, `cell` = mesh only. Full English words, no abbreviations (e.g. `displacement` not `u`, `cell_gll_node_index` not `cgi`).
- **Format before commit:** Run `bash format.sh` before every `git add` / `git commit`.
- **Config source of truth:** All simulation parameters live in `config.py`. No duplicated constants.
- **No backward compatibility:** Old partition files and record files are incompatible; re-preprocess required.
- **Test commands:** Python: `pytest tests/path/test.py -v`. C++: `cd build && ctest -R test_name --output-on-failure`.
- **Build:** `source $HOME/.spack/share/spack/setup_env.sh && spack load cuda && spack load /zkrqzmds` before building forward/postprocess.
- **Spack env:** `source env_setup.sh` for full environment.

______________________________________________________________________

### Task 1: recording_map.py - GLL Node Selection

**Files:**

- Modify: `preprocess/recording_map.py` (rewrite `_build_rank_recording` + `build_recording_map`)
- Test: `tests/preprocess/test_recording_map.py`

**Interfaces:**

- Consumes: `global_cell2global_node` from `model.h5` `[n_cell, 125]`, `coords` from `model.h5` `[n_cell, 5, 5, 5, 3]`

- Produces: `build_recording_map()` returns dict with keys: `rec_cell_global_ids`, `rec_cell_local_index`, `cell_gll_node_ids`, `gll_node_ids`, `gll_node_coords`, `cell_gll_node_index`

- [ ] **Step 1: Write failing test for GLL node selection**

Create `tests/preprocess/test_recording_map.py`:

```python
"""Test GLL-node recording map generation."""
import numpy as np
from preprocess.recording_map import build_recording_map
from preprocess.topology import TopologyData


def test_build_recording_map_selects_gll_nodes():
    """Recording map should select all 125 GLL nodes per recording cell."""
    # Minimal 2x2x1 topology: 2 cells, NGLL=2 (8 nodes/cell)
    # Shared face means some GLL nodes are shared
    ngl = 2
    n_cell = 2
    n_node_per_cell = ngl ** 3  # 8

    # global_cell2global_node: cell 0 has nodes 0-7, cell 1 has nodes 4-11 (4 shared)
    global_cell2global_node = np.array([
        list(range(8)),        # cell 0: 0,1,2,3,4,5,6,7
        list(range(4, 12)),    # cell 1: 4,5,6,7,8,9,10,11
    ], dtype=np.int64)

    # coords: 12 unique nodes on a line (x = 0,1,2,...)
    coords = np.zeros((n_cell, ngl, ngl, ngl, 3), dtype=np.float64)
    for c in range(n_cell):
        for i in range(ngl):
            for j in range(ngl):
                for k in range(ngl):
                    node = global_cell2global_node[c, i, j, k]
                    coords[c, i, j, k] = [float(node), 0.0, 0.0]

    topology = TopologyData(
        n_cell=n_cell,
        global_cell2global_node=global_cell2global_node,
    )

    # Both cells in recording region (depth = 0, all surface)
    is_pml = np.zeros(n_cell, dtype=np.int8)
    domain_bounds = {"zmin": 0.0, "zmax": 100.0, "xmin": 0.0, "xmax": 200.0,
                     "ymin": 0.0, "ymax": 100.0}

    result = build_recording_map(
        topology=topology,
        domain_bounds=domain_bounds,
        is_pml=is_pml,
        record_depth_max_m=100.0,
        gll_node_coords=coords,
    )

    rec = result["per_rank_recording"][0]
    # Both cells recorded
    assert len(rec["rec_cell_global_ids"]) == 2
    assert len(rec["rec_cell_local_index"]) == 2
    # cell_gll_node_ids: [2, 8]
    assert len(rec["cell_gll_node_ids"]) == 2
    assert len(rec["cell_gll_node_ids"][0]) == 8
    # Unique GLL nodes: 0-11 = 12 unique
    assert len(rec["gll_node_ids"]) == 12
    # cell_gll_node_index: [2, 8], values index into gll_node_ids
    assert len(rec["cell_gll_node_index"]) == 2
    for ci in range(2):
        for n in range(8):
            idx = rec["cell_gll_node_index"][ci][n]
            assert 0 <= idx < 12
            # Index maps to correct global node ID
            assert rec["gll_node_ids"][idx] == global_cell2global_node[ci, n // 4, (n // 2) % 2, n % 2]
    # gll_node_coords: [12, 3]
    assert len(rec["gll_node_coords"]) == 12
    assert len(rec["gll_node_coords"][0]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/preprocess/test_recording_map.py::test_build_recording_map_selects_gll_nodes -v`
Expected: FAIL (function signature changed or `gll_node_coords` param missing)

- [ ] **Step 3: Rewrite `_build_rank_recording` for GLL nodes**

Replace the entire `_build_rank_recording` function in `preprocess/recording_map.py`:

```python
def _build_rank_recording(
    rank: int,
    local_cell_ids: list[int],
    recording_cell_set: set[int],
    global_cell2global_node: npt.NDArray[np.int64],
    gll_node_coords_all: npt.NDArray[np.float64],
    ngll: int,
) -> dict[str, Any]:
    """Build GLL-node recording map for one rank.

    Selects all 125 GLL nodes per recording cell. Deduplicates shared nodes
    across cells to produce a unique global node list.
    """
    n_node = ngll * ngll * ngll

    # Select recording cells that are local to this rank
    rec_cell_local = [
        idx for idx, cid in enumerate(local_cell_ids) if cid in recording_cell_set
    ]
    rec_cell_global = [local_cell_ids[idx] for idx in rec_cell_local]

    # Collect GLL node IDs for each recording cell
    cell_gll_node_ids: list[list[int]] = []
    all_node_ids: set[int] = set()
    for gcell in rec_cell_global:
        nodes = global_cell2global_node[gcell].tolist()  # [n_node]
        cell_gll_node_ids.append(nodes)
        all_node_ids.update(nodes)

    # Deduplicate: unique global GLL node IDs
    gll_node_ids = sorted(all_node_ids)
    node_id_to_index = {nid: idx for idx, nid in enumerate(gll_node_ids)}

    # cell_gll_node_index: [n_rec_cell, n_node] -> index into gll_node_ids
    cell_gll_node_index = [
        [node_id_to_index[n] for n in nodes] for nodes in cell_gll_node_ids
    ]

    # gll_node_coords: [n_unique_gll, 3]
    # Build global node ID -> coord map from first occurrence in any cell
    node_coord_map: dict[int, list[float]] = {}
    for ci, gcell in enumerate(rec_cell_global):
        coords_cell = gll_node_coords_all[gcell]  # [ngll, ngll, ngll, 3]
        for i in range(ngll):
            for j in range(ngll):
                for k in range(ngll):
                    nid = cell_gll_node_ids[ci][i * ngll * ngll + j * ngll + k]
                    if nid not in node_coord_map:
                        node_coord_map[nid] = coords_cell[i, j, k].tolist()
    gll_node_coords = [node_coord_map[nid] for nid in gll_node_ids]

    return {
        "rec_cell_global_ids": rec_cell_global,
        "rec_cell_local_index": rec_cell_local,
        "cell_gll_node_ids": cell_gll_node_ids,
        "gll_node_ids": gll_node_ids,
        "gll_node_coords": gll_node_coords,
        "cell_gll_node_index": cell_gll_node_index,
    }
```

- [ ] **Step 4: Update `build_recording_map` signature and body**

Update `build_recording_map` to accept `gll_node_coords` and `global_cell2global_node`, select recording cells by depth, and call `_build_rank_recording`. Replace the cell-vertex-map logic (which used `_get_cell_vertex_ids`) with GLL-node logic. The function should:

1. Determine recording cells: cells whose centroid z is within `record_depth_max_m` of `zmin`, excluding PML cells.
1. For single-rank (no `element_to_rank`), call `_build_rank_recording` with all cells.
1. For multi-rank, call per-rank with local cell lists.
1. Return `{"record_depth_actual_m": ..., "per_rank_recording": {rank: {...}}}`.

Key changes: remove `cell_vertex_map` construction, remove `_GMSH_CORNER_TO_BITFLAG` usage, add `gll_node_coords` and `global_cell2global_node` parameters.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/preprocess/test_recording_map.py::test_build_recording_map_selects_gll_nodes -v`
Expected: PASS

- [ ] **Step 6: Update callers of `build_recording_map`**

Search for callers: `grep -rn 'build_recording_map' preprocess/`. Update `config_writer.py` and any pipeline scripts to pass `global_cell2global_node` and `gll_node_coords` (both available from `model.h5`). Remove references to old output keys (`vertex_ids`, `source_element_local_index`, `source_corner_index`).

- [ ] **Step 7: Commit**

```bash
bash format.sh
git add preprocess/recording_map.py tests/preprocess/test_recording_map.py preprocess/config_writer.py
git commit -m "feat: recording_map selects GLL nodes instead of corner vertices"
```

______________________________________________________________________

### Task 2: types.hpp + io.cpp + config_writer.py - Data Structure and I/O

**Files:**

- Modify: `forward/share/include/gf/types.hpp:67-72` (RecordingMap struct)
- Modify: `forward/share/src/io.cpp` (read recording map from partition)
- Modify: `preprocess/partition.py` (write new recording map to partition files)
- Test: `tests/test_io.cpp`

**Interfaces:**

- Consumes: Task 1 output (`gll_node_ids`, `gll_node_coords`, `rec_cell_local`, `cell_gll_node_index`)

- Produces: C++ `RankData::RecordingMap` with GLL fields, populated by `io.cpp` from partition files

- [ ] **Step 1: Update RecordingMap struct in types.hpp**

Replace the struct at `types.hpp:67-72`:

```cpp
struct RecordingMap {
    bool has_recording = false;
    // GLL node data (global, for displacement/velocity/acceleration dedup)
    std::vector<int64_t> gll_node_ids;         // [n_unique_gll]
    std::vector<double> gll_node_coords;       // [n_unique_gll * 3]
    // Cell data (for strain element-local extraction + GLL interpolation)
    std::vector<int32_t> rec_cell_local;       // [n_rec_cell]
    std::vector<int32_t> cell_gll_node_index;  // [n_rec_cell * 125]
};
```

- [ ] **Step 2: Update partition.py to write new recording map**

In `preprocess/partition.py`, find where recording data is written to partition files (search for `recording` group). Replace the old datasets (`vertex_ids`, `source_element_local_index`, `source_corner_index`) with:

```python
rec_grp = f.create_group("recording")
rec_grp.attrs["has_recording"] = rec_data["has_recording"]
rec_grp.create_dataset("gll_node_ids", data=np.array(rec_data["gll_node_ids"], dtype=np.int64))
rec_grp.create_dataset("gll_node_coords", data=np.array(rec_data["gll_node_coords"], dtype=np.float64))
rec_grp.create_dataset("rec_cell_local", data=np.array(rec_data["rec_cell_local_index"], dtype=np.int32))
rec_grp.create_dataset("cell_gll_node_index", data=np.array(rec_data["cell_gll_node_index"], dtype=np.int32).flatten())
rec_grp.attrs["n_unique_gll"] = len(rec_data["gll_node_ids"])
rec_grp.attrs["n_rec_cell"] = len(rec_data["rec_cell_local_index"])
```

- [ ] **Step 3: Update io.cpp to read new recording map**

In `forward/share/src/io.cpp`, find the recording-map reading code (search for `recording`). Replace reading of `vertex_ids`/`src_elem_local`/`src_corner` with reading of the new datasets. Use the existing `read_attr_string` / HDF5 helpers pattern:

```cpp
// Read recording map (GLL-node format)
if (H5Lexists(part_file, "recording", H5P_DEFAULT) > 0) {
    hid_t rec_gid = H5Gopen2(part_file, "recording", H5P_DEFAULT);
    part.recording.has_recording = true;
    // gll_node_ids [n_unique_gll]
    read_int64_1d(rec_gid, "gll_node_ids", part.recording.gll_node_ids);
    // gll_node_coords [n_unique_gll * 3]
    read_double_1d(rec_gid, "gll_node_coords", part.recording.gll_node_coords);
    // rec_cell_local [n_rec_cell]
    read_int32_1d(rec_gid, "rec_cell_local", part.recording.rec_cell_local);
    // cell_gll_node_index [n_rec_cell * 125]
    read_int32_1d(rec_gid, "cell_gll_node_index", part.recording.cell_gll_node_index);
    H5Gclose(rec_gid);
}
```

If `read_int32_1d` does not exist, add it following the `read_int64_1d` / `read_double_1d` pattern already in `io.cpp`.

- [ ] **Step 4: Write C++ test for new recording map I/O**

Add to `tests/test_io.cpp` a test that writes a partition file with the new recording format and reads it back, verifying all fields. Follow the existing `test_io.cpp` partition round-trip test pattern:

```cpp
TEST_CASE("Recording map GLL format round-trip", "[io]") {
    // Write a minimal partition file with recording group
    // Read it back, verify gll_node_ids, gll_node_coords, rec_cell_local, cell_gll_node_index
    // ...
}
```

- [ ] **Step 5: Build and run test**

Run: `cd forward && cmake --build build && cd ../tests && cmake --build build && ctest -R test_io --output-on-failure`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
bash format.sh
git add forward/share/include/gf/types.hpp forward/share/src/io.cpp preprocess/partition.py tests/test_io.cpp
git commit -m "feat: RecordingMap struct + I/O for GLL-node format"
```

______________________________________________________________________

### Task 3: solver.cpp - Extract 125 Nodes

**Files:**

- Modify: `forward/share/src/solver.cpp:398-443` (record extraction loop)

**Interfaces:**

- Consumes: Task 2 `RecordingMap` (rec_cell_local, cell_gll_node_index, gll_node_ids)

- Produces: `rec_strain [n_rec_cell * 125 * 6]`, `rec_displacement [n_rec_cell * 125 * 3]`, etc.

- [ ] **Step 1: Replace corner extraction with 125-node extraction**

In `solver.cpp`, find the record extraction block (lines ~398-443, inside `if (cfg.snapshot_stride > 0 && step % cfg.snapshot_stride == 0)`). Replace the `for (size_t vertex_idx = 0; vertex_idx < n_vertices; ++vertex_idx)` loop with:

```cpp
if (has_recording) {
    if (gpu_state.use_global_dof) {
        cuda_gather_from_rank(gpu_state);
    }
    size_t n_rec_cell = part.recording.rec_cell_local.size();
    int n_node = ngll * ngll * ngll;  // 125
    rec_strain.resize(n_rec_cell * n_node * 6, 0.0);
    rec_displacement.resize(n_rec_cell * n_node * 3, 0.0);
    rec_velocity.resize(n_rec_cell * n_node * 3, 0.0);
    rec_acceleration.resize(n_rec_cell * n_node * 3, 0.0);

    // GPU: strain already computed for all elements by cuda_compute_strain
    // CPU: element_strain already available
    cuda_compute_strain(gpu_state, D_mat.data(), ngll, part.dxi_dx);
    cuda_copy_strain_to_host(gpu_state, rec_strain.data());
    // Note: rec_strain now holds [n_local_cell * 125 * 6]; we extract recording subset below

    cuda_copy_state_to_host(gpu_state, displacement, velocity, acceleration);
    for (size_t ci = 0; ci < n_rec_cell; ++ci) {
        int elem = part.recording.rec_cell_local[ci];
        for (int n = 0; n < n_node; ++n) {
            // strain: element-local, copy from element_strain
            for (int c = 0; c < 6; ++c)
                rec_strain[(ci * n_node + n) * 6 + c] =
                    element_strain[elem * n_node * 6 + n * 6 + c];
            // displacement/velocity/acceleration: global DOF
            int node_id = part.local_cell2rank_node[elem * n_node + n];
            for (int d = 0; d < 3; ++d) {
                rec_displacement[(ci * n_node + n) * 3 + d] = displacement[node_id * 3 + d];
                rec_velocity[(ci * n_node + n) * 3 + d] = velocity[node_id * 3 + d];
                rec_acceleration[(ci * n_node + n) * 3 + d] = acceleration[node_id * 3 + d];
            }
        }
    }
    record.write_step(step, rec_strain.data(), rec_displacement.data(),
                      rec_velocity.data(), rec_acceleration.data());
}
```

**Important:** The CPU path has a similar block (search for `// --- CPU path ---`). Apply the same change there. The CPU path uses `element_strain` computed by `compute_element_residual`. Verify `element_strain` is available as a flat `[n_local_cell * 125 * 6]` array in both paths.

- [ ] **Step 2: Build solver**

Run: `cd forward && cmake --build build`
Expected: Build succeeds. Fix any compile errors from the changed RecordingMap fields.

- [ ] **Step 3: Verify no test regressions**

Run: `cd tests && cmake --build build && ctest --output-on-failure`
Expected: Existing tests pass (some may need updates if they reference old vertex fields - fix as needed).

- [ ] **Step 4: Commit**

```bash
bash format.sh
git add forward/share/src/solver.cpp
git commit -m "feat: solver extracts 125 GLL nodes per recording cell"
```

______________________________________________________________________

### Task 4: record.hpp + record.cpp - 4D Record Format

**Files:**

- Modify: `forward/share/include/gf/record.hpp` (RecordWriter constructor + write_step signature)
- Modify: `forward/share/src/record.cpp` (write_step writes 4D + new datasets)

**Interfaces:**

- Consumes: Task 2 `RecordingMap` (gll_node_ids, gll_node_coords, cell_gll_node_index)

- Produces: `record_{rank}_{step}.h5` with 4D datasets + mesh metadata

- [ ] **Step 1: Update RecordWriter constructor in record.hpp**

Change the constructor to accept the new RecordingMap and store GLL metadata. Update `write_step` to accept `n_rec_cell` and `n_node` (or store them from constructor):

```cpp
class RecordWriter {
public:
    RecordWriter(const std::string& output_dir, const std::string& source_direction,
                 int rank, const RankData::RecordingMap& rec_map, int ngll,
                 CompressionConfig compression, bool use_float32 = false,
                 double record_depth_max_m = 0.0, double record_depth_actual_m = 0.0);

    void write_step(int step, const double* strain, const double* displacement = nullptr,
                    const double* velocity = nullptr, const double* acceleration = nullptr);

    int n_rec_cell() const { return static_cast<int>(rec_cell_local_.size()); }
    int n_unique_gll() const { return static_cast<int>(gll_node_ids_.size()); }
    // ... existing close(), n_vertices() removed
private:
    // ... existing fields
    std::string basis_ = "gll";
    std::vector<int64_t> gll_node_ids_;
    std::vector<double> gll_node_coords_;
    std::vector<int32_t> rec_cell_local_;
    std::vector<int32_t> cell_gll_node_index_;
    int ngll_;
};
```

- [ ] **Step 2: Update write_step in record.cpp to write 4D datasets**

Replace `write_field` (which writes `[1, n_vertices, ncomp]`) with a 4D version `[1, n_rec_cell, 125, ncomp]`. Add writing of mesh metadata datasets:

```cpp
void RecordWriter::write_step(int step, const double* strain, const double* displacement,
                              const double* velocity, const double* acceleration) {
    // Create file record_{rank}_{step}.h5
    std::string filename = output_dir_ + "/record_" + std::to_string(rank_) + "_"
                         + std::to_string(step) + ".h5";
    hid_t file_id = H5Fcreate(filename.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    int n_node = ngll_ * ngll_ * ngll_;
    hsize_t n_rec = n_rec_cell();

    // Write 4D field datasets [1, n_rec_cell, n_node, ncomp]
    write_field_4d(file_id, "strain", 6, n_rec, n_node, use_float32_, strain);
    write_field_4d(file_id, "displacement", 3, n_rec, n_node, use_float32_, displacement);
    write_field_4d(file_id, "velocity", 3, n_rec, n_node, use_float32_, velocity);
    write_field_4d(file_id, "acceleration", 3, n_rec, n_node, use_float32_, acceleration);

    // Write mesh metadata (same for every step, but per-file for self-containment)
    write_int64_1d(file_id, "gll_node_ids", gll_node_ids_);
    write_double_1d(file_id, "gll_node_coords", gll_node_coords_);
    write_int32_2d(file_id, "cell_gll_node_index", cell_gll_node_index_, n_rec, n_node);

    // Write attributes
    write_string_attr(file_id, "basis", "gll");
    write_int_attr(file_id, "ngll", ngll_);
    write_int_attr(file_id, "n_rec_cell", (int)n_rec);
    write_int_attr(file_id, "n_unique_gll", (int)gll_node_ids_.size());
    // ... record_depth_max_m, record_depth_actual_m, source_direction, rank

    H5Fclose(file_id);
}
```

Add helper `write_field_4d` (4D `[1, n_rec, n_node, ncomp]` version of existing `write_field`). Add `write_int32_2d` if not present.

- [ ] **Step 3: Update C++ test for record format**

Update `tests/test_record.cpp` to write and read the 4D format. Verify dataset shapes are `[1, n_rec_cell, 125, ncomp]` and mesh metadata is present.

- [ ] **Step 4: Build and run test**

Run: `cd tests && cmake --build build && ctest -R test_record --output-on-failure`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
bash format.sh
git add forward/share/include/gf/record.hpp forward/share/src/record.cpp tests/test_record.cpp
git commit -m "feat: record files use 4D GLL-node format [n_rec_cell, 125, ncomp]"
```

______________________________________________________________________

### Task 5: postprocess reader.hh + main.cpp - L2 Projection

**Files:**

- Modify: `postprocess/cpp/reader.hh` (read 4D records + mesh metadata)
- Modify: `postprocess/cpp/main.cpp` (merge_direction: L2 projection + deduplication)

**Interfaces:**

- Consumes: Task 4 record files (4D + mesh metadata), `model.h5 /field/cell/mass`

- Produces: `MergedDirection` with `[n_steps, n_unique_gll, ncomp]` (continuous strain, deduped displacement)

- [ ] **Step 1: Update reader.hh to read 4D records**

Add a function to read `[n_rec_cell, 125, ncomp]` datasets from record files. Also read `gll_node_ids`, `gll_node_coords`, `cell_gll_node_index` from the first record file (same for all steps). Add reading of `n_rec_cell`, `n_unique_gll`, `ngll` attributes.

- [ ] **Step 2: Add mass-weight reading from model.h5**

In `reader.hh`, add a function to read `/field/cell/mass` `[n_cell, 5, 5, 5]` from `model.h5`. This is the L2 projection weight `M = density × jacobian × w_i × w_j × w_k` (already computed).

- [ ] **Step 3: Implement L2 projection in main.cpp merge_direction**

Replace the current merge logic (which merges by `vertex_id` into `[n_steps, n_vertex, ncomp]`) with:

```cpp
// Read cell-level data: strain[n_steps, n_rec_cell, 125, 6]
// For each step:
//   1. Strain L2 projection:
//      strain_weighted[global_node] += strain[ci, n] × mass[rec_cell, n]
//      weight_total[global_node] += mass[rec_cell, n]
//      strain_global[global_node] = strain_weighted / weight_total
//   2. Displacement deduplication:
//      displacement_global[global_node] = displacement[ci, n] (take first)
// Result: [n_steps, n_unique_gll, ncomp]
```

Key: `global_node = cell_gll_node_index[ci * 125 + n]`. Mass weight: `mass[rec_cell_global_id, i, j, k]` where `(i,j,k) = unravel(n, ngll, ngll, ngll)`.

- [ ] **Step 4: Build postprocess**

Run: `cd postprocess && cmake --build build`
Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
bash format.sh
git add postprocess/cpp/reader.hh postprocess/cpp/main.cpp
git commit -m "feat: postprocess L2-projects strain to continuous global nodes"
```

______________________________________________________________________

### Task 6: postprocess writer.hh - GLL Tile Schema

**Files:**

- Modify: `postprocess/cpp/writer.hh` (write GLL tile with new schema)

**Interfaces:**

- Consumes: Task 5 `MergedDirection` (continuous, `[n_unique_gll]`), mesh metadata

- Produces: `tile_{src}_{dir}.h5` with `basis="gll"`, `/mesh/` group, `/field/` 4D tensors

- [ ] **Step 1: Update write_tile for GLL schema**

Update the tile writer to write the schema from spec §7:

```
attrs: basis="gll", source_xyz_m, source_directions, ngll, n_unique_gll, n_rec_cell
/time/t [nt]
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

The writer needs `gll_node_ids`, `gll_node_coords`, `cell_ids`, `cell_gll_node_index` passed from `main.cpp` (read from record files in Task 5).

- [ ] **Step 2: Build and verify tile output**

Run: `cd postprocess && cmake --build build`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
bash format.sh
git add postprocess/cpp/writer.hh postprocess/cpp/main.cpp
git commit -m "feat: postprocess writes GLL-basis tiles"
```

______________________________________________________________________

### Task 7: greenfun interpolator.py + library.py - GLL Interpolation

**Files:**

- Create: `greenfun/gll_interpolator.py` (GLLInterpolator class)
- Modify: `greenfun/library.py` (detect tile basis, select interpolator)
- Test: `tests/greenfun/test_gll_interpolator.py`

**Interfaces:**

- Consumes: Task 6 tile files (`basis="gll"`, `/mesh/cell_gll_node_index`, `/mesh/gll_node_coords`)

- Produces: `GLLInterpolator.interpolate(point, values)` -> spectral-accuracy interpolation

- [ ] **Step 1: Write failing test for GLL interpolation**

Create `tests/greenfun/test_gll_interpolator.py`:

```python
"""Test GLL Lagrange interpolation."""
import numpy as np
from greenfun.gll_interpolator import GLLInterpolator


def test_gll_interpolation_polynomial_exactness():
    """GLL interpolation of a polynomial of degree <= N should be exact."""
    ngll = 5
    # Single cell [-1,1]^3, 125 GLL nodes
    from preprocess.gll_geometry import gll_quadrature_points, gll_weights
    pts = gll_quadrature_points(ngll - 1)  # 5 points in [-1,1]

    # GLL node coords: tensor product of pts
    coords = np.array([[x, y, z] for x in pts for y in pts for z in pts],
                      dtype=np.float64)

    # cell_gll_node_index: single cell, 125 nodes = 0..124
    cell_gll_node_index = np.arange(125).reshape(1, 125)

    # Test polynomial: f(x,y,z) = 1 + x + y + z + x*y*z (degree 3 <= 4)
    values = 1 + coords[:, 0] + coords[:, 1] + coords[:, 2] + coords[:, 0] * coords[:, 1] * coords[:, 2]

    interp = GLLInterpolator(coords, cell_gll_node_index, ngll=ngll,
                             cell_origin=[-1, -1, -1], cell_size=[2, 2, 2])

    # Query at interior point
    query = np.array([0.3, -0.2, 0.15])
    result = interp.interpolate(query, values)
    expected = 1 + query[0] + query[1] + query[2] + query[0] * query[1] * query[2]
    assert abs(result - expected) < 1e-10, f"GLL interp {result} != exact {expected}"


def test_gll_direct_node_value():
    """Query at a GLL node should return the exact node value."""
    ngll = 5
    from preprocess.gll_geometry import gll_quadrature_points
    pts = gll_quadrature_points(ngll - 1)
    coords = np.array([[x, y, z] for x in pts for y in pts for z in pts], dtype=np.float64)
    cell_gll_node_index = np.arange(125).reshape(1, 125)
    values = np.random.rand(125)

    interp = GLLInterpolator(coords, cell_gll_node_index, ngll=ngll,
                             cell_origin=[-1, -1, -1], cell_size=[2, 2, 2])

    # Query at node 42
    result = interp.interpolate(coords[42], values)
    assert abs(result - values[42]) < 1e-10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/greenfun/test_gll_interpolator.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement GLLInterpolator**

Create `greenfun/gll_interpolator.py`:

```python
"""Spectral-accuracy GLL Lagrange interpolation for Green's function tiles."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree


class GLLInterpolator:
    """GLL Lagrange interpolation over recording-region cells.

    Uses cell_gll_node_index to gather 125 GLL node values per cell,
    then evaluates the tensor-product Lagrange polynomial at the query point.
    """

    def __init__(
        self,
        gll_node_coords: npt.NDArray[np.float64],
        cell_gll_node_index: npt.NDArray[np.int32],
        ngll: int = 5,
        cell_origins: npt.NDArray[np.float64] | None = None,
        cell_sizes: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.ngll = ngll
        self.n_node = ngll ** 3
        self.gll_node_coords = np.asarray(gll_node_coords, dtype=np.float64)
        self.cell_gll_node_index = np.asarray(cell_gll_node_index, dtype=np.int32)

        # GLL reference points in [-1, 1]
        from preprocess.gll_geometry import gll_quadrature_points
        self.gll_pts = gll_quadrature_points(ngll - 1)

        # Precompute Lagrange polynomials evaluated at GLL points (identity)
        # We need L_i(xi) for arbitrary xi, computed on-the-fly
        self._lagrange_cache: dict[float, np.ndarray] = {}

        # Cell locator: KDTree of cell centers (or grid for Cartesian)
        n_rec_cell = self.cell_gll_node_index.shape[0]
        if cell_origins is not None and cell_sizes is not None:
            self.cell_origins = np.asarray(cell_origins, dtype=np.float64)
            self.cell_sizes = np.asarray(cell_sizes, dtype=np.float64)
            self.cell_centers = self.cell_origins + self.cell_sizes / 2.0
        else:
            # Infer from GLL node coords: cell center = mean of 125 nodes
            self.cell_centers = np.zeros((n_rec_cell, 3))
            for ci in range(n_rec_cell):
                idx = self.cell_gll_node_index[ci]
                self.cell_centers[ci] = self.gll_node_coords[idx].mean(axis=0)
            # Infer cell size from first cell (Cartesian assumption)
            # ... (compute from GLL node spacing)

        self._cell_kdtree = KDTree(self.cell_centers)
        self._exact_kdtree = KDTree(self.gll_node_coords)

    def _lagrange_basis(self, xi: float) -> np.ndarray:
        """Evaluate all NGLL Lagrange polynomials at xi. Cached."""
        if xi not in self._lagrange_cache:
            n = self.ngll
            result = np.ones(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        result[i] *= (xi - self.gll_pts[j]) / (self.gll_pts[i] - self.gll_pts[j])
            self._lagrange_cache[xi] = result
        return self._lagrange_cache[xi]

    def locate_cell(self, point_xyz: np.ndarray) -> int:
        """Find the cell containing the query point."""
        dist, idx = self._cell_kdtree.query(point_xyz)
        return int(idx)

    def to_reference(self, point_xyz: np.ndarray, cell_idx: int) -> np.ndarray:
        """Map physical -> reference coord (xi, eta, zeta) in [-1,1]^3.

        For Cartesian cell: xi = 2*(x - origin)/size - 1
        """
        origin = self.cell_centers[cell_idx] - self.cell_sizes[cell_idx] / 2.0 \
            if hasattr(self, 'cell_sizes') else \
            self.gll_node_coords[self.cell_gll_node_index[cell_idx, 0]]
        # For Cartesian: use cell extent from GLL node coords
        cell_nodes = self.gll_node_coords[self.cell_gll_node_index[cell_idx]]
        origin = cell_nodes.min(axis=0)
        extent = cell_nodes.max(axis=0) - origin
        return 2.0 * (point_xyz - origin) / extent - 1.0

    def interpolate(self, point_xyz: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Interpolate values at point using GLL Lagrange basis.

        values: [n_unique_gll, ...]. Returns interpolated value(s).
        """
        point_xyz = np.asarray(point_xyz, dtype=np.float64)

        # Direct node-value mode: check if point matches a GLL node
        dist, node_idx = self._exact_kdtree.query(point_xyz)
        if dist < 1e-6:
            return values[node_idx]

        # Locate cell and map to reference coords
        ci = self.locate_cell(point_xyz)
        xi_eta_zeta = self.to_reference(point_xyz, ci)
        xi, eta, zeta = xi_eta_zeta

        # Lagrange basis at query point
        Lx = self._lagrange_basis(xi)
        Ly = self._lagrange_basis(eta)
        Lz = self._lagrange_basis(zeta)

        # Gather 125 node values for this cell
        idx = self.cell_gll_node_index[ci]  # [125]
        gathered = values[idx]  # [125, ...]

        # Reshape to [ngll, ngll, ngll, ...] and contract with Lx ⊗ Ly ⊗ Lz
        shape = gathered.shape
        gathered_3d = gathered.reshape(self.ngll, self.ngll, self.ngll, *shape[1:])
        # result = sum_{i,j,k} gathered[i,j,k] * Lx[i] * Ly[j] * Lz[k]
        result = np.tensordot(Lz, gathered_3d, axes=([0], [2]))  # [ngll, ngll, ...]
        result = np.tensordot(Ly, result, axes=([0], [1]))  # [ngll, ...]
        result = np.tensordot(Lx, result, axes=([0], [0]))  # [...]
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/greenfun/test_gll_interpolator.py -v`
Expected: PASS

- [ ] **Step 5: Update library.py to detect tile basis**

In `greenfun/library.py`, update the `query` method (or `SourceRun` class) to read the tile `basis` attribute and select the interpolator:

```python
# When loading a tile:
basis = tile_attrs.get("basis", "mesh_vertices")
if basis == "gll":
    from greenfun.gll_interpolator import GLLInterpolator
    interpolator = GLLInterpolator(
        gll_node_coords=tile["/mesh/gll_node_coords"][:],
        cell_gll_node_index=tile["/mesh/cell_gll_node_index"][:],
        ngll=tile_attrs["ngll"],
    )
elif basis == "mesh_vertices":
    interpolator = TrilinearInterpolator(vertex_coords=tile["/mesh/vertex_coords"][:])
```

Keep `TrilinearInterpolator` for backward compatibility with old corner tiles.

- [ ] **Step 6: Run all greenfun tests**

Run: `pytest tests/greenfun/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
bash format.sh
git add greenfun/gll_interpolator.py greenfun/library.py tests/greenfun/test_gll_interpolator.py
git commit -m "feat: GLLInterpolator with spectral-accuracy Lagrange interpolation"
```

______________________________________________________________________

### Task 8: config.py + End-to-End Integration

**Files:**

- Modify: `examples/halfspace/config.py` (add `record_basis`)
- Modify: `examples/layer/config.py` (same)
- Verify: `examples/halfspace/` end-to-end pipeline

**Interfaces:**

- Consumes: Tasks 1-7 (full pipeline)

- Produces: Working GLL-tile pipeline with verified accuracy

- [ ] **Step 1: Add record_basis to config.py**

In `examples/halfspace/config.py` and `examples/layer/config.py`, add:

```python
# Recording basis: "gll" for full GLL-node tiles (spectral interpolation),
# "mesh_vertices" for legacy 8-corner tiles (trilinear, deprecated).
record_basis = "gll"
```

- [ ] **Step 2: Run halfspace preprocess**

Run: `cd examples/halfspace && bash preprocess.sh`
Expected: `model.h5` + `config.h5` + `partition_*.h5` generated with GLL recording map. Verify partition file has `/recording/gll_node_ids`, `/recording/cell_gll_node_index`.

- [ ] **Step 3: Run halfspace forward**

Run: `cd examples/halfspace && bash forward.sh`
Expected: `wavefields/*/record_{rank}_{step}.h5` with 4D datasets. Verify with:

```bash
python3 -c "import h5py; f=h5py.File('wavefields/x/record_0_0.h5','r'); print(f['/strain'].shape, f.attrs['basis'])"
```

Expected: `(1, n_rec_cell, 125, 6)` and `b'gll'`

- [ ] **Step 4: Run halfspace postprocess**

Run: `cd examples/halfspace && bash postprocess.sh`
Expected: `greenfun/tile_*.h5` with `basis="gll"`, `/mesh/cell_gll_node_index`, `/field/greens_tensor [nt, n_unique_gll, 6, 3]`. Verify L2-projected strain is continuous (shared nodes have same value from all cells).

- [ ] **Step 5: Run halfspace compare**

Run: `cd examples/halfspace && bash compare.sh`
Expected: GLL-interpolated query vs analytic reference. Target: rel_l2 < 0.3 (improvement over current trilinear ~0.58). If using direct node-value mode (query at a GLL node), expect rel_l2 < 0.1.

- [ ] **Step 6: Run layer example (optional, if time permits)**

Run: `cd examples/layer && bash preprocess.sh && bash forward.sh && bash postprocess.sh && bash compare.sh`
Expected: Similar results.

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v && cd tests && cmake --build build && ctest --output-on-failure`
Expected: All tests pass (update any tests that reference old vertex-based recording fields).

- [ ] **Step 8: Commit**

```bash
bash format.sh
git add examples/halfspace/config.py examples/layer/config.py
git commit -m "feat: GLL tile pipeline end-to-end verified on halfspace example"
```

______________________________________________________________________

## Self-Review Notes

**Spec coverage:** All 7 spec layers covered by Tasks 1-7. Task 8 covers integration (spec §10.2).

**Dependency order:** Task 1 (preprocess) -> Task 2 (types/IO) -> Tasks 3-4 (solver/record) -> Tasks 5-6 (postprocess) -> Task 7 (greenfun) -> Task 8 (integration). Tasks 3-4 can be done together (solver writes what record outputs). Tasks 5-6 together (reader feeds writer).

**Type consistency:** `cell_gll_node_index` is `int32` in C++ (`types.hpp`) and `int32` in Python (partition.py). `gll_node_ids` is `int64` everywhere. `gll_node_coords` is `float64` everywhere.

**Key risk:** Task 3 solver extraction - verify `element_strain` layout is `[n_local_cell * 125 * 6]` (flat, cell-major) in both CPU and GPU paths. If layout differs, adjust indexing.
