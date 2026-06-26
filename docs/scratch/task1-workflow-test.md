Everything is clean. Here's the summary.

______________________________________________________________________

**Implemented both tasks:**

### Task 1: Integration test (`tests/workflows/test_halfspace_workflow.py`)

Wrote end-to-end pipeline test exercising all 9 preprocessor steps: mesh generation → topology read → GLL geometry → material load → boundary detection → PML damping → partition → model write → config write. Validates 20 assertions covering mesh.h5, config.h5, and partition_0.h5 output.

### Task 2: Fix `_get_cell_vertex_ids` sorting bug

Replaced the old sorted-set approach with GMSH-ordered extraction using signed edge topology. The new function reconstructs the canonical hex vertex ordering from the bottom and top face signed-edge cycles, splitting into `[v0, v1, v2, v3]` (bottom) and `[v4, v5, v6, v7]` (top) per GMSH spec. Updated the test fixture in `test_gll_geometry.py` to use `extract_topology()` instead of the hand-crafted (incorrect) topology.

**Changed files:**

- `preprocess/gll_geometry.py` — replaced `_get_cell_vertex_ids` (sorted → face-cycle GMSH order)
- `tests/preprocess/test_gll_geometry.py` — `_make_unit_cube_topo` now uses `extract_topology()` for GMSH-compliant topology
- `tests/workflows/test_halfspace_workflow.py` — new end-to-end integration test (59 total tests pass)

**Validation:**

- 59/59 tests pass (9 gll_geometry + 1 workflow + 49 other preprocess), 0 regressions
- No staged files, no untracked side-effects outside expected files

**Open risks:**

- `_face_vertex_cycle` assumes signed edges in `s2e` form a contiguous loop; malformed topology could produce wrong ordering (fallback to sorted set covers this)
- `extract_topology` dependency in test fixture couples `test_gll_geometry.py` to `tools/gmsh_to_hdf5.py` (acceptable, both are project-internal)

**Recommended next step:** The half-space integration test now exercises the pipeline with a single element. After the vertex-id fix, the test can be extended to multi-element meshes (`nx=2, ny=2, nz=2`).
