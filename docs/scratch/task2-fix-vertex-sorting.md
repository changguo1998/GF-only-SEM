# Task 2: Fix \_get_cell_vertex_ids Sorting Bug — Result

## Summary

Replaced `_get_cell_vertex_ids` with a face-membership approach, fixed related test fixtures, and added a multi-element test that catches the bug.

## Bug Root Cause

The old implementation used `sorted(all_verts)` to order the 8 corner vertices. This only works by coincidence for the unit cube test (where vertex IDs 1-8 happen to be in GMSH order). For multi-element meshes with vertex IDs not in sequential order, the sorted order diverges from GMSH order, mangling physical coordinates for shape-function interpolation.

## Changes

### 1. `preprocess/gll_geometry.py` — `_get_cell_vertex_ids` rewrite

**Old approach**: Collect all vertices in a set, sort numerically. Works only for single-element meshes with vertices numbered 1-8 in GMSH order.

**New approach**: Face-membership identification. Each of the 8 hex corners belongs to exactly 3 faces. The 6 faces are ordered [-z, +z, -y, +y, -x, +x] (indices 0..5). By checking which 3 faces each vertex appears on, the 8 GMSH corners are uniquely identified regardless of vertex ID numbering.

The GMSH corner → face membership mapping:

```
v0(-z,-y,-x) → {0, 2, 4}
v1(-z,-y,+x) → {0, 2, 5}
v2(-z,+y,+x) → {0, 3, 5}
v3(-z,+y,-x) → {0, 3, 4}
v4(+z,-y,-x) → {1, 2, 4}
v5(+z,-y,+x) → {1, 2, 5}
v6(+z,+y,+x) → {1, 3, 5}
v7(+z,+y,-x) → {1, 3, 4}
```

### 2. `tests/preprocess/test_pml.py` — fixed broken face topology

The PML test's `_make_two_cube_topo` had incorrect s2e entries for faces 8-11 (top element's -y, +y, +x faces). Edge 14 was used where edge 18 was required, and similar issues. These faces contained 5 vertices each instead of the required 4, causing the face-membership lookup to fail.

Fixed to match `test_boundary_detector.py`'s already-corrected topology.

### 3. `tests/preprocess/test_gll_geometry.py` — multi-element test

Added `TestMultiElementGLLGeometry` class with 4 tests using a 2x1x1 element mesh:

- `test_element0_corners_correct_order` — verifies all 8 corners of element 0 map to correct physical coordinates
- `test_element1_corners_correct_order` — verifies all 8 corners of element 1
- `test_jacobian_consistent_across_elements` — both elements (identical unit cubes) have equal Jacobians
- `test_mass_sum_per_element` — each element's full mass matrix sums to 1.0

For element 0, GMSH vertex order is `[1, 2, 5, 4, 7, 8, 11, 10]` but sorted would be `[1, 2, 4, 5, 7, 8, 10, 11]` — swapping vertices 4↔5 and 10↔11 on the +y face. This test would fail with the old `sorted()` approach.

## Validation

All 85 tests pass (62 preprocess + 22 tools + 1 workflow):

- Existing unit cube tests: pass (GMSH order = sorted order for unit cube with vertex IDs 1-8)
- New multi-element tests: pass (GMSH order ≠ sorted order — would fail with old code)
- PML tests: pass (fixed topology)
- Boundary detector tests: pass
- All other preprocess tests: pass

No staged files (clean working tree with only tracked modifications).

## Residual Risks

- The face-membership approach requires that each face has exactly 4 unique vertices. Hand-crafted topologies in tests that violate this will produce `ValueError` with a diagnostic message. The auto-generated topology from `extract_topology` (used by the test_gll_geometry fixture) is always correct.
