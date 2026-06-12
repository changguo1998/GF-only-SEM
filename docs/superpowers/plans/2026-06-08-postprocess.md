# Postprocess Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Python postprocess module that reads HDF5 strain record files from the C++ SEM solver and mesh.h5 for GLL-node geometry, locates receiver positions within mesh elements, performs GLL basis interpolation at receiver positions, and outputs strain Green's functions as spatially tiled HDF5 files in `greenfun/`.

> **Design**: Technical decisions (HDF5 checkpoint schema, Green's function output schema, receiver search algorithm, GLL interpolation strategy) are documented in [`docs/superpowers/design/postprocess.md`](../design/postprocess.md). This file contains only the implementation plan.

---

> **Note**: Strain is the primary scientific output — no displacement integration in postprocess. Green's function extraction requires 3 independent forward runs per source location (one per orthogonal force direction: x, y, z). Postprocess assembles the 6×3 strain Green's tensor from these 3 record sets.

---

### Phase 0: Package Skeleton

- [ ] **T0.1** Create `postprocess/pyproject.toml` with package metadata, dependencies (numpy, h5py, scipy, click, pytest), and CLI entry point `gf-postprocess = gf_post.cli:main`
- [ ] **T0.2** Create `postprocess/src/gf_post/__init__.py` with `__version__ = "0.1.0"` and public exports
- [ ] **T0.3** Create `postprocess/tests/conftest.py` with `pytest` marker setup and synthetic data helper imports
- [ ] **T0.4** Verify `pip install -e .` works and `python -m gf_post --help` raises `ModuleNotFoundError` (expected — CLI not yet built)
- [ ] **T0.5** Commit: "feat(postprocess): package skeleton with pyproject.toml"

---

### Phase 1: GLL Geometry Utilities (`geometry.py`)

GLL quadrature and Lagrange basis — the mathematical foundation. All other modules depend on these.

- [ ] **T1.1** Write failing test: `test_geometry.py::test_gll_nodes_1d` — verify N=1 returns [-1, 1] and N=2 returns [-1, 0, 1]
- [ ] **T1.2** Implement `gll_nodes_1d(N: int) -> np.ndarray` — compute 1D GLL quadrature points in [-1, 1] using root-polling + Newton refinement of Legendre polynomial derivative
- [ ] **T1.3** Run test to confirm T1.2 passes
- [ ] **T1.4** Write test: `test_geometry.py::test_gll_weights_1d` — verify N=2 weights are [2/6, 4/6, 2/6]
- [ ] **T1.5** Implement `gll_weights_1d(N: int) -> np.ndarray` — compute 1D GLL quadrature weights
- [ ] **T1.6** Run test to confirm T1.5 passes
- [ ] **T1.7** Write test: `test_geometry.py::test_gll_nodes_3d` — verify `gll_nodes_3d(N)` returns `(n, 3)` array with n=(N+1)³ points
- [ ] **T1.8** Implement `gll_nodes_3d(N: int) -> np.ndarray` — tensor-product 3D GLL nodes in natural coordinates
- [ ] **T1.9** Write test: `test_lagrange_1d` — verify ℓ_i(ξ_j) = δ_{ij} for N=3
- [ ] **T1.10** Implement `lagrange_basis_1d(xi: float, nodes: np.ndarray) -> np.ndarray` — evaluate all 1D Lagrange polynomials at scalar xi
- [ ] **T1.11** Write test: `test_lagrange_3d` — verify 3D tensor-product basis sums to 1 at all GLL nodes (partition of unity)
- [ ] **T1.12** Implement `lagrange_basis_3d(xi: np.ndarray, nodes_1d: np.ndarray) -> np.ndarray` — evaluate 3D Lagrange basis at (ξ, η, ζ)
- [ ] **T1.13** Commit: "feat(postprocess): GLL geometry utilities (nodes, weights, Lagrange basis)"

---

### Phase 2: Record Reader (`reader.py`)

Reads strain + local_element_ids + attrs from record files. **Does NOT read geometry** — geometry comes from `GeometryReader` reading mesh.h5.

- [ ] **T2.1** Write failing test: `test_reader.py::test_reader_raises_missing_file` — Reader should raise on nonexistent path
- [ ] **T2.2** Implement class `RecordReader(path: str)` with `__init__` that validates HDF5 file exists and opens it
- [ ] **T2.3** Write failing test: `test_reader.py::test_reader_metadata` — verify `dt`, `source_direction`, `record_interval`, `nsteps` attributes
- [ ] **T2.4** Implement `RecordReader` properties: `dt`, `source_direction`, `record_interval`, `nsteps`, `local_element_ids`
- [ ] **T2.5** Write failing test: `test_reader.py::test_reader_strain` — verify strain tensor read: shape `[n_records, n_elem_local, NGLL, NGLL, NGLL, 6]`
- [ ] **T2.6** Implement `RecordReader.read_strain(step: int) -> np.ndarray` — read strain data for a single record step
- [ ] **T2.7** Implement `RecordReader.read_all_strain() -> np.ndarray` — read all steps
- [ ] **T2.8** Implement `RecordReader.close()` and `__enter__`/`__exit__` (context manager protocol)
- [ ] **T2.9** Write conftest fixture: `synthetic_record(tmp_path)` — creates a minimal valid HDF5 record file with N=3 hex element, local_element_ids, and strain at a few steps
- [ ] **T2.10 Commit: "feat(postprocess): RecordReader with strain-only HDF5 I/O"**

---

### Phase 2B: Geometry Reader (`reader.py` — add `GeometryReader`)

Reads GLL node coords, dξ/dx, and is_pml from mesh.h5. NGLL extracted from array shapes.

- [ ] **T2B.1** Write failing test: `test_reader.py::test_geometry_reader` — verify coords shape `[n_cell, NGLL, NGLL, NGLL, 3]`, verify NGLL=4 for N=3
- [ ] **T2B.2** Implement `GeometryReader` — reads `/field/element/coords`, `/field/element/dxi_dx`, `/field/element/is_pml` from mesh.h5. Exposes `coords`, `dxi_dx`, `is_pml`, `ngll` (extracted from shape).
- [ ] **T2B.3** Commit: "feat(postprocess): GeometryReader for mesh.h5 coords + dxi_dx + is_pml"

---

### Phase 2C: Multi-Rank Merge (`reader.py` — add merge logic)

Combines all rank files by mapping `local_element_ids` → global element IDs, merging strain into unified `[n_records, n_cell, NGLL, NGLL, NGLL, 6]` view.

- [ ] **T2C.1** Write test — 2 rank files with non-overlapping element IDs, verify merged strain covers all elements
- [ ] **T2C.2** Implement `merge_records(rank_files: list[str]) -> tuple` — returns merged strain array + time info
- [ ] **T2C.3** Commit: "feat(postprocess): multi-rank record merge"

---

### Phase 3: Spatial Index (`index.py`)

KD-tree over element centroids for candidate element filtering.

- [ ] **T3.1** Write failing test: `test_index.py::test_index_build` — verify index builds over N synthetic elements
- [ ] **T3.2** Implement `class ElementIndex`: constructor takes `element_gll_coords: np.ndarray` (shape `[nelem, NGLL³, 3]`), computes element centroids and bounding boxes, builds `scipy.spatial.KDTree` on centroids
- [ ] **T3.3** Write failing test: `test_index.py::test_bounding_boxes` — verify bounding boxes are computed correctly for a unit cube
- [ ] **T3.4** Implement `ElementIndex.bounding_boxes` — compute `[nelem, 2, 3]` min/max boxes from GLL coordinates
- [ ] **T3.5** Write failing test: `test_index.py::test_query` — verify query returns correct index for exact centroid match
- [ ] **T3.6** Implement `ElementIndex.query(point: np.ndarray, k: int) -> tuple` — return indices and distances of k nearest centroids
- [ ] **T3.7** Commit: "feat(postprocess): KD-tree spatial index over element centroids"

---

### Phase 4: Point-in-Hexahedron Search (`search.py`)

Newton iteration in natural coordinates (ξ, η, ζ) ∈ [-1, 1]³ using precomputed dξ/dx from mesh.h5.

- [ ] **T4.1** Write failing test: `test_search.py::test_point_inside_hex` — point at element center should converge to (0, 0, 0)
- [ ] **T4.2** Write failing test: `test_search.py::test_point_on_face` — point on face should have one coord = ±1
- [ ] **T4.3** Write failing test: `test_search.py::test_point_outside_hex` — point far away should raise ConvergenceError
- [ ] **T4.4** Implement `find_containing_element(point, candidates, gll_coords, dxi_dx, tol, max_iter)` — for each candidate element, run Newton iteration. Return (element_id, xi, eta, zeta) or raise if not found.
- [ ] **T4.5** Commit: "feat(postprocess): point-in-hexahedron search via Newton iteration"

---

### Phase 5: GLL Interpolation (`interpolate.py`)

Evaluate strain at receiver position using GLL Lagrange basis.

- [ ] **T5.1** Write failing test: `test_interpolate.py::test_interpolate_constant_strain` — constant strain at all GLL nodes → interpolated value matches constant
- [ ] **T5.2** Write failing test: `test_interpolate.py::test_interpolate_at_gll_node` — receiver at exact GLL node → interpolated value matches that node's strain
- [ ] **T5.3** Implement `interpolate_strain(strain_at_element, xi, eta, zeta)` — evaluate 6 strain components via `Σ l_i(ξ)·l_j(η)·l_k(ζ)·ε_ijk`
- [ ] **T5.4** Commit: "feat(postprocess): GLL strain interpolation at receiver position"

---

### Phase 6: Green's Tensor Assembly (`assembly.py`)

Assembles 6×3 strain Green's tensor from 3 forward runs (fx, fy, fz).

Each forward run with a different source direction produces 6 strain components (εxx, εyy, εzz, εxy, εxz, εyz). 3 runs fill the 3 columns of the 6×3 strain Green's tensor: column j = strain response to unit force in direction j.

- [ ] **T6.1** Write failing test: `test_assembly.py::test_assemble_3x3` — verify 3 runs → 6×3 tensor at each timestep
- [ ] **T6.2** Implement `assemble_greens_tensor(waveforms: dict[str, np.ndarray])` — waveforms = {"fx": ..., "fy": ..., "fz": ...}, returns `[nt, 6, 3]`
- [ ] **T6.3** Commit: "feat(postprocess): Green's tensor assembly from 3 runs"

---

### Phase 7: Writer (`writer.py`)

Write spatially-tiled Green's function output to `greenfun/` directory, sorted by lat/lon bounding boxes.

- [ ] **T7.1** Write failing test: `test_writer.py::test_writer_output` — verify output file has correct structure
- [ ] **T7.2** Implement `GFWriter.write(output_dir, receivers, time, waveforms, tile_size)` — writes `greenfun/tile_{i}.h5` with attrs (minlat, maxlat, minlon, maxlon), receivers/positions, receivers/names, receivers/element_ids, time/t, time/dt, waveforms/{fx,fy,fz}/{recv}/strain_{xx,yy,zz,xy,xz,yz}
- [ ] **T7.3** Commit: "feat(postprocess): strain Green's function tile writer"

---

### Phase 8: CLI (`cli.py`)

- [ ] **T8.1** Implement `main()` entry point: reads receivers.csv, record files from 3 wavefields/ directories (--fx wavefields/x/, --fy wavefields/y/, --fz wavefields/z/), mesh.h5, runs full pipeline → writes tiles to greenfun/
- [ ] **T8.2** Commit: "feat(postprocess): CLI entry point"

---

## File Layout (Final)

```
postprocess/
├── pyproject.toml
├── src/gf_post/
│   ├── __init__.py         — package exports, version
│   ├── reader.py           — RecordReader (strain) + GeometryReader (mesh.h5) + merge
│   ├── geometry.py         — GLL: nodes, weights, Lagrange basis
│   ├── index.py            — Spatial index (KD-tree over element centroids)
│   ├── search.py           — Point-in-hexahedron (Newton iteration using dξ/dx)
│   ├── interpolate.py      — GLL interpolation of strain at arbitrary point
│   ├── assembly.py         — Green's tensor assembly from 3 runs
│   ├── writer.py           — Strain GF tile writer
│   └── cli.py              — CLI entry point
└── tests/
    ├── conftest.py         — Synthetic record fixtures
    ├── test_reader.py, test_geometry.py, test_index.py, test_search.py
    ├── test_interpolate.py, test_assembly.py, test_writer.py, test_cli.py
```