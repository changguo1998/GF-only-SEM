# Deferred Designs and Plans

This file lists known deferred work. When work resumes, use the linked design or plan.

______________________________________________________________________

## 1. SLS Viscoelastic Attenuation [ARCHIVED]

**Status: COMPLETE & VERIFIED.** Implemented via 7 commits (Jul 2026).
Elastic-limit regression (Q→∞) verified 2026-07-26: max_rel_l2=0.0 across
all 4 MPI C++/Python variants (halfspace + layer).

- `c90268e`: SLS namespace, constants, `precompute_sls_coefficients()`, RankData fields
- `7ec4d18`: Preprocess τ computation (`preprocess/attenuation.py`), I/O + restart
- `a265668`: Viscoelastic solver skeleton (`forward/viscoelastic/`)
- `a8b083a`: CPU viscoelastic element kernel with SLS memory update
- `b0749ac`: Shared solver SLS integration + viscoelastic runner
- `f493e5e`: CUDA viscoelastic element kernel + SLS runtime
- `590d774`: Unit tests (Python + C++), 204 tests pass

**Architecture:**

- Independent solver binary `gf_solver_viscoelastic_mpi` reuses `libgf_shared`
  and the 5 shared `kernel_helpers`; only the stress computation differs
- Exact piecewise-linear SLS memory update inline in the element kernel
- Independent Qμ/Qκ per GLL node, n_sls = 3 (compile-time)
- `has_attenuation` flag gates all SLS code paths
- Propagation-level analytical attenuation/dispersion validation in
  `examples/finite-q-propagation/`. The current 20×14×14, 1 Hz CPU/CUDA runs have maximum
  amplitude/phase errors of 6.25%/0.0084 rad and pass (see `test-reports/05-finite-q.md`).

**Preprocess:** `preprocess/attenuation.py`

- `compute_tau_from_q()`: τ-method with log-spaced τ_σ, least-squares fit for τ_ε
- `write_attenuation_to_model()`: writes shear/bulk SLS relaxation times to model.h5

**Forward solver executables:**

- `gf_solver_viscoelastic_mpi` (CPU + MPI) — verified: 16 ranks, 1000 steps
- `gf_solver_viscoelastic_cuda` (CUDA, no MPI)
- `gf_solver_viscoelastic_mpi_cuda` (CUDA + MPI)

**Spec:** [`docs/_archive/specs/2026-07-21-sls-viscoelastic-design.md`](../_archive/specs/2026-07-21-sls-viscoelastic-design.md)
**Plan:** [`docs/_archive/plans/2026-07-21-sls-viscoelastic.md`](../_archive/plans/2026-07-21-sls-viscoelastic.md)

______________________________________________________________________

## 2. Compress Module (Placeholder)

**Status:** Placeholder - not in current scope. The `compress/` module (header-only
HDF5 compression/chunking/precision utilities) has been removed. Record files are
written uncompressed. If compression is needed in the future, re-implement from
[`design/compress.md`](design/compress.md) (kept as historical design reference).

______________________________________________________________________

## 3. Full C-PML Implementation (Strain Correction) [ARCHIVED]

**Status: COMPLETE & VERIFIED.** Displacement-based C-PML (acceleration correction,
3 memory variables/node) and strain-based correction (A₆…A₂₃, 18 memory
variables/node) are both implemented. All 4 bugs (1a-1d) fixed. Key commits:

- `18b89ca`: CUDA C-PML strain correction (element kernel + runtime)
- `71aac4a`: C-PML strain correction unit tests (7 new, 1443 assertions)

### Implemented

- [`docs/design/cpml.md`](design/cpml.md): Full design document.
- `preprocess/pml_cpml.py`: C-PML profile computation (K, d, α per direction)
  and convolution coefficients (α/β 9 each, Ā₁…Ā₅, A₆…A₂₃).
- `forward/`: C-PML data structures in `types.hpp`, I/O in `io.cpp`,
  memory variable update + accel contribution + strain correction in `pml.hpp/cpp`.
- `solver.cpp`: C-PML accel correction + strain memory update integrated.
- Backward compatible: falls back to old damping when C-PML data absent.
- Strain-based correction: element kernel (CPU + CUDA) modifies physical gradient
  via A₆…A₂₃ convolution, restart I/O for memory state.
- Unit tests: 7 Catch2 tests covering all CpmlStrain helper functions.

### Remaining: none

C-PML implementation verified correct (2026-07-22):

- Non-symmetric stress, parameter separation, alpha-convolved lx/ly/lz memory
  all match SPECFEM3D exactly
- Solver physics validated: 99.1% scaled waveform correlation with Lamb reference. The former
  ~3× factor was a postprocess count bug and is fixed; see [`bugs.md`](bugs.md).
- Halfspace 1000-step test stable, max|u|≈2.4e5 (no inf/nan)
- All 4 bugs (1a-1d) fixed and verified

Absorption quality benchmark not needed — the C-PML is correct by construction
(matches SPECFEM3D) and the solver produces physically correct wavefields.

## 4. Compression Benchmark Tool

**Status:** Obsolete (2026-08-09) — compression is DISABLED project-wide
(see `docs/design-decisions.md` §10 and AGENTS.md); the benchmark tool is only
relevant if compression is re-introduced after a design review.

______________________________________________________________________

## 5. GPU/DCU Backends (HIP, SYCL)

**Status:** CUDA backend implemented. HIP and SYCL backends deferred.

Same pattern as CUDA: add tag struct, source file, CMake branch. See [`design/gpu.md`](design/gpu.md).

______________________________________________________________________

## 6. SEM Amplitude & Radiation Pattern Discrepancy [ARCHIVED]

**Status: RESOLVED.** Three issues were found and fixed:

- **E-W wavefield asymmetry (1.77×)**: CG-SEM element-interface assembly bug.
  Fixed by global GLL node numbering (`cff2cd1`, 2026-07-17).
- **P-SV coupling "bias" (0.5–2×)**: misdiagnosed as Cartesian mesh anisotropy;
  actual cause was a Green tensor index convention mismatch (transpose bug) in
  the postprocess. Fixed 2026-07-19 (`postprocess/cpp/main.cpp`).

#### Third issue found and fixed (2026-07-23)

A **postprocess mass-weighting bug** was discovered: `merge_direction()` in
`postprocess/cpp/main.cpp` applied mass-weighted averaging to displacement,
velocity, and acceleration (same normalization as strain). Since CG-SEM
shared nodes have identical displacement across elements, count-based
averaging is correct. The mass-weighted path divided by ~3.7e9 kg per
node, suppressing displacement by ~1.9e9×.

**Fix:** commit `6f90c12` — separate normalization: strain keeps
mass-weighted, displacement/velocity/acceleration use count-based average.

**After fix:** best-fit scale reduced from 1.92e9 to 2.95 (halfspace)
and 9.50e7 to 2.60 (layer). Waveform correlation improved from 0.945
to 0.991 (halfspace).

### Interpolation accuracy

The library now supports two interpolation modes:

- **GLL-basis interpolation** (current default for `basis="gll"` tiles):
  spectral accuracy via tensor-product Lagrange basis. Exact GLL-node
  matches return directly via KDTree (zero interpolation error). This
  is the output format of `gf_postprocess` (commit `18087ef`).
- **Trilinear interpolation** (legacy fallback for `basis="mesh_vertices"`
  tiles): degrades off-diagonal components (rel_l2≈0.58 halfspace, ~0.88
  layer). See [`docs/design/postprocess.md`](design/postprocess.md).

**Mitigation:** Use current GLL-format tiles (re-run `gf_postprocess`).
For vertex-only tiles, query at recorded vertices or re-postprocess.
This is a tile-format limitation, not a solver bug.

### Verified correct (solver physics)

1. GLL Lagrange weights normalized to sum=1 across shared elements ✓
1. Mass = ρ·J·w_i·w_j·w_k, density applied in `cli.py` ✓
1. Newmark explicit central difference (β=0, γ=½) in solver.cpp ✓
1. Total force = stf_val × Σ(weights) = 1.0 N ✓
1. Element residual: isotropic elastic stress correct ✓
1. E-W axisymmetry: centered source gives identical E/W displacement ✓

______________________________________________________________________

## 7. Postprocess MPI Tile-Parallel Refactoring [ARCHIVED]

**Status: COMPLETE & VERIFIED (2026-09-21).** Serial and MPI targets now compile the
same tile-batched pipeline in `postprocess/cpp/main.cpp`. Memory was redesigned:
`merge_direction()` (full replication, ~331 GB for 16 ranks -> OOM/reboot) split
into metadata-only layout construction plus worker-local field extraction. The original one-tile
loop was further optimized on 2026-09-23: tile ownership is assigned by record-rank overlap and
balanced estimated work; each worker reads a record once and scatters it to all assigned tiles.
Every tile still has one writer, and idle ranks exit before field allocation.
Multi-rank verification (halfspace, 9 tiles): `mpirun -n 1` and `mpirun -n 4`
outputs are numerically bit-identical to serial `gf_postprocess` across all
datasets of all 9 tiles (HDF5 file bytes differ only by serialization, not data).
The unified 2026-09-21 regression used the complete halfspace records (500 output
steps in all three force directions). Serial and 4-rank MPI produced 16 tiles;
all 160 datasets and all attributes were bit-identical. Runtime was 158.4 s serial
and 61.1 s MPI (2.59× speedup). HDF5 serialization bytes need not be identical.
On the 20³/800-step profile, worker-local reuse reduced MPI-4 runtime from 34.3 s to
19.0 s and HDF5 reading from 59.187% to 17.693% of cumulative worker time.

## Summary

有限 Q SLS、完整 C-PML、CUDA 后端、后处理 MPI tile 并行和集总质量 L2 应变投影均已
完成并验证。目前延期事项如下：

| 事项 | 模块 | 优先级 | 状态 |
|---|---|---|---|
| 降低紧凑域边界/C-PML 残差 | forward + examples | 中 | 已知限制；仅在需要更高全空间波形精度时继续研究 |
| 超大模型 GPU 网格流式计算 | forward | 中 | 延期；当前实现要求模型可装入显存 |
| CUDA-aware MPI 与残差常驻显存 | forward | 低 | 可选性能优化 |
| HIP/SYCL 后端 | forward | 低 | 延期；CUDA 是当前支持的加速后端 |

压缩不是待实现任务：项目规定 HDF5 不压缩。重新引入压缩必须单独进行设计评审。
