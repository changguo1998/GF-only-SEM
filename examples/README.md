# Examples

End-to-end computation examples for gf-calculation.

Each example is self-contained and demonstrates the computational pipeline:

```
mesh generation → preprocess (GLL + material + PML + SLS attenuation) →
forward solver → postprocess (Green's ftn extraction)
```

The preprocessor auto-injects the config's viscoelastic (SLS) parameters
(`q_mu`, `q_kappa`, `n_sls`) into `model.h5`. The solver to run is chosen by
the user inside each example's `forward.sh` (commented-out, switchable).

Green's function extraction uses configured shallow mesh vertices. No receivers.

All current example configurations use a 1 Hz Ricker source. Their
`f0_for_pml_hz` and, when present, SLS reference frequency match that source.

## Finite-Q Propagation

`finite-q-propagation` is the small analytical attenuation/dispersion regression. It runs an
elastic baseline and a Qμ=20 viscoelastic y-force on a 20×14×14 full-space mesh, then compares
the finite-Q/elastic transfer between one and two S wavelengths against the exact complex SLS
Green function at 0.75–1.0 Hz. The x direction and time are a two-times scaling of the former
2 Hz case. The transverse section was widened from 12 to 14 km so its non-PML width exceeds
one S wavelength without changing the 1 km element size.

```bash
bash examples/finite-q-propagation/run.sh auto  # CUDA when available, else 2-rank CPU
bash examples/finite-q-propagation/run.sh cpu
bash examples/finite-q-propagation/run.sh cuda
```

Acceptance limits are 12% relative amplitude error and 0.11 rad phase error. A 2026-09-24
current-code rerun gives CPU/CUDA metrics identical at the reported precision: maximum
amplitude error 6.25% and maximum phase error 0.0084 rad, so all points pass. The former
20×12×12 model failed because its 5-element PML on each side left only a 2 km transverse
physical section. See `docs/test-reports/05-finite-q.md`.

## Half-Space

Homogeneous elastic half-space with a buried point force at 278 m depth.

| File | Purpose |
|------|---------|
| `halfspace/config.py` | Simulation configuration (Python config script) |
| `halfspace/mesh_gen.py` | Regular hex mesh generator (standalone) |
| `halfspace/compare.sh` | **Full validation pipeline** — SEM → analytic Lamb reference → comparison |
| `halfspace/reference.py` | Analytic Lamb (Johnson 1974) reference waveform (self-contained, no PYTHONPATH needed) |
| `halfspace/compare.py` | Compare reference vs SEM GreenFunctionLibrary result |
| `halfspace/setenv.sh` | Environment init (Spack MPI/Eigen/HDF5) |
| `halfspace/mesh.sh` | Stage 1: mesh generation |
| `halfspace/preprocess.sh` | Stage 2: GLL geometry, materials, PML, partition |
| `halfspace/forward.sh` | Stage 3: forward solver (3 force directions) — **solver switchable inside** (uncomment one) |
| `halfspace/postprocess.sh` | Stage 4: Green's function tile extraction |

**Quick start — full validation:**

```bash
# End-to-end: SEM pipeline → Lamb analytic reference → comparison plot
bash examples/halfspace/compare.sh

# Or step by step:
source examples/halfspace/setenv.sh
bash examples/halfspace/preprocess.sh   # Stage 1+2: mesh + preprocess
bash examples/halfspace/forward.sh      # Stage 3: forward solver
bash examples/halfspace/postprocess.sh  # Stage 4: Green's function extraction

# Generate analytic reference + compare (no PYTHONPATH needed)
# --source = displacement observation point; --receiver = point matching SEM source
python examples/halfspace/reference.py examples/halfspace/greenfun \
  --source 5556 5556 0 --receiver 5278 5278 278 --source-depth-m 278.0 \
  --output /tmp/lamb_ref.npz

python examples/halfspace/compare.py examples/halfspace/greenfun \
  --source 5556 5556 0 --receiver 5278 5278 278 \
  --reference /tmp/lamb_ref.npz --output /tmp/lamb_cmp.npz --fit-scale

# Compare 10 deterministic surface points in several directions and distances
python examples/halfspace/multi_compare.py \
  --library examples/halfspace/greenfun --n-points 10 --early-end-s 2.0 \
  --output examples/halfspace/multi_comparison.npz
```

**What it does:**

1. Generates a 22×22×11 regular hex mesh (5324 elements, 10km×10km×5km)
1. Runs preprocessor: GLL geometry, constant material (Vp=5000, Vs=3000, ρ=2700), PML boundaries, 16-rank METIS partition
1. Runs the selected forward solver in 3 directions (x, y, z)
1. Extracts Green's functions into spatial tiles with 4 MPI ranks
1. Generates analytic Lamb (Johnson 1974) reference waveform
1. Compares SEM result with analytic reference (relative L² error, best-fit amplitude scaling)

**22×22×11、1 Hz 验证（2026-09-24）：** 16-rank CPU SLS 弹性极限三个力方向耗时分别为
56.83/55.94/56.83 秒。4-rank MPI Debug 后处理在共享 60 GiB 限制内用时 75.0 秒。
与 Lamb 解析解相比，0–2 秒位移相关系数为 0.999792，SEM/参考缩放系数为
1.007655，拟合相对 L² 为 0.020396。

10 个固定地表点覆盖 0.5–1.5 km 距离及正负 x/y、对角方向。0–2 秒主波
平均相关系数为 0.9999，平均拟合相对 L² 为 0.0171；0–5 秒平均相关系数为
0.9992、平均拟合相对 L² 为 0.0320。1 Hz 长波降低了该网格上的离散误差；最远
1.5 km 点的全时段误差仍较高，主要反映晚期有限边界/PML 返回。

**Output layout:**

```
examples/halfspace/
├── model.h5                  # Extended mesh (topology + GLL + materials + PML)
├── config.h5                 # Simulation parameters + STF
├── partitions/
│   └── partition_{r}.h5      # Per-rank local elements + exchange patterns
├── wavefields/
│   ├── x/record_*.h5         # shallow mesh-vertex strain, force x
│   ├── y/record_*.h5         # shallow mesh-vertex strain, force y
│   └── z/record_*.h5         # shallow mesh-vertex strain, force z
├── greenfun/
│   └── tile_x*_y*.h5         # horizontal Green tiles
├── lamb_reference.npz        # Analytic reference (500, 3, 3)
├── lamb_comparison.npz       # Comparison with SEM
└── vtk/                      # Visualization outputs
```

**Build requirements:**

```bash
# Python dependencies
uv sync --group dev

# C++ forward solver — all targets auto-detected:
cmake -B build && cmake --build build
# All binaries go to bin/

# MPI environment (if using Spack)
source env_setup.sh
```

## Layered Half-Space

Two-layer model (soft 500 m layer over stiff half-space) with PyFK reference.

| File | Purpose |
|------|---------|
| `layer/config.py` | SEM + PyFK configuration (depth-dependent vp/vs/rho) |
| `layer/mesh_gen.py` | Layer-aligned hex mesh generator (standalone) |
| `layer/reference.py` | PyFK layered reference Green tensor (needs Python 3.9 venv) |
| `layer/compare.py` | Compare reference vs SEM GreenFunctionLibrary result |
| `layer/compare.sh` | **Full validation pipeline** — SEM → PyFK reference → comparison |

**Quick start:**

```bash
# End-to-end: SEM pipeline → PyFK layered reference → comparison
bash examples/layer/compare.sh

# Or manually (after SEM pipeline has run):
# --source = displacement observation point; --receiver = point matching SEM source
examples/layer/.venv/bin/python examples/layer/reference.py examples/layer/greenfun \
  --source 5778 5278 0 --receiver 5278 5278 278 --output /tmp/layer_ref.npz

python examples/layer/compare.py examples/layer/greenfun \
  --source 5778 5278 0 --receiver 5278 5278 278 \
  --reference /tmp/layer_ref.npz --output /tmp/layer_cmp.npz --fit-scale
```

**Model:**

| Layer | Thickness (km) | Vs (km/s) | Vp (km/s) | ρ (g/cm³) |
|-------|---------------|-----------|-----------|-----------|
| 1 | 0.5 | 1.5 | 2.5 | 2.2 |
| 2 (∞) | 0.0 | 3.0 | 5.0 | 2.7 |

Material functions (`vp_m_s`, `vs_m_s`, `density_kg_m3`) are depth-dependent
piecewise functions compatible with the SEM preprocessor. The 22×22×11 mesh uses
one 500 m surface-layer element and ten 450 m lower-layer elements, keeping the
500 m material interface on an element boundary.

**22×22×11、1 Hz 验证（2026-09-18）：** 震源位于
`(5278, 5278, 278) m`，比较点与震源水平相距 500 m。由于 CUDA
设备节点当时不可用，本次使用 16-rank CPU 弹性求解器，三个方向分别
耗时 55.8/56.0/54.2 秒；4-rank MPI 后处理在共享 60 GiB 限制下耗时
70.0 秒。

本次同时修复了界面材料判定：500 m 界面坐标的浮点舍入误差曾使同一
GLL 面上的节点被不一致地分到上、下两层；现用 1 μm 容差确保界面及所有
表层单元的材料一致。

0–2 秒主波窗全张量相关系数为 0.9914，SEM 最佳拟合缩放系数为
0.9506，原始/拟合相对 L² 为 0.1409/0.1312。0–5 秒全时段相关系数为
0.5935，缩放系数为 0.9277，拟合相对 L² 为 0.8049。1 Hz 下软表层 S 波
波长为 1500 m，当前网格为 3.3 单元/波长；主波精度明显改善，但 2 秒后
仍由有限边界/PML 回波主导。

**PyFK environment setup:**

```bash
cd examples/layer
uv venv .venv --python 3.9
uv pip install --python .venv/bin/python \
  'cython<3' poetry-core setuptools wheel numpy scipy h5py obspy
uv pip install --python .venv/bin/python --no-build-isolation pyfk
```

## 浅源层状半空间

`layer-shallow-source/` 是两层介质模型的浅源版本。保持相同的
22×22×11 层界面对齐网格和 1 Hz Ricker 震源，将点力放在
500 m 软表层内的 `(5278, 5278, 100) m`。完整运行命令为：

```bash
bash examples/layer-shallow-source/compare.sh
```

震源位于非 PML 单元 253，参考坐标为 `(0.2232, 0.2232, -0.6)`。
125 个插值权重均非零且权重和为 1.0，浅部非网格点震源加载有效。

**22×22×11、1 Hz 验证（2026-09-18）：** 16-rank CPU 三方向正演耗时
57.64/54.98/56.39 秒；60 GiB 限制下的 4-rank MPI 后处理耗时 83.9 秒。
在距震源水平 500 m 的地表点，0–2 秒主波窗的全张量相关系数为 0.9930，
原始/拟合相对 L² 为 0.1248/0.1182，最佳缩放系数为 0.9611。0–5 秒全时段
相关系数为 0.5813、拟合相对 L² 为 0.8137；晚期误差仍主要来自有限边界
与 PML 回波。

完整参数、震源权重和结果见
[`layer-shallow-source/VERIFICATION.md`](layer-shallow-source/VERIFICATION.md)。

## 全 C++ 层状地表震源

`layer-surface-source-cpp/` 使用与层状模型相同的 22×22×11 层界面对齐
网格和 1 Hz Ricker 点力，但将震源自动定位到 z-min 自由表面。从拓扑网格
生成到预处理、CUDA 正演和 MPI 后处理均不调用 Python，且使用独立构建目录，
不会覆盖 `layer/` 用例或主构建中的预处理器。

```bash
bash examples/layer-surface-source-cpp/run.sh
```

## Canonical Examples & Solver Selection

Three canonical, self-contained examples replace the earlier
`{model}-{physics}-{runtime}-{language}` test-case matrix (GPU/CPU and solver
are no longer encoded in directory names):

| Case | Purpose |
|------|---------|
| `halfspace/` | Homogeneous half-space — vs analytic Lamb (Johnson 1974) reference |
| `layer/` | Two-layer half-space — vs PyFK reference |
| `fullspace-cubic/` | Full-space (all-PML) — **solver / backend (GPU vs CPU) comparison** |

All configs use **viscoelastic (SLS) parameters**: `q_mu`, `q_kappa`, `n_sls`
(and `f0_for_pml_hz`). The preprocessor auto-injects them into `model.h5`
(`field/cell/tau_*`); selecting a viscoelastic solver therefore exercises the SLS code path.
With the default Q→∞ (elastic limit) visco output is bit-identical to elastic; set
`q_mu`/`q_kappa` to a finite value (e.g. `100.0`) for real attenuation.

默认回归组合刻意避免全排列：`halfspace` 使用 SLS CPU+MPI，`layer` 使用
elastic CPU+MPI，`fullspace-cubic` 使用 elastic CUDA。前两者的 0–2 s 主体波
比较现在包含相关系数、拟合 L2 和绝对幅值硬门限；全空间继续使用 Stokes
解析比较门限。完整选择理由见 [`docs/testing.md`](../docs/testing.md)。

**Solver selection** is done by the user, not by test-case naming. Edit
`examples/<case>/forward.sh` and uncomment ONE solver entry — options cover:

| Option | Solver | Backend |
|--------|--------|---------|
| (A) `gf_solver_viscoelastic_mpi` | viscoelastic (SLS) | CPU + MPI (default) |
| (B) `gf_solver_viscoelastic_cuda` | viscoelastic (SLS) | CUDA single-GPU |
| (C) `gf_solver_viscoelastic_mpi_cuda` | viscoelastic (SLS) | CUDA + MPI |
| (D) `gf_solver_elastic_mpi` | elastic | CPU + MPI |
| (E) `gf_solver_elastic_cuda` | elastic | CUDA single-GPU |
| (F) `gf_solver_elastic_mpi_cuda` | elastic | CUDA + MPI |
| (G) custom ranks | any `*_mpi` | override `config.py:n_ranks` |
| (H) custom path | any | build variant |

The chosen solver runs all 3 force directions. MPI rank count comes from
`config.py:n_ranks` by default.

**Usage:**

```bash
source scripts/env.sh && scripts/build.sh --backend cpu
bash examples/halfspace/compare.sh                # single case

bash scripts/run_all_examples.sh --dry-run        # list the 3 cases
bash scripts/run_all_examples.sh --case halfspace # run one case
bash scripts/run_all_examples.sh                  # run all cases
```

`fullspace-cubic` is the GPU vs CPU comparison case: its `compare.sh` runs the
CUDA solver (skips gracefully with "SKIP: no GPU available" when no GPU is
present), and `compare_solvers.py` aligns the CUDA and MPI-CPU strain fields
(rel_l2, Pearson correlation). See `fullspace-cubic/VERIFICATION.md`.

`meshsize/` is the fixed-receiver resolution study for the compact 18 km
full-space domain. Its regenerated 18³/20³/22³/24³/28³ results show fitted
relative L2 = 0.3414–0.3468 with no refinement trend and amplitude scale
1.013–1.038. See `meshsize/README.md`.

`fullspace-expanded/` is a non-canonical PML-distance and grid-size diagnostic:
a 28 km homogeneous full space with a 1 Hz source. Its current configuration is
22³ elements; `fullspace-expanded/VERIFICATION.md` compares 20³, 22³, 24³, 26³,
and 28³ runs and records accuracy, runtime, memory constraints, and storage use.

## Adding a New Example

1. Create `examples/<name>/` with:
   - `config.py` — Python config (see `preprocess/config_loader.py` for schema)
   - `mesh_gen.py` — Mesh generator
   - `reference.py` — Analytic/numerical reference (if applicable)
   - `compare.sh` — Orchestration script
1. Follow the halfspace example as a template
1. Update this README
