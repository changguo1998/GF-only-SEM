# examples/layer-shallow-source — Shallow-source layered benchmark

## Purpose

PyFK reference waveform generation for a layered elastic half-space.
The model consists of a soft surface layer over a stiffer half-space,
designed to produce Love and Rayleigh waves for SEM validation.

## Model

| Layer | Thickness (km) | Vs (km/s) | Vp (km/s) | ρ (g/cm³) | Qs | Qp |
|-------|---------------|-----------|-----------|-----------|-------|-------|
| 1 | 0.5 | 1.5 | 2.5 | 2.2 | ∞ | ∞ |
| 2 (∞) | 0.0 | 3.0 | 5.0 | 2.7 | ∞ | ∞ |

## Usage

```bash
# Generate PyFK reference (Green function in m/N)
bash examples/layer-shallow-source/compare.sh

# Or manually:
examples/layer-shallow-source/.venv/bin/python examples/layer-shallow-source/reference.py \
  examples/layer-shallow-source/greenfun --source 5778 5278 0 --receiver 5278 5278 100 \
  --output /tmp/layer_ref.npz

# With Ricker wavelet synthetic
examples/layer-shallow-source/.venv/bin/python examples/layer-shallow-source/reference.py \
  examples/layer-shallow-source/greenfun --source 5778 5278 0 --receiver 5278 5278 100 \
  --output /tmp/layer_synth.npz --ricker-freq 2
```

## Prerequisites

PyFK environment at `examples/layer-shallow-source/.venv/` (Python 3.9):

```bash
cd examples/layer-shallow-source
uv venv .venv --python 3.9
uv pip install --python .venv/bin/python \
  'cython<3' poetry-core setuptools wheel numpy scipy h5py obspy
uv pip install --python .venv/bin/python --no-build-isolation pyfk
```

## 验证

2026-09-18 测试使用 16-rank CPU 弹性求解器和 4-rank MPI 后处理，
两者均限制在 60 GiB 内存内。三个正演方向耗时 57.64/54.98/56.39 s，
后处理耗时 83.9 s。

震源定位到非 PML 单元 253，参考坐标为
`(xi, eta, zeta) = (0.2232, 0.2232, -0.6)`。125 个张量积震源权重均非零，
权重和严格等于 1.0，证明 100 m 深度的非网格点震源加载有效。

在距震源水平 500 m 的地表点，0–2 s 主波窗的全张量相关系数为
0.9930，原始相对 L2 为 0.1248，SEM 最佳缩放系数为 0.9611，拟合相对
L2 为 0.1182。0–5 s 全时段的相关系数为 0.5813，拟合相对 L2 为
0.8137，误差主要由晚期有限边界/PML 回波主导。
