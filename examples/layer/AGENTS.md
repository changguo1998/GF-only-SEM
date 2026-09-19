# examples/layer — Layered half-space benchmark

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
bash examples/layer/compare.sh

# Or manually:
examples/layer/.venv/bin/python examples/layer/reference.py \
  examples/layer/greenfun --source 5778 5278 0 --receiver 5278 5278 278 \
  --output /tmp/layer_ref.npz

# With Ricker wavelet synthetic
examples/layer/.venv/bin/python examples/layer/reference.py \
  examples/layer/greenfun --source 5778 5278 0 --receiver 5278 5278 278 \
  --output /tmp/layer_synth.npz --ricker-freq 2
```

## Prerequisites

PyFK environment at `examples/layer/.venv/` (Python 3.9):

```bash
cd examples/layer
uv venv .venv --python 3.9
uv pip install --python .venv/bin/python \
  'cython<3' poetry-core setuptools wheel numpy scipy h5py obspy
uv pip install --python .venv/bin/python --no-build-isolation pyfk
```
