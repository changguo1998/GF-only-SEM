# examples/layer — Layered half-space benchmark

## Purpose

PyFK reference waveform generation for a layered elastic half-space.
The model consists of a soft surface layer over a stiffer half-space,
designed to produce Love and Rayleigh waves for SEM validation.

## Model

| Layer | Thickness (km) | Vs (km/s) | Vp (km/s) | ρ (g/cm³) | Qs | Qp |
|-------|---------------|-----------|-----------|-----------|-------|-------|
| 1 | 0.5 | 1.5 | 2.5 | 2.2 | 100 | 200 |
| 2 (∞) | 0.0 | 3.0 | 5.0 | 2.7 | 500 | 1000 |

## Usage

```bash
# Generate PyFK reference (Green function in m/N)
bash examples/layer/compare.sh

# Or manually:
examples/layer/.pyfk-venv/bin/python examples/layer/reference.py \
  --source 0 0 490 --receiver 5000 0 0 \
  --output /tmp/layer_ref.npz

# With Ricker wavelet synthetic
examples/layer/.pyfk-venv/bin/python examples/layer/reference.py \
  --source 0 0 490 --receiver 5000 0 0 \
  --output /tmp/layer_synth.npz --ricker-freq 5
```

## Prerequisites

PyFK environment at `examples/layer/.pyfk-venv/` (Python 3.9):

```bash
cd examples/layer
uv venv .pyfk-venv --python 3.9
.pyfk-venv/bin/python -m ensurepip --upgrade
.pyfk-venv/bin/python -m pip install pyfk obspy
```
