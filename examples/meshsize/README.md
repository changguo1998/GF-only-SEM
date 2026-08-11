# Mesh-size study: fullspace 18 / 20 / 22 / 24 / 28

Five full-space SEM cases that differ ONLY in the grid. Everything else —
domain, source, frequency, duration, receivers — is identical, so the only
axis being varied is spatial resolution (elements per S wavelength).

**28³ is the agreed ceiling** — no denser grids will be added. Memory is the
binding constraint: estimated postprocess peak ~57 GB at 28³ vs ~89 GB for a
hypothetical 32³ on this 125 GB host.

## Layout

```
examples/meshsize/
├── gen_grids.py               # single source of truth: writes all case dirs'
│                              #   config.py + config_user_fullspace.cpp and
│                              #   receivers_fixed.npy
├── receivers_fixed.npy        # 64 fixed receiver points, seed 0,
│                              #   uniform in [5500, 12500]³ m — the SAME
│                              #   physical locations compared on every grid
├── run_study.sh               # driver: each case pipeline + fixed-receiver
│                              #   analytical comparison + fullspace-cubic restore
├── results/                   # per-case logs + postprocess peak RSS
├── fullspace18/               # 1000 m elements, PML 5 (5.00 km)
├── fullspace20/               #  900 m elements, PML 6 (5.40 km)
├── fullspace22/               #  818 m elements, PML 6 (4.91 km)
├── fullspace24/               #  750 m elements, PML 7 (5.25 km)
└── fullspace28/               #  643 m elements, PML 8 (5.14 km) — coarsest ceiling
```

Each case dir is a self-contained example: `config.py`,
`config_user_fullspace.cpp`, `compare.sh`, `mesh_gen.py`, and its own
artifacts (`model.h5`, `config.h5`, `wavefields/`, `greenfun/`).

## Fixed parameters (identical in all five)

| parameter | value |
|-----------|-------|
| Domain | 18 km × 18 km × 18 km |
| GLL order | N=4 |
| Source | point force at (9375, 9375, 9375) m — inside every element (never on a GLL node) of every grid |
| Force | 1e20 N |
| STF | Ricker f0=1 Hz, t0=2 s, 8 s duration |
| Time step | solver_dt = output_dt_s = 0.01 s, 800 steps |
| Material | vp=5000 / vs=3000 / rho=2700 |
| PML | ≈1.0 λp in meters (5.00 / 5.40 / 4.91 / 5.25 / 5.14 km) |
| Receivers | 64 fixed points; each grid compares at its NEAREST recorded GLL node, and the analytical solution is evaluated at that node's true coordinates |
| HDF5 | uncompressed (project rule, AGENTS.md / design-decisions §10) |

## Grid-varying parameters

| case | elements | elem. size | elem/λs | PML | interior | tiles |
|------|----------|-----------|---------|-----|----------|-------|
| 18 | 5832 | 1000 m | 3.0 | 5 | 8 elem | [2,2,2,2]² |
| 20 | 8000 | 900 m | 3.3 | 6 | 8 elem | [2,2,2,2]² |
| 22 | 10648 | 818 m | 3.7 | 6 | 10 elem | [2,2,3,3]² |
| 24 | 13824 | 750 m | 4.0 | 7 | 10 elem | [2,2,3,3]² |
| 28 | 21952 | 643 m | 4.7 | 8 | 12 elem | [3,3,3,3]² |

## How to run

```bash
# regenerate configs + fixed receivers from the single source
python3 examples/meshsize/gen_grids.py

# one case end-to-end (solver: CUDA elastic; postprocess MPI)
cd examples/meshsize/fullspace18 && bash compare.sh

# the whole study (all five cases + fixed-receiver comparison + logs)
bash examples/meshsize/run_study.sh
```

Parallelism follows the 16-core host: postprocess MPI ranks per grid are
18/20³ → 12 ranks, 22³/24³ → 4, 28³ → 3 (peak memory scales ~linearly
with surviving ranks — tile-local extraction, see
`docs/design/postprocess-tile-parallel.md`), with caps 64/64/96/96/72 GB.

## Results (2026-08-10, five grids, fixed receivers, ≤64 GB budget)

| grid | elem/λs | ranks | peak GiB | mean_corr | shape L2 | best-fit scale | solver s/dir |
|------|---------|-------|----------|-----------|----------|----------------|--------------|
| 18³ | 3.0 | 12 | 48.4 | 0.8502 | 0.1230 | 0.339 | 63 |
| 20³ | 3.3 | 12 | 52.0 | 0.8476 | 0.1275 | 0.346 | — |
| 22³ | 3.7 | 4 | 32.3 | 0.8405 | 0.1235 | 0.338 | 117 |
| 24³ | 4.0 | 4 | 34.3 | 0.8313 | 0.1267 | 0.344 | 148 |
| 28³ | 4.7 | 3 | 40.2 | 0.8467 | 0.1255 | 0.342 | 236 |

Key findings: (1) shape error does NOT improve from 3.0→4.7 elem/λs
(scale-fitted L2 flat at 0.123–0.127) — the residual is PML-reflection +
near-field dominated, not resolution. (2) best-fit scale ~0.34 everywhere
(the ~2.9× amplitude factor is mesh-independent). (3) 28³ is the ceiling.
Detailed logs: `results/stage6_fixed.<n>.log`, `results/compare.<n>.log`,
`results/peak.<n>.txt`; full analysis in
`examples/fullspace-cubic/VERIFICATION.md`.
