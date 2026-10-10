# 50×50×25 layered half-space source-depth test

This case uses the shared 50×50×25, 1 km-element geometry. The 1 km material
interface coincides with the first horizontal mesh face. A moderately soft
surface layer keeps the 1 Hz wavefield resolved on this mesh while retaining
a measurable contrast with the half-space.

| Region | Thickness | Vp | Vs | Density |
| --- | ---: | ---: | ---: | ---: |
| Surface layer | 1 km | 4000 m/s | 2500 m/s | 2500 kg/m³ |
| Half-space | Below 1 km | 5000 m/s | 3000 m/s | 2700 kg/m³ |

The model has a free surface, five-element side and bottom C-PML, N=4,
1 Hz Ricker source, 50 ms snapshots, and a 10 s record. The source is at
`(18.5, 25.5, z) km`; the depth sweep is 0, 1, 10, 100, 500, 900, and 1100 m.
The 900 m and 1100 m cases lie on opposite sides of the interface.

Run the C++ preprocessing, Debug CUDA forward solver, and selected-tile
postprocessing for every depth:

```bash
source scripts/env.sh
python examples/layered-50x50x25/run_depth_sweep.py
```

Each depth archive retains its configuration, logs, and three long-axis tiles;
full-domain wavefields are removed after postprocessing. The run requires about
263 GiB of temporary space for one depth and checks for at least 360 GiB free.
The source-depth comparison reuses the PyFK implementation and layer Python
environment from `examples/layer`:

```bash
examples/layer/.venv/bin/python examples/layered-50x50x25/compare_depths.py
```

The comparison reports direct SEM differences from the strict-surface source
at 2/5/8/11/14 km. It also compares each depth with PyFK at 2 and 14 km using
the same layered model and source time function. Comparison windows end at
3.5/4.5/5.5/6.5/7.3 s, before the estimated earliest side-PML return. Results
are written to `docs/test-reports/data/layered-50x50x25-source-depth.csv`.
PyFK rejects a point source exactly on the free-surface interface, so its
reference for the 0 m SEM case uses a 1 m source-depth approximation; the
direct SEM-to-SEM comparison still uses the exact 0 m source.
