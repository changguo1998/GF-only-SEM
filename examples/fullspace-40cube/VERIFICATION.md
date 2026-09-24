# 40³ Expanded Full-Space Debug Verification

## Configuration

This diagnostic case keeps the verified 1 km element size and expands the
physical domain to 40 km on each axis. The five-element C-PML therefore starts
5 km from each boundary, while the source is 14.5 km from the nearest PML
interface. Debug output records all 64,000 elements and all four dynamic
fields. To keep the generated records within local storage, snapshots are
written every 0.05 s; the solver timestep remains CFL-controlled at 0.0166667 s.

| Parameter | Value |
|---|---:|
| Domain / mesh | 40 × 40 × 40 km / 40³ elements |
| Element size | 1 km |
| Polynomial order | 4 |
| Source | 1 Hz Ricker at (20.5, 20.5, 20.5) km |
| Material | vp=5 km/s, vs=3 km/s, density=2700 kg/m³ |
| C-PML | 5 elements on all six faces |
| Solver timestep / duration | 0.0166667 s / 8 s |
| Output interval / frames | 0.05 s / 160 |
| Build | Debug CUDA (`bin-debug`) |

Preprocessing passed the SEM consistency check. It produced 64,000 elements,
8,000,000 GLL points, 16 partitions, and 2,064,381 recorded GLL nodes. The
preprocessor storage estimate was 232.77 GB.

## CUDA forward results

All 480 solver steps and 160 full-domain records completed in every direction.
Every record contains `strain`, `displacement`, `velocity`, and `acceleration`
as uncompressed float32 datasets with shape `[1, 64000, 125, ncomp]`.

| Force direction | Wall time |
|---|---:|
| x | 246.1 s |
| y | 378.3 s |
| z | 513.8 s |

The increasing wall time is dominated by filesystem throughput while writing
the three approximately 76 GB direction directories. GPU C-PML and residual
stages remained finite and stable.

## Postprocess and analytical comparison

MPI-2 Debug postprocessing generated all 16 Green-function tiles from the
three full-domain directions in 3698.9 s (61.6 min). The workers used three
memory batches under an 80 GiB aggregate field budget; peak resident memory
was below the 100 GiB cgroup limit.

The comparison tool now validates `solver_dt × snapshot_stride = output_dt_s`
and downsamples the solver-grid STF at `snapshot_stride`; this is required when
the output interval is larger than the CFL timestep.

| Receiver set | Mean correlation | Scale-fitted L2 | SEM / analytical scale |
|---|---:|---:|---:|
| Source-centred 7 km box `[17,24] km³` | 0.9558 | 0.1848 | 0.982 |
| Full non-PML interior | 0.9753 | 0.1516 | 0.990 |

All three force directions passed the 0.95 correlation gate. The result shows
that moving the C-PML farther away reduces returned-boundary contamination while
preserving the 1 km spatial resolution.
