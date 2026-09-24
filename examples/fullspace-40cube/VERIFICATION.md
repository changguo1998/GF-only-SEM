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
| x | 307.1 s |
| y | 446.1 s |
| z | 610.1 s |

The increasing wall time is dominated by filesystem throughput while writing
the three approximately 76 GB direction directories. GPU C-PML and residual
stages remained finite and stable.

## Postprocess and analytical comparison

MPI-2 Debug postprocessing generated all 16 Green-function tiles from the
three full-domain directions in 2346.7 s (39.1 min). The workers used four
memory batches in total under an 80 GiB aggregate field budget; peak resident
memory was below the 95 GiB cgroup limit.

The comparison tool now validates `solver_dt × snapshot_stride = output_dt_s`
and downsamples the solver-grid STF at `snapshot_stride`; this is required when
the output interval is larger than the CFL timestep.

| Receiver set | Mean correlation | Scale-fitted L2 | SEM / analytical scale |
|---|---:|---:|---:|
| Source-centred 7 km box `[17,24] km³` | 0.9558 | 0.1848 | 0.982 |
| Full non-PML interior | 0.9759 | 0.1516 | 0.989 |

All three force directions passed the 0.95 correlation gate. The result shows
that moving the C-PML farther away reduces returned-boundary contamination while
preserving the 1 km spatial resolution.

## Far field before the earliest PML return

The six planar PML interfaces were represented by mirrored source positions.
Receivers were required to be at least two S wavelengths from the source and
to leave 0.1 s between the end of the direct P/S Ricker support and the start
of the earliest possible reflected P-wave support. This leaves 79,201 unique
candidates at 2.000--3.176 S wavelengths. Five fixed seeds selected 100 points
each; metrics were evaluated only before the predicted return.

| Metric | Five-seed mean | Range |
|---|---:|---:|
| Concatenated waveform correlation | 0.9811 | 0.9801--0.9819 |
| SEM / analytical scale | 0.9898 | 0.9881--0.9911 |
| Raw relative L2 | 0.1936 | 0.1898--0.1989 |
| Scale-fitted relative L2 | 0.1933 | 0.1896--0.1985 |
| Median component correlation | 0.9839 | 0.9829--0.9851 |

Removing the PML-return window raises the dominant waveform agreement to about
98.1% and leaves only about 1% amplitude bias, but the energy error remains
about 19%. PML-returned energy is therefore not the only residual source;
point-source representation, SEM dispersion, and analytical time integration
remain candidates. This 40 km box cannot retain the complete direct P/S wave
support before the earliest return beyond four S wavelengths.
