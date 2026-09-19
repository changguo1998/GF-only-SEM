# Expanded Full-Space Verification

## Configuration

Run on 2026-09-17 with the CUDA elastic solver and the Spack-managed CUDA
toolchain:

| Parameter | Value |
|---|---:|
| Domain | 28 × 28 × 28 km |
| Mesh | 28³ elements, 1 km per element |
| Polynomial order | 4 |
| Source | 1 Hz Ricker at (14.5, 14.5, 14.5) km |
| Material | vp=5 km/s, vs=3 km/s, density=2700 kg/m³ |
| PML | 5 elements (5 km) on every face |
| Nearest source-to-PML-interface distance | 8.5 km |
| Duration / output interval | 8 s / 0.01 s |
| Preprocess partitions / output tiles | 16 / 16 |

The mesh has 21,952 elements and 1,442,897 global GLL nodes. All preprocessing
parameter-consistency checks passed.

## Runtime and storage

| Stage | Result |
|---|---:|
| CUDA forward, x force | 312.505 s |
| CUDA forward, y force | 397.988 s |
| CUDA forward, z force | 479.196 s |
| MPI postprocess, 3 ranks | OOM at 90 GiB (peak demand >90 GiB) |
| MPI postprocess, 2 ranks | 2,088.8 s, passed under 90 GiB |
| MPI postprocess, 1 rank | 4,129.2 s, passed under 60 GiB |
| Recorded nodes | 495,597 |
| Wavefields | 170 GiB |
| Green-function tiles | 75 GiB |

All 16 output tiles contain finite displacement values; the maximum absolute
displacement is `1.76854e6`.

The analytical comparison originally loaded all 75 GiB displacement tensors
and then allocated a second concatenated copy. It was changed to load only the
50 selected receiver traces. This removes the comparison-stage OOM without
changing the metric calculation.

The 90 GiB postprocess run reached approximately 90 GiB cgroup usage while
writing large tiles, but recorded no OOM events. Two ranks are suitable at
90 GiB; three ranks exceed that limit. For the default 60 GiB project limit,
the script uses one rank.

## Stokes full-space comparison

The default comparison samples 50 nodes throughout the true non-PML box
`[5, 23] km³` (150 component comparisons per force direction):

| Metric | Result |
|---|---:|
| Mean correlation | **0.9106** |
| Mean relative L2 | 0.3663 |
| Best-fit SEM / analytical scale | 1.016 |
| Scale-fitted L2 | 0.2708 |
| Correlation, distance \<2 λs | 0.9885 (n=18) |
| Correlation, distance 2–4 λs | 0.9050 (n=126) |
| Correlation, distance 4–6 λs | 0.7932 (n=6) |

The default score includes distant points approaching the PML and therefore
measures both interior accuracy and the remaining late PML contamination.

For a closer comparison with the former 18 km / 24³ case, the receiver box was
shifted to preserve its position relative to the source: `[10.625, 17.625] km³`. The result passes the 0.95 analytical threshold:

| Metric | Expanded 28 km case |
|---|---:|
| Mean correlation | **0.9672** |
| Mean relative L2 | 0.1186 |
| Best-fit SEM / analytical scale | 0.999 |
| Scale-fitted L2 | **0.0882** |

## Conclusion

Expanding the domain improves the default full-interior correlation from the
previous 24³ result of 0.8430 to 0.9106, while the comparable source-centred
region reaches 0.9672. This happens despite reducing resolution from four to
three elements per S wavelength. The result supports the earlier diagnosis:
PML-returned late energy and receiver proximity to the PML dominate the
remaining whole-domain error; spatial dispersion is secondary at this mesh
size. The stable fitted amplitude near one also rules out a global source-scale
error for this run.

## Grid comparison: 28³ / 26³ / 24³ / 22³ / 20³ (2026-09-18)

The physical domain, 1 Hz source, polynomial order, material, duration, and
five-element PML were retained. The element sizes are 1.0, 1.077, 1.167, 1.273,
and 1.4 km, giving 3.0, 2.79, 2.57, 2.36, and 2.14 elements per S wavelength.
Each source is at the nearest element centre to the domain centre.

| Quantity | 28³ | 26³ | 24³ | 22³ | 20³ |
|---|---:|---:|---:|---:|---:|
| Elements | 21,952 | 17,576 | 13,824 | 10,648 | 8,000 |
| Global GLL nodes | 1,442,897 | 1,157,625 | 912,673 | 704,969 | 531,441 |
| Recorded nodes | 495,597 | 359,125 | 250,173 | 165,669 | 102,541 |
| CUDA forward x / y / z | 312.5 / 398.0 / 479.2 s | 242.9 / 263.4 / 341.3 s | 189.6 / 203.1 / 214.5 s | 145.2 / 146.2 / 142.4 s | 99.4 / 112.3 / 105.5 s |
| Postprocess | 4,129.2 s (1 rank) | 3,153.3 s (1 rank) | 1,981.7 s (1 rank) | 560.0 s (4 ranks) | 345.5 s (1 rank) |
| Wavefields | 170 GiB | 123 GiB | 85 GiB | 56 GiB | 35 GiB |
| Green-function tiles | 75 GiB | 55 GiB | 39 GiB | 26 GiB | 17 GiB |

Default non-PML-box metrics are similar because the receiver distribution and
physical PML geometry both change with the grid:

| Metric | 28³ | 26³ | 24³ | 22³ | 20³ |
|---|---:|---:|---:|---:|---:|
| Mean correlation | 0.9106 | 0.9106 | 0.8972 | 0.8742 | 0.9093 |
| Mean relative L2 | 0.3663 | 0.3533 | 0.3640 | 0.3711 | 0.3545 |
| Best-fit SEM / analytical scale | 1.016 | 1.003 | 1.005 | 1.014 | 1.007 |
| Scale-fitted L2 | 0.2708 | 0.2170 | 0.2419 | 0.2209 | 0.2245 |

A source-relative 7 km comparison box is more diagnostic. The 28³ box
`[10.625, 17.625] km³` was shifted with each source to
`[10.663, 17.663] km³` for 26³, `[10.708, 17.708] km³` for 24³,
`[10.761, 17.761] km³` for 22³, and `[10.825, 17.825] km³` for 20³:

| Metric | 28³ | 26³ | 24³ | 22³ | 20³ |
|---|---:|---:|---:|---:|---:|
| Mean correlation | **0.9672** | **0.9318** | **0.9309** | **0.9179** | **0.8968** |
| Mean relative L2 | 0.1186 | 0.1541 | 0.1616 | 0.1902 | 0.2267 |
| Best-fit SEM / analytical scale | 0.999 | 1.001 | 1.000 | 0.999 | 0.999 |
| Scale-fitted L2 | **0.0882** | **0.0973** | **0.0995** | **0.1026** | **0.1200** |

Scale-fitted L2 improves monotonically with grid count:
`0.1200 → 0.1026 → 0.0995 → 0.0973 → 0.0882`. Mean correlation at 24³ and
26³ differs by only 0.0009, within the noise expected from sampling 50
different mesh nodes; the fitted L2 still shows a small improvement. The 22³
grid has a modest drop from 24³ (correlation 0.9309→0.9179 and fitted L2
0.0995→0.1026), whereas the 20³ loss is clearer. The stable amplitude scale
shows that the loss is waveform shape rather than source normalisation. The
26³ mesh remains below the 0.95 correlation criterion; 28³ is the first tested
grid to pass it. Coarser meshes combine stronger spatial dispersion with
earlier PML returns: the nearest PML interface is 8.5 km from the 28³ source,
8.08 km for 26³, 7.58 km for 24³, 7.0 km for 22³, and 6.3 km for 20³.

The 26³ postprocess completed with one rank under the 60 GiB limit, using
approximately 54 GiB. Its additional model and partition files occupy 977 MiB
and 378 MiB, respectively. All preprocessing consistency checks and the
finite-output guard passed.

The 22³ postprocess completed with four ranks in 560.0 s under the shared
60 GiB cgroup limit. The same work would be substantially slower with one
rank. Its model and partition files occupy 592 MiB and 228 MiB, respectively;
all consistency checks and the 16-tile finite-output guard passed.
