# Known Limitations

## 1. Full-Space Waveform Shape Error

The five-grid full-space study reports mean component-wise correlation of about 0.83–0.85.
All five grids were regenerated on 2026-09-20 after the postprocess averaging and fitted-L2
fixes. From 18³ to 28³ (3.0–4.7 elements per S wavelength), the corrected fitted relative L2
is flat at 0.3414–0.3468, while the SEM/analytical amplitude scale remains 1.013–1.038. Thus
neither the waveform residual nor the absolute amplitude improves with compact-domain grid
refinement.

The expanded-domain source-centred comparison reaches correlation 0.9672 and fitted relative
L2 0.0882 even at only 3.0 elements per S wavelength. This isolates finite-boundary/C-PML
returned energy as the dominant source of the compact-domain residual. Near-field point-source
representation and analytical time/integral discretization may contribute to the remaining
expanded-domain error, but the five-grid compact-domain study cannot separate them further.

## Resolved: Systematic ~3× Amplitude Factor

**Status:** Fixed in postprocess (2026-09-17).

This was not an inherent SEM/GLL discretization effect. The serial and MPI postprocessors
used one shared `node_count` while independently accumulating displacement, velocity, and
acceleration. With all three quantities present, the count was incremented three times per
cell contribution, so each vector field was divided by three.

Each field now owns its accumulator and sample count. A regression test verifies that
enabling all three fields cannot change any field's average. The full-space analytical
comparison also rejects best-fit SEM/analytical amplitude scales outside [0.8, 1.2]; the old
correlation-only gate could not detect uniform amplitude errors.

Historical scales map exactly as follows because the faulty division was exactly three:

| Case | Historical scale | Corrected scale |
|------|------------------|-----------------|
| Full-space grids | 0.338–0.346 (SEM/reference) | 1.014–1.038 |
| Half-space | 2.95 (reference/SEM) | 0.983 |
| Layer | 2.60 (reference/SEM) | 0.867 |

The historical correlations are unchanged because they are invariant under uniform scaling.
The historical "scale-fitted shape L2" was not actually scale-invariant and is invalid. The
regenerated five-grid study gives SEM/reference scale 1.013–1.038 and corrected fitted relative
L2 0.3414–0.3468 at the 64 fixed receivers.
