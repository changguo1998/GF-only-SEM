# Known Limitations

## 1. Full-Space Waveform Shape Error

The five-grid full-space study reports mean component-wise correlation of about 0.83–0.85.
The residual is likely dominated by the finite-domain C-PML tail, near-field source
representation, and the analytical reference's time/integral discretization. The old
"scale-fitted shape L2" values are invalid: they were normalized by the unscaled analytical
norm and therefore inherited the amplitude error. The corrected 20³ fixed-receiver value is
0.3458 (mean correlation 0.8476); the other grids must be regenerated before drawing a new
resolution-convergence conclusion.

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
The historical "scale-fitted shape L2" was not actually scale-invariant and is invalid. A
20³ production GPU/MPI rerun gives SEM/reference scale 1.038 and corrected fitted relative
L2 0.3458 at the 64 fixed receivers.
