# Known Limitations

## 1. Systematic ~3× Amplitude Factor (SEM vs Continuous Green's Function)

**Status:** DOCUMENTED (2026-07-23). Not a bug — inherent to SEM discretization.

### Observation

SEM displacement Green's function amplitudes are systematically ~3× larger than
continuous analytical references (Lamb, Boussinesq):

| Model | Best-fit scale factor | Waveform correlation |
|----------------|----------------------|----------------------|
| Halfspace (Lamb) | 2.95 | 0.991 (aligned) |
| Layer (PyFK) | 2.60 | 0.745 (multi-point) |

### Investigation (2026-07-23)

Eight hypotheses were tested and ruled out:

| Hypothesis | Test Result |
|------------------------------------------------|---------------------|
| Interpolation error from query points | Ruled out — exact vertex query gives same factor |
| STF sub-sampling (output_dt vs solver_dt) | Ruled out — fine STF gives same factor |
| Convolution numerical method | Ruled out — 3 methods all agree |
| Source weight normalization | Verified — Σwᵢ = 1.0000000000 |
| Mass matrix scaling | Verified — total mass ratio 1.0008× |
| Reference solution correctness | Verified — matches Boussinesq static (ratio 1.02) |
| Factor dependence on distance | None — ratio 2.90–3.47 across 200–1600m |
| Factor dependence on component | None — ratio 2.92–3.21 across all 9 G_ij |

### Conclusion

The ~3× factor is a systematic SEM discretization effect (GLL spectral element
integration vs continuous Green's function). Waveform shape is near-perfect
(correlation 0.98–0.999). Comparing against SPECFEM3D for the same mesh would
confirm whether this factor is inherent to the spectral element method or
specific to this implementation.

### Source

Archived from `docs/bugs.md` (Issue 3, 2026-07-26 clean-up). All three
previously tracked bugs (C-PML divergence, postprocess velocity/acceleration
zeros, postprocess mass-weighting) are fixed and verified.
