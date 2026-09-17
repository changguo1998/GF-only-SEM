# Known Bugs (Archived)

All previously tracked bugs are **fixed and verified** (2026-07-26).

| # | Bug | Resolution |
|---|-----|------------|
| 1 | C-PML numerical divergence | Fixed — non-symmetric stress, parameter separation, alpha-convolved memory |
| 2 | Velocity/acceleration = 0 in postprocess | Fixed — added velocity/acceleration reads in postprocess merge |
| 3 | Displacement amplitude ~1.9e9× mismatch | Fixed — postprocess mass-weighting bug (commit `6f90c12`) |
| 4 | Displacement/velocity/acceleration uniformly divided by three | Fixed — independent per-field averaging counts (2026-09-17) |

The residual ~3× factor was also caused by postprocessing, not SEM discretization. See
[`docs/design/known-limitations.md`](design/known-limitations.md).

No active bugs remain.
