# forward/viscoelastic/ — AGENTS.md

## Purpose

Viscoelastic CG-SEM solver (SLS attenuation). Extends the elastic solver with
standard linear solid (SLS) memory variables for frequency-independent Q
attenuation.

**Status:** Skeleton only — implementation deferred. See
[`docs/deferred.md`](../docs/deferred.md) for design details.

## Architecture

TBD — will follow `forward/share/` + `forward/elastic/` structure.

When implemented, the viscoelastic solver will share the infrastructure library
(`libgf_shared` from `forward/share/`) and add:

- SLS memory arrays and stress-update terms in its element residual
- Additional HDF5 datasets for SLS memory state in restart files
- A runtime flag in `ConfigData`/`RankData` to enable solver-specific data
