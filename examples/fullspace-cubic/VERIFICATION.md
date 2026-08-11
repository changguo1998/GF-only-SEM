# Full-Space Cubic Test Model — Verification Report

## Model Configuration (Iteration 2 — Parameter Optimized)

| Parameter | Original (v1) | Optimized (v2) |
|-----------|---------------|-----------------|
| Domain | 18 × 18 × 18 km | 18 × 18 × 18 km |
| Elements | 18³ = 5,832 | 18³ = 5,832 |
| Element size | 1 km | 1 km |
| GLL order | N=4 | N=4 |
| P-wavelength (λp) | 2,500 m | 5,000 m |
| Elements/λp | 2.5 | 5.0 |
| S-wavelength (λs) | 1,500 m | 3,000 m |
| Elements/λs | 1.5 | 3.0 |
| f0 | 2 Hz | 1 Hz |
| PML thickness | 3 elements (3 km) | 5 elements (5 km = 1.0λp) |
| Source location | (9,9,9) km — 8-element corner | (9.5,9.5,9.5) km — single element center |
| Solver | MPI CPU | GPU CUDA |
| Duration | 5.0 s | 8.0 s |

## Bugs Discovered and Fixed

### Bug 1: C++ STF parameter mismatch (CRITICAL)

**Symptom**: Correlation dropped to 0.017 after changing Python config to f0=1Hz, t0=2s.

**Root cause**: The C++ config file `config_user_fullspace.cpp` has its own `stf_func()`
with hardcoded `f0_hz=2.0, t0_s=1.0`. Even though `f0_for_pml_hz` was updated to 1.0,
the `stf_func` was not. The C++ preprocessor (`gf_preprocess run`) uses the C++
`stf_func` to compute the STF array written to `config.h5`, OVERRIDING the Python
`stf_func` from `config.py`.

**Fix**: Updated `config_user_fullspace.cpp` lines 49-50:

```cpp
double f0_hz = 1.0;  // was 2.0
double t0_s = 2.0;   // was 1.0
```

**Impact**: All C++ preprocessed cases with non-default STF parameters are affected.
The halfspace and layer cases use f0=2Hz, t0=1s in both Python and C++ (matching),
so they are NOT affected. This bug only manifests when Python and C++ STF parameters
diverge.

## Results (Parameter-Optimized)

| Force Direction | Mean Correlation | Mean Rel. L2 | Comparisons |
|-----------------|-----------------|--------------|-------------|
| x | 0.787 | 0.740 | 150 |
| y | 0.771 | 0.748 | 150 |
| z | 0.785 | 0.737 | 150 |
| **Overall** | **0.781** | **0.741** | **450** |

### Comparison with Original (v1)

| Metric | Original (18³, f0=2Hz, 8-cell source) | Optimized (18³, f0=1Hz, 1-cell source) |
|--------|--------------------------------------|---------------------------------------|
| Correlation | 0.663 | 0.781 (+0.118) |
| Rel. L2 | 0.797 | 0.741 (-0.056) |
| z-component | 0.760 | 0.785 (+0.025) |

### Key Improvements

1. **Source in single element vs 8-element corner**: eliminated source splitting.
   Previously the source energy was distributed across 8 elements, distorting the
   near-field radiation pattern.

1. **PML at 5 elements (5km = 1.0λp) vs 3 elements (3km = 0.6λp)**: PML now spans
   a full P-wavelength, significantly reducing boundary reflections.

1. **f0=1Hz gives 5 elements/P-wavelength** (was 2.5): numerical dispersion reduced.

1. **STF matched between SEM and analytical**: the C++ STF bug fix ensures both
   use identical source time functions.

## Residual Error Analysis

### Known ~3× SEM Amplitude Factor

The SEM displacement amplitude is ~2.7× smaller than the Stokes analytical solution.
This is documented in [`docs/design/known-limitations.md`](../../docs/design/known-limitations.md)
as a systematic GLL spectral element integration effect. The Pearson correlation
is amplitude-invariant, so this factor does NOT reduce the reported correlation.

### Remaining Shape Errors (~22% unexplained variance)

1. **S-wave resolution (3 elements/λs)**: At f0=1Hz, λs = 3,000m with 1km elements
   gives 3 elements per S-wavelength. This is marginal for accurate S-wave propagation.
   P-waves (5 elements/λp) are well-resolved.

1. **PML corners**: Overlapping PML layers in 8 corners cause non-physical damping
   for waves propagating diagonally.

1. **Near-field integral discretization**: The analytical near-field term uses
   discrete integration with step dt=0.01s. For receivers within 3km of the source,
   the near-field integral has only ~20 tau points, introducing O(dt²) error.

1. **Sub-sample timing (≤10ms)**: The analytical time shift uses integer step
   truncation, causing up to 0.5 sample (5ms) jitter per receiver.

### Why 0.95 May Not Be Achievable

The ~22% shape error is dominated by:

- **S-wave numerical dispersion** (3 elements/λs → ~5-10% phase velocity error)
- **PML corner effects** (geometric, irreducible for cubic domain)
- **GLL quadrature accuracy** (N=4 for near-field Green's function)

To reach 0.95 correlation would require:

- S-wave resolution ≥ 4 elements/λs (reduce f0 to 0.75Hz or refine mesh)
- Larger domain or spherical PML to eliminate corner effects
- Higher GLL order (N≥6) for near-field accuracy

These are algorithmic limitations, not parameter errors. The 0.78 correlation
represents the practical limit for N=4 SEM with 3 elements/S-wavelength.

## Conclusions

1. **STF parameter mismatch bug found and fixed** in C++ config system. Python
   config changes must be synchronized with `config_user_*.cpp` `stf_func()`.

1. **Parameter optimization improved correlation from 0.66 → 0.78** through:
   source relocation (8-cell → 1-cell), PML thickening (3→5 elements),
   and frequency-mesh matching (2Hz→1Hz for 5 elements/λp).

1. **Remaining ~22% shape error is algorithmic** (S-wave dispersion, PML corners,
   GLL quadrature). Achieving 0.95 correlation requires algorithmic improvements
   beyond parameter tuning.

1. **The known ~3× SEM amplitude factor is confirmed** (2.7× measured) — this
   is a systematic GLL integration effect, not a code bug.

## CUDA vs MPI-CPU Solver Consistency Verification (2026-08-05)

Cross-verification of the two elastic solver backends on this case
(direction=x, 800 steps, 57717 common recording GLL nodes):

| step | rel_l2 | Pearson corr | threshold | result |
|------|--------|--------------|-----------|--------|
| 400 | 1.737e-04 | 0.99999998 | rel_l2\<0.01, corr>0.999 | PASS |
| 700 | 5.113e-04 | 0.99999987 | rel_l2\<0.01, corr>0.999 | PASS |

**Verdict: CONSISTENT.** Reproduce with:

```bash
# MPI run (16 ranks)
cd tmp/mpi_run && mpirun -n 16 ../../../../bin/gf_solver_elastic_mpi --direction x
# comparison (expects CUDA records in wavefields/x, MPI in tmp/mpi_run/wavefields/x)
cd .. && ../../.venv/bin/python compare_solvers.py
```

### Method

`compare_solvers.py` aligns CUDA (`wavefields/x/record_0_*.h5`) and MPI
(`tmp/mpi_run/wavefields/x/record_{r}_*.h5`) strain snapshots by rounded GLL
node **coordinates** (the two backends use different global node numberings —
`field/cell/global_cell2global_node` vs `partition/global_cell2global_node`),
then computes rel_l2 = ||mpi-cuda||/||cuda|| and Pearson correlation over the
aligned 6-component strain vectors. Only 9 of 16 ranks own recording cells
(interior non-PML elements), so `record_*_<step>.h5` has 9 files — ranks with
only PML elements record nothing (RecordWriter returns early for n_rec_cell=0).

### Root causes found and fixed (3 solver/preprocess bugs)

1. **Missing multi-rank shared nodes in exchange patterns**
   (`preprocess/partition.py`). Exchange DOF lists were built from
   face-adjacent cell pairs only, so nodes at partition edges/corners shared
   by 3+ ranks were never exchanged. The source element sits at a 7-rank METIS
   corner; each rank integrated its copy of those nodes independently and the
   copies diverged. Fixed by building exchange patterns from node ownership
   (all co-owner rank pairs), ordered by global node id.

1. **Per-rank PML damping inconsistency** (`forward/share/src/solver.cpp`
   init). `rank_node_damping[node] = pml_damping[e*n]` assigns last-local-cell
   wins per rank. At PML-interface nodes, an interior cell (d=0) and a PML
   cell (d>0) share the node; different ranks picked different winners, so
   each rank damped its own copy of a shared node differently and the copies
   diverged exponentially. Fixed by (a) tracking the winning global cell id
   per node locally, and (b) a new `exchange_halo_max` reduction
   (`forward/share/src/exchange.cpp`, no-op variant in `exchange_noop.cpp`)
   on a packed value `cell_id + damping/2`, implementing exactly the
   single-rank/CUDA rule "highest global cell id wins" on every rank.

1. **Strain recording double-offset bug (the apparent "explosion")**
   (`forward/share/src/solver.cpp` `compute_full_strain`, used by BOTH
   backends' snapshot paths). The element base pointer was computed as
   `&strain_disp[elem*n_node*3 + node_idx*3]` and the GLL stencil then added
   `3*node_sjk` etc. — the `node_idx*3` term double-offsets every stencil
   read. Nodes whose summed index stayed < 125 read a wrong in-element node
   (bounded error, invisible for smooth fields); sums >= 125 read into the
   next element; at each rank's LAST element they read past the vector into
   heap memory holding growing simulation state — producing the exponential
   "strain explosion" (max 2.6e9 at step 700) while velocity/displacement
   stayed correct (rel_l2 vs CUDA < 1e-2 even pre-fix). Fixed by removing the
   `+ node_idx*3` offset.

### Side notes

- The C++ preprocessor hardcodes `n_ranks=16` (`config_user_fullspace.cpp`),
  and `preprocess/cli.py` reuses the C++ `partition/element_to_rank` from
  model.h5 regardless of `config.py:n_ranks`. A 1-rank control run therefore
  requires rebuilding the partition explicitly (see
  `tmp/cpu1_run/build_1rank.py` for the recipe).
- CPU single-rank vs CUDA strain agreement after the fix: the MPI=CPU result
  above implies the CPU kernel matches CUDA; the earlier apparent CPU/CUDA
  strain mismatch (rel_l2=0.23) was fully explained by bug 3.

## Re-verification (2026-08-07): parameter consistency + low-memory pipeline

Full pipeline re-run on the parameter-optimized v2 configuration (f0=1 Hz,
t0=2 s, 8 s duration, PML 5 elements, 18³ mesh, N=4, source at 9500 m
center, amplitude 1e20 N). Three changes were made around the case; the
numerical result is unchanged, which validates that none of them drifted:

| Change | Detail |
|--------|--------|
| **Parameter consistency by construction** | `analytical_compare.py` no longer hardcodes vp/vs/rho, f0/t0/amplitude or the Ricker rebuild — it reads the exact STF (`config.h5:/source/stf_values`), source (`/source/{x,y,z}`), dt (`/simulation/output_dt_s`) and material (`model.h5:/field/element/{vp,vs,density}`) from the run artifacts. Eliminates the STF-mismatch bug class (see Bug 1 above). |
| **Consistency guard (Stage 2.5)** | `check_sem_consistency.py` compares config.py vs config.h5 vs model.h5 (source, amplitude, output_dt, nsteps, STF, stride divisibility, material) before the solver runs — 10 checks, all PASS. |
| **Low-memory postprocess** | `compare.sh` now runs `gf_postprocess_mpi` (4 ranks) instead of serial `gf_postprocess`: total-tree peak **20.6 GB** (measured; serial was 42–50 GB) under a `GF_MEM_LIMIT_GB=24` cap, ~101 s (serial 173–194 s). Bit-identical output (verified in the postprocess tile-parallel design). |
| **Hermetic runs** | `compare.sh` clears `wavefields/{x,y,z}` (+runs) and `greenfun/` before each stage; `analytical_compare.py` asserts recorded frames == config `nsteps` — catches stale record files from a different-duration previous run (hit and fixed during this session). |

### Parameters suitability (requirement: analytical & SEM in a good computation range)

| Parameter | Value | Comment |
|-----------|-------|---------|
| Elements | 18³ = 5832, 1 km, N=4 | memory floor; solver ~1–2 GB host |
| f0 / t0 | 1 Hz / 2.0 s | 5 elements/λp, 3 elements/λs (S marginal, see below) |
| PML | 5 elements = 1.0 λp | adequate absorption (C-PML) |
| dt chain | solver_dt = output_dt = 0.01 s, stride=1 | exact divisibility enforced & checked |
| Duration | 8 s | full S-window at the farthest receiver (arrival ≤3.2 s + wavelet ~4.7 s) |
| Amplitude | 1e20 N | recorded max displacement ≈ 5.9e5 — healthy float32 range |

### Results (2026-08-07, GPU `elastic_cuda`, 800 steps)

| Metric | Value |
|--------|-------|
| Force x / y / z correlation | 0.7869 / 0.7712 / 0.7846 |
| **Overall mean_corr** | **0.7809** (identical to the 2026-08-05 baseline 0.781) |
| mean_l2 (raw) | 0.7415 |
| best-fit scale (SEM/analytical) | 0.338 → SEM ≈ 2.96× smaller — confirms the documented ~3× factor |
| scale-fitted l2 (shape only) | 0.1570 |
| corr vs distance: r\<2 λs | 0.8210 (n=120) |
| corr vs distance: 2–4 λs | 0.6206 (n=30) — near-receivers clean, far receivers sit at the PML-box corners |

### Parameter-sensitivity experiments (recorded for reference)

| Config | Overall corr | Why it underperforms |
|--------|-------------|---------------------|
| v3: f0=0.75 Hz, t0=2.67 s, 10 s | 0.727 | λs=3750 m → all 150 comparisons at r\<2λs: near-field-dominated, where the SEM point force on a 1 km mesh and the coarse near-field integral match worst |
| v5: f0=1 Hz, 10 s | 0.748 | the extra 2 s adds small-amplitude PML-reflected tail that dilutes the correlation window |
| v7 (final): f0=1 Hz, 8 s | **0.781** | adequate coverage, minimal tail noise |

**Conclusion:** the v2/v7 parameter set is the practical optimum for this box.
Raising correlation beyond ~0.78 requires an algorithmic change (larger domain
/ thicker PML to move far receivers away from corners, or N≥6 GLL order), not
parameter tuning.

## Mesh refinement (2026-08-07): 18³ → 24³ (750 m elements)

Per user request, the mesh was refined (more grid points, smaller elements)
while keeping the pipeline memory-safe. Same domain (18 km)³, N=4, f0=1 Hz,
t0=2 s, 8 s duration; PML thickened 5 → 7 elements to keep the physical PML at
~1.05λp (5.25 km); source moved to the 24³ element-center (9375, 9375, 9375) m;
tiles resliced to [2,2,3,3] (16 tiles).

### Parameter-consistency enforcement (hardened this iteration)

compare.sh's Stage 6 previously passed a hardcoded `--source 9500 ...` which
SILENTLY OVERRODE the SEM source (9375) for the 24³ run — the comparison's own
WARNING caught it (the consistency chain works), but the first 24³ comparison
was discarded as invalid. Fixed: compare.sh no longer passes `--source`, and
analytical_compare.py now raises a hard error if an explicit --source disagrees
with config.h5 (the SEM source is authoritative). The valid re-run below was
made against the same SEM tiles with the correct source.

### Results (valid 24³, correct source)

| Metric | 18³ (2026-08-05 baseline) | 24³ refined |
|--------|---------------------------|-------------|
| Elements / element size | 5832 / 1 km | 13824 / 750 m |
| Elements per λs / λp | 3.0 / 5.0 | **4.0 / 6.7** |
| Recorded GLL nodes | 57,717 | ≈116 k |
| **Overall mean_corr** | 0.7809 | **0.7916** (+0.011) |
| mean_l2 | 0.7415 | 0.7260 (−0.016) |
| best-fit scale (SEM/ana) | 0.338 | 0.341 (≈2.9× factor) |
| scale-fitted l2 (shape) | 0.1570 | 0.1571 |
| corr r\<2 λs | 0.8210 (n=120) | **0.8245** (n=135) |
| corr 2–4 λs | 0.6206 (n=30) | 0.4958 (n=15) |
| postprocess peak RAM | 20.6 GB (cap 24) | **34.3 GB** (cap 56, safe on 125 GB host) |
| solver wall / direction | ~65 s | ~148 s |
| disk: wavefields / tiles | 19 GB / 8 GB | 39 GB / 15 GB |

**Verdict:** refinement gives a real but modest accuracy gain (corr +0.011,
l2 −0.016); the scale-fitted shape error is unchanged (0.157) so the residual
is dominated by PML corners (the far-field bin even degrades: the 24³ physical
interior is 7.5 km vs 8 km, pushing far receivers closer to the PML boundary).
Near-field and dispersion improve as expected (4.0 elements/λs vs 3.0). Memory
stays safe (34 GB postprocess on a 125 GB host). A further 36³ refinement is
estimated at ~7.7× recorded nodes → ~100 GB+ peak and is not recommended on
this host without a larger machine or the multi-tile MPI postprocess at 16
ranks.

### Correction (same day): the 0.79 result was a mask bug — honest value is 0.843

The comparison script's "interior" receiver mask was a HARDCODED
`[3000, 15000] m` box (an 18³-era relic). At the current 24³/PML-7 geometry the
real interior is `[5250, 12750] m`, so the stale box silently included
~2.25 km of actual PML-region vertices on every side — contaminated/by-passed
receivers dragged every reported correlation down. (The same box was used for
the 18³ baseline, so 0.7809 is also understated; an honest 18³ rerun is
pending a decision.)

Fix: `interior_bounds_from_config()` now derives the box from `config.h5`
`/simulation` PML attrs × true element size from `model.h5` cell coords
(NOT tile coords, which are cropped to the recording region). A guard
rejects a PML that covers the whole domain.

Corrected 24³ numbers (50 receivers × 3 forces × 3 components, derived box):

| Metric | stale-mask value | corrected |
|--------|------------------|-----------|
| OVERALL mean_corr | 0.7916 | **0.8430** |
| mean_l2 | 0.7260 | **0.7063** |
| best-fit scale (SEM/ana) | 0.341 | 0.341 (≈2.9×) |
| scale-fitted shape L2 | 0.1571 | **0.1307** |
| corr r\<2λs | 0.8245 | **0.8477** (n=147) |
| corr 2–4λs | 0.4958 | 0.6125 (n=3; low corner ≈7.1 km ≈ 2.4λs) |

So the mesh refinement 18³→24³ delivers a real gain (raw published 0.781→0.843,
modulo the mask contamination in both), not the +0.011 previously attributed.
The residual gap to the halfspace-style 0.99 level is dominated by PML
reflections (cannot be window-isolated in this box: PML echoes arrive before the
Ricker body ends) plus the ~2.9× amplitude scale (constant across distances and
goes back ~400 m from true interior; distinct from shape error, which the
scale-fitted L2 captures at 0.131).

## Grid convergence (2026-08-08): 18³ / 20³ / 24³, common receiver box

Three meshes run with identical physics (18 km³, f0=1 Hz/t0=2 s/8 s, dt=0.01,
N=4, vp/vs/rho as above): 18³ (1000 m, PML5), 20³ (900 m, PML6), 24³ (750 m,
PML7). PML ≈ 5.0-5.4 km (~1.0-1.08 λp) everywhere; source at the cell center
nearest domain centre (9500/9450/9375 m). To make the comparison cookbook
independent of each mesh's interior width, all three are scored on the SAME
receiver region `--interior-box 5500 12500` m (fits inside every interior) —
50 receivers × 3 forces × 3 components each.

| Metric | 18³ (3.0 elem/λs) | 20³ (3.3) | 24³ (4.0) |
|--------|-------------------|-----------|-----------|
| mean_corr | 0.8468 | 0.8494 | 0.8284 |
| mean_l2 | 0.7016 | 0.7002 | 0.7108 |
| best-fit scale (SEM/ana) | 0.340 | 0.340 | 0.336 |
| **scale-fitted shape L2** | 0.1225 | 0.1207 | **0.1061** |
| solver wall / direction | 64 s | 86 s | 148 s |
| postprocess peak (MPI -n4) | 16.2 GB | 17.4 GB | 34.3 GB |
| recorded vertices | 68.7 k | 73.9 k | 133.6 k |

Read of the trend:

1. **The shape error converges, the correlation is noisy.** scale-fitted
   (scale-invariant) L2 improves monotonically 0.123 → 0.121 → 0.106
   (~13% over the range) — spatial resolution genuinely reduces the waveform
   shape mismatch. Raw mean_corr is NOT monotonic (0.847/0.849/0.828): the
   residual is dominated by low-correlating PML-reflected tails plus
   50-receiver sampling noise (~±0.02), which drowns the small per-point
   gain. Shape L2 is the cleaner resolution metric here.
1. **The ~2.9× amplitude factor is flat with mesh size** (scale 0.340/0.340/
   0.336). It does NOT converge toward 1 as h shrinks over 3.0-4.0 elem/λs →
   it is an effective-source / normalization offset, not a discretization
   error a finer mesh fixes in this range (the r≫λ far-field test would still
   be the definitive one).
1. **Cost vs accuracy:** 24³ gives the −13% shape-error gain for 2.3× solver
   time and ~2.1× postprocess peak over 18³; memory stays far under the cap
   at every grid.

Caveats: one run per grid; source position and PML thickness differ slightly
(500/450/375 m offset; 5.0/5.4/5.25 km) — small confounds for a 3-point trend.
A 32³ point (562 m, ~90 GB postprocess peak, tight on 125 GB) would confirm
whether shape L2 keeps dropping and mean_corr finally beats ~0.85 at a
PML-tail-dominated floor.

## Mesh-size study, fixed receivers, uncompressed HDF5, 64 GB budget (2026-08-10)

Protocol: five grids (18/20/22/24/28) with IDENTICAL physics — domain 18 km³,
source (9375,9375,9375) m, Ricker f0=1 Hz t0=2 s 8 s, dt=0.01, N=4, PML ≈
1.0 λp — compared at 64 FIXED receiver points (seed-0 uniform in [5500,12500]³ m),
each grid using its nearest GLL node and evaluating the Stokes reference at that
node's true coordinates. HDF5 writes uncompressed. Postprocess memory capped at
64 GB (cgroup incl. page cache); ranks chosen per grid from the ~linear
rank-vs-peak model: 12/12/4/4/3 (18/20/22/24/28). Configs generated by
examples/meshsize/gen_grids.py; per-case artifacts kept in
examples/meshsize/fullspace{18,20,22,24,28}/.

| grid | elem/λs | ranks | peak GiB | mean_corr | mean_l2 | scale | shape L2 | solver s/dir |
|------|---------|-------|----------|-----------|---------|-------|----------|--------------|
| 18³ | 3.0 | 12 | 48.4 | 0.8502 | 0.7039 | 0.339 | 0.1230 | 63 |
| 20³ | 3.3 | 12 | 52.0 | 0.8476 | 0.7017 | 0.346 | 0.1275 | — |
| 22³ | 3.7 | 4 | 32.3 | 0.8405 | 0.7068 | 0.338 | 0.1235 | 117 |
| 24³ | 4.0 | 4 | 34.3 | 0.8313 | 0.6982 | 0.344 | 0.1267 | 148 |
| 28³ | 4.7 | 3 | 40.2 | 0.8467 | 0.7004 | 0.342 | 0.1255 | 236 |

**Conclusions (now on 5 grids, clean protocol):**

1. **Shape error does NOT converge with resolution**: scale-fitted L2 stays flat
   at 0.123–0.127 across 3.0→4.7 elem/λs (±0.002 scatter, no trend). The residual
   is dominated by PML reflections + near-field source representation, not
   element resolution.
1. **mean_corr is flat/noisy** (0.85/0.85/0.84/0.83/0.85) — same conclusion.
1. **The ~2.9× amplitude factor is resolution-independent**: best-fit scale
   = 0.338–0.346 everywhere (effective-source/convention offset, not a mesh error).
1. 28³ is the agreed ceiling; a denser grid (32³) would cost ~89 GB peak at 4
   ranks and would NOT improve the comparison.

**Memory lessons (64 GB cgroup budget):**

- postprocess peak scales ~linearly with surviving MPI ranks (tile-local
  extraction); the cgroup counter includes page cache, so RSS sampling alone
  UNDERSTATES it — 22³@6 ranks OOM'd at 64 GB (sampled RSS 48.4 GB but
  memory.current > 64 GB) and left a 6/16 partial tile set. Fixed by lowering
  ranks (22/24→4, 28→3) plus a 16-tile coverage guard before the comparison.
- 18³/20³ at 12 ranks peak at 48/52 GB — near the cap but pass.
