# forward/viscoelastic/ — AGENTS.md

## Purpose

Viscoelastic CG-SEM solver with SLS attenuation. Extends the elastic solver
with standard linear solid (SLS) memory variables for frequency-independent Q
attenuation.

**Status: COMPLETE.** All 9 implementation tasks done (7 commits, Jul 2026).
End-to-end verified: 16 MPI ranks, 1000 steps, 0 crashes.

## Architecture

Follows the structure of `forward/elastic/` but with the SLS viscoelastic
element kernel instead of the elastic one.

- **Infrastructure**: `libgf_shared` from `forward/share/` (solver, assembly,
  I/O, exchange, Newmark, C-PML, Newmark, restart, recording)
- **Element kernel**: `src/element_cpu.cpp` — calls 5 shared `kernel_helpers`
  functions, computes elastic trial stress, subtracts SLS memory stress R_l,
  and updates R = a·R + b·Δσ inline
- **CUDA kernel**: `src/element_cuda.cu` — same logic as CPU, `__device__`
  helpers from `kernel_helpers.cuh`, `atomicAdd` for residual scatter
- **Solver**: `src/solver.cpp` — thin wrapper: `run_viscoelastic_forward()`
  delegates to the shared `run_forward()` (from `forward/share/src/solver.cpp`).
  The viscoelastic behaviour comes from linking `libgf_visco` instead of `libgf`.

## Memory Layout

- `rmemory_sls`: [n_total_nodes × N_SLS × 6] doubles per rank — SLS memory
  stress tensors R_l^v (Voigt components)
- `sigma_old`: [n_total_nodes × 6] doubles per rank — previous-step elastic
  stress for Δσ computation
- `sls_coef_a/b`: [n_total_nodes × N_SLS] doubles — precomputed coefficients

## Governing Equations

```
sigma_ij(t) = sigma_ij^elastic(t) - sum_{l=1}^{N_SLS} R_ij^l(t)
```

where:

```
R_ij^l(t+dt) = a_l * R_ij^l(t) + b_l * (sigma_ij(t+dt) - sigma_ij(t))
a_l = exp(-dt / tau_sigma^l)
b_l = (tau_epsilon^l / tau_sigma^l - 1) * (1 - a_l)
```

## Build

```bash
cd forward
cmake -B build -DGF_DEVICE_BACKEND=CPU
cmake --build build
# Produces: gf_solver_viscoelastic_mpi, gf_solver_viscoelastic_cuda,
#           gf_solver_viscoelastic_mpi_cuda
```

## Solver Executables

| Executable | Backend | MPI |
|---|---|---|
| `gf_solver_viscoelastic_mpi` | CPU | Yes |
| `gf_solver_viscoelastic_cuda` | CUDA | No |
| `gf_solver_viscoelastic_mpi_cuda` | CUDA | Yes |

## Related Documents

- Design spec: `docs/superpowers/specs/2026-07-21-sls-viscoelastic-design.md`
- Implementation plan: `docs/superpowers/plans/2026-07-21-sls-viscoelastic.md`
- SLS preprocessor: `preprocess/attenuation.py`
- SLS namespace + helpers: `forward/share/include/gf/attenuation.hpp`
- Shared kernel helpers: `forward/share/include/gf/kernel_helpers.hpp`/.cuh
