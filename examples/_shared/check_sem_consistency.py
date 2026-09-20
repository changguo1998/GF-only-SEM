#!/usr/bin/env python3
"""Check SEM-internal parameter consistency: config.py vs config.h5 vs model.h5.

Runs AFTER preprocessing and BEFORE the solver, verifying that the Python
config (the project's single source of truth) exactly matches what the
preprocessor materialized into config.h5 / model.h5:

  * STF:        config.stf_func(t) matched against config.h5:/source/stf_values
                (peak must equal config.source_force_amplitude_n)
  * source:     config.source_{x,y,z}_m                    ==  config.h5:/source/{x,y,z}
  * timing:     config.output_dt_s / total_duration_s      ==  config.h5:/simulation/*
  * material:   config.vp/vs/density callables (at domain center, since the
                fullspace model is homogeneous)            ==  model.h5:/field/element/*
  * stride:     solver_dt * snapshot_stride == output_dt_s (exact-divisibility)

This catches the class of bug documented in VERIFICATION.md ("C++ STF
parameter mismatch"): Python config changes that were not propagated to
the compiled user config, which silently diverge.

Usage:
    python3 check_sem_consistency.py <case_dir>
"""

import importlib.util
import math
import os
import sys

import numpy as np
import h5py

RTOL = 1e-9
ATOL_STF = 1e-6  # absolute tolerance scaled to the ~1e20 amplitude


def _to_float(value, name: str) -> float:
    """Convert a value to float, giving a descriptive error on invalid input."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot parse {name} as float: {value!r}") from exc


def _to_int(value, name: str) -> int:
    """Convert a value to int, giving a descriptive error on invalid input."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot parse {name} as int: {value!r}") from exc


def _attr_float(f: h5py.File, group: str, key: str) -> float:
    return _to_float(np.asarray(f[group].attrs[key]), f"{group}.{key}")


def _rtol_match(name: str, expected: float, actual: float, rtol: float = RTOL) -> bool:
    ok = bool(np.isclose(expected, actual, rtol=rtol, atol=1e-12 * max(abs(expected), 1.0)))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: config={expected:.10g}, artifact={actual:.10g}")
    return ok


def _load_config(case_dir: str):
    config_path = os.path.join(case_dir, "config.py")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.py not found: {config_path}")
    spec = importlib.util.spec_from_file_location("sem_case_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load config module: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_sem_consistency.py <case_dir>")
        return 2
    case_dir = os.path.abspath(sys.argv[1])
    config = _load_config(case_dir)
    config_h5 = os.path.join(case_dir, "config.h5")
    model_h5 = os.path.join(case_dir, "model.h5")

    print(f"Checking SEM-internal consistency in {case_dir} ...")
    results = []

    with h5py.File(config_h5, "r") as f:
        stf_time = np.asarray(f["source/stf_t"], dtype=np.float64)
        stf_values = np.asarray(f["source/stf_values"], dtype=np.float64)
        src = {k: _attr_float(f, "source", k) for k in ("x", "y", "z")}
        sim = {
            k: _attr_float(f, "simulation", k)
            for k in ("output_dt_s", "solver_dt", "snapshot_stride")
        }

    nsteps_stored = len(stf_values)

    # Source location
    results.append(_rtol_match("source.x_m", config.source_x_m, src["x"]))
    results.append(_rtol_match("source.y_m", config.source_y_m, src["y"]))
    results.append(_rtol_match("source.z_m", config.source_z_m, src["z"]))

    # Time stepping
    results.append(_rtol_match("output_dt_s", config.output_dt_s, sim["output_dt_s"]))
    try:
        expected_nsteps = math.ceil(config.total_duration_s / sim["solver_dt"])
    except ZeroDivisionError as exc:
        raise ValueError("config.h5 simulation.solver_dt is zero") from exc
    ok = expected_nsteps == nsteps_stored
    results.append(ok)
    print(
        f"  [{'PASS' if ok else 'FAIL'}] nsteps: config-duration={expected_nsteps}, "
        f"artifact={nsteps_stored}"
    )

    # STF: config.stf_func sampled on the artifact grid == artifact stf_values.
    stf_expected = np.asarray(
        [config.stf_func(_to_float(ti, "stf time")) for ti in stf_time], dtype=np.float64
    )
    max_diff = _to_float(np.max(np.abs(stf_expected - stf_values)), "stf max diff")
    scale = _to_float(max(1.0, np.max(np.abs(stf_values))), "stf scale")
    ok = max_diff <= RTOL * scale + ATOL_STF
    results.append(ok)
    print(
        f"  [{'PASS' if ok else 'FAIL'}] stf_func vs /source/stf_values: "
        f"max_abs_diff={max_diff:.4g} (tol={RTOL * scale + ATOL_STF:.4g})"
    )

    # STF peak (Ricker at t0) == config.source_force_amplitude_n
    # (t0 rarely lands on the output grid, so the sampled peak is up to ~5e-4
    # below A for f0=0.75/dt=0.01 — use rtol=1e-3)
    stf_peak = _to_float(np.max(np.abs(stf_values)), "stf peak")
    results.append(
        _rtol_match(
            "stf_peak == force_amplitude", config.source_force_amplitude_n, stf_peak, rtol=1e-3
        )
    )

    # solver_dt * snapshot_stride == output_dt_s (exact divisibility)
    stride = _to_int(sim["snapshot_stride"], "snapshot_stride")
    ok = bool(np.isclose(sim["solver_dt"] * stride, sim["output_dt_s"], rtol=1e-12, atol=0.0))
    results.append(ok)
    print(
        f"  [{'PASS' if ok else 'FAIL'}] stride divisibility: "
        f"solver_dt={sim['solver_dt']:.6e} x {stride} == output_dt={sim['output_dt_s']}"
    )

    # Material (homogeneous fullspace: evaluate callables at domain center)
    with h5py.File(model_h5, "r") as f:

        def _median(path: str) -> float:
            return _to_float(
                np.median(np.asarray(f[path], dtype=np.float64).ravel()), "model material"
            )

        model_vp = _median("field/cell/vp")
        model_vs = _median("field/cell/vs")
        model_rho = _median("field/cell/density")

    center = (
        _to_float(config.lx, "lx") / 2.0,
        _to_float(config.ly, "ly") / 2.0,
        _to_float(config.lz, "lz") / 2.0,
    )
    results.append(_rtol_match("vp_m_s", _to_float(config.vp_m_s(*center), "vp"), model_vp))
    results.append(_rtol_match("vs_m_s", _to_float(config.vs_m_s(*center), "vs"), model_vs))
    results.append(
        _rtol_match(
            "density_kg_m3", _to_float(config.density_kg_m3(*center), "density"), model_rho
        )
    )

    passed = all(results)
    print(f"\nSEM consistency: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
