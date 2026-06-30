"""Stage2 runner — wraps gf_preprocess_stage2 C++ executable.

Reads material arrays from model.h5 (written by Python after material interp),
computes λ/μ, solver_dt, nsteps, and pre-flight stats in C++, then writes
λ/μ back to model.h5 and prints stats to stdout.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess

logger = logging.getLogger("preprocess")


def _find_stage2_binary() -> str | None:
    """Locate gf_preprocess_stage2 binary."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)

    candidates = [
        os.path.join(project_root, "bin", "gf_preprocess_stage2"),
        os.path.join(this_dir, "cpp", "bin", "gf_preprocess_stage2"),
        os.path.join(this_dir, "cpp", "build", "gf_preprocess_stage2"),
        "gf_preprocess_stage2",  # PATH
    ]
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            logger.debug(f"Found stage2 binary: {cand}")
            return os.path.abspath(cand)
    return None


def run_stage2(model_path: str) -> dict[str, str]:
    """Run gf_preprocess_stage2, return parsed STAT_* values.

    Args:
        model_path: Path to model.h5 (must have /config/, /field/element/{vp,vs,density}).

    Returns:
        Dict of STAT_key → value string from C++ stdout.

    Raises:
        RuntimeError: If binary not found or fails.
    """
    binary = _find_stage2_binary()
    if binary is None:
        raise RuntimeError(
            "gf_preprocess_stage2 not found. Build with: cmake --build preprocess/cpp/build"
        )

    cmd = [binary, os.path.abspath(model_path)]
    logger.info(f"Running stage2: {' '.join(shlex.quote(str(x)) for x in cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise RuntimeError(f"stage2 binary not found: {binary}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("stage2 timed out after 600s")
    except Exception as e:
        raise RuntimeError(f"stage2 subprocess error: {e}")

    if proc.returncode != 0:
        raise RuntimeError(
            f"stage2 exited with code {proc.returncode}:\n  stderr: {proc.stderr[:500]}"
        )

    # Log stderr
    if proc.stderr.strip():
        for line in proc.stderr.strip().split("\n"):
            logger.debug(f"  [stage2] {line}")

    # Parse STAT_* from stdout
    stats: dict[str, str] = {}
    for line in proc.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("STAT_"):
            try:
                key, val = line.split("=", 1)
                stats[key] = val
            except ValueError:
                pass

    if not stats:
        raise RuntimeError(
            f"stage2 produced no STAT_* output:\n  stdout: {proc.stdout[:300]}"
        )

    n_cell = int(stats.get("STAT_NCELL", "0"))
    ngll = int(stats.get("STAT_NGLL", "0"))
    solver_dt = float(stats.get("STAT_SOLVER_DT", "0"))
    nsteps = int(stats.get("STAT_NSTEPS", "0"))
    snapshot_stride = int(stats.get("STAT_SNAPSHOT_STRIDE", "1"))
    logger.info(
        f"Stage2 done: n_cell={n_cell}, ngll={ngll}, "
        f"solver_dt={solver_dt:.6e}, nsteps={nsteps}, stride={snapshot_stride}"
    )

    return stats