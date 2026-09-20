#!/usr/bin/env python3
"""Apply numerical acceptance criteria to a saved SEM/reference comparison.

The reported scale uses the project-wide ``SEM = scale * reference`` convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path, help="NPZ written by examples/*/compare.py")
    parser.add_argument(
        "--end-time-s", type=float, required=True, help="Exclusive time-window end"
    )
    parser.add_argument("--min-correlation", type=float, required=True)
    parser.add_argument("--max-fitted-rel-l2", type=float, required=True)
    parser.add_argument("--min-scale", type=float, required=True)
    parser.add_argument("--max-scale", type=float, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with np.load(args.comparison) as comparison:
        time = np.asarray(comparison["time"], dtype=np.float64)
        reference = np.asarray(comparison["reference_displacement"], dtype=np.float64)
        sem = np.asarray(comparison["sem_displacement"], dtype=np.float64)

    time_mask = time < args.end_time_s
    if not np.any(time_mask):
        raise ValueError(f"no samples before end_time_s={args.end_time_s}")

    reference_samples = reference[time_mask].reshape(-1)
    sem_samples = sem[time_mask].reshape(-1)
    reference_norm_squared = float(np.dot(reference_samples, reference_samples))
    if reference_norm_squared < 1.0e-300:
        raise ValueError("reference waveform has zero norm in the selected time window")

    scale = float(np.dot(sem_samples, reference_samples) / reference_norm_squared)
    fitted_reference = scale * reference_samples
    fitted_rel_l2 = float(
        np.linalg.norm(sem_samples - fitted_reference)
        / max(np.linalg.norm(sem_samples), np.linalg.norm(fitted_reference), 1.0e-300)
    )
    correlation = float(np.corrcoef(sem_samples, reference_samples)[0, 1])

    checks = [
        correlation >= args.min_correlation,
        fitted_rel_l2 <= args.max_fitted_rel_l2,
        args.min_scale <= scale <= args.max_scale,
    ]
    verdict = "PASS" if all(checks) else "FAIL"
    print(
        f"{verdict}: correlation={correlation:.6f} "
        f"(min={args.min_correlation:.6f}), fitted_rel_l2={fitted_rel_l2:.6f} "
        f"(max={args.max_fitted_rel_l2:.6f}), scale={scale:.6f} "
        f"(range=[{args.min_scale:.6f}, {args.max_scale:.6f}]), "
        f"window=[0, {args.end_time_s:g}) s"
    )
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
