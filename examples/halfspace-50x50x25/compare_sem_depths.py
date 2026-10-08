"""Compare archived buried-source SEM displacement against strict-surface SEM."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from compare_grid import _metrics  # noqa: E402

from greenfun.library import GreenFunctionLibrary  # noqa: E402


def read_config(directory):
    """Load the saved simulation contract and actual source, not environment defaults."""
    with h5py.File(directory / "config.h5", "r") as stream:
        source = stream["source"].attrs
        return (
            np.array([source[axis] for axis in ("x", "y", "z")]),
            {group: dict(stream[group].attrs) for group in ("domain", "simulation")},
            stream["source/stf_t"][...],
            stream["source/stf_values"][...],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 10, 50, 100, 200, 500])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/test-reports/data/halfspace-sem-source-depth.csv",
    )
    args = parser.parse_args()
    if any(depth <= 0 for depth in args.depths):
        parser.error("Buried source depths must be positive")

    # Cache only the five surface receiver traces, not whole tile arrays.
    archives = CASE / "depth_sweep"
    surface_directory = archives / "depth_0000m"
    surface_source, contract, stf_time, stf_values = read_config(surface_directory)
    if surface_source[2] != 0.0:
        raise ValueError("Baseline is not a strict-surface source")
    surface_library = GreenFunctionLibrary(surface_directory / "greenfun")
    surface_results = []
    for offset in config.validation_offsets_m:
        receiver = surface_source + [offset, 0.0, 0.0]
        surface_results.append(
            (receiver, surface_library.query(receiver, surface_source, quantity="displacement"))
        )
    rows = []
    for depth in args.depths:
        directory = archives / f"depth_{depth:04d}m"
        source, saved_contract, saved_time, saved_values = read_config(directory)
        if saved_contract != contract:
            raise ValueError(f"Simulation/domain mismatch for depth {depth}")
        np.testing.assert_array_equal(source, surface_source + [0.0, 0.0, depth])
        np.testing.assert_array_equal(saved_time, stf_time)
        np.testing.assert_array_equal(saved_values, stf_values)
        library = GreenFunctionLibrary(directory / "greenfun")

        # Compare unshifted, unnormalized tensors in identical pre-return windows.
        for offset, end_time, (receiver, surface) in zip(
            config.validation_offsets_m,
            config.validation_end_times_s,
            surface_results,
            strict=True,
        ):
            buried = library.query(receiver, source, quantity="displacement")
            np.testing.assert_array_equal(buried.time, surface.time)
            if buried.displacement is None or surface.displacement is None:
                raise ValueError("Displacement is missing")
            if not all(
                np.isfinite(values).all() for values in (buried.displacement, surface.displacement)
            ):
                raise ValueError("Non-finite displacement")
            for direction, component in (("all", slice(None)), ("x", 0), ("y", 1), ("z", 2)):
                baseline = np.asarray(surface.displacement[:, :, component], dtype=np.float64)
                candidate = np.asarray(buried.displacement[:, :, component], dtype=np.float64)
                correlation, scale, fitted = _metrics(surface.time, baseline, candidate, end_time)
                mask = surface.time < end_time
                raw = np.linalg.norm(candidate[mask] - baseline[mask]) / np.linalg.norm(
                    baseline[mask]
                )
                rows.append(
                    dict(
                        source_depth_m=depth,
                        offset_m=offset,
                        receiver_depth_m=0.0,
                        end_time_s=end_time,
                        force_direction=direction,
                        correlation=correlation,
                        scale=scale,
                        raw_relative_l2=float(raw),
                        fitted_relative_l2=fitted,
                    )
                )
                if direction == "all":
                    print(
                        f"depth={depth}m offset={offset:g}m raw={raw:.8%} fitted={fitted:.8%} scale={scale:.9f} corr={correlation:.9f}",
                        flush=True,
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
