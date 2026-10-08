"""Check Lamb quadrature and time sampling against retained depth-sweep tiles."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))

from examples.halfspace import reference  # noqa: E402
from examples.halfspace.reference_sampling import (  # noqa: E402
    integrate_batched,
    sampled_displacement,
)
from greenfun.library import GreenFunctionLibrary  # noqa: E402

SCALAR_INTEGRATE = reference._integrate_segments


def validate_batch():
    """Compare each integral with the scalar implementation before using it."""
    nodes, weights = reference._gauss_quad(24)
    maximum_error = 0.0
    for depth_m in (1.0, 10.0, 500.0):
        radius = np.hypot(14000.0, depth_m)
        theta = np.arctan2(14000.0, depth_m)
        for time_s in (3.25, 4.75, 5.10):
            args = (theta, 0.0, radius, 5000.0, 3000.0, 2700.0, time_s)
            for integrand, speed in (
                (reference._integrand_p, 5000.0),
                (reference._integrand_s_cagniard, 3000.0),
                (reference._integrand_s_branch, 3000.0),
            ):
                radicand = (time_s / radius) ** 2 - 1.0 / speed**2
                if integrand is reference._integrand_s_branch:
                    lower = np.sqrt(max(radicand, 0.0))
                    numerator = time_s / radius - np.sqrt(
                        1.0 / 3000.0**2 - 1.0 / 5000.0**2
                    ) * np.cos(theta)
                    upper = np.sqrt(max((numerator / np.sin(theta)) ** 2 - 1.0 / 5000.0**2, 0.0))
                else:
                    lower, upper = 0.0, np.sqrt(max(radicand, 0.0))
                scalar = SCALAR_INTEGRATE(integrand, lower, upper, nodes, weights, args, 40)
                batched = integrate_batched(integrand, lower, upper, nodes, weights, args, 40)
                error = np.linalg.norm(batched - scalar) / max(np.linalg.norm(scalar), 1e-300)
                maximum_error = max(maximum_error, error)
                if error > 1e-9:
                    raise ValueError(f"Batch mismatch {integrand.__name__}: {error}")
    print(f"BATCH_VALIDATION max_relative_error={maximum_error:.3e}", flush=True)


def metrics(sem, analytical):
    sem = sem.ravel()
    analytical = analytical.ravel()
    reference_norm = np.linalg.norm(analytical)
    sem_norm = np.linalg.norm(sem)
    if min(reference_norm, sem_norm) < 1e-100:
        return dict(
            correlation=float("nan"),
            scale=float("nan"),
            fitted_l2=float("nan"),
            raw_l2=float("nan"),
            sem_norm=sem_norm,
            reference_norm=reference_norm,
        )
    scale = np.dot(sem, analytical) / reference_norm**2
    fitted = scale * analytical
    return dict(
        correlation=np.corrcoef(sem, analytical)[0, 1],
        scale=scale,
        fitted_l2=np.linalg.norm(sem - fitted) / max(sem_norm, np.linalg.norm(fitted)),
        raw_l2=np.linalg.norm(sem - analytical) / reference_norm,
        sem_norm=sem_norm,
        reference_norm=reference_norm,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--depths", nargs="+", type=int, default=[500, 100, 50, 10])
    parser.add_argument("--offsets", nargs="+", type=int, default=[5000, 14000])
    parser.add_argument("--time-refinements", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--fine-points", type=int, default=48)
    parser.add_argument("--fine-segments", type=int, default=1600)
    args = parser.parse_args()
    validate_batch()
    if args.validate_only:
        return
    output = CASE / "depth_sweep/reference_convergence"
    output.mkdir(exist_ok=True)
    rows = []
    levels = [(24, 800, 1), (48, 800, 1), (24, 1600, 1), (48, 1600, 1)]
    levels += [
        (args.fine_points, args.fine_segments, refinement) for refinement in args.time_refinements
    ]
    end_times = {2000: 3.5, 5000: 4.5, 8000: 5.5, 11000: 6.5, 14000: 7.3}
    for depth_m in args.depths:
        saved = CASE / "depth_sweep" / f"depth_{depth_m:04d}m"
        library = GreenFunctionLibrary(saved / "greenfun")
        with h5py.File(saved / "config.h5") as config_file:
            source = config_file["source"].attrs
            source_xyz = np.array([source["x"], source["y"], source["z"]])
            stf_time = config_file["source/stf_t"][...]
            stf_values = config_file["source/stf_values"][...]
        if source_xyz[2] != depth_m:
            raise ValueError("Saved source depth mismatch")
        for offset_m in args.offsets:
            receiver = source_xyz + [offset_m, 0.0, -depth_m]
            result = library.query(receiver, source_xyz, quantity="displacement")
            time_s = np.asarray(result.time)
            sem = np.asarray(result.displacement)
            if not np.isclose(time_s[0], 0.0):
                raise ValueError("Expected zero-based output times")
            # Include the saved CLI resolution when validating newer, refined runs.
            point_levels = list(levels)
            metrics_path = saved / "metrics.csv"
            if metrics_path.exists():
                with metrics_path.open() as stream:
                    published = next(
                        (
                            row
                            for row in csv.DictReader(stream)
                            if float(row["offset_m"]) == offset_m
                        ),
                        {},
                    )
                if published.get("reference_dt_s"):
                    saved_refinement = round(
                        float(time_s[1] - time_s[0]) / float(published["reference_dt_s"])
                    )
                    saved_level = (48, 1600, saved_refinement)
                    if saved_level not in point_levels:
                        point_levels.append(saved_level)
            baseline = None
            for points, segments, refinement in point_levels:
                started = time.monotonic()
                cache = (
                    output
                    / f"depth{depth_m}_offset{offset_m}_q{points}_s{segments}_t{refinement}.npz"
                )
                if cache.exists():
                    with np.load(cache) as data:
                        analytical = data["analytical"]
                else:
                    analytical = sampled_displacement(
                        5000.0,
                        3000.0,
                        2700.0,
                        source_xyz,
                        receiver,
                        time_s,
                        stf_time,
                        stf_values,
                        refinement,
                        n_quad_pts=points,
                        n_seg=segments,
                    )
                    if not np.isfinite(analytical).all():
                        raise ValueError(f"Nonfinite reference: {cache}")
                    np.savez(cache, time=time_s, analytical=analytical, sem=sem)
                if baseline is None:
                    baseline = analytical
                mask = time_s < end_times[offset_m]
                row = dict(
                    source_depth_m=depth_m,
                    offset_m=offset_m,
                    quadrature_points=points,
                    segments=segments,
                    time_refinement=refinement,
                    **metrics(sem[mask], analytical[mask]),
                    reference_change=np.linalg.norm((analytical - baseline)[mask])
                    / np.linalg.norm(baseline[mask]),
                )
                rows.append(row)
                with (output / "metrics.csv").open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"RESULT {row} elapsed={time.monotonic() - started:.1f}s", flush=True)
    print("CONVERGENCE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
