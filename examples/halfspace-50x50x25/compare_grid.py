#!/usr/bin/env python3
"""Compare the large half-space run with Lamb solutions over distance and depth."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np

CASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CASE_DIR.parents[1]
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from examples.halfspace.reference import (  # noqa: E402
    compute_step_response,
    convolve_with_stf,
)
from greenfun.library import GreenFunctionLibrary  # noqa: E402


def _metrics(
    time_s: np.ndarray, reference: np.ndarray, sem: np.ndarray, end_time_s: float
) -> tuple[float, float, float]:
    mask = time_s < end_time_s
    reference_samples = reference[mask].reshape(-1)
    sem_samples = sem[mask].reshape(-1)
    reference_norm_squared = float(np.dot(reference_samples, reference_samples))
    if reference_norm_squared < 1.0e-300:
        raise ValueError(f"reference norm is zero before {end_time_s:g} s")
    scale = float(np.dot(sem_samples, reference_samples) / reference_norm_squared)
    fitted_reference = scale * reference_samples
    fitted_relative_l2 = float(
        np.linalg.norm(sem_samples - fitted_reference)
        / max(np.linalg.norm(sem_samples), np.linalg.norm(fitted_reference), 1.0e-300)
    )
    correlation = float(np.corrcoef(sem_samples, reference_samples)[0, 1])
    return correlation, scale, fitted_relative_l2


def _earliest_pml_return_s(query_xyz_m: np.ndarray) -> float:
    element_x_m = config.lx / config.nx_elements
    element_y_m = config.ly / config.ny_elements
    element_z_m = config.lz / config.nz_elements
    source_depth_m = 0.0 if config.source_z_m is None else config.source_z_m
    source_xyz_m = np.array(
        [config.source_x_m, config.source_y_m, source_depth_m], dtype=np.float64
    )
    pml_planes = (
        (0, config.pml_thickness["xmin"] * element_x_m),
        (0, config.lx - config.pml_thickness["xmax"] * element_x_m),
        (1, config.pml_thickness["ymin"] * element_y_m),
        (1, config.ly - config.pml_thickness["ymax"] * element_y_m),
        (2, config.lz - config.pml_thickness["zmax"] * element_z_m),
    )
    path_lengths_m = []
    for axis, plane_m in pml_planes:
        mirrored_source = source_xyz_m.copy()
        mirrored_source[axis] = 2.0 * plane_m - mirrored_source[axis]
        path_lengths_m.append(float(np.linalg.norm(query_xyz_m - mirrored_source)))
    return min(path_lengths_m) / float(config.vp_m_s(0.0, 0.0, 0.0))


def main() -> int:
    library_root = CASE_DIR / "greenfun"
    library = GreenFunctionLibrary(library_root, rebuild_index=True)
    source_depth_m = 0.0 if config.source_z_m is None else config.source_z_m
    sem_source_xyz_m = np.array(
        [config.source_x_m, config.source_y_m, source_depth_m], dtype=np.float64
    )
    rows: list[dict[str, float]] = []
    with h5py.File(CASE_DIR / "config.h5", "r") as config_file:
        source_attributes = config_file["source"].attrs
        preprocessed_source_xyz_m = np.array(
            [source_attributes["x"], source_attributes["y"], source_attributes["z"]]
        )
        if not np.allclose(preprocessed_source_xyz_m, sem_source_xyz_m):
            raise ValueError(
                f"preprocessed source {preprocessed_source_xyz_m} does not match "
                f"config.py {sem_source_xyz_m}"
            )
        stf_time_s = np.asarray(config_file["source/stf_t"], dtype=np.float64)
        stf_values = np.asarray(config_file["source/stf_values"], dtype=np.float64)

    for offset_m, end_time_s in zip(
        config.validation_offsets_m, config.validation_end_times_s, strict=True
    ):
        for depth_m in config.validation_depths_m:
            query_xyz_m = np.array(
                [config.source_x_m + offset_m, config.source_y_m, depth_m], dtype=np.float64
            )
            sem_result = library.query(query_xyz_m, sem_source_xyz_m, quantity="displacement")
            if sem_result.displacement is None:
                raise ValueError(f"missing displacement at {query_xyz_m}")

            time_s = np.asarray(sem_result.time, dtype=np.float64)
            if sem_source_xyz_m[2] <= 0.0 or query_xyz_m[2] != 0.0:
                raise ValueError("Johnson reference requires a buried source and surface receiver")
            step_response = compute_step_response(
                vp=float(config.vp_m_s(*query_xyz_m)),
                vs=float(config.vs_m_s(*query_xyz_m)),
                density=float(config.density_kg_m3(*query_xyz_m)),
                source_xyz=sem_source_xyz_m,
                receiver_xyz=query_xyz_m,
                time=time_s,
                n_quad_pts=24,
                n_seg=800,
            )
            source_values = np.interp(time_s, stf_time_s, stf_values)
            reference_displacement = convolve_with_stf(
                step_response, source_values, float(config.output_dt_s)
            )
            correlation, scale, fitted_relative_l2 = _metrics(
                time_s, reference_displacement, np.asarray(sem_result.displacement), end_time_s
            )
            row = {
                "offset_m": offset_m,
                "depth_m": depth_m,
                "end_time_s": end_time_s,
                "earliest_pml_return_s": _earliest_pml_return_s(query_xyz_m),
                "correlation": correlation,
                "scale": scale,
                "fitted_relative_l2": fitted_relative_l2,
            }
            rows.append(row)
            print(
                f"offset={offset_m / 1000:g} km depth={depth_m / 1000:g} km "
                f"corr={correlation:.6f} scale={scale:.6f} "
                f"fitted_l2={fitted_relative_l2:.6f}"
            )

    output_path = CASE_DIR / "log/lamb_distance_depth.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path}")

    baseline = next(row for row in rows if row["offset_m"] == 11000.0 and row["depth_m"] == 0.0)
    passed = (
        baseline["correlation"] >= 0.95
        and baseline["fitted_relative_l2"] <= 0.30
        and 0.8 <= baseline["scale"] <= 1.2
    )
    print(f"11 km surface baseline: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
