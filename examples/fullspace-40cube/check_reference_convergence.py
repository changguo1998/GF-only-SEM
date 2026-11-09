#!/usr/bin/env python3
"""Check time convergence of the full-space Stokes reference solution."""

from __future__ import annotations

import argparse
import csv
import os
import sys

import h5py
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SHARED_DIR = os.path.join(PROJECT_ROOT, "examples", "_shared")
sys.path.insert(0, SHARED_DIR)

from analytical_compare import (  # noqa: E402
    find_artifacts,
    interior_bounds_from_config,
    load_sem_parameters,
    load_sem_tiles,
)
from analytical_green import stokes_displacement_green_tensor  # noqa: E402
from stokes_reference import (  # noqa: E402
    relative_reference_change,
    sampled_stokes_displacement,
    stokes_displacement_piecewise_linear,
)

REFINEMENTS = (1, 2, 4, 8, 16, 32, 64, 128)
N_RECEIVERS = 15


def _dominant_frequency(source_values: np.ndarray, solver_dt_s: float) -> float:
    centered = source_values - np.mean(source_values)
    frequencies = np.fft.rfftfreq(len(source_values), solver_dt_s)
    spectrum = np.abs(np.fft.rfft(centered))
    return float(frequencies[1:][np.argmax(spectrum[1:])])


def _select_receivers(
    coordinates: np.ndarray,
    source_xyz: np.ndarray,
    shear_wavelength_m: float,
    config_path: str,
    model_path: str,
) -> np.ndarray:
    bounds = interior_bounds_from_config(config_path, model_path)
    inside = np.all((coordinates >= bounds[:, 0]) & (coordinates <= bounds[:, 1]), axis=1)
    distances = np.linalg.norm(coordinates - source_xyz, axis=1)
    candidates = np.flatnonzero(inside & (distances >= 2.0 * shear_wavelength_m))
    if len(candidates) < N_RECEIVERS:
        raise RuntimeError(f"only {len(candidates)} valid interior receivers found")
    candidate_order = candidates[np.argsort(distances[candidates])]
    selected_positions = np.linspace(0, len(candidate_order) - 1, N_RECEIVERS, dtype=int)
    return candidate_order[selected_positions]


def _load_output_times(greenfun_dir: str, n_steps: int, dt_s: float) -> np.ndarray:
    tile_paths = sorted(
        os.path.join(greenfun_dir, name)
        for name in os.listdir(greenfun_dir)
        if name.startswith("tile_") and name.endswith(".h5")
    )
    if not tile_paths:
        raise FileNotFoundError(f"no tile_*.h5 files in {greenfun_dir}")
    with h5py.File(tile_paths[0], "r") as tile_file:
        times = np.asarray(tile_file["/time/t"], dtype=np.float64)
    if len(times) != n_steps:
        raise ValueError(f"tile time count {len(times)} != displacement frame count {n_steps}")
    if not np.allclose(times, np.arange(n_steps) * dt_s, rtol=1e-9, atol=1e-12):
        raise ValueError("tile time grid is inconsistent with its recorded timestep")
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "greenfun_dir", nargs="?", default=os.path.join(os.path.dirname(__file__), "greenfun")
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "reference_convergence.csv"),
        help="CSV path for convergence metrics",
    )
    args = parser.parse_args()
    greenfun_dir = os.path.abspath(args.greenfun_dir)

    try:
        sem = load_sem_tiles(greenfun_dir)
        config_path, model_path = find_artifacts(greenfun_dir, None, None)
        parameters = load_sem_parameters(config_path, model_path, sem["n_steps"])
        output_time = _load_output_times(greenfun_dir, sem["n_steps"], sem["dt_s"])
        dominant_frequency_hz = _dominant_frequency(
            parameters["stf_values"], parameters["solver_dt_s"]
        )
        receiver_indices = _select_receivers(
            sem["coords"],
            parameters["source_xyz"],
            parameters["vs_m_s"] / dominant_frequency_hz,
            config_path,
            model_path,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1

    reference_by_refinement: dict[int, np.ndarray] = {}
    for refinement in REFINEMENTS:
        receiver_values = []
        for receiver_index in receiver_indices:
            tensor = sampled_stokes_displacement(
                sem["coords"][receiver_index],
                parameters["source_xyz"],
                parameters["vp_m_s"],
                parameters["vs_m_s"],
                parameters["density_kg_m3"],
                output_time,
                parameters["stf_time"],
                parameters["stf_values"],
                refinement,
            )
            receiver_values.append(tensor)
        reference_by_refinement[refinement] = np.stack(receiver_values, axis=0)

    exact_reference = []
    legacy_reference = []
    output_stf = parameters["output_stf_values"]
    for receiver_index in receiver_indices:
        exact_tensor = stokes_displacement_piecewise_linear(
            sem["coords"][receiver_index],
            parameters["source_xyz"],
            parameters["vp_m_s"],
            parameters["vs_m_s"],
            parameters["density_kg_m3"],
            output_time,
            parameters["stf_time"],
            parameters["stf_values"],
        )
        legacy_tensor = np.stack(
            [
                stokes_displacement_green_tensor(
                    sem["coords"][receiver_index],
                    parameters["source_xyz"],
                    force_direction,
                    parameters["vp_m_s"],
                    parameters["vs_m_s"],
                    parameters["density_kg_m3"],
                    output_stf,
                    sem["dt_s"],
                )
                for force_direction in range(3)
            ],
            axis=-1,
        )
        exact_reference.append(exact_tensor)
        legacy_reference.append(legacy_tensor)

    exact_array = np.stack(exact_reference, axis=0)
    legacy_array = np.stack(legacy_reference, axis=0)
    output_path = os.path.abspath(args.output)
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "refinement",
                "reference_dt_s",
                "max_relative_change",
                "p95_relative_change",
                "max_exact_relative_error",
                "legacy_relative_error",
            ],
        )
        writer.writeheader()
        previous = None
        final_change = None
        final_exact_error = None
        for refinement in REFINEMENTS:
            current = reference_by_refinement[refinement]
            changes = []
            if previous is not None:
                changes = [
                    relative_reference_change(current[index], previous[index])
                    for index in range(len(current))
                ]
            exact_errors = [
                relative_reference_change(current[index], exact_array[index])
                for index in range(len(current))
            ]
            legacy_errors = [
                relative_reference_change(legacy_array[index], exact_array[index])
                for index in range(len(current))
            ]
            writer.writerow(
                {
                    "refinement": refinement,
                    "reference_dt_s": sem["dt_s"] / refinement,
                    "max_relative_change": max(changes, default=0.0),
                    "p95_relative_change": float(np.percentile(changes, 95)) if changes else 0.0,
                    "max_exact_relative_error": max(exact_errors),
                    "legacy_relative_error": max(legacy_errors),
                }
            )
            if previous is not None:
                final_change = max(changes)
            if refinement == REFINEMENTS[-1]:
                final_exact_error = max(exact_errors)
            previous = current

    print(f"Wrote {output_path}")
    print(f"Receivers: {len(receiver_indices)}, refinements: {len(REFINEMENTS)}")
    assert final_change is not None and final_exact_error is not None
    print(f"final relative change: {final_change:.6g}")
    print(f"final exact relative error: {final_exact_error:.6g}")
    if final_change > 0.002 or final_exact_error > 0.0005:
        print("FAILED: full-space reference has not met convergence gates")
        return 1
    print("PASSED: full-space reference time convergence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
