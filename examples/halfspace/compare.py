#!/usr/bin/env python3
"""Compare Lamb reference output with the SEM GreenFunctionLibrary result."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Make the project root importable without setting PYTHONPATH externally.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = (_SCRIPT_DIR / "../..").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from greenfun.library import GreenFunctionLibrary  # noqa: E402

_COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2}
_FORCE_DIRECTIONS = ("x", "y", "z")


def _as_vector(values: list[float], name: str) -> npt.NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    return vector


def _load_reference(path: Path) -> dict[str, npt.NDArray[np.float64]]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _relative_l2_error(
    reference: npt.NDArray[np.float64], candidate: npt.NDArray[np.float64]
) -> float:
    denominator = float(np.linalg.norm(reference.ravel()))
    numerator = float(np.linalg.norm((candidate - reference).ravel()))
    if denominator <= 0.0:
        return numerator
    return numerator / denominator


def _normalized_relative_l2_error(
    reference: npt.NDArray[np.float64], candidate: npt.NDArray[np.float64]
) -> tuple[float, float]:
    denominator = float(np.dot(candidate.ravel(), candidate.ravel()))
    if denominator <= 0.0:
        scale = 0.0
        return _relative_l2_error(reference, candidate), scale
    scale = float(np.dot(reference.ravel(), candidate.ravel()) / denominator)
    return _relative_l2_error(reference, scale * candidate), scale


def _max_abs_error(
    reference: npt.NDArray[np.float64], candidate: npt.NDArray[np.float64]
) -> float:
    return float(np.max(np.abs(candidate - reference)))


def _validate_coordinates(
    reference: dict[str, npt.NDArray[np.float64]],
    source_xyz_m: npt.NDArray[np.float64],
    receiver_xyz_m: npt.NDArray[np.float64],
) -> None:
    if not np.allclose(reference["source_xyz_m"], source_xyz_m):
        raise ValueError(
            f"Reference source {reference['source_xyz_m']} does not match requested source {source_xyz_m}"
        )
    if not np.allclose(reference["receiver_xyz_m"], receiver_xyz_m):
        raise ValueError(
            "Reference receiver "
            f"{reference['receiver_xyz_m']} does not match requested receiver {receiver_xyz_m}"
        )


def _select_displacement_series(
    data: npt.NDArray[np.float64], component: str, force_direction: str
) -> npt.NDArray[np.float64]:
    return data[:, _COMPONENT_INDEX[component], _COMPONENT_INDEX[force_direction]]


def _write_scalar_csv(
    path: Path,
    time_s: npt.NDArray[np.float64],
    reference_series: npt.NDArray[np.float64],
    sem_series: npt.NDArray[np.float64],
    scaled_sem_series: npt.NDArray[np.float64] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = ["time_s", "reference", "sem", "difference"]
        if scaled_sem_series is not None:
            header += ["scaled_sem", "scaled_difference"]
        writer.writerow(header)
        for index, time_value in enumerate(time_s):
            row = [
                time_value,
                reference_series[index],
                sem_series[index],
                sem_series[index] - reference_series[index],
            ]
            if scaled_sem_series is not None:
                row += [
                    scaled_sem_series[index],
                    scaled_sem_series[index] - reference_series[index],
                ]
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Lamb reference output with SEM greenfun output."
    )
    parser.add_argument("library_root", type=Path, help="Path to greenfun library root")
    parser.add_argument("--source", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--receiver", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--quantity",
        choices=("displacement",),
        default="displacement",
        help="Only displacement is supported by the Lamb reference script",
    )
    parser.add_argument("--reference", type=Path, required=True, help="Reference .npz path")
    parser.add_argument("--output", type=Path, required=True, help="Comparison .npz output path")
    parser.add_argument(
        "--rebuild-index", action="store_true", help="Force rebuild of the greenfun index cache"
    )
    parser.add_argument(
        "--component",
        choices=tuple(_COMPONENT_INDEX),
        help="Optional displacement component for CSV export",
    )
    parser.add_argument(
        "--force-direction",
        choices=_FORCE_DIRECTIONS,
        default="z",
        help="Force direction for optional scalar CSV export",
    )
    parser.add_argument("--csv-output", type=Path, help="Optional scalar comparison CSV path")
    parser.add_argument(
        "--fit-scale",
        action="store_true",
        help="Also report best-fit scalar amplitude correction for SEM vs reference",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_xyz_m = _as_vector(args.source, "source")
    receiver_xyz_m = _as_vector(args.receiver, "receiver")
    reference = _load_reference(args.reference)
    _validate_coordinates(reference, source_xyz_m, receiver_xyz_m)

    library = GreenFunctionLibrary(args.library_root, rebuild_index=args.rebuild_index)
    sem_result = library.query(source_xyz_m, receiver_xyz_m, quantity="displacement")
    if sem_result.displacement is None:
        raise ValueError("GreenFunctionLibrary did not return displacement")
    if not np.allclose(reference["time"], sem_result.time):
        raise ValueError("Reference and SEM time axes do not match")

    reference_displacement = np.asarray(reference["displacement"], dtype=np.float64)
    sem_displacement = np.asarray(sem_result.displacement, dtype=np.float64)
    displacement_difference = sem_displacement - reference_displacement

    normalized_rel_l2 = None
    amplitude_scale = None
    scaled_sem_displacement = None
    if args.fit_scale:
        normalized_rel_l2, amplitude_scale = _normalized_relative_l2_error(
            reference_displacement, sem_displacement
        )
        scaled_sem_displacement = amplitude_scale * sem_displacement

    output: dict[str, npt.NDArray[np.float64]] = {
        "time": np.asarray(sem_result.time, dtype=np.float64),
        "source_xyz_m": np.asarray(sem_result.source_xyz, dtype=np.float64),
        "receiver_xyz_m": np.asarray(sem_result.receiver_xyz, dtype=np.float64),
        "sem_source_xyz_m": np.asarray(sem_result.sem_source_xyz, dtype=np.float64),
        "reference_displacement": reference_displacement,
        "sem_displacement": sem_displacement,
        "displacement_difference": displacement_difference,
    }
    if scaled_sem_displacement is not None:
        output["scaled_sem_displacement"] = scaled_sem_displacement
        output["amplitude_scale"] = np.array(amplitude_scale, dtype=np.float64)

    print(f"source_xyz_m     : {output['source_xyz_m']}")
    print(f"receiver_xyz_m   : {output['receiver_xyz_m']}")
    print(f"sem_source_xyz_m : {output['sem_source_xyz_m']}")
    print(f"interpolated SEM : {sem_result.interpolation_used}")
    print(
        "displacement rel_l2/max_abs: "
        f"{_relative_l2_error(reference_displacement, sem_displacement):.6e} / "
        f"{_max_abs_error(reference_displacement, sem_displacement):.6e}"
    )
    if normalized_rel_l2 is not None and amplitude_scale is not None:
        print(f"best-fit SEM scale          : {amplitude_scale:.6e}")
        print(f"scaled displacement rel_l2  : {normalized_rel_l2:.6e}")

    if args.csv_output is not None:
        if args.component is None:
            raise ValueError("--component is required when --csv-output is set")
        reference_series = _select_displacement_series(
            reference_displacement, args.component, args.force_direction
        )
        sem_series = _select_displacement_series(
            sem_displacement, args.component, args.force_direction
        )
        scaled_sem_series = None
        if scaled_sem_displacement is not None:
            scaled_sem_series = _select_displacement_series(
                scaled_sem_displacement, args.component, args.force_direction
            )
        _write_scalar_csv(
            args.csv_output, output["time"], reference_series, sem_series, scaled_sem_series
        )
        print(f"Wrote {args.csv_output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
