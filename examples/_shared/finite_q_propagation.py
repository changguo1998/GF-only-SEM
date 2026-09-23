#!/usr/bin/env python3
"""Finite-Q propagation validation against the analytical SLS Green function."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

AMPLITUDE_RELATIVE_ERROR_LIMIT = 0.12
PHASE_ERROR_LIMIT_RAD = 0.11


@dataclass(frozen=True)
class TransferError:
    """Amplitude and phase errors for one frequency."""

    amplitude_relative_error: float
    phase_error_rad: float
    passed: bool


@dataclass(frozen=True)
class FrequencyResult:
    """Measured and analytical propagation transfer at one frequency."""

    frequency_hz: float
    measured: complex
    expected: complex
    error: TransferError


def _sls_modulus_ratio(
    angular_frequency_rad_s: float, tau_sigma_s: np.ndarray, tau_epsilon_s: np.ndarray
) -> complex:
    """Return relaxed/unrelaxed modulus ratio for the SPECFEM SLS convention."""
    tau_ratio = tau_epsilon_s / tau_sigma_s
    weights = (tau_ratio - 1.0) / np.sum(tau_ratio)
    return complex(1.0 - np.sum(weights / (1.0 + 1j * angular_frequency_rad_s * tau_sigma_s)))


def _transverse_green(
    distance_m: float,
    angular_frequency_rad_s: float,
    density_kg_m3: float,
    shear_modulus_pa: complex,
    bulk_modulus_pa: complex,
) -> complex:
    """Frequency-domain transverse point-force Green function on the ray axis."""
    longitudinal_modulus_pa = bulk_modulus_pa + 4.0 * shear_modulus_pa / 3.0
    shear_wavenumber = angular_frequency_rad_s * np.sqrt(density_kg_m3 / shear_modulus_pa)
    pressure_wavenumber = angular_frequency_rad_s * np.sqrt(
        density_kg_m3 / longitudinal_modulus_pa
    )

    shear_exponential = np.exp(-1j * shear_wavenumber * distance_m)
    pressure_exponential = np.exp(-1j * pressure_wavenumber * distance_m)
    far_field = shear_wavenumber**2 * shear_exponential / distance_m
    near_field = (
        shear_exponential * (-1j * shear_wavenumber * distance_m - 1.0)
        - pressure_exponential * (-1j * pressure_wavenumber * distance_m - 1.0)
    ) / distance_m**3
    return complex((far_field + near_field) / (density_kg_m3 * angular_frequency_rad_s**2))


def analytical_relative_transfer(
    *,
    frequency_hz: float,
    near_distance_m: float,
    far_distance_m: float,
    density_kg_m3: float,
    shear_modulus_unrelaxed_pa: float,
    bulk_modulus_unrelaxed_pa: float,
    tau_sigma_s: np.ndarray,
    tau_epsilon_mu_s: np.ndarray,
) -> complex:
    """Return (finite-Q far/near transfer) / (elastic far/near transfer)."""
    angular_frequency_rad_s = 2.0 * np.pi * frequency_hz
    modulus_ratio = _sls_modulus_ratio(
        angular_frequency_rad_s,
        np.asarray(tau_sigma_s, dtype=np.float64),
        np.asarray(tau_epsilon_mu_s, dtype=np.float64),
    )
    viscoelastic_shear_modulus = shear_modulus_unrelaxed_pa * modulus_ratio

    def propagation_transfer(shear_modulus_pa: complex) -> complex:
        near_green = _transverse_green(
            near_distance_m,
            angular_frequency_rad_s,
            density_kg_m3,
            shear_modulus_pa,
            bulk_modulus_unrelaxed_pa,
        )
        far_green = _transverse_green(
            far_distance_m,
            angular_frequency_rad_s,
            density_kg_m3,
            shear_modulus_pa,
            bulk_modulus_unrelaxed_pa,
        )
        return far_green / near_green

    return propagation_transfer(viscoelastic_shear_modulus) / propagation_transfer(
        complex(shear_modulus_unrelaxed_pa)
    )


def evaluate_transfer_error(measured: complex, expected: complex) -> TransferError:
    """Evaluate the finite-Q amplitude attenuation and phase-dispersion gates."""
    amplitude_relative_error = abs(abs(measured) / abs(expected) - 1.0)
    phase_error_rad = abs(float(np.angle(measured / expected)))
    passed = (
        amplitude_relative_error <= AMPLITUDE_RELATIVE_ERROR_LIMIT
        and phase_error_rad <= PHASE_ERROR_LIMIT_RAD
    )
    return TransferError(amplitude_relative_error, phase_error_rad, passed)


def load_receiver_signals(
    record_dir: str | Path, *, receiver_xyz_m: np.ndarray, component: int, solver_dt_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Load displacement at physical coordinates from public record HDF5 files."""
    record_paths = sorted(Path(record_dir).glob("record_*_*.h5"))
    if not record_paths:
        raise FileNotFoundError(f"no record files found in {record_dir}")

    path_by_step: dict[int, list[Path]] = {}
    for record_path in record_paths:
        match = re.fullmatch(r"record_\d+_(\d+)\.h5", record_path.name)
        if match is None:
            continue
        path_by_step.setdefault(int(match.group(1)), []).append(record_path)
    if not path_by_step:
        raise ValueError(f"no valid record_<rank>_<step>.h5 files in {record_dir}")

    def rank_from_path(record_path: Path) -> int:
        match = re.fullmatch(r"record_(\d+)_\d+\.h5", record_path.name)
        if match is None:
            raise ValueError(f"invalid record filename: {record_path.name}")
        return int(match.group(1))

    def load_recording_layout(record_path: Path) -> tuple[np.ndarray, np.ndarray]:
        rank = rank_from_path(record_path)
        with h5py.File(record_path, "r") as record:
            partition_start = int(record.attrs.get("source_partition_start", rank))
            partition_count = int(record.attrs.get("source_partition_count", 1))

        partition_dir = record_path.parents[2] / "partitions"
        global_to_merged: dict[int, int] = {}
        merged_coordinates: list[np.ndarray] = []
        merged_cell_indexes: list[np.ndarray] = []
        for partition_index in range(partition_start, partition_start + partition_count):
            partition_path = partition_dir / f"partition_{partition_index}.h5"
            with h5py.File(partition_path, "r") as partition:
                if "recording" not in partition:
                    continue
                recording = partition["recording"]
                node_ids = np.asarray(recording["gll_node_ids"], dtype=np.int64)
                node_coordinates = np.asarray(recording["gll_node_coords"], dtype=np.float64)
                cell_indexes = np.asarray(recording["cell_gll_node_index"], dtype=np.int64)

            partition_to_merged = np.empty(node_ids.size, dtype=np.int64)
            for node_index, node_id in enumerate(node_ids):
                merged_index = global_to_merged.get(int(node_id))
                if merged_index is None:
                    merged_index = len(merged_coordinates)
                    global_to_merged[int(node_id)] = merged_index
                    merged_coordinates.append(node_coordinates[node_index])
                partition_to_merged[node_index] = merged_index
            merged_cell_indexes.append(partition_to_merged[cell_indexes])

        if not merged_coordinates or not merged_cell_indexes:
            raise ValueError(f"no recording map found for {record_path}")
        return np.asarray(merged_coordinates), np.concatenate(merged_cell_indexes, axis=0)

    layout_by_rank: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for record_path in record_paths:
        rank = rank_from_path(record_path)
        if rank in layout_by_rank:
            continue
        layout_by_rank[rank] = load_recording_layout(record_path)

    first_step = min(path_by_step)
    available_coordinates = []
    for record_path in path_by_step[first_step]:
        coordinates, _ = layout_by_rank[rank_from_path(record_path)]
        available_coordinates.append(coordinates)
    all_coordinates = np.concatenate(available_coordinates)

    requested_coordinates = np.asarray(receiver_xyz_m, dtype=np.float64)
    actual_coordinates = []
    for requested_coordinate in requested_coordinates:
        distances = np.linalg.norm(all_coordinates - requested_coordinate, axis=1)
        nearest_index = int(np.argmin(distances))
        if distances[nearest_index] > 1.0e-6:
            raise ValueError(
                f"receiver {requested_coordinate.tolist()} is not a recorded GLL node; "
                f"nearest distance={distances[nearest_index]:.6g} m"
            )
        actual_coordinates.append(all_coordinates[nearest_index])

    steps = sorted(path_by_step)
    signals = np.zeros((len(actual_coordinates), len(steps)), dtype=np.float64)
    for step_position, step in enumerate(steps):
        value_sums = np.zeros(len(actual_coordinates), dtype=np.float64)
        value_counts = np.zeros(len(actual_coordinates), dtype=np.int64)
        for record_path in path_by_step[step]:
            coordinates, cell_node_index = layout_by_rank[rank_from_path(record_path)]
            with h5py.File(record_path, "r") as record:
                flat_coordinates = coordinates[cell_node_index.reshape(-1)]
                displacement = np.asarray(record["displacement"][0], dtype=np.float64).reshape(
                    -1, 3
                )
            for receiver_index, actual_coordinate in enumerate(actual_coordinates):
                matches = np.all(
                    np.isclose(flat_coordinates, actual_coordinate, atol=1.0e-6), axis=1
                )
                value_sums[receiver_index] += float(np.sum(displacement[matches, component]))
                value_counts[receiver_index] += int(np.count_nonzero(matches))
        if np.any(value_counts == 0):
            raise ValueError(f"step {step}: receiver missing from record files")
        signals[:, step_position] = value_sums / value_counts

    return np.asarray(steps, dtype=np.float64) * solver_dt_s, signals


def _uniform_scalar(dataset: h5py.Dataset, name: str) -> float:
    values = np.asarray(dataset, dtype=np.float64)
    value = float(np.median(values))
    if not np.allclose(values, value, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"{name} must be uniform for the analytical propagation check")
    return value


def _uniform_mechanisms(dataset: h5py.Dataset, name: str) -> np.ndarray:
    values = np.asarray(dataset, dtype=np.float64)
    mechanisms = values.reshape(-1, values.shape[-1])
    if not np.allclose(mechanisms, mechanisms[0], rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"{name} must be uniform for the analytical propagation check")
    return mechanisms[0]


def _windowed_transfer(
    time_s: np.ndarray,
    signals: np.ndarray,
    frequency_hz: float,
    arrival_centers_s: np.ndarray,
    window_half_width_s: float,
) -> complex:
    spectra = []
    fourier_weight = np.exp(-2j * np.pi * frequency_hz * time_s)
    for signal, center_s in zip(signals, arrival_centers_s):
        offset = time_s - center_s
        window = np.where(
            np.abs(offset) <= window_half_width_s,
            0.5 * (1.0 + np.cos(np.pi * offset / window_half_width_s)),
            0.0,
        )
        spectra.append(np.dot(signal * window, fourier_weight))
    return complex(spectra[1] / spectra[0])


def validate_case(case_dir: str | Path) -> list[FrequencyResult]:
    """Validate one homogeneous finite-Q propagation example from its artifacts."""
    case_path = Path(case_dir)
    config_path = case_path / "config.h5"
    model_path = case_path / "model.h5"

    with h5py.File(config_path, "r") as config:
        solver_dt_s = float(config["simulation"].attrs["solver_dt"])
        source_xyz_m = np.array(
            [config["source"].attrs[axis] for axis in ("x", "y", "z")], dtype=np.float64
        )
        source_time_s = np.asarray(config["source/stf_t"], dtype=np.float64)
        source_values = np.asarray(config["source/stf_values"], dtype=np.float64)
        source_peak_time_s = float(source_time_s[np.argmax(np.abs(source_values))])

    with h5py.File(model_path, "r") as model:
        fields = model["field/cell"]
        density_kg_m3 = _uniform_scalar(fields["density"], "density")
        shear_speed_m_s = _uniform_scalar(fields["vs"], "vs")
        shear_modulus_unrelaxed_pa = _uniform_scalar(fields["mu"], "mu")
        lambda_unrelaxed_pa = _uniform_scalar(fields["lambda"], "lambda")
        bulk_modulus_unrelaxed_pa = lambda_unrelaxed_pa + 2.0 * shear_modulus_unrelaxed_pa / 3.0
        tau_sigma_s = _uniform_mechanisms(fields["tau_sigma"], "tau_sigma")
        tau_epsilon_mu_s = _uniform_mechanisms(fields["tau_epsilon_mu"], "tau_epsilon_mu")
        tau_epsilon_kappa_s = _uniform_mechanisms(fields["tau_epsilon_kappa"], "tau_epsilon_kappa")
        reference_frequency_hz = float(fields["tau_sigma"].attrs["f0_Hz"])

    if not np.allclose(tau_epsilon_kappa_s, tau_sigma_s, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(
            "finite-Q propagation case must use elastic bulk modulus (Q_kappa >= 1e8)"
        )

    shear_wavelength_m = shear_speed_m_s / reference_frequency_hz
    distances_m = np.array([shear_wavelength_m, 2.0 * shear_wavelength_m])
    receiver_xyz_m = source_xyz_m + np.column_stack(
        [distances_m, np.zeros((2, 2), dtype=np.float64)]
    )
    elastic_time_s, elastic_signals = load_receiver_signals(
        case_path / "elastic_wavefields" / "y",
        receiver_xyz_m=receiver_xyz_m,
        component=1,
        solver_dt_s=solver_dt_s,
    )
    viscoelastic_time_s, viscoelastic_signals = load_receiver_signals(
        case_path / "viscoelastic_wavefields" / "y",
        receiver_xyz_m=receiver_xyz_m,
        component=1,
        solver_dt_s=solver_dt_s,
    )
    if not np.array_equal(elastic_time_s, viscoelastic_time_s):
        raise ValueError("elastic and viscoelastic record times differ")

    arrival_centers_s = source_peak_time_s + distances_m / shear_speed_m_s
    window_half_width_s = 1.2 / reference_frequency_hz
    results = []
    for frequency_factor in (0.75, 0.875, 1.0):
        frequency_hz = frequency_factor * reference_frequency_hz
        elastic_transfer = _windowed_transfer(
            elastic_time_s, elastic_signals, frequency_hz, arrival_centers_s, window_half_width_s
        )
        viscoelastic_transfer = _windowed_transfer(
            viscoelastic_time_s,
            viscoelastic_signals,
            frequency_hz,
            arrival_centers_s,
            window_half_width_s,
        )
        measured = viscoelastic_transfer / elastic_transfer
        expected = analytical_relative_transfer(
            frequency_hz=frequency_hz,
            near_distance_m=distances_m[0],
            far_distance_m=distances_m[1],
            density_kg_m3=density_kg_m3,
            shear_modulus_unrelaxed_pa=shear_modulus_unrelaxed_pa,
            bulk_modulus_unrelaxed_pa=bulk_modulus_unrelaxed_pa,
            tau_sigma_s=tau_sigma_s,
            tau_epsilon_mu_s=tau_epsilon_mu_s,
        )
        results.append(
            FrequencyResult(
                frequency_hz=frequency_hz,
                measured=measured,
                expected=expected,
                error=evaluate_transfer_error(measured, expected),
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    args = parser.parse_args()

    results = validate_case(args.case_dir)
    print(
        f"{'frequency':>10} {'amp_meas':>10} {'amp_ref':>10} {'amp_err':>10} "
        f"{'phase_meas':>12} {'phase_ref':>10} {'phase_err':>10}"
    )
    for result in results:
        print(
            f"{result.frequency_hz:10.3f} {abs(result.measured):10.6f} "
            f"{abs(result.expected):10.6f} {result.error.amplitude_relative_error:10.6f} "
            f"{np.angle(result.measured):12.6f} {np.angle(result.expected):10.6f} "
            f"{result.error.phase_error_rad:10.6f}"
        )
    passed = all(result.error.passed for result in results)
    print(
        f"VERDICT: {'PASS' if passed else 'FAIL'} "
        f"(limits: amplitude relative error <= {AMPLITUDE_RELATIVE_ERROR_LIMIT:.3f}, "
        f"phase error <= {PHASE_ERROR_LIMIT_RAD:.3f} rad)"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
