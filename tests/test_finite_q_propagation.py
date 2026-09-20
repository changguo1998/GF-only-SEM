import h5py
import numpy as np
import pytest

from examples._shared.finite_q_propagation import (
    AMPLITUDE_RELATIVE_ERROR_LIMIT,
    PHASE_ERROR_LIMIT_RAD,
    analytical_relative_transfer,
    evaluate_transfer_error,
    load_receiver_signals,
)

TAU_SIGMA_S = np.array([1.5915494309189535, 0.20546814802049992, 0.026525823848649224])
TAU_EPSILON_MU_S = np.array([1.990132281541118, 0.24760030595211538, 0.034294139543322895])


def test_analytical_relative_transfer_matches_worked_sls_result():
    transfer = analytical_relative_transfer(
        frequency_hz=2.0,
        near_distance_m=1500.0,
        far_distance_m=3000.0,
        density_kg_m3=2700.0,
        shear_modulus_unrelaxed_pa=26344864296.918156,
        bulk_modulus_unrelaxed_pa=35100000000.0,
        tau_sigma_s=TAU_SIGMA_S,
        tau_epsilon_mu_s=TAU_EPSILON_MU_S,
    )

    assert abs(transfer) == pytest.approx(0.8477162798231204, rel=1.0e-12)
    assert np.angle(transfer) == pytest.approx(-0.20576866536609997, abs=1.0e-12)


def test_transfer_error_accepts_production_tolerances():
    expected = 0.85 * np.exp(-0.2j)
    measured = expected * 1.10 * np.exp(0.03j)

    result = evaluate_transfer_error(measured, expected)

    assert result.amplitude_relative_error == pytest.approx(0.10)
    assert result.phase_error_rad == pytest.approx(0.03)
    assert result.passed


def test_transfer_error_rejects_amplitude_or_phase_regression():
    expected = 0.85 * np.exp(-0.2j)

    amplitude_result = evaluate_transfer_error(
        expected * (1.0 + AMPLITUDE_RELATIVE_ERROR_LIMIT + 0.01), expected
    )
    phase_result = evaluate_transfer_error(
        expected * np.exp(1j * (PHASE_ERROR_LIMIT_RAD + 0.01)), expected
    )

    assert not amplitude_result.passed
    assert not phase_result.passed


def test_load_receiver_signals_reads_public_record_schema(tmp_path):
    record_dir = tmp_path / "wavefields" / "y"
    record_dir.mkdir(parents=True)
    coordinates = np.array([[1500.0, 500.0, 500.0], [3000.0, 500.0, 500.0]])
    cell_node_index = np.array([[0, 1]], dtype=np.int64)

    for step, values in ((0, [1.0, 2.0]), (5, [3.0, 4.0])):
        with h5py.File(record_dir / f"record_0_{step}.h5", "w") as record:
            record.create_dataset("gll_node_coords", data=coordinates, compression=None)
            record.create_dataset("cell_gll_node_index", data=cell_node_index, compression=None)
            displacement = np.zeros((1, 1, 2, 3))
            displacement[0, 0, :, 1] = values
            record.create_dataset("displacement", data=displacement, compression=None)

    times_s, signals = load_receiver_signals(
        record_dir, receiver_xyz_m=coordinates, component=1, solver_dt_s=0.008
    )

    np.testing.assert_allclose(times_s, [0.0, 0.04])
    np.testing.assert_allclose(signals, [[1.0, 3.0], [2.0, 4.0]])
