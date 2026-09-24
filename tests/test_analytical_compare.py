"""Regression tests for the full-space analytical comparison."""

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_PATH = Path(__file__).parents[1] / "examples" / "_shared" / "analytical_compare.py"
SPEC = importlib.util.spec_from_file_location("analytical_compare", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYTICAL_COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYTICAL_COMPARE)


def test_amplitude_scale_rejects_threefold_postprocess_error():
    assert ANALYTICAL_COMPARE.amplitude_scale_is_acceptable(1.0)
    assert not ANALYTICAL_COMPARE.amplitude_scale_is_acceptable(1.0 / 3.0)


def test_best_fit_scale_uses_sem_over_analytical_convention():
    analytical = np.array([1.0, -2.0, 3.0])
    sem = 1.1 * analytical
    assert np.isclose(ANALYTICAL_COMPARE.best_fit_scale(sem, analytical), 1.1)


def test_scale_fitted_l2_is_invariant_to_sem_amplitude():
    analytical = np.array([1.0, -2.0, 3.0, -1.0])
    sem = np.array([1.0, -1.8, 3.2, -0.9])
    _, original_error = ANALYTICAL_COMPARE.scale_fitted_relative_l2(sem, analytical)
    _, scaled_error = ANALYTICAL_COMPARE.scale_fitted_relative_l2(3.0 * sem, analytical)
    assert np.isclose(original_error, scaled_error)


def test_fixed_receivers_does_not_skip_following_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["analytical_compare.py", "greenfun", "--fixed-receivers", "points.npy", "--fullspace"],
    )
    args = ANALYTICAL_COMPARE.parse_args()
    assert args["fixed_receivers"] == "points.npy"
    assert args["fullspace"] is True


def test_load_sem_tiles_reads_only_selected_displacement_traces(tmp_path):
    expected = []
    for tile_index in range(2):
        tile_path = tmp_path / f"tile_{tile_index}.h5"
        displacement = np.arange(3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3)
        displacement += tile_index * 1000
        coords = np.full((2, 3), tile_index, dtype=np.float64)
        with h5py.File(tile_path, "w") as output:
            output.create_dataset(
                "/field/displacement_tensor", data=displacement, compression=None
            )
            output.create_dataset("/mesh/gll_node_coords", data=coords, compression=None)
            output.create_dataset("/time/t", data=np.array([0.0, 0.1, 0.2]), compression=None)
        expected.append(displacement)

    sem = ANALYTICAL_COMPARE.load_sem_tiles(str(tmp_path))
    sampled = ANALYTICAL_COMPARE.load_sampled_displacement(sem, [0, 0, 3])

    assert "displacement" not in sem
    assert sampled.shape == (3, 3, 3, 3)
    np.testing.assert_array_equal(sampled[:, 0], expected[0][:, 0])
    np.testing.assert_array_equal(sampled[:, 1], expected[0][:, 0])
    np.testing.assert_array_equal(sampled[:, 2], expected[1][:, 1])


def test_nearest_node_indices_preserve_fixed_receiver_count():
    coordinates = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    receivers = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])

    indices = ANALYTICAL_COMPARE._nearest_node_indices(coordinates, receivers)

    assert indices == [0, 0]


def test_load_sem_parameters_reads_canonical_cell_material_schema(tmp_path):
    config_path = tmp_path / "config.h5"
    model_path = tmp_path / "model.h5"
    with h5py.File(config_path, "w") as config:
        source = config.create_group("source")
        source.attrs.update({"x": 1.0, "y": 2.0, "z": 3.0})
        source.create_dataset("stf_t", data=[0.0, 0.1], compression=None)
        source.create_dataset("stf_values", data=[4.0, 5.0], compression=None)
        simulation = config.create_group("simulation")
        simulation.attrs.update({"output_dt_s": 0.1, "nsteps": 2})
    with h5py.File(model_path, "w") as model:
        cell = model.create_group("field/cell")
        cell.create_dataset("vp", data=[5000.0], compression=None)
        cell.create_dataset("vs", data=[3000.0], compression=None)
        cell.create_dataset("density", data=[2700.0], compression=None)

    parameters = ANALYTICAL_COMPARE.load_sem_parameters(
        str(config_path), str(model_path), n_steps=2
    )

    assert parameters["vp_m_s"] == 5000.0
    assert parameters["vs_m_s"] == 3000.0
    assert parameters["density_kg_m3"] == 2700.0


def test_load_sem_parameters_downsamples_solver_stf_to_output_frames(tmp_path):
    config_path = tmp_path / "config.h5"
    model_path = tmp_path / "model.h5"
    with h5py.File(config_path, "w") as config:
        source = config.create_group("source")
        source.attrs.update({"x": 1.0, "y": 2.0, "z": 3.0})
        source.create_dataset("stf_t", data=np.arange(6) * 0.01, compression=None)
        source.create_dataset("stf_values", data=np.arange(6), compression=None)
        simulation = config.create_group("simulation")
        simulation.attrs.update(
            {"solver_dt": 0.01, "snapshot_stride": 2, "output_dt_s": 0.02, "nsteps": 5}
        )
    with h5py.File(model_path, "w") as model:
        cell = model.create_group("field/cell")
        cell.create_dataset("vp", data=[5000.0], compression=None)
        cell.create_dataset("vs", data=[3000.0], compression=None)
        cell.create_dataset("density", data=[2700.0], compression=None)

    parameters = ANALYTICAL_COMPARE.load_sem_parameters(str(config_path), str(model_path), 3)

    np.testing.assert_array_equal(parameters["stf"], [0, 2, 4])
    assert parameters["output_dt_s"] == 0.02
