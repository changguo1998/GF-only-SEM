"""Regression tests for complete C-PML parameter generation."""

from __future__ import annotations

import numpy as np

from preprocess.pml_cpml import CPML_X_ONLY, CPML_XY_ONLY, compute_cpml_parameters


def test_min_face_profile_grows_toward_boundary_when_both_faces_are_enabled() -> None:
    """The opposite max face must not overwrite an xmin PML element."""
    axis_points = np.array([0.0, 500.0, 1000.0])
    transverse_points = np.array([1000.0, 1500.0, 2000.0])
    x_coord, y_coord, z_coord = np.meshgrid(
        axis_points, transverse_points, transverse_points, indexing="ij"
    )
    coordinates = np.stack((x_coord, y_coord, z_coord), axis=-1)[None, ...]
    parameters = compute_cpml_parameters(
        coordinates,
        np.array([True]),
        {"xmin": 0.0, "xmax": 3000.0, "ymin": 0.0, "ymax": 3000.0, "zmin": 0.0, "zmax": 3000.0},
        {"xmin": 1000.0, "xmax": 1000.0, "ymin": 0.0, "ymax": 0.0, "zmin": 0.0, "zmax": 0.0},
        np.full((1, 3, 3, 3), 3000.0),
        f0_hz=5.0,
        dt=0.01,
    )

    assert parameters["pml_region"][0] == CPML_X_ONLY
    damping_x = parameters["pml_d"][0, :, 0].reshape(3, 3, 3)
    assert np.all(damping_x[0] > 0.0)
    assert np.all(damping_x[-1] == 0.0)
    for name in ("pml_coef_alpha", "pml_coef_beta", "pml_coef_abar", "pml_coef_strain"):
        assert np.all(np.isfinite(parameters[name]))


def test_parameter_separation_uses_minimum_adjacent_gll_distance() -> None:
    """A layer-wide distance sum must not trigger unnecessary alpha separation."""
    pml_points = np.linspace(0.0, 1000.0, 5)
    z_points = np.linspace(1000.0, 2000.0, 5)
    x_coord, y_coord, z_coord = np.meshgrid(pml_points, pml_points, z_points, indexing="ij")
    coordinates = np.stack((x_coord, y_coord, z_coord), axis=-1)[None, ...]
    parameters = compute_cpml_parameters(
        coordinates,
        np.array([True]),
        {"xmin": 0.0, "xmax": 3000.0, "ymin": 0.0, "ymax": 3000.0, "zmin": 0.0, "zmax": 3000.0},
        {"xmin": 1000.0, "xmax": 0.0, "ymin": 1000.0, "ymax": 0.0, "zmin": 0.0, "zmax": 0.0},
        np.full((1, 5, 5, 5), 3000.0),
        f0_hz=5.0,
        dt=0.01,
    )

    assert parameters["pml_region"][0] == CPML_XY_ONLY
    alpha = parameters["pml_alpha"].reshape(1, 5, 5, 5, 3)
    assert alpha[0, 2, 2, 2, 0] == np.pi * 5.0 * 0.9 * 0.5
    assert alpha[0, 2, 2, 2, 1] == np.pi * 5.0 * 0.5
