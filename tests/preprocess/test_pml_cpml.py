"""Regression tests for complete C-PML parameter generation."""

from __future__ import annotations

import numpy as np

from preprocess.pml_cpml import (
    CPML_X_ONLY,
    CPML_XY_ONLY,
    CPML_XYZ,
    _l_parameter,
    apply_cpml_mass_correction,
    compute_cpml_parameters,
)


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
    alpha_x = parameters["pml_alpha"][0, :, 0].reshape(3, 3, 3)
    assert np.all(damping_x[0] > 0.0)
    assert np.all(damping_x[-1] == 0.0)
    assert np.all(alpha_x[0] == 0.0)
    assert np.max(np.abs(parameters["pml_coef_abar"])) > 3.0
    assert np.max(np.abs(parameters["pml_coef_strain"])) > 3.0
    strain_x_group = parameters["pml_coef_strain"][0, 0, :4]
    np.testing.assert_allclose(strain_x_group, [1.0, 0.0, 0.0, -damping_x[0, 0, 0]])
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


def test_cpml_mass_correction_matches_specfem_direction_products() -> None:
    mass = np.ones((2, 1, 1, 1), dtype=np.float64)
    K_store = np.array([[[2.0, 3.0, 5.0]], [[2.0, 3.0, 5.0]]])
    d_store = np.array([[[7.0, 11.0, 13.0]], [[7.0, 11.0, 13.0]]])
    regions = np.array([CPML_X_ONLY, CPML_XYZ], dtype=np.int32)

    corrected = apply_cpml_mass_correction(mass, K_store, d_store, regions, dt=0.2)

    assert corrected[0, 0, 0, 0] == 2.0 + 0.1 * 7.0
    assert corrected[1, 0, 0, 0] == 30.0 + 0.1 * (
        7.0 * 3.0 * 5.0 + 11.0 * 2.0 * 5.0 + 13.0 * 2.0 * 3.0
    )


def test_xyz_acceleration_coefficients_match_partial_fraction_identity() -> None:
    """The XYZ residues must reconstruct SPECFEM's rational PML operator."""
    kappa = np.array([1.0, 1.0, 1.0])
    alpha = np.array([1.3, 2.1, 3.7])
    damping = np.array([8.0, 11.0, 17.0])
    beta = alpha + damping / kappa
    coefficients = _l_parameter(
        CPML_XYZ,
        kappa[0],
        damping[0],
        alpha[0],
        kappa[1],
        damping[1],
        alpha[1],
        kappa[2],
        damping[2],
        alpha[2],
    )

    for laplace_frequency in (0.25, 1.0, 4.0, 9.0):
        lhs = (
            laplace_frequency**2
            * np.prod(laplace_frequency + beta)
            / np.prod(laplace_frequency + alpha)
        )
        rhs = laplace_frequency**2 + coefficients[0] * laplace_frequency + coefficients[1]
        rhs += sum(
            residue / (laplace_frequency + pole) for residue, pole in zip(coefficients[2:], alpha)
        )
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-10)
