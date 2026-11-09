import numpy as np
import pytest
from scipy.integrate import quad

from examples._shared.stokes_reference import (
    relative_reference_change,
    sampled_stokes_displacement,
    stokes_displacement_piecewise_linear,
)


def test_piecewise_linear_reference_preserves_fractional_arrivals():
    source_time = np.arange(5, dtype=np.float64)
    source_values = source_time.copy()
    output_time = np.arange(5, dtype=np.float64)
    tensor = stokes_displacement_piecewise_linear(
        np.array([1.25, 0.0, 0.0]),
        np.zeros(3),
        5.0,
        3.0,
        2700.0,
        output_time,
        source_time,
        source_values,
    )

    p_arrival = 1.25 / 5.0
    expected_p = np.interp(
        output_time - p_arrival, source_time, source_values, left=0.0, right=0.0
    )
    s_arrival = 1.25 / 3.0
    near_integral = np.zeros_like(output_time)
    valid = output_time >= s_arrival
    near_integral[valid] = (
        output_time[valid] * (s_arrival**2 - p_arrival**2) / 2.0
        - (s_arrival**3 - p_arrival**3) / 3.0
    )
    constant = 1.0 / (4.0 * np.pi * 2700.0)
    expected = constant * (expected_p / (5.0**2 * 1.25) + 2.0 * near_integral / 1.25**3)
    np.testing.assert_allclose(tensor[:, 0, 0], expected)


def test_piecewise_linear_near_field_matches_constant_source_integral():
    source_time = np.arange(11, dtype=np.float64)
    source_values = np.ones_like(source_time)
    output_time = np.arange(5, dtype=np.float64)
    tensor = stokes_displacement_piecewise_linear(
        np.array([1.0, 0.0, 0.0]),
        np.zeros(3),
        2.0,
        1.0,
        2700.0,
        output_time,
        source_time,
        source_values,
    )

    p_arrival = 0.5
    s_arrival = 1.0
    expected_near = 0.5 * (s_arrival**2 - p_arrival**2)
    near_amplitude = (3.0 - 1.0) / 1.0**3
    constant = 1.0 / (4.0 * np.pi * 2700.0)
    expected = np.full_like(output_time, constant * (expected_near * near_amplitude))
    expected[0] = 0.0
    expected[1:] += constant / (2.0**2) * 1.0
    np.testing.assert_allclose(tensor[:, 0, 0], expected)


def test_sampled_reference_converges_to_piecewise_linear_reference():
    source_time = np.arange(501, dtype=np.float64) * 0.01
    source_values = np.sin(source_time * 4.0)
    output_time = np.arange(101, dtype=np.float64) * 0.05
    exact = stokes_displacement_piecewise_linear(
        np.array([4.3, 1.7, 2.1]),
        np.zeros(3),
        5.0,
        3.0,
        2700.0,
        output_time,
        source_time,
        source_values,
    )
    coarse = sampled_stokes_displacement(
        np.array([4.3, 1.7, 2.1]),
        np.zeros(3),
        5.0,
        3.0,
        2700.0,
        output_time,
        source_time,
        source_values,
        1,
    )
    fine = sampled_stokes_displacement(
        np.array([4.3, 1.7, 2.1]),
        np.zeros(3),
        5.0,
        3.0,
        2700.0,
        output_time,
        source_time,
        source_values,
        64,
    )
    assert relative_reference_change(fine, exact) < relative_reference_change(coarse, exact)


def test_piecewise_linear_stokes_matches_independent_near_field_quadrature():
    source_time = np.arange(61, dtype=np.float64) * 0.05
    phase = np.pi * (source_time - 1.2)
    source_values = (1 - 2 * phase**2) * np.exp(-(phase**2))
    receiver = np.array([1200.0, 700.0, 900.0])
    output_time = source_time.copy()
    vp_m_s, vs_m_s, density_kg_m3 = 5000.0, 3000.0, 2700.0
    radius = float(np.linalg.norm(receiver))
    direction = receiver / radius
    p_arrival, s_arrival = radius / vp_m_s, radius / vs_m_s
    p_amplitude = np.outer(direction, direction) / (vp_m_s**2 * radius)
    s_amplitude = (np.eye(3) - np.outer(direction, direction)) / (vs_m_s**2 * radius)
    near_amplitude = (3 * np.outer(direction, direction) - np.eye(3)) / radius**3
    actual = stokes_displacement_piecewise_linear(
        receiver,
        np.zeros(3),
        vp_m_s,
        vs_m_s,
        density_kg_m3,
        output_time,
        source_time,
        source_values,
    )

    for time_index in (20, 23, 27, 44):
        instant = output_time[time_index]
        knots = instant - source_time
        breakpoints = knots[(knots > p_arrival) & (knots < s_arrival)]
        near_integral, _ = quad(
            lambda delay: (
                delay * np.interp(instant - delay, source_time, source_values, left=0.0, right=0.0)
            ),
            p_arrival,
            s_arrival,
            points=breakpoints.tolist(),
            epsabs=1e-12,
        )
        p_value = np.interp(instant - p_arrival, source_time, source_values, left=0.0, right=0.0)
        s_value = np.interp(instant - s_arrival, source_time, source_values, left=0.0, right=0.0)
        expected = (
            p_value * p_amplitude + s_value * s_amplitude + near_integral * near_amplitude
        ) / (4 * np.pi * density_kg_m3)
        np.testing.assert_allclose(actual[time_index], expected, rtol=1e-9, atol=1e-22)


@pytest.mark.parametrize("refinement", [0, -1, 1.5])
def test_sampled_reference_rejects_invalid_refinement(refinement):
    with pytest.raises(ValueError):
        sampled_stokes_displacement(
            np.array([1.0, 0.0, 0.0]),
            np.zeros(3),
            5.0,
            3.0,
            2700.0,
            np.array([0.0, 0.1]),
            np.array([0.0, 0.1]),
            np.array([0.0, 1.0]),
            refinement,
        )
