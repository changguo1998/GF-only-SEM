"""Time-refined full-space Stokes reference evaluation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_time_grid(time_s: NDArray[np.float64], name: str) -> float:
    """Validate a uniformly sampled time grid and return its interval."""
    values = np.asarray(time_s, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"{name} must be a one-dimensional grid with at least two samples")
    if not np.isfinite(values).all() or not np.isclose(values[0], 0.0):
        raise ValueError(f"{name} must be finite and start at zero")
    interval = float(values[1] - values[0])
    if interval <= 0.0 or not np.allclose(np.diff(values), interval, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{name} must be uniformly increasing")
    return interval


def _validate_source_grid(
    source_time_s: NDArray[np.float64], source_values: NDArray[np.float64]
) -> float:
    """Validate the solver source grid and return its interval."""
    source_time = np.asarray(source_time_s, dtype=np.float64)
    source = np.asarray(source_values, dtype=np.float64)
    interval = _validate_time_grid(source_time, "source_time_s")
    if source.ndim != 1 or len(source) != len(source_time):
        raise ValueError("source_values must match source_time_s in shape")
    if not np.isfinite(source).all():
        raise ValueError("source_values must be finite")
    return interval


def _piecewise_linear_primitives(
    time_s: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return exact cumulative integrals for a piecewise-linear signal."""
    intervals = np.diff(time_s)
    slopes = np.diff(values) / intervals
    cumulative_values = np.zeros_like(values)
    cumulative_time_values = np.zeros_like(values)

    for index, interval in enumerate(intervals):
        start_time = time_s[index]
        start_value = values[index]
        slope = slopes[index]
        cumulative_values[index + 1] = cumulative_values[index] + (
            start_value * interval + 0.5 * slope * interval**2
        )
        cumulative_time_values[index + 1] = cumulative_time_values[index] + (
            start_time * start_value * interval
            + 0.5 * (start_time * slope + start_value) * interval**2
            + slope * interval**3 / 3.0
        )
    return cumulative_values, cumulative_time_values


def _evaluate_primitives(
    query_time_s: NDArray[np.float64],
    source_time_s: NDArray[np.float64],
    source_values: NDArray[np.float64],
    cumulative_values: NDArray[np.float64],
    cumulative_time_values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the two piecewise-linear source primitives at arbitrary times."""
    query = np.asarray(query_time_s, dtype=np.float64)
    values = np.zeros_like(query)
    time_values = np.zeros_like(query)
    valid = query >= source_time_s[0]
    if not np.any(valid):
        return values, time_values

    valid_query = query[valid]
    after_source = valid_query >= source_time_s[-1]

    inside = ~after_source
    if np.any(inside):
        inside_query = valid_query[inside]
        indexes = np.searchsorted(source_time_s, inside_query, side="right") - 1
        indexes = np.clip(indexes, 0, len(source_time_s) - 2)
        start_time = source_time_s[indexes]
        interval = source_time_s[indexes + 1] - start_time
        start_value = source_values[indexes]
        slope = (source_values[indexes + 1] - start_value) / interval
        offset = inside_query - start_time
        primitive_values = cumulative_values[indexes] + start_value * offset
        primitive_values += 0.5 * slope * offset**2
        primitive_time_values = cumulative_time_values[indexes] + start_time * start_value * offset
        primitive_time_values += 0.5 * (start_time * slope + start_value) * offset**2
        primitive_time_values += slope * offset**3 / 3.0

        valid_indices = np.flatnonzero(valid)
        values[valid_indices[inside]] = primitive_values
        time_values[valid_indices[inside]] = primitive_time_values

    valid_indices = np.flatnonzero(valid)
    values[valid_indices[after_source]] = cumulative_values[-1]
    time_values[valid_indices[after_source]] = cumulative_time_values[-1]
    return values, time_values


def stokes_displacement_piecewise_linear(
    receiver_xyz_m: NDArray[np.float64],
    source_xyz_m: NDArray[np.float64],
    vp_m_s: float,
    vs_m_s: float,
    density_kg_m3: float,
    output_time_s: NDArray[np.float64],
    source_time_s: NDArray[np.float64],
    source_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate the full-space Stokes tensor for a piecewise-linear source.

    The result has shape ``(time, response_component, force_component)``.
    Fractional P/S delays and the near-field integral are evaluated directly
    from exact primitives of the piecewise-linear source signal.
    """
    _validate_time_grid(output_time_s, "output_time_s")
    _validate_source_grid(source_time_s, source_values)
    output_time = np.asarray(output_time_s, dtype=np.float64)
    source_time = np.asarray(source_time_s, dtype=np.float64)
    source = np.asarray(source_values, dtype=np.float64)

    receiver = np.asarray(receiver_xyz_m, dtype=np.float64)
    source_position = np.asarray(source_xyz_m, dtype=np.float64)
    radius_vector = receiver - source_position
    radius = float(np.linalg.norm(radius_vector))
    if radius < 1e-6:
        return np.zeros((len(output_time), 3, 3), dtype=np.float64)

    direction = radius_vector / radius
    p_amplitude = np.outer(direction, direction) / (vp_m_s**2 * radius)
    s_amplitude = (np.eye(3) - np.outer(direction, direction)) / (vs_m_s**2 * radius)
    near_amplitude = (3.0 * np.outer(direction, direction) - np.eye(3)) / radius**3
    p_arrival_s = radius / vp_m_s
    s_arrival_s = radius / vs_m_s
    cumulative_values, cumulative_time_values = _piecewise_linear_primitives(source_time, source)

    p_wave = np.interp(output_time - p_arrival_s, source_time, source, left=0.0, right=0.0)
    s_wave = np.interp(output_time - s_arrival_s, source_time, source, left=0.0, right=0.0)
    near_upper, near_upper_time = _evaluate_primitives(
        output_time - p_arrival_s, source_time, source, cumulative_values, cumulative_time_values
    )
    near_lower, near_lower_time = _evaluate_primitives(
        output_time - s_arrival_s, source_time, source, cumulative_values, cumulative_time_values
    )
    near_wave = output_time * (near_upper - near_lower) - (near_upper_time - near_lower_time)

    displacement = np.empty((len(output_time), 3, 3), dtype=np.float64)
    constant = 1.0 / (4.0 * np.pi * density_kg_m3)
    for response_component in range(3):
        displacement[:, response_component, :] = constant * (
            p_wave[:, None] * p_amplitude[response_component, :]
            + s_wave[:, None] * s_amplitude[response_component, :]
            + near_wave[:, None] * near_amplitude[response_component, :]
        )
    return displacement


def sampled_stokes_displacement(
    receiver_xyz_m: NDArray[np.float64],
    source_xyz_m: NDArray[np.float64],
    vp_m_s: float,
    vs_m_s: float,
    density_kg_m3: float,
    output_time_s: NDArray[np.float64],
    source_time_s: NDArray[np.float64],
    source_values: NDArray[np.float64],
    refinement: int,
) -> NDArray[np.float64]:
    """Evaluate a uniformly refined reference and sample it at output times."""
    output_dt_s = _validate_time_grid(output_time_s, "output_time_s")
    _validate_source_grid(source_time_s, source_values)
    if not isinstance(refinement, (int, np.integer)) or refinement < 1:
        raise ValueError("refinement must be a positive integer")

    output_time = np.asarray(output_time_s, dtype=np.float64)
    fine_dt_s = output_dt_s / refinement
    fine_time = np.arange((len(output_time) - 1) * refinement + 1, dtype=np.float64)
    fine_time *= fine_dt_s
    fine_source = np.interp(
        fine_time,
        np.asarray(source_time_s, dtype=np.float64),
        np.asarray(source_values, dtype=np.float64),
        left=0.0,
        right=0.0,
    )
    fine_displacement = stokes_displacement_piecewise_linear(
        receiver_xyz_m,
        source_xyz_m,
        vp_m_s,
        vs_m_s,
        density_kg_m3,
        fine_time,
        fine_time,
        fine_source,
    )
    sampled = fine_displacement[::refinement]
    if not np.isfinite(sampled).all():
        raise ValueError("reference calculation produced nonfinite displacement")
    return sampled


def relative_reference_change(
    current: NDArray[np.float64], previous: NDArray[np.float64]
) -> float:
    """Return the relative change between two reference tensors."""
    current_array = np.asarray(current, dtype=np.float64)
    previous_array = np.asarray(previous, dtype=np.float64)
    denominator = max(float(np.linalg.norm(current_array)), 1e-300)
    return float(np.linalg.norm(current_array - previous_array) / denominator)
