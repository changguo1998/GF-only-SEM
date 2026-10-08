"""Fine-time Lamb response and output-grid sampling, independent of SEM output dt."""

import numpy as np
from scipy.signal import fftconvolve

from examples.halfspace import reference


def integrate_batched(integrand, lower, upper, nodes, weights, args, n_seg=20):
    """Batch the original x-axis quadrature; retain scalar fallback off axis."""
    if upper <= lower:
        return np.zeros((3, 3))
    theta, phi, radius, vp, vs, density, time_s = args
    if phi != 0.0:
        return reference._integrate_segments(integrand, lower, upper, nodes, weights, args, n_seg)
    edges = np.linspace(lower, upper, n_seg + 1)
    half = np.diff(edges) * 0.5
    midpoint = (edges[:-1] + edges[1:]) * 0.5
    parameter = (midpoint[:, None] + half[:, None] * nodes).ravel()
    quadrature_weights = (half[:, None] * weights).ravel()
    branch_cut = integrand is reference._integrand_s_branch
    p_wave = integrand is reference._integrand_p
    speed = vp if p_wave else vs
    radicand = (time_s / radius) ** 2 - 1.0 / speed**2 - parameter**2
    if branch_cut:
        denominator = np.sqrt(-radicand + 0.0j)
        contour = -(time_s / radius) * np.sin(theta) + denominator.real * np.cos(theta)
        active = radicand < 0.0
    else:
        denominator = np.sqrt(radicand + 0.0j)
        contour = -(time_s / radius) * np.sin(theta) + 1j * denominator * np.cos(theta)
        active = radicand > 0.0

    # These five nonzero components are the original matrices at phi=0.
    eta_p = np.sqrt(1.0 / vp**2 + parameter**2 - contour**2 + 0.0j)
    eta_s = np.sqrt(1.0 / vs**2 + parameter**2 - contour**2 + 0.0j)
    gamma = eta_s**2 + parameter**2 - contour**2
    sigma = gamma**2 + 4.0 * eta_s * eta_p * (contour**2 - parameter**2)
    matrix = np.zeros((len(parameter), 3, 3), dtype=np.complex128)
    if p_wave:
        matrix[:, 0, 0] = 2.0 * eta_s * contour**2
        matrix[:, 1, 1] = -2.0 * eta_s * parameter**2
        matrix[:, 0, 2] = 2.0 * contour * eta_s * eta_p
        matrix[:, 2, 0] = contour * gamma
        matrix[:, 2, 2] = eta_p * gamma
        factor = eta_p / sigma / denominator
    else:
        extra = gamma - 4.0 * eta_p * eta_s
        matrix[:, 0, 0] = (eta_s**2 * gamma + extra * parameter**2) / eta_s
        matrix[:, 1, 1] = (eta_s**2 * gamma - extra * contour**2) / eta_s
        matrix[:, 0, 2] = -contour * gamma
        matrix[:, 2, 0] = -2.0 * contour * eta_p * eta_s
        matrix[:, 2, 2] = 2.0 * eta_p * (contour**2 - parameter**2)
        factor = eta_s / sigma / denominator
    values = factor[:, None, None] * matrix
    values = values.imag if branch_cut else values.real
    values[~active] = 0.0
    return np.sum(values * quadrature_weights[:, None, None], axis=0) / (
        np.pi**2 * density * vs**2 * radius
    )


def sampled_displacement(
    vp,
    vs,
    density,
    source_xyz,
    receiver_xyz,
    time_s,
    stf_time_s,
    stf_values,
    refinement,
    n_quad_pts=48,
    n_seg=1600,
):
    """Differentiate and convolve on the fine grid, then sample at output times."""
    time_s = np.asarray(time_s, dtype=np.float64)
    if len(time_s) < 3 or not np.isfinite(time_s).all() or time_s[0] != 0.0:
        raise ValueError("Reference output times must start at zero and have at least 3 samples")
    output_dt = time_s[1] - time_s[0]
    if output_dt <= 0.0 or not np.allclose(np.diff(time_s), output_dt, rtol=1e-9, atol=1e-12):
        raise ValueError("Reference output times must be uniformly increasing")
    if not isinstance(refinement, (int, np.integer)) or refinement < 1:
        raise ValueError("Reference time refinement must be a positive integer")
    fine_dt = output_dt / refinement
    fine_time = np.arange((len(time_s) - 1) * refinement + 1) * fine_dt

    # Retain the physical source samples while independently refining the response.
    step = reference.compute_step_response(
        vp,
        vs,
        density,
        source_xyz,
        receiver_xyz,
        fine_time,
        n_quad_pts=n_quad_pts,
        n_seg=n_seg,
        integrate_segments=integrate_batched,
    )
    force = np.interp(fine_time, stf_time_s, stf_values)
    impulse = reference.impulse_response_from_step(step, fine_dt)
    displacement = fftconvolve(impulse, force[:, None, None], mode="full", axes=0)
    sampled = displacement[: len(fine_time) : refinement] * fine_dt
    if not np.isfinite(sampled).all():
        raise ValueError("Reference calculation produced nonfinite displacement")
    return sampled


def converged_displacement(
    evaluate, time_s, end_time_s, initial_refinement=16, max_refinement=1024, tolerance=0.002
):
    """Compare consecutive reference resolutions without inspecting SEM values."""
    if initial_refinement < 1 or max_refinement < 2 * initial_refinement:
        raise ValueError("At least two positive reference refinement levels are required")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Reference convergence tolerance must be positive and finite")
    mask = np.asarray(time_s) < end_time_s
    if not mask.any():
        raise ValueError("Reference comparison window is empty")
    previous = None
    refinement = initial_refinement
    while refinement <= max_refinement:
        current = evaluate(refinement)
        if previous is not None:
            norm = np.linalg.norm(current[mask])
            change = np.linalg.norm((current - previous)[mask]) / max(norm, 1e-300)
            if not np.isfinite(change) or norm == 0.0:
                raise ValueError("Reference convergence window is nonfinite or zero")
            print(f"reference refinement={refinement} relative_change={change:.6g}", flush=True)
            if change <= tolerance:
                return current, refinement, float(change)
        previous = current
        refinement *= 2
    raise RuntimeError(f"Reference did not converge through refinement {max_refinement}")
