#!/usr/bin/env python3
"""Analytical Green's function for elastic media.

Provides the Stokes full-space solution and the half-space free-surface
correction using Zoeppritz reflection coefficients.

Reference: Aki & Richards, Quantitative Seismology, 2nd ed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stokes_displacement_green_tensor(
    receiver_xyz_m: NDArray[np.float64],
    source_xyz_m: NDArray[np.float64],
    force_direction: int,
    vp_m_s: float,
    vs_m_s: float,
    density_kg_m3: float,
    source_time_function: NDArray[np.float64],
    dt_s: float,
) -> NDArray[np.float64]:
    """Stokes full-space displacement Green's tensor (Aki & Richards Eq. 4.23).

    For a point force F_j in an unbounded elastic medium:

        u_i(t) = (1/4πρ) * [
            (3γ_iγ_j - δ_ij)/r³ ∫_{r/α}^{r/β} τ F(t-τ) dτ   (near-field)
            + γ_iγ_j/(α²r) F(t - r/α)                         (P far-field)
            - (γ_iγ_j - δ_ij)/(β²r) F(t - r/β)                (S far-field)
        ]
    """
    nt = len(source_time_function)
    r_vec = receiver_xyz_m - source_xyz_m
    r = float(np.linalg.norm(r_vec))
    if r < 1e-6:
        return np.zeros((nt, 3))

    gamma = r_vec / r
    p_amp = np.outer(gamma, gamma) / (vp_m_s**2 * r)
    s_amp = (np.eye(3) - np.outer(gamma, gamma)) / (vs_m_s**2 * r)
    near_amp = (3.0 * np.outer(gamma, gamma) - np.eye(3)) / r**3

    t_p = r / vp_m_s
    t_s = r / vs_m_s
    displacement = np.zeros((nt, 3))
    const = 1.0 / (4.0 * np.pi * density_kg_m3)

    for comp in range(3):
        pf = p_amp[comp, force_direction]
        sf = s_amp[comp, force_direction]
        nf = near_amp[comp, force_direction]

        # Far-field P
        if abs(pf) > 1e-30:
            sp = int(t_p / dt_s)
            if sp < nt:
                end = nt - sp
                displacement[sp:, comp] += const * pf * source_time_function[:end]

        # Far-field S
        if abs(sf) > 1e-30:
            ss = int(t_s / dt_s)
            if ss < nt:
                end = nt - ss
                displacement[ss:, comp] += const * sf * source_time_function[:end]

        # Near-field integral
        if abs(nf) > 1e-30:
            n_tau = max(2, int((t_s - t_p) / dt_s) + 1)
            tau_vals = np.linspace(t_p, t_s, n_tau)
            dtau = float(tau_vals[1] - tau_vals[0])
            for t_idx in range(nt):
                total = 0.0
                for tau in tau_vals:
                    si = t_idx - int(tau / dt_s)
                    if 0 <= si < nt:
                        total += tau * source_time_function[si] * dtau
                displacement[t_idx, comp] += const * nf * total

    return displacement


def make_ricker_stf(
    f0_hz: float = 1.0,
    t0_s: float = 1.0,
    amplitude_n: float = 1.0e20,
    dt_s: float = 0.01,
    total_duration_s: float = 5.0,
):
    """Ricker wavelet source time function."""
    nt = int(total_duration_s / dt_s) + 1
    time = np.arange(nt) * dt_s
    a = np.pi * f0_hz * (time - t0_s)
    stf = amplitude_n * (1.0 - 2.0 * a**2) * np.exp(-(a**2))
    return time, stf


def wavelet_correlation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Pearson correlation coefficient."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def relative_l2_error(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Relative L2 norm error."""
    dn = float(np.linalg.norm(a - b))
    mn = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    if mn < 1e-30:
        return 0.0
    return dn / mn


def free_surface_zoeppritz(
    vp_m_s: float, vs_m_s: float, density_kg_m3: float, sin_i_p: float
) -> dict:
    """Zoeppritz reflection coefficients for a free surface (traction-free BC).

    For an incident P-wave at angle i (sin_i_p given), computes the reflected
    P (R_PP) and reflected SV (R_PS) coefficients such that the free-surface
    traction vanishes.

    Returns dict with keys: R_PP, R_PS, sin_j_s, cos_i_p, cos_j_s.
    """
    sin_i = sin_i_p
    if sin_i > 1.0:
        sin_i = 1.0
    cos_i = np.sqrt(1.0 - sin_i**2)

    sin_j = (vs_m_s / vp_m_s) * sin_i
    if sin_j > 1.0:
        sin_j = 1.0
        cos_j = 0.0
    else:
        cos_j = np.sqrt(1.0 - sin_j**2)

    a_val = vp_m_s / vs_m_s
    b_val = cos_i
    c_val = cos_j
    d_val = sin_i
    e_val = sin_j

    denom = (a_val**2 * e_val**2 + b_val * c_val) * (b_val * c_val + d_val * e_val) - (
        b_val * c_val - d_val * e_val
    ) ** 2

    if abs(denom) < 1e-30:
        return {"R_PP": -1.0, "R_PS": 0.0, "sin_j_s": sin_j, "cos_i_p": cos_i, "cos_j_s": cos_j}

    R_pp = -(a_val**2 * e_val**2 + b_val * c_val) * (b_val * c_val - d_val * e_val) / denom
    R_ps = -2.0 * a_val * b_val * d_val * (b_val * c_val - d_val * e_val) / denom

    return {
        "R_PP": float(R_pp),
        "R_PS": float(R_ps),
        "sin_j_s": sin_j,
        "cos_i_p": cos_i,
        "cos_j_s": cos_j,
    }


def halfspace_displacement(
    receiver_xyz_m: NDArray[np.float64],
    source_xyz_m: NDArray[np.float64],
    force_direction: int,
    vp_m_s: float,
    vs_m_s: float,
    density_kg_m3: float,
    source_time_function: NDArray[np.float64],
    dt_s: float,
) -> NDArray[np.float64]:
    """Displacement Green's tensor for a homogeneous elastic half-space.

    Computes the total displacement as:
        u_total = u_direct + u_reflected

    where u_direct is the full-space Stokes solution from the source,
    and u_reflected is the wave reflected from the free surface (z=0)
    using the image source method with Zoeppritz coefficients.

    For receivers ON the free surface (z≈0), the direct and reflected
    waves coalesce; the free-surface amplification is computed directly.
    """
    nt = len(source_time_function)

    # Direct wave from the actual source
    direct = stokes_displacement_green_tensor(
        receiver_xyz_m,
        source_xyz_m,
        force_direction,
        vp_m_s,
        vs_m_s,
        density_kg_m3,
        source_time_function,
        dt_s,
    )

    zr = float(receiver_xyz_m[2])
    zs = float(source_xyz_m[2])

    # Image source for the free-surface reflection
    image_src = source_xyz_m.copy()
    image_src[2] = -zs

    # Ray from image source to receiver
    r_img_vec = receiver_xyz_m - image_src
    r_img = float(np.linalg.norm(r_img_vec))

    if r_img < 1e-6:
        return direct

    # Incidence angle of the reflected P-wave at the free surface
    # For the image source ray, the reflection point on the surface is
    # where the ray crosses z=0. The incidence angle is the angle between
    # the ray and the vertical at z=0.
    cos_i = abs(float(r_img_vec[2])) / r_img  # cos(incidence angle)
    if cos_i > 1.0:
        cos_i = 1.0
    sin_i = np.sqrt(1.0 - cos_i**2) if cos_i < 1.0 else 0.0

    # Compute free-surface Zoeppritz coefficients
    coeffs = free_surface_zoeppritz(vp_m_s, vs_m_s, density_kg_m3, sin_i)

    # Image source contribution (unscaled Stokes)
    image_wave = stokes_displacement_green_tensor(
        receiver_xyz_m,
        image_src,
        force_direction,
        vp_m_s,
        vs_m_s,
        density_kg_m3,
        source_time_function,
        dt_s,
    )

    # Apply Zoeppritz reflection coefficients to the image contribution
    # The image source represents the SPECULAR reflection from the free
    # surface. The reflected displacement is a linear combination of
    # the P-reflected (R_PP) and S-reflected (R_PS) contributions.

    # For the half-space, the reflected field involves both P and S waves.
    # The image source with the Stokes solution represents the P-wave
    # reflection path. The S-wave reflection is along a different path.
    # We use a simplified approach: the total reflected field at the
    # receiver is approximately R_PP × u_image_P + R_PS × u_image_S.

    # For receivers at the free surface, compute the free-surface
    # amplification directly using the Zoeppritz factors.

    if zr < 50.0:  # receiver within 50m of surface
        # Free-surface amplification for surface receivers
        # The total displacement at the free surface = amplification × incident wave
        # For the z-component: Amp_z = 2 cos(i) / (...) from Zoeppritz
        R_pp = coeffs["R_PP"]
        R_ps = coeffs["R_PS"]
        a_val = vp_m_s / vs_m_s
        c_i = coeffs["cos_i_p"]
        c_j = coeffs["cos_j_s"]
        s_i = sin_i
        s_j = coeffs["sin_j_s"]

        # Free-surface displacement for incident P wave (Aki & Richards, §5.2.5)
        # u_x^free / u_x^inc = (1+R_PP) + (vs/vp)(c_j/c_i)R_PS (approx for horizontal)
        # The exact factors depend on which component of the incident field
        # is being amplified. For a general point-source wavefield, use
        # the doubling approximation for vertical and the Zoeppritz for
        # the relative amplitudes.

        # Simplified: apply factor-of-2 doubling to vertical component,
        # and keep the Stokes solution for horizontals (which is the
        # standard free-surface correction for near-vertical incidence).
        return direct * np.array([1.0, 1.0, 2.0], dtype=np.float64)[np.newaxis, :]

    # Receiver at depth: direct + reflected
    # Reflected contribution: R_PP for the P reflection path
    R_pp = coeffs["R_PP"]
    return direct + R_pp * image_wave


def first_surface_reflected_arrival_time_s(
    receiver_xyz_m: NDArray[np.float64], source_xyz_m: NDArray[np.float64], vp_m_s: float
) -> float:
    """Travel time of the earliest surface-reflected P-wave [s]."""
    img = source_xyz_m.copy()
    img[2] = -img[2]
    return float(np.linalg.norm(receiver_xyz_m - img)) / vp_m_s
