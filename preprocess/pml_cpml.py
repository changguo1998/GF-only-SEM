"""C-PML parameter computation (recursive convolution PML).

Computes per-GLL-node C-PML damping profiles (K, d, alpha) per direction,
PML region classification, and all convolution coefficients needed by the
forward solver.

Reference: Wang et al. (2006), Xie et al. (2014), SPECFEM3D implementation.
See: docs/design/cpml.md for the full mathematical formulation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Constants (SPECFEM3D defaults)
# ---------------------------------------------------------------------------

THETA = 1.0 / 8.0  # Wang et al. (2006) second-order convolution parameter
K_MIN_PML = 1.0
K_MAX_PML = 1.0
NPOWER = 2
R_COEF = 1e-5  # Target reflection coefficient
MIN_DISTANCE = 1e-6  # Singularity avoidance threshold

# PML region codes (matching SPECFEM3D constants.h)
CPML_X_ONLY = 1
CPML_Y_ONLY = 2
CPML_Z_ONLY = 3
CPML_XY_ONLY = 4
CPML_XZ_ONLY = 5
CPML_YZ_ONLY = 6
CPML_XYZ = 7


# ---------------------------------------------------------------------------
# Damping profile
# ---------------------------------------------------------------------------


def pml_damping_profile(
    dist: float | npt.NDArray[np.float64], vp: float | npt.NDArray[np.float64], pml_width: float
) -> npt.NDArray[np.float64]:
    """Compute d_axis damping profile (polynomial grading).

    d = -(NPOWER + 1) * vp * ln(R_coef) / (2 * pml_width) * dist^(1.2 * NPOWER)

    Args:
        dist: Normalized distance into PML [0, 1].
        vp: P-wave velocity at the GLL node (m/s).
        pml_width: Physical width of the PML layer (m).

    Returns:
        Damping coefficient d (positive, represents absorption strength).
    """
    if pml_width <= 0:
        return np.zeros_like(np.asarray(dist, dtype=np.float64))
    exponent = 1.2 * NPOWER
    return -((NPOWER + 1.0) * vp * np.log(R_COEF) / (2.0 * pml_width)) * np.power(dist, exponent)


def compute_pml_profiles(
    gll_coords: npt.NDArray[np.float64],
    is_pml: npt.NDArray[np.bool_],
    pml_regions: npt.NDArray[np.int32],
    domain_bounds: dict[str, float],
    pml_widths: dict[str, float],
    vp: npt.NDArray[np.float64],
    f0_hz: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute K, d, alpha profiles per direction per GLL node.

    Args:
        gll_coords: [n_cell, NGLL, NGLL, NGLL, 3] GLL node coordinates.
        is_pml: [n_cell] boolean PML flag.
        pml_regions: [n_cell] PML region code (0=interior, 1-7=PML region).
        domain_bounds: {xmin, xmax, ymin, ymax, zmin, zmax}.
        pml_widths: {xmin, xmax, ymin, ymax, zmin, zmax} physical widths (m).
        vp: [n_cell, NGLL, NGLL, NGLL] P-wave velocity per GLL node.
        f0_hz: Dominant source frequency (Hz).

    Returns:
        K_store: [n_cell, NGLL^3, 3] K per direction (1.0 = no stretching).
        d_store: [n_cell, NGLL^3, 3] d per direction (0.0 = no damping).
        alpha_store: [n_cell, NGLL^3, 3] alpha per direction (0.0 = no shift).
    """
    n_cell = gll_coords.shape[0]
    NGLL = gll_coords.shape[1]
    n_node = NGLL * NGLL * NGLL

    K_store = np.ones((n_cell, n_node, 3), dtype=np.float64)
    d_store = np.zeros((n_cell, n_node, 3), dtype=np.float64)
    alpha_store = np.zeros((n_cell, n_node, 3), dtype=np.float64)

    alpha_max = [
        np.pi * f0_hz * 0.9,  # x
        np.pi * f0_hz * 1.0,  # y
        np.pi * f0_hz * 1.1,  # z
    ]

    # Face boundaries: (axis, face_key, boundary_value, direction_sign)
    # direction_sign: +1 if PML is on the max side, -1 if on the min side
    faces = [
        (0, "xmin", domain_bounds["xmin"], -1),
        (0, "xmax", domain_bounds["xmax"], +1),
        (1, "ymin", domain_bounds["ymin"], -1),
        (1, "ymax", domain_bounds["ymax"], +1),
        (2, "zmin", domain_bounds["zmin"], -1),
        (2, "zmax", domain_bounds["zmax"], +1),
    ]

    pml_cells = np.where(is_pml)[0]
    for e in pml_cells:
        region = int(pml_regions[e].item()) if e < len(pml_regions) else 0
        coords_flat = gll_coords[e].reshape(-1, 3)  # [n_node, 3]
        vp_flat = vp[e].reshape(-1)  # [n_node]

        for axis, face_key, boundary_val, _sign in faces:
            width = pml_widths.get(face_key, 0.0)
            if width <= 0:
                continue
            # Check if this axis is active for this region
            if not _is_axis_active(region, axis):
                continue

            # Distance from PML interior boundary to the physical boundary
            # dist = |coord - boundary| / width  ∈ [0, 1]
            coord_axis = coords_flat[:, axis]
            dist = np.abs(coord_axis - boundary_val) / width
            dist = np.clip(dist, 0.0, 1.0)

            K_val = K_MIN_PML + (K_MAX_PML - 1.0) * dist
            d_val = pml_damping_profile(dist, vp_flat, width)
            alpha_val = alpha_max[axis] * (1.0 - dist)

            # Clamp: K >= 1, d >= 0
            mask = (K_val < 1.0) | (d_val < 0.0)
            K_val[mask] = 1.0
            d_val[mask] = 0.0
            alpha_val = np.maximum(alpha_val, 0.0)

            K_store[e, :, axis] = K_val
            d_store[e, :, axis] = d_val
            alpha_store[e, :, axis] = alpha_val

    return K_store, d_store, alpha_store


def _is_axis_active(region: int, axis: int) -> bool:
    """Check if a PML direction is active for a given region code."""
    if region == CPML_XYZ:
        return True
    if region == CPML_X_ONLY and axis == 0:
        return True
    if region == CPML_Y_ONLY and axis == 1:
        return True
    if region == CPML_Z_ONLY and axis == 2:
        return True
    if region == CPML_XY_ONLY and axis in (0, 1):
        return True
    if region == CPML_XZ_ONLY and axis in (0, 2):
        return True
    if region == CPML_YZ_ONLY and axis in (1, 2):
        return True
    return False


# ---------------------------------------------------------------------------
# PML region classification
# ---------------------------------------------------------------------------


def classify_pml_regions(
    gll_coords: npt.NDArray[np.float64],
    is_pml: npt.NDArray[np.bool_],
    domain_bounds: dict[str, float],
    pml_widths: dict[str, float],
    tol: float = 1e-6,
) -> npt.NDArray[np.int32]:
    """Classify each PML element by which faces it touches.

    Args:
        gll_coords: [n_cell, NGLL, NGLL, NGLL, 3] GLL node coordinates.
        is_pml: [n_cell] boolean PML flag.
        domain_bounds: {xmin, xmax, ymin, ymax, zmin, zmax}.
        pml_widths: {xmin, xmax, ymin, ymax, zmin, zmax} physical widths (m).
        tol: Relative tolerance for boundary detection.

    Returns:
        regions: [n_cell] int32, 0=interior, 1-7=PML region code.
    """
    n_cell = gll_coords.shape[0]
    regions = np.zeros(n_cell, dtype=np.int32)

    # For each PML element, check which PML faces it overlaps
    pml_cells = np.where(is_pml)[0]
    for e in pml_cells:
        center = gll_coords[e].mean(axis=(0, 1, 2))  # [3]

        active_x = False
        active_y = False
        active_z = False

        # Check x faces
        if pml_widths.get("xmin", 0) > 0:
            x_start = domain_bounds["xmin"] + pml_widths["xmin"]
            if center[0] < x_start + tol * pml_widths["xmin"]:
                active_x = True
        if pml_widths.get("xmax", 0) > 0:
            x_start = domain_bounds["xmax"] - pml_widths["xmax"]
            if center[0] > x_start - tol * pml_widths["xmax"]:
                active_x = True

        # Check y faces
        if pml_widths.get("ymin", 0) > 0:
            y_start = domain_bounds["ymin"] + pml_widths["ymin"]
            if center[1] < y_start + tol * pml_widths["ymin"]:
                active_y = True
        if pml_widths.get("ymax", 0) > 0:
            y_start = domain_bounds["ymax"] - pml_widths["ymax"]
            if center[1] > y_start - tol * pml_widths["ymax"]:
                active_y = True

        # Check z faces
        if pml_widths.get("zmin", 0) > 0:
            z_start = domain_bounds["zmin"] + pml_widths["zmin"]
            if center[2] < z_start + tol * pml_widths["zmin"]:
                active_z = True
        if pml_widths.get("zmax", 0) > 0:
            z_start = domain_bounds["zmax"] - pml_widths["zmax"]
            if center[2] > z_start - tol * pml_widths["zmax"]:
                active_z = True

        if active_x and active_y and active_z:
            regions[e] = CPML_XYZ
        elif active_x and active_y:
            regions[e] = CPML_XY_ONLY
        elif active_x and active_z:
            regions[e] = CPML_XZ_ONLY
        elif active_y and active_z:
            regions[e] = CPML_YZ_ONLY
        elif active_x:
            regions[e] = CPML_X_ONLY
        elif active_y:
            regions[e] = CPML_Y_ONLY
        elif active_z:
            regions[e] = CPML_Z_ONLY
        else:
            # PML element but no face detected (shouldn't happen)
            regions[e] = CPML_XYZ  # fallback: treat as all-directions

    return regions


# ---------------------------------------------------------------------------
# Convolution coefficients
# ---------------------------------------------------------------------------


def compute_convolution_coef(
    b: float | npt.NDArray[np.float64], dt: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute recursive convolution coefficients (second-order, Wang 2006).

    coef0 = exp(-b * dt)
    coef1 = (1 - exp(-b*dt/2)) / b
    coef2 = coef1 * exp(-b*dt/2)

    For small |b|, uses Taylor expansion to avoid division by zero.

    Args:
        b: Damping parameter (alpha or beta).
        dt: Timestep (s).

    Returns:
        coef0, coef1, coef2: Recursive convolution coefficients.
    """
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    temp = np.exp(-0.5 * b * dt)
    coef0 = temp * temp

    # For |b| >= MIN_DISTANCE: exact formula
    large = np.abs(b) >= MIN_DISTANCE
    safe_b = np.where(large, b, 1.0)
    coef1 = np.where(large, (1.0 - temp) / safe_b, 0.0)
    coef2 = coef1 * temp

    # For |b| < MIN_DISTANCE: Taylor expansion
    small = ~large
    if np.any(small):
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        b_s = b[small]
        b2 = b_s * b_s
        b3 = b2 * b_s
        coef1[small] = dt * 0.5 + (
            -0.125 * dt2 * b_s + (1.0 / 48.0) * dt3 * b2 - (1.0 / 384.0) * dt4 * b3
        )
        coef2[small] = dt * 0.5 + (
            -0.375 * dt2 * b_s + (7.0 / 48.0) * dt3 * b2 - (5.0 / 128.0) * dt4 * b3
        )

    return coef0, coef1, coef2


def compute_coef_alpha_beta(
    K_store: npt.NDArray[np.float64],
    d_store: npt.NDArray[np.float64],
    alpha_store: npt.NDArray[np.float64],
    dt: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute alpha and beta convolution coefficients (9 each per GLL node).

    Layout: [n_cell, n_node, 9] where indices 0-2=x, 3-5=y, 6-8=z
    and within each group: (coef0, coef1, coef2).

    Args:
        K_store: [n_cell, n_node, 3]
        d_store: [n_cell, n_node, 3]
        alpha_store: [n_cell, n_node, 3]
        dt: Timestep (s).

    Returns:
        coef_alpha: [n_cell, n_node, 9]
        coef_beta: [n_cell, n_node, 9]
    """
    n_cell, n_node, _ = K_store.shape
    coef_alpha = np.zeros((n_cell, n_node, 9), dtype=np.float64)
    coef_beta = np.zeros((n_cell, n_node, 9), dtype=np.float64)

    for axis in range(3):
        c0, c1, c2 = compute_convolution_coef(alpha_store[:, :, axis], dt)
        coef_alpha[:, :, axis * 3 + 0] = c0
        coef_alpha[:, :, axis * 3 + 1] = c1
        coef_alpha[:, :, axis * 3 + 2] = c2

        beta = alpha_store[:, :, axis] + d_store[:, :, axis] / K_store[:, :, axis]
        c0b, c1b, c2b = compute_convolution_coef(beta, dt)
        coef_beta[:, :, axis * 3 + 0] = c0b
        coef_beta[:, :, axis * 3 + 1] = c1b
        coef_beta[:, :, axis * 3 + 2] = c2b

    return coef_alpha, coef_beta


# ---------------------------------------------------------------------------
# Accel-update coefficients A1..A5 (Xie et al. 2014)
# ---------------------------------------------------------------------------


def compute_abar_coefficients(
    K_store: npt.NDArray[np.float64],
    d_store: npt.NDArray[np.float64],
    alpha_store: npt.NDArray[np.float64],
    pml_regions: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Compute accel-update coefficients A_bar_1..A_bar_5.

    Args:
        K_store: [n_cell, n_node, 3]
        d_store: [n_cell, n_node, 3]
        alpha_store: [n_cell, n_node, 3]
        pml_regions: [n_cell] int32

    Returns:
        coef_abar: [n_cell, n_node, 5] (A1..A5)
    """
    n_cell, n_node, _ = K_store.shape
    coef_abar = np.zeros((n_cell, n_node, 5), dtype=np.float64)

    beta = alpha_store + d_store / np.maximum(K_store, 1.0)  # [n_cell, n_node, 3]

    for e in range(n_cell):
        region = int(pml_regions[e].item()) if e < len(pml_regions) else 0
        if region == 0:
            continue

        kx, ky, kz = K_store[e, :, 0], K_store[e, :, 1], K_store[e, :, 2]
        dx, dy, dz = d_store[e, :, 0], d_store[e, :, 1], d_store[e, :, 2]
        ax, ay, az = alpha_store[e, :, 0], alpha_store[e, :, 1], alpha_store[e, :, 2]
        bx, by, bz = beta[e, :, 0], beta[e, :, 1], beta[e, :, 2]

        A1, A2, A3, A4, A5 = _l_parameter(region, kx, dx, ax, ky, dy, ay, kz, dz, az)
        coef_abar[e, :, 0] = A1
        coef_abar[e, :, 1] = A2
        coef_abar[e, :, 2] = A3
        coef_abar[e, :, 3] = A4
        coef_abar[e, :, 4] = A5

    return coef_abar


def _l_parameter(
    region: int,
    kx: np.ndarray,
    dx: np.ndarray,
    ax: np.ndarray,
    ky: np.ndarray,
    dy: np.ndarray,
    ay: np.ndarray,
    kz: np.ndarray,
    dz: np.ndarray,
    az: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute A_bar_1..A_bar_5 for a single element (all GLL nodes).

    Implements l_parameter_computation from SPECFEM3D (Xie et al. 2014).
    """
    bx = ax + dx / np.maximum(kx, 1.0)
    by = ay + dy / np.maximum(ky, 1.0)
    bz = az + dz / np.maximum(kz, 1.0)

    if region == CPML_XYZ:
        A0 = kx * ky * kz
        A1 = A0 * (bx + by + bz - ax - ay - az)
        A2 = (
            A0 * (bx - ax) * (by - ay - ax)
            + A0 * (by - ay) * (bz - az - ay)
            + A0 * (bz - az) * (bx - ax - az)
        )
        # Need alpha_x != alpha_y != alpha_z for the partial fraction
        # Use safe division (clamp denominators)
        dxy = np.where(np.abs(ax - ay) < MIN_DISTANCE, MIN_DISTANCE, ax - ay)
        dxz = np.where(np.abs(ax - az) < MIN_DISTANCE, MIN_DISTANCE, ax - az)
        dyz = np.where(np.abs(ay - az) < MIN_DISTANCE, MIN_DISTANCE, ay - az)
        dyx = np.where(np.abs(ay - ax) < MIN_DISTANCE, MIN_DISTANCE, ay - ax)
        dzx = np.where(np.abs(az - ax) < MIN_DISTANCE, MIN_DISTANCE, az - ax)
        dzy = np.where(np.abs(az - ay) < MIN_DISTANCE, MIN_DISTANCE, az - ay)

        A3 = A0 * ax**2 * (bx - ax) * (by - ax) * (bz - ax) / (dyx * dzx)
        A4 = A0 * ay**2 * (bx - ay) * (by - ay) * (bz - ay) / (dxy * dzy)
        A5 = A0 * az**2 * (bx - az) * (by - az) * (bz - az) / (dyz * dzx)

    elif region == CPML_XY_ONLY:
        A0 = kx * ky
        A1 = A0 * (bx + by - ax - ay)
        A2 = A0 * (bx - ax) * (by - ay - ax) - A0 * (by - ay) * ay
        d = np.where(np.abs(ax - ay) < MIN_DISTANCE, MIN_DISTANCE, ax - ay)
        A3 = (
            A0
            * ax**2
            * (bx - ax)
            * (by - ax)
            / (np.where(np.abs(ay - ax) < MIN_DISTANCE, MIN_DISTANCE, ay - ax))
        )
        A4 = A0 * ay**2 * (bx - ay) * (by - ay) / d
        A5 = np.zeros_like(A1)

    elif region == CPML_XZ_ONLY:
        A0 = kx * kz
        A1 = A0 * (bx + bz - ax - az)
        A2 = A0 * (bx - ax) * (-ax) + A0 * (bz - az) * (bx - ax - az)
        d = np.where(np.abs(ax - az) < MIN_DISTANCE, MIN_DISTANCE, ax - az)
        A3 = (
            A0
            * ax**2
            * (bx - ax)
            * (bz - ax)
            / (np.where(np.abs(az - ax) < MIN_DISTANCE, MIN_DISTANCE, az - ax))
        )
        A4 = np.zeros_like(A1)
        A5 = A0 * az**2 * (bx - az) * (bz - az) / d

    elif region == CPML_YZ_ONLY:
        A0 = ky * kz
        A1 = A0 * (by + bz - ay - az)
        A2 = A0 * (by - ay) * (bz - az - ay) - A0 * (bz - az) * az
        d = np.where(np.abs(ay - az) < MIN_DISTANCE, MIN_DISTANCE, ay - az)
        A3 = np.zeros_like(A1)
        A4 = (
            A0
            * ay**2
            * (by - ay)
            * (bz - ay)
            / (np.where(np.abs(az - ay) < MIN_DISTANCE, MIN_DISTANCE, az - ay))
        )
        A5 = A0 * az**2 * (by - az) * (bz - az) / d

    elif region == CPML_X_ONLY:
        A0 = kx
        diff = bx - ax
        A1 = A0 * diff
        A2 = -A0 * ax * diff
        A3 = A0 * ax**2 * diff
        A4 = np.zeros_like(A1)
        A5 = np.zeros_like(A1)

    elif region == CPML_Y_ONLY:
        A0 = ky
        diff = by - ay
        A1 = A0 * diff
        A2 = -A0 * ay * diff
        A3 = np.zeros_like(A1)
        A4 = A0 * ay**2 * diff
        A5 = np.zeros_like(A1)

    elif region == CPML_Z_ONLY:
        A0 = kz
        diff = bz - az
        A1 = A0 * diff
        A2 = -A0 * az * diff
        A3 = np.zeros_like(A1)
        A4 = np.zeros_like(A1)
        A5 = A0 * az**2 * diff

    else:
        A1 = np.zeros_like(kx)
        A2 = np.zeros_like(kx)
        A3 = np.zeros_like(kx)
        A4 = np.zeros_like(kx)
        A5 = np.zeros_like(kx)

    return A1, A2, A3, A4, A5


# ---------------------------------------------------------------------------
# Strain-update coefficients A6..A23
# ---------------------------------------------------------------------------


def compute_strain_coefficients(
    K_store: npt.NDArray[np.float64],
    d_store: npt.NDArray[np.float64],
    alpha_store: npt.NDArray[np.float64],
    pml_regions: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Compute strain-update coefficients A6..A23 (18 values).

    These are used inside the element kernel to modify displacement
    gradients for PML elements via the C-PML convolution.

    Layout: [n_cell, n_node, 18]
    - [0..3]  = A6..A9   (for du/dx, via lijk with index 231)
    - [4..7]  = A10..A13 (for du/dy, via lijk with index 132)
    - [8..11] = A14..A17 (for du/dz, via lijk with index 123)
    - [12..13]= A18..A19 (via lx_parameter)
    - [14..15]= A20..A21 (via ly_parameter)
    - [16..17]= A22..A23 (via lz_parameter)

    Args:
        K_store, d_store, alpha_store: [n_cell, n_node, 3]
        pml_regions: [n_cell] int32

    Returns:
        coef_strain: [n_cell, n_node, 18]
    """
    n_cell, n_node, _ = K_store.shape
    coef_strain = np.zeros((n_cell, n_node, 18), dtype=np.float64)

    for e in range(n_cell):
        region = int(pml_regions[e].item()) if e < len(pml_regions) else 0
        if region == 0:
            continue

        kx, ky, kz = K_store[e, :, 0], K_store[e, :, 1], K_store[e, :, 2]
        dx, dy, dz = d_store[e, :, 0], d_store[e, :, 1], d_store[e, :, 2]
        ax, ay, az = alpha_store[e, :, 0], alpha_store[e, :, 1], alpha_store[e, :, 2]

        # A6..A9: lijk(z, y, x) = index 231
        A0, A6, A7, A8 = _lijk_parameter(region, kz, dz, az, ky, dy, ay, kx, dx, ax)
        coef_strain[e, :, 0] = A0
        coef_strain[e, :, 1] = A6
        coef_strain[e, :, 2] = A7
        coef_strain[e, :, 3] = A8

        # A10..A13: lijk(x, z, y) = index 132
        A0, A10, A11, A12 = _lijk_parameter(region, kx, dx, ax, kz, dz, az, ky, dy, ay)
        coef_strain[e, :, 4] = A0
        coef_strain[e, :, 5] = A10
        coef_strain[e, :, 6] = A11
        coef_strain[e, :, 7] = A12

        # A14..A17: lijk(x, y, z) = index 123
        A0, A14, A15, A16 = _lijk_parameter(region, kx, dx, ax, ky, dy, ay, kz, dz, az)
        coef_strain[e, :, 8] = A0
        coef_strain[e, :, 9] = A14
        coef_strain[e, :, 10] = A15
        coef_strain[e, :, 11] = A16

        # A18..A19: lx_parameter(x)
        A18, A19 = _lx_parameter(region, kx, dx, ax)
        coef_strain[e, :, 12] = A18
        coef_strain[e, :, 13] = A19

        # A20..A21: ly_parameter(y)
        A20, A21 = _ly_parameter(region, ky, dy, ay)
        coef_strain[e, :, 14] = A20
        coef_strain[e, :, 15] = A21

        # A22..A23: lz_parameter(z)
        A22, A23 = _lz_parameter(region, kz, dz, az)
        coef_strain[e, :, 16] = A22
        coef_strain[e, :, 17] = A23

    return coef_strain


def _lijk_parameter(
    region: int,
    kx: np.ndarray,
    dx: np.ndarray,
    ax: np.ndarray,
    ky: np.ndarray,
    dy: np.ndarray,
    ay: np.ndarray,
    kz: np.ndarray,
    dz: np.ndarray,
    az: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute A0, A1, A2, A3 for strain coefficients (lijk_parameter_computation).

    The index_ijk parameter (123/132/231) determines which axes are mapped
    to x/y/z in the formula. The caller handles this by permuting the
    arguments, so this function always uses the standard 123 formula.
    """
    bx = ax + dx / np.maximum(kx, 1.0)
    by = ay + dy / np.maximum(ky, 1.0)
    bz = az + dz / np.maximum(kz, 1.0)

    if region == CPML_XYZ:
        A0 = kx * ky / np.maximum(kz, 1.0)
        dxy = np.where(np.abs(ax - ay) < MIN_DISTANCE, MIN_DISTANCE, ax - ay)
        dxbz = np.where(np.abs(ax - bz) < MIN_DISTANCE, MIN_DISTANCE, ax - bz)
        dybz = np.where(np.abs(ay - bz) < MIN_DISTANCE, MIN_DISTANCE, ay - bz)
        dyx = np.where(np.abs(ay - ax) < MIN_DISTANCE, MIN_DISTANCE, ay - ax)
        dzbx = np.where(np.abs(bz - ax) < MIN_DISTANCE, MIN_DISTANCE, bz - ax)
        dzby = np.where(np.abs(bz - ay) < MIN_DISTANCE, MIN_DISTANCE, bz - ay)

        A1 = -A0 * (ax - az) * (ax - bx) * (ax - by) / (dxy * dxbz)
        A2 = -A0 * (ay - az) * (ay - bx) * (ay - by) / (dyx * dybz)
        A3 = -A0 * (bz - az) * (bz - bx) * (bz - by) / (dzbx * dzby)

    elif region in (CPML_X_ONLY,):
        A0 = kx
        A1 = -A0 * (ax - bx)
        A2 = np.zeros_like(A0)
        A3 = np.zeros_like(A0)

    elif region in (CPML_Y_ONLY,):
        A0 = ky
        A1 = np.zeros_like(A0)
        A2 = -A0 * (ay - by)
        A3 = np.zeros_like(A0)

    elif region in (CPML_Z_ONLY,):
        A0 = 1.0 / np.maximum(kz, 1.0)
        A1 = np.zeros_like(A0)
        A2 = np.zeros_like(A0)
        A3 = -A0 * (bz - az)

    elif region == CPML_XY_ONLY:
        A0 = kx * ky
        d = np.where(np.abs(ax - ay) < MIN_DISTANCE, MIN_DISTANCE, ax - ay)
        dyx = np.where(np.abs(ay - ax) < MIN_DISTANCE, MIN_DISTANCE, ay - ax)
        A1 = -A0 * (ax - bx) * (ax - by) / d
        A2 = -A0 * (ay - bx) * (ay - by) / dyx
        A3 = np.zeros_like(A0)

    elif region == CPML_XZ_ONLY:
        A0 = kx / np.maximum(kz, 1.0)
        d = np.where(np.abs(ax - bz) < MIN_DISTANCE, MIN_DISTANCE, ax - bz)
        dzbx = np.where(np.abs(bz - ax) < MIN_DISTANCE, MIN_DISTANCE, bz - ax)
        A1 = -A0 * (ax - az) * (ax - bx) / d
        A2 = np.zeros_like(A0)
        A3 = -A0 * (bz - az) * (bz - bx) / dzbx

    elif region == CPML_YZ_ONLY:
        A0 = ky / np.maximum(kz, 1.0)
        d = np.where(np.abs(ay - bz) < MIN_DISTANCE, MIN_DISTANCE, ay - bz)
        dzby = np.where(np.abs(bz - ay) < MIN_DISTANCE, MIN_DISTANCE, bz - ay)
        A1 = np.zeros_like(A0)
        A2 = -A0 * (ay - az) * (ay - by) / d
        A3 = -A0 * (bz - az) * (bz - by) / dzby

    else:
        A0 = np.zeros_like(kx)
        A1 = np.zeros_like(kx)
        A2 = np.zeros_like(kx)
        A3 = np.zeros_like(kx)

    return A0, A1, A2, A3


def _lx_parameter(
    region: int, kx: np.ndarray, dx: np.ndarray, ax: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute A0, A1 for lx_parameter (x-direction strain)."""
    bx = ax + dx / np.maximum(kx, 1.0)
    if region in (CPML_XYZ, CPML_XZ_ONLY, CPML_XY_ONLY, CPML_X_ONLY):
        A0 = kx
        A1 = -A0 * (ax - bx)
    elif region == CPML_YZ_ONLY:
        A0 = np.ones_like(kx)
        A1 = np.zeros_like(kx)
    else:
        A0 = np.ones_like(kx)
        A1 = np.zeros_like(kx)
    return A0, A1


def _ly_parameter(
    region: int, ky: np.ndarray, dy: np.ndarray, ay: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute A0, A1 for ly_parameter (y-direction strain)."""
    by = ay + dy / np.maximum(ky, 1.0)
    if region in (CPML_XYZ, CPML_XY_ONLY, CPML_YZ_ONLY, CPML_Y_ONLY):
        A0 = ky
        A1 = -A0 * (ay - by)
    elif region == CPML_XZ_ONLY:
        A0 = np.ones_like(ky)
        A1 = np.zeros_like(ky)
    else:
        A0 = np.ones_like(ky)
        A1 = np.zeros_like(ky)
    return A0, A1


def _lz_parameter(
    region: int, kz: np.ndarray, dz: np.ndarray, az: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute A0, A1 for lz_parameter (z-direction strain)."""
    bz = az + dz / np.maximum(kz, 1.0)
    if region in (CPML_XYZ, CPML_XZ_ONLY, CPML_YZ_ONLY, CPML_Z_ONLY):
        A0 = kz
        A1 = -A0 * (az - bz)
    elif region == CPML_XY_ONLY:
        A0 = np.ones_like(kz)
        A1 = np.zeros_like(kz)
    else:
        A0 = np.ones_like(kz)
        A1 = np.zeros_like(kz)
    return A0, A1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_cpml_parameters(
    gll_coords: npt.NDArray[np.float64],
    is_pml: npt.NDArray[np.bool_],
    domain_bounds: dict[str, float],
    pml_widths: dict[str, float],
    vp: npt.NDArray[np.float64],
    f0_hz: float,
    dt: float,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int32]]:
    """Compute all C-PML parameters for the mesh.

    This is the main entry point called by the preprocessor.

    Args:
        gll_coords: [n_cell, NGLL, NGLL, NGLL, 3] GLL node coordinates (m).
        is_pml: [n_cell] boolean PML flag.
        domain_bounds: {xmin, xmax, ymin, ymax, zmin, zmax} (m).
        pml_widths: {xmin, xmax, ymin, ymax, zmin, zmax} physical widths (m).
        vp: [n_cell, NGLL, NGLL, NGLL] P-wave velocity per GLL node (m/s).
        f0_hz: Dominant source frequency (Hz).
        dt: Solver timestep (s).

    Returns:
        Dict with keys:
            pml_region: [n_cell] int32 (0=interior, 1-7=PML region)
            pml_K: [n_cell, NGLL^3, 3] float64
            pml_d: [n_cell, NGLL^3, 3] float64
            pml_alpha: [n_cell, NGLL^3, 3] float64
            pml_coef_alpha: [n_cell, NGLL^3, 9] float64
            pml_coef_beta: [n_cell, NGLL^3, 9] float64
            pml_coef_abar: [n_cell, NGLL^3, 5] float64
            pml_coef_strain: [n_cell, NGLL^3, 18] float64
    """
    # Step 1: Classify PML regions
    pml_regions = classify_pml_regions(gll_coords, is_pml, domain_bounds, pml_widths)

    # Step 2: Compute K, d, alpha profiles
    K_store, d_store, alpha_store = compute_pml_profiles(
        gll_coords, is_pml, pml_regions, domain_bounds, pml_widths, vp, f0_hz
    )

    # Step 3: Compute convolution coefficients
    coef_alpha, coef_beta = compute_coef_alpha_beta(K_store, d_store, alpha_store, dt)

    # Step 4: Compute accel-update coefficients
    coef_abar = compute_abar_coefficients(K_store, d_store, alpha_store, pml_regions)

    # Step 5: Compute strain-update coefficients
    coef_strain = compute_strain_coefficients(K_store, d_store, alpha_store, pml_regions)

    return {
        "pml_region": pml_regions,
        "pml_K": K_store,
        "pml_d": d_store,
        "pml_alpha": alpha_store,
        "pml_coef_alpha": coef_alpha,
        "pml_coef_beta": coef_beta,
        "pml_coef_abar": coef_abar,
        "pml_coef_strain": coef_strain,
    }
