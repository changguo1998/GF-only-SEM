"""C-PML parameter computation (recursive convolution PML).

Computes per-GLL-node C-PML damping profiles (K, d, alpha) per direction,
PML region classification, and all convolution coefficients needed by the
forward solver.

Reference: Wang et al. (2006), Xie et al. (2014), SPECFEM3D implementation.
See: docs/design/cpml.md for the full mathematical formulation.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Constants (SPECFEM3D defaults)
# ---------------------------------------------------------------------------

THETA = 1.0 / 8.0  # Wang et al. (2006) second-order convolution parameter
K_MIN_PML = 1.0
# SPECFEM3D reference implementation uses no coordinate stretching.
K_MAX_PML = 1.0
NPOWER = 2
R_COEF = 1e-5  # Target reflection coefficient
# SPECFEM permits alpha=0 at the physical PML boundary. Parameter separation
# handles coincident partial-fraction poles; do not move the boundary inward.
# Fallback denominator for partial-fraction formulas — SPECFEM3D's parameter
# separation in compute_pml_profiles() should prevent this from being needed.
MIN_DISTANCE = 1e-12
# Factor for min_distance_between_CPML_parameter computation (SPECFEM3D: 1/8).
MIN_DISTANCE_FACTOR = 1.0 / 8.0

# PML region codes (matching SPECFEM3D constants.h)
CPML_X_ONLY = 1
CPML_Y_ONLY = 2
CPML_Z_ONLY = 3
CPML_XY_ONLY = 4
CPML_XZ_ONLY = 5
CPML_YZ_ONLY = 6
CPML_XYZ = 7

_REGION_TO_AXES = {
    CPML_X_ONLY: frozenset({0}),
    CPML_Y_ONLY: frozenset({1}),
    CPML_Z_ONLY: frozenset({2}),
    CPML_XY_ONLY: frozenset({0, 1}),
    CPML_XZ_ONLY: frozenset({0, 2}),
    CPML_YZ_ONLY: frozenset({1, 2}),
    CPML_XYZ: frozenset({0, 1, 2}),
}
_AXES_TO_REGION = {axes: region for region, axes in _REGION_TO_AXES.items()}


def _permute_region(region: int, original_axis_for_local: tuple[int, int, int]) -> int:
    """Map a physical PML region into a permuted lijk coordinate system."""
    active_original_axes = _REGION_TO_AXES.get(region, frozenset())
    active_local_axes = frozenset(
        local_axis
        for local_axis, original_axis in enumerate(original_axis_for_local)
        if original_axis in active_original_axes
    )
    return _AXES_TO_REGION.get(active_local_axes, 0)


def _get_region(pml_regions: npt.NDArray[np.int32], e: int | np.integer) -> int:
    """Safely get PML region code for element e, defaulting to 0 (interior)."""
    try:
        return int(pml_regions[e])
    except (IndexError, ValueError):
        return 0


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
        region = _get_region(pml_regions, e)
        coords_flat = gll_coords[e].reshape(-1, 3)  # [n_node, 3]
        # SPECFEM3D uses the global maximum P-wave speed for all PML profiles.
        vp_flat = np.full(n_node, float(np.max(vp)), dtype=np.float64)

        for axis, face_key, boundary_val, direction_sign in faces:
            width = pml_widths.get(face_key, 0.0)
            if width <= 0:
                continue
            # Check if this axis is active for this region
            if not _is_axis_active(region, axis):
                continue

            center = float(np.mean(coords_flat[:, axis]))
            tolerance = 1.0e-6 * width
            if direction_sign < 0 and center >= boundary_val + width + tolerance:
                continue
            if direction_sign > 0 and center <= boundary_val - width - tolerance:
                continue

            # Normalized depth from the PML interior interface toward the boundary.
            # dist=0 at the interior interface and dist=1 at the physical boundary.
            coord_axis = coords_flat[:, axis]
            dist = 1.0 - np.abs(coord_axis - boundary_val) / width
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

    # --- SPECFEM3D: robust parameter separation for PML damping parameters ---
    # After computing initial profiles, adjust alpha/d values at nodes where the
    # partial-fraction denominators (alpha_x - alpha_y, alpha_x - beta_z, etc.)
    # would be too small, causing coefficient explosion in _l_parameter() and
    # _lijk_parameter(). This follows pml_set_local_dampingcoeff.f90 lines 1378-1833.
    _separate_pml_parameters(gll_coords, pml_widths, pml_regions, K_store, d_store, alpha_store)

    return K_store, d_store, alpha_store


# ---------------------------------------------------------------------------
# SPECFEM3D PML parameter separation (pml_set_local_dampingcoeff.f90:1378-1833)
# ---------------------------------------------------------------------------


def _separate_pml_parameters(
    gll_coords: npt.NDArray[np.float64],
    pml_widths: dict[str, float],
    pml_regions: npt.NDArray[np.int32],
    K_store: npt.NDArray[np.float64],
    d_store: npt.NDArray[np.float64],
    alpha_store: npt.NDArray[np.float64],
) -> None:
    """Adjust alpha/d values to prevent partial-fraction denominator collapse.

    Matches SPECFEM3D's pml_set_local_dampingcoeff.f90 separation logic
    for XY, XZ, YZ and XYZ CPML regions.

    Modifies alpha_store and d_store in-place.
    """
    n_cell, n_node, _ = K_store.shape
    # Compute min_distance_between_CPML_parameter (lines 1378-1444)
    distance_min = np.inf
    pml_cells = np.where(
        np.any(K_store[..., 0] > 1.0, axis=1)
        | np.any(d_store[..., 0] > 0.0, axis=1)
        | np.any(d_store[..., 1] > 0.0, axis=1)
        | np.any(d_store[..., 2] > 0.0, axis=1)
    )[0]
    if len(pml_cells) == 0:
        pml_cells = np.arange(n_cell)
    for e in pml_cells:
        coords = gll_coords[e]  # [NGLL, NGLL, NGLL, 3]
        d2_x = np.sum((coords[1:, :, :] - coords[:-1, :, :]) ** 2, axis=-1)
        if np.any(d2_x > 0):
            distance_min = min(distance_min, float(np.min(d2_x[d2_x > 0])))
        d2_y = np.sum((coords[:, 1:, :] - coords[:, :-1, :]) ** 2, axis=-1)
        if np.any(d2_y > 0):
            distance_min = min(distance_min, float(np.min(d2_y[d2_y > 0])))
        d2_z = np.sum((coords[:, :, 1:] - coords[:, :, :-1]) ** 2, axis=-1)
        if np.any(d2_z > 0):
            distance_min = min(distance_min, float(np.min(d2_z[d2_z > 0])))
    distance_min = np.sqrt(distance_min)
    if distance_min <= 0.0 or not np.isfinite(distance_min):
        warnings.warn("Cannot compute min GLL distance for PML separation; skipping.")
        return

    # SPECFEM3D derives the separation scale from ALPHA_MAX_PML_x, the
    # smallest directional alpha maximum (0.9 * pi * f0).
    alpha_max_pml = float(np.max(alpha_store[..., 0]))
    if alpha_max_pml <= 0:
        alpha_max_pml = np.pi * 10.0 * 0.9

    pml_w = max(
        pml_widths.get("xmin", 0.0),
        pml_widths.get("xmax", 0.0),
        pml_widths.get("ymin", 0.0),
        pml_widths.get("ymax", 0.0),
        pml_widths.get("zmin", 0.0),
        pml_widths.get("zmax", 0.0),
    )
    if pml_w <= 0:
        return

    min_sep = alpha_max_pml * distance_min / pml_w * MIN_DISTANCE_FACTOR
    const_sep_two = min_sep * 2.0
    const_sep_four = min_sep * 4.0

    if min_sep <= 0:
        return

    # Per-element, per-node separation (lines 1447-1833)
    for e in pml_cells:
        region = int(pml_regions[e])
        if region == 0:
            continue

        for n in range(n_node):
            kx = float(K_store[e, n, 0])
            ky = float(K_store[e, n, 1])
            kz = float(K_store[e, n, 2])
            dx = float(d_store[e, n, 0])
            dy = float(d_store[e, n, 1])
            dz = float(d_store[e, n, 2])
            ax = float(alpha_store[e, n, 0])
            ay = float(alpha_store[e, n, 1])
            az = float(alpha_store[e, n, 2])

            if dx == 0.0 and dy == 0.0 and dz == 0.0:
                continue

            if region == CPML_XY_ONLY:
                ax, ay, bx, by, dx, dy = _separate_xy_node(
                    ax, ay, kx, ky, dx, dy, min_sep, const_sep_two, const_sep_four
                )
            elif region == CPML_XZ_ONLY:
                ax, az, bx, bz, dx, dz = _separate_xz_node(
                    ax, az, kx, kz, dx, dz, min_sep, const_sep_two, const_sep_four
                )
            elif region == CPML_YZ_ONLY:
                ay, az, by, bz, dy, dz = _separate_yz_node(
                    ay, az, ky, kz, dy, dz, min_sep, const_sep_two, const_sep_four
                )
            elif region == CPML_XYZ:
                ax, ay, az, bx, by, bz, dx, dy, dz = _separate_xyz_node(
                    ax, ay, az, kx, ky, kz, dx, dy, dz, min_sep, const_sep_two, const_sep_four
                )

            alpha_store[e, n, 0] = ax
            alpha_store[e, n, 1] = ay
            alpha_store[e, n, 2] = az
            d_store[e, n, 0] = dx
            d_store[e, n, 1] = dy
            d_store[e, n, 2] = dz


def _separate_two_changeable(a: float, b: float, sep2: float) -> tuple[float, float]:
    """Adjust both values to ensure |a - b| >= sep2. Keeps larger value fixed."""
    if a >= b:
        return b + sep2, b
    else:
        return a, a + sep2


def _separate_one_changeable(a: float, b: float, sep2: float, sep4: float) -> tuple[float, float]:
    """Adjust value a (only) to be safely away from fixed b."""
    if a >= b:
        return b + sep2, b
    else:
        return b + sep4, b


def _separate_xy_node(
    ax: float,
    ay: float,
    kx: float,
    ky: float,
    dx: float,
    dy: float,
    min_sep: float,
    sep2: float,
    sep4: float,
) -> tuple[float, float, float, float, float, float]:
    """SPECFEM3D XY_ONLY separation (lines 1453-1499)."""
    if abs(ax - ay) < min_sep:
        ax, ay = _separate_two_changeable(ax, ay, sep2)

    bx = ax + dx / max(kx, 1.0)
    by = ay + dy / max(ky, 1.0)

    if abs(bx - ay) < min_sep:
        bx, ay = _separate_one_changeable(bx, ay, sep2, sep4)
    if abs(by - ax) < min_sep:
        by, ax = _separate_one_changeable(by, ax, sep2, sep4)

    if abs(ax - ay) < min_sep or abs(bx - ay) < min_sep or abs(by - ax) < min_sep:
        raise ValueError(f"CPML XY parameter separation failed: ax={ax:.6e} ay={ay:.6e}")

    dx = (bx - ax) * max(kx, 1.0)
    dy = (by - ay) * max(ky, 1.0)
    return ax, ay, bx, by, dx, dy


def _separate_xz_node(
    ax: float,
    az: float,
    kx: float,
    kz: float,
    dx: float,
    dz: float,
    min_sep: float,
    sep2: float,
    sep4: float,
) -> tuple[float, float, float, float, float, float]:
    """SPECFEM3D XZ_ONLY separation (lines 1501-1547)."""
    if abs(ax - az) < min_sep:
        ax, az = _separate_two_changeable(ax, az, sep2)

    bx = ax + dx / max(kx, 1.0)
    bz = az + dz / max(kz, 1.0)

    if abs(bx - az) < min_sep:
        bx, az = _separate_one_changeable(bx, az, sep2, sep4)
    if abs(bz - ax) < min_sep:
        bz, ax = _separate_one_changeable(bz, ax, sep2, sep4)

    if abs(ax - az) < min_sep or abs(bx - az) < min_sep or abs(bz - ax) < min_sep:
        raise ValueError(f"CPML XZ parameter separation failed: ax={ax:.6e} az={az:.6e}")

    dx = (bx - ax) * max(kx, 1.0)
    dz = (bz - az) * max(kz, 1.0)
    return ax, az, bx, bz, dx, dz


def _separate_yz_node(
    ay: float,
    az: float,
    ky: float,
    kz: float,
    dy: float,
    dz: float,
    min_sep: float,
    sep2: float,
    sep4: float,
) -> tuple[float, float, float, float, float, float]:
    """SPECFEM3D YZ_ONLY separation (lines 1549-1595)."""
    if abs(ay - az) < min_sep:
        ay, az = _separate_two_changeable(ay, az, sep2)

    by = ay + dy / max(ky, 1.0)
    bz = az + dz / max(kz, 1.0)

    if abs(by - az) < min_sep:
        by, az = _separate_one_changeable(by, az, sep2, sep4)
    if abs(bz - ay) < min_sep:
        bz, ay = _separate_one_changeable(bz, ay, sep2, sep4)

    if abs(ay - az) < min_sep or abs(by - az) < min_sep or abs(bz - ay) < min_sep:
        raise ValueError(f"CPML YZ parameter separation failed: ay={ay:.6e} az={az:.6e}")

    dy = (by - ay) * max(ky, 1.0)
    dz = (bz - az) * max(kz, 1.0)
    return ay, az, by, bz, dy, dz


def _separate_xyz_node(
    ax: float,
    ay: float,
    az: float,
    kx: float,
    ky: float,
    kz: float,
    dx: float,
    dy: float,
    dz: float,
    min_sep: float,
    sep2: float,
    sep4: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """SPECFEM3D XYZ separation (lines 1597-1825)."""
    # Stage 1: ax vs ay
    if abs(ax - ay) < min_sep:
        if ax > ay:
            ax = ay + sep2
        else:
            ay = ax + sep2
        maxtemp = max(ax, ay)
        mintemp = min(ax, ay)
        if az > maxtemp:
            if abs(az - maxtemp) < min_sep:
                az = maxtemp + sep2
        elif az < mintemp:
            if abs(az - mintemp) < min_sep:
                if ax > ay:
                    ax = az + sep4
                    ay = az + sep2
                else:
                    ay = az + sep4
                    ax = az + sep2
        else:
            if ax > ay:
                ax = ay + sep4
                az = ay + sep2
            else:
                ay = ax + sep4
                az = ax + sep2

    # Stage 2: ax vs az
    if abs(ax - az) < min_sep:
        if ax > az:
            ax = az + sep2
        else:
            az = ax + sep2
        maxtemp = max(ax, az)
        mintemp = min(ax, az)
        if ay > maxtemp:
            if abs(ay - maxtemp) < min_sep:
                ay = maxtemp + sep2
        elif ay < mintemp:
            if abs(ay - mintemp) < min_sep:
                if ax > az:
                    ax = ay + sep4
                    az = ay + sep2
                else:
                    az = ay + sep4
                    ax = ay + sep2
        else:
            if ax > az:
                ax = az + sep4
                ay = az + sep2
            else:
                az = ax + sep4
                ay = ax + sep2

    # Stage 3: ay vs az
    if abs(ay - az) < min_sep:
        if ay > az:
            ay = az + sep2
        else:
            az = ay + sep2
        maxtemp = max(ay, az)
        mintemp = min(ay, az)
        if ax > maxtemp:
            if abs(ax - maxtemp) < min_sep:
                ax = maxtemp + sep2
        elif ax < mintemp:
            if abs(ax - mintemp) < min_sep:
                if ay > az:
                    ay = ax + sep4
                    az = ax + sep2
                else:
                    az = ax + sep4
                    ay = ax + sep2
        else:
            if ay > az:
                ay = az + sep4
                ax = az + sep2
            else:
                az = ay + sep4
                ax = ay + sep2

    if abs(ax - ay) < min_sep or abs(ay - az) < min_sep or abs(ax - az) < min_sep:
        raise ValueError("CPML XYZ alpha parameter separation failed")

    # Beta adjustments
    bx = ax + dx / max(kx, 1.0)
    maxtemp = max(ay, az)
    mintemp = min(ay, az)
    if bx > maxtemp:
        if abs(bx - maxtemp) < min_sep:
            bx = maxtemp + sep2
    elif bx < mintemp:
        if abs(bx - mintemp) < min_sep:
            bx = (ay if ay > az else az) + sep2
    else:
        if abs(bx - maxtemp) < min_sep:
            bx = maxtemp + sep2
        if abs(bx - mintemp) < min_sep:
            bx = mintemp + sep2
            if abs(bx - maxtemp) < min_sep:
                bx = maxtemp + sep2

    by = ay + dy / max(ky, 1.0)
    maxtemp = max(ax, az)
    mintemp = min(ax, az)
    if by > maxtemp:
        if abs(by - maxtemp) < min_sep:
            by = maxtemp + sep2
    elif by < mintemp:
        if abs(by - mintemp) < min_sep:
            by = (ax if ax > az else az) + sep2
    else:
        if abs(by - maxtemp) < min_sep:
            by = maxtemp + sep2
        if abs(by - mintemp) < min_sep:
            by = mintemp + sep2
            if abs(by - maxtemp) < min_sep:
                by = maxtemp + sep2

    bz = az + dz / max(kz, 1.0)
    maxtemp = max(ax, ay)
    mintemp = min(ax, ay)
    if bz > maxtemp:
        if abs(bz - maxtemp) < min_sep:
            bz = maxtemp + sep2
    elif bz < mintemp:
        if abs(bz - mintemp) < min_sep:
            bz = (ax if ax > ay else ay) + sep2
    else:
        if abs(bz - maxtemp) < min_sep:
            bz = maxtemp + sep2
        if abs(bz - mintemp) < min_sep:
            bz = mintemp + sep2
            if abs(bz - maxtemp) < min_sep:
                bz = maxtemp + sep2

    if (
        abs(bx - ay) < min_sep
        or abs(bx - az) < min_sep
        or abs(by - ax) < min_sep
        or abs(by - az) < min_sep
        or abs(bz - ax) < min_sep
        or abs(bz - ay) < min_sep
    ):
        raise ValueError("CPML XYZ beta parameter separation failed")

    dx = (bx - ax) * max(kx, 1.0)
    dy = (by - ay) * max(ky, 1.0)
    dz = (bz - az) * max(kz, 1.0)
    return ax, ay, az, bx, by, bz, dx, dy, dz


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
        region = _get_region(pml_regions, e)
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

    if not np.all(np.isfinite(coef_abar)):
        raise ValueError("C-PML acceleration coefficients contain non-finite values")

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
        # Need alpha_x != alpha_y != alpha_z for the partial fraction.
        dxy = np.where(np.abs(ax - ay) < MIN_DISTANCE, MIN_DISTANCE, ax - ay)
        dxz = np.where(np.abs(ax - az) < MIN_DISTANCE, MIN_DISTANCE, ax - az)
        dyz = np.where(np.abs(ay - az) < MIN_DISTANCE, MIN_DISTANCE, ay - az)
        dyx = np.where(np.abs(ay - ax) < MIN_DISTANCE, MIN_DISTANCE, ay - ax)
        dzx = np.where(np.abs(az - ax) < MIN_DISTANCE, MIN_DISTANCE, az - ax)
        dzy = np.where(np.abs(az - ay) < MIN_DISTANCE, MIN_DISTANCE, az - ay)

        A3 = A0 * ax**2 * (bx - ax) * (by - ax) * (bz - ax) / (dyx * dzx)
        A4 = A0 * ay**2 * (bx - ay) * (by - ay) * (bz - ay) / (dxy * dzy)
        A5 = A0 * az**2 * (bx - az) * (by - az) * (bz - az) / (dyz * dxz)

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
        region = _get_region(pml_regions, e)
        if region == 0:
            continue

        kx, ky, kz = K_store[e, :, 0], K_store[e, :, 1], K_store[e, :, 2]
        dx, dy, dz = d_store[e, :, 0], d_store[e, :, 1], d_store[e, :, 2]
        ax, ay, az = alpha_store[e, :, 0], alpha_store[e, :, 1], alpha_store[e, :, 2]

        # A6..A9: lijk(z, y, x) = index 231
        region_231 = _permute_region(region, (2, 1, 0))
        A0, A6, A7, A8 = _lijk_parameter(region_231, kz, dz, az, ky, dy, ay, kx, dx, ax)
        coef_strain[e, :, 0] = A0
        coef_strain[e, :, 1] = A6
        coef_strain[e, :, 2] = A7
        coef_strain[e, :, 3] = A8

        # A10..A13: lijk(x, z, y) = index 132
        region_132 = _permute_region(region, (0, 2, 1))
        A0, A10, A11, A12 = _lijk_parameter(region_132, kx, dx, ax, kz, dz, az, ky, dy, ay)
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

    if not np.all(np.isfinite(coef_strain)):
        raise ValueError("C-PML strain coefficients contain non-finite values")

    return coef_strain


def apply_cpml_mass_correction(
    mass: npt.NDArray[np.float64],
    K_store: npt.NDArray[np.float64],
    d_store: npt.NDArray[np.float64],
    pml_regions: npt.NDArray[np.int32],
    dt: float,
) -> npt.NDArray[np.float64]:
    """Apply SPECFEM's explicit C-PML mass stabilization factor."""
    corrected_mass = np.array(mass, dtype=np.float64, copy=True)
    corrected_flat = corrected_mass.reshape(K_store.shape[:2])

    for region in range(CPML_X_ONLY, CPML_XYZ + 1):
        cell_mask = pml_regions == region
        if not np.any(cell_mask):
            continue
        active_axes = [axis for axis in range(3) if _is_axis_active(region, axis)]
        K_active = K_store[cell_mask][:, :, active_axes]
        d_active = d_store[cell_mask][:, :, active_axes]
        K_product = np.prod(K_active, axis=2)
        damping_sum = np.zeros_like(K_product)
        for local_axis in range(len(active_axes)):
            other_K_product = np.prod(np.delete(K_active, local_axis, axis=2), axis=2)
            damping_sum += d_active[:, :, local_axis] * other_K_product
        corrected_flat[cell_mask] *= K_product + 0.5 * dt * damping_sum

    return corrected_mass


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
