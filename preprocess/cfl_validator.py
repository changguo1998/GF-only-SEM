"""CFL validator — compute CFL-limited timestep and derive solver timestep.

After GLL geometry and material (vp) are known, compute the minimum GLL
node spacing h_min, the elastic CFL limit, and the C-PML damping limit:

    cfl_dt = cfl_safety × h_min / vp_max
    cpml_dt = stability_number / max(sum(d_axis))

Then derive the solver timestep by searching for an integer stride such
that output_dt_s / stride is no greater than either limit. The solver_dt is
the largest timestep that satisfies both constraints while keeping output_dt_s
as an integer multiple.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from preprocess.pml_cpml import NPOWER as CPML_NPOWER
from preprocess.pml_cpml import R_COEF as CPML_R_COEF

# SPECFEM3D reference implementation uses K_MAX_PML=K_MIN_PML=1.
CPML_K_MAX_PML = 1.0

# The measured XYZ-corner instability starts near dt * sum(d / K) = 1.4.
# Use a conservative dimensionless limit rather than encoding that threshold.
CPML_STABILITY_NUMBER = 1.0

MAX_STRIDE = 100


def compute_cfl_dt(
    gll_coords: npt.NDArray[np.float64], vp_array: npt.NDArray[np.float64], cfl_safety: float
) -> float:
    """Compute the CFL-limited time step.

    h_min = minimum Euclidean distance between adjacent GLL nodes in
            the i, j, or k direction within any element.
    vp_max = maximum P-wave velocity across all GLL nodes.

    Args:
        gll_coords: [n_cell, NGLL, NGLL, NGLL, 3] GLL node positions.
        vp_array:   [n_cell, NGLL, NGLL, NGLL] P-wave speed at each node.
        cfl_safety: CFL safety factor (0 < cfl_safety < 1).

    Returns:
        cfl_dt: CFL-limited time step in seconds.
    """
    n_cell, NGLL, _, _, _ = gll_coords.shape

    # Compute minimum GLL node spacing
    h_min = float("inf")

    for e in range(n_cell):
        for i in range(NGLL):
            for j in range(NGLL):
                for k in range(NGLL):
                    x_node = gll_coords[e, i, j, k]

                    # Neighbor in +i direction
                    if i + 1 < NGLL:
                        diff = x_node - gll_coords[e, i + 1, j, k]
                        dist = np.sqrt(np.dot(diff, diff))
                        h_min = min(h_min, dist)

                    # Neighbor in +j direction
                    if j + 1 < NGLL:
                        diff = x_node - gll_coords[e, i, j + 1, k]
                        dist = np.sqrt(np.dot(diff, diff))
                        h_min = min(h_min, dist)

                    # Neighbor in +k direction
                    if k + 1 < NGLL:
                        diff = x_node - gll_coords[e, i, j, k + 1]
                        dist = np.sqrt(np.dot(diff, diff))
                        h_min = min(h_min, dist)

    if h_min == float("inf") or h_min <= 0:
        raise ValueError(f"Invalid minimum GLL node spacing: h_min = {h_min}")

    vp_max = float(np.max(vp_array))
    if vp_max <= 0:
        raise ValueError(f"Invalid maximum vp: {vp_max}")

    return cfl_safety * h_min / (vp_max * np.sqrt(CPML_K_MAX_PML))


def compute_cpml_stability_dt(pml_widths: dict[str, float], vp_max: float) -> float:
    """Compute the C-PML damping-limited timestep.

    At an edge or corner, damping from each active axis contributes to the
    explicit acceleration update. The worst case uses the thinner active face
    on each axis because it has the largest maximum damping coefficient.

    Args:
        pml_widths: Physical PML width per face in metres.
        vp_max: Global maximum P-wave velocity in metres per second.

    Returns:
        Conservative timestep limit in seconds, or infinity when no PML face
        is active.
    """
    active_widths = []
    for axis in ("x", "y", "z"):
        face_widths = [
            float(pml_widths.get(f"{axis}min", 0.0)),
            float(pml_widths.get(f"{axis}max", 0.0)),
        ]
        positive_widths = [width for width in face_widths if width > 0.0]
        if positive_widths:
            active_widths.append(min(positive_widths))

    if not active_widths:
        return float("inf")
    if vp_max <= 0.0:
        raise ValueError(f"vp_max must be positive when PML is active, got {vp_max}")

    damping_sum_max = sum(
        -((CPML_NPOWER + 1.0) * vp_max * np.log(CPML_R_COEF) / (2.0 * width))
        for width in active_widths
    )
    return CPML_STABILITY_NUMBER / damping_sum_max


def compute_solver_dt(
    output_dt_s: float, cfl_dt: float, max_stride: int = MAX_STRIDE
) -> tuple[float, int]:
    """Derive solver timestep and snapshot stride.

    Search for the smallest stride such that output_dt_s / stride ≤ cfl_dt.
    This ensures the output snapshot interval is an integer multiple of the
    solver timestep.

    Args:
        output_dt_s: User-specified snapshot interval (seconds).
        cfl_dt: CFL-limited timestep from compute_cfl_dt().
        max_stride: Maximum stride to search (default: MAX_STRIDE=100).

    Returns:
        Tuple of (solver_dt, snapshot_stride):
            solver_dt:      Timestep used by the Newmark loop (seconds).
            snapshot_stride: Number of solver steps per output snapshot.

    Raises:
        ValueError: If no integer stride satisfies the CFL constraint.
    """
    if cfl_dt <= 0:
        raise ValueError(f"cfl_dt must be positive, got {cfl_dt}")
    if output_dt_s <= 0:
        raise ValueError(f"output_dt_s must be positive, got {output_dt_s}")
    if max_stride < 1:
        raise ValueError(f"max_stride must be >= 1, got {max_stride}")

    for stride in range(1, max_stride + 1):
        solver_dt = output_dt_s / stride
        if solver_dt <= cfl_dt:
            return solver_dt, stride

    raise ValueError(
        f"output_dt_s={output_dt_s} too large for CFL limit (cfl_dt={cfl_dt:.6e}). "
        f"No integer stride 1..{max_stride} gives solver_dt ≤ cfl_dt. "
        "Increase cfl_safety, reduce output_dt_s, or increase element size."
    )
