"""CFL validator — compute CFL-limited timestep and derive solver timestep.

After GLL geometry and material (vp) are known, compute the minimum GLL
node spacing h_min and the CFL-limited time step:

    cfl_dt = cfl_safety × h_min / vp_max

Then derive the solver timestep by searching for an integer stride such
that output_dt_s / stride ≤ cfl_dt. The solver_dt is the largest timestep
that satisfies CFL while keeping output_dt_s as an integer multiple.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

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

    return cfl_safety * h_min / vp_max


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
