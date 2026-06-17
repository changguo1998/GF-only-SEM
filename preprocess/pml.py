"""C-PML damping profiles (simplified ramp placeholder).

Computes per-GLL-node damping profiles for PML elements.
Non-PML elements return zero.  PML elements receive a linear ramp
from zero at the PML-entry interface to a maximum at the physical
domain boundary.

NOTE: This is a placeholder.  The full C-PML damping profile
formulas (d/K/alpha per direction, convolution coefficients) are
deferred to the forward-solver implementation phase.  The actual
C-PML implementation in the C++ kernel will follow the SPECFEM3D
conventions documented in the design doc.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from preprocess.topology_reader import TopologyData


def compute_pml_damping(
    topology: TopologyData,
    gll_coords: npt.NDArray[np.float64],
    pml_thickness: dict[str, int],
    domain_bounds: dict[str, float],
    is_pml: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Compute PML damping profile per GLL node (simplified ramp).

    For PML elements the damping increases linearly from 0 at the
    PML-entry interface to 1.0 at the physical domain boundary.
    For non-PML elements the result is 0.0 everywhere.

    Args:
        topology:      Mesh topology.
        gll_coords:    GLL node coords  [n_cell, NGLL, NGLL, NGLL, 3].
        pml_thickness: Element count per face (str → int).
        domain_bounds: Dict {xmin, xmax, ymin, ymax, zmin, zmax}.
        is_pml:        Boolean flag per cell [n_cell].

    Returns:
        damping: [n_cell, NGLL, NGLL, NGLL] scalar in [0, 1].

    NOTE: Full C-PML (d/K/alpha per direction, convolution
    coefficients) is deferred.
    """
    n_cell = topology.n_cell
    shape = gll_coords.shape
    NGLL = shape[1]
    damping = np.zeros((n_cell, NGLL, NGLL, NGLL), dtype=np.float64)

    # For each active PML direction, compute a normalized ramp.
    # PML zone occupies the outermost portion of the domain.
    # Physical start (interface) / end (boundary):
    pairs = [
        ("xmin", domain_bounds["xmin"], domain_bounds["xmax"], 0),
        ("xmax", domain_bounds["xmax"], domain_bounds["xmin"], 0),
        ("ymin", domain_bounds["ymin"], domain_bounds["ymax"], 1),
        ("ymax", domain_bounds["ymax"], domain_bounds["ymin"], 1),
        ("zmin", domain_bounds["zmin"], domain_bounds["zmax"], 2),
        ("zmax", domain_bounds["zmax"], domain_bounds["zmin"], 2),
    ]

    for face_key, bound_val, opp_val, axis in pairs:
        thickness = pml_thickness.get(face_key, 0)
        if thickness <= 0:
            continue

        # PML physical width for this face (approximate)
        domain_extent = domain_bounds[["xmax", "ymax", "zmax"][axis]] - \
                        domain_bounds[["xmin", "ymin", "zmin"][axis]]
        # Estimate element size along this axis
        n_cells_approx = max(1, int(np.ceil(n_cell ** (1.0 / 3.0))))
        cell_size = domain_extent / n_cells_approx
        pml_width = thickness * cell_size

        if pml_width <= 0:
            continue

        # PML region: [pml_start, pml_end]
        # Is it a positive or negative face?
        is_negative = face_key.endswith("min")
        if is_negative:
            # PML layer extends from xmin to xmin + width
            pml_start = bound_val + pml_width   # PML entry (interior face)
            pml_end = bound_val                  # domain boundary
        else:
            # PML layer extends from xmax - width to xmax
            pml_start = bound_val - pml_width   # PML entry (interior face)
            pml_end = bound_val                  # domain boundary

        # Extract coordinates along this axis
        coords_axis = gll_coords[:, :, :, :, axis]  # [n_cell, NGLL, NGLL, NGLL]

        for e in range(n_cell):
            if not is_pml[e]:
                continue

            # Check if this element overlaps the PML band
            e_min = coords_axis[e].min()
            e_max = coords_axis[e].max()

            if is_negative:
                # Element must be near the negative boundary
                if e_max > pml_start + 1e-12 and e_max > bound_val + pml_width + 1e-12:
                    # Element is too far from boundary, skip
                    continue
                # Ramp: dist from interior, normalized from 0 at pml_start to 1 at pml_end
                # coord decreasing toward boundary → (pml_start - coord) / (pml_start - pml_end)
                ramp = np.clip((pml_start - coords_axis[e]) / pml_width, 0.0, 1.0)
            else:
                # Element must be near the positive boundary
                if e_min < pml_start - 1e-12:
                    continue
                # Ramp: dist from interior, 0 at pml_start to 1 at pml_end
                ramp = np.clip((coords_axis[e] - pml_start) / pml_width, 0.0, 1.0)

            damping[e] = np.maximum(damping[e], ramp)

    return damping