"""Boundary detector — auto-detect boundary tags from surface geometry.

For each surface in the mesh, compute the face center coordinates and
classify as free surface (z ≈ z_min), absorbing (x/y/z at domain bounds),
or interior (tag 0).  Also flag cells that touch absorbing boundaries
for PML treatment.
"""

import numpy as np
import numpy.typing as npt

from preprocess.topology_reader import TopologyData

# Tolerance as fraction of domain size for boundary proximity checks
_BOUND_TOL = 1e-6


def detect_boundaries(
    topology: TopologyData,
    domain_bounds: dict[str, float],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Detect boundary types for each surface.

    Args:
        topology: Mesh topology with surface-to-edge and edge-to-vertex.
        domain_bounds: Dict with xmin, xmax, ymin, ymax, zmin, zmax.

    Returns:
        boundary_tag: [n_surface] int64 — 0=interior, 1=free surface, 2=absorbing
        is_pml: [n_cell] bool — True if cell has any absorbing surface
    """
    s2e = topology.surface_to_edge
    e2v = topology.edge_to_vertex
    v2c = topology.vertex_to_coord
    c2s = topology.cell_to_surface
    n_surface = topology.n_surface
    n_cell = topology.n_cell

    # Compute domain size for tolerance
    dx = domain_bounds["xmax"] - domain_bounds["xmin"]
    dy = domain_bounds["ymax"] - domain_bounds["ymin"]
    dz = domain_bounds["zmax"] - domain_bounds["zmin"]
    max_dim = max(dx, dy, dz)
    tol = _BOUND_TOL * max_dim if max_dim > 0 else 1e-9

    boundary_tag = np.zeros(n_surface, dtype=np.int64)

    for surf_idx in range(n_surface):
        # Collect unique vertex IDs for this face
        vids: set[int] = set()
        for signed_eid in s2e[surf_idx]:
            abs_eid = abs(int(signed_eid)) - 1
            vids.add(int(e2v[abs_eid, 0]))
            vids.add(int(e2v[abs_eid, 1]))

        face_coords = np.array([v2c[v - 1] for v in sorted(vids)])
        center = face_coords.mean(axis=0)

        # Check z ≈ zmin → free surface
        if np.isclose(center[2], domain_bounds["zmin"], atol=tol):
            boundary_tag[surf_idx] = 1
            continue

        # Check other domain bounds → absorbing
        if (np.isclose(center[0], domain_bounds["xmin"], atol=tol) or
                np.isclose(center[0], domain_bounds["xmax"], atol=tol) or
                np.isclose(center[1], domain_bounds["ymin"], atol=tol) or
                np.isclose(center[1], domain_bounds["ymax"], atol=tol) or
                np.isclose(center[2], domain_bounds["zmax"], atol=tol)):
            boundary_tag[surf_idx] = 2
            continue

        # Interior
        boundary_tag[surf_idx] = 0

    # is_pml: True if any surface of a cell is tagged absorbing
    is_pml = np.zeros(n_cell, dtype=np.bool_)
    for cell_idx in range(n_cell):
        for signed_surf in c2s[cell_idx]:
            abs_surf = abs(int(signed_surf)) - 1
            if boundary_tag[abs_surf] == 2:
                is_pml[cell_idx] = True
                break

    return boundary_tag, is_pml