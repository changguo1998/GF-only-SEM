"""GLL geometry computation for hexahedral elements.

Computes GLL node positions via linear shape function interpolation,
Jacobian determinants, dxi/dx derivatives, and lumped mass diagonal.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import legendre

from preprocess.topology_reader import TopologyData

# GMSH hex reference corners in [-1,1]^3 (unit cube mapping)
HEX_REF_CORNERS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)


def gll_quadrature_points(N: int) -> npt.NDArray[np.float64]:
    """GLL quadrature points in [-1, 1] for poly order N (N+1 points)."""
    if N == 0:
        return np.array([0.0], dtype=np.float64)
    if N == 1:
        return np.array([-1.0, 1.0], dtype=np.float64)

    dpoly = np.polyder(legendre(N))
    xi_roots = np.sort(np.roots(dpoly)).real
    xi_roots = xi_roots[(xi_roots > -1 + 1e-12) & (xi_roots < 1 - 1e-12)]
    points = np.concatenate([[-1.0], xi_roots, [1.0]])
    return np.ascontiguousarray(points, dtype=np.float64)


def gll_weights(pts: npt.NDArray[np.float64], N: int) -> npt.NDArray[np.float64]:
    """GLL quadrature weights."""
    n = N + 1
    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        pn = legendre(N)(pts[i])
        w[i] = 2.0 / (N * (N + 1) * pn * pn)
    return w


def _linear_shape_derivs(xi: float, eta: float, zeta: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear hex shape functions and derivatives at (xi,eta,zeta).

    Returns (N_vals [8], dN [8,3]) where dN[a,m] = ∂N_a/∂ξ_m.
    """
    N_vals = np.zeros(8, dtype=np.float64)
    dN = np.zeros((8, 3), dtype=np.float64)
    for a in range(8):
        ca, cb, cc = HEX_REF_CORNERS[a]
        t0 = 0.125
        N_vals[a] = t0 * (1 + ca * xi) * (1 + cb * eta) * (1 + cc * zeta)
        dN[a, 0] = t0 * ca * (1 + cb * eta) * (1 + cc * zeta)
        dN[a, 1] = t0 * (1 + ca * xi) * cb * (1 + cc * zeta)
        dN[a, 2] = t0 * (1 + ca * xi) * (1 + cb * eta) * cc
    return N_vals, dN


def _get_cell_vertex_ids(
    e: int, c2s: npt.NDArray[np.int64], s2e: npt.NDArray[np.int64], e2v: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Extract the 8 vertex IDs (1-based, GMSH hex order) for cell e.

    Returns vertices in GMSH hex ordering:
    v0(-1,-1,-1), v1(1,-1,-1), v2(1,1,-1), v3(-1,1,-1),
    v4(-1,-1,1),  v5(1,-1,1),  v6(1,1,1),  v7(-1,1,1).

    Uses face-membership: each corner belongs to exactly 3 faces.
    The 6 faces in cell_to_surface are ordered -z, +z, -y, +y, -x, +x
    (indices 0..5).  By checking which 3 faces each vertex appears on,
    the 8 GMSH corners are uniquely identified.
    """
    # Collect vertex sets for each face (indices 0..5)
    face_vertices: list[set[int]] = [set() for _ in range(6)]
    for fi, signed_sid in enumerate(c2s[e]):
        abs_sid = abs(int(signed_sid)) - 1
        for sedge in s2e[abs_sid]:
            abs_eid = abs(int(sedge)) - 1
            face_vertices[fi].add(int(e2v[abs_eid, 0]))
            face_vertices[fi].add(int(e2v[abs_eid, 1]))

    # Build vertex → face membership (set of face indices 0..5)
    vertex_to_faces: dict[int, set[int]] = {}
    for fi, fv in enumerate(face_vertices):
        for v in fv:
            if v not in vertex_to_faces:
                vertex_to_faces[v] = set()
            vertex_to_faces[v].add(fi)

    # GMSH corner → face membership
    # Face 0=-z, 1=+z, 2=-y, 3=+y, 4=-x, 5=+x
    corner_faces = [
        {0, 2, 4},  # v0: -z, -y, -x
        {0, 2, 5},  # v1: -z, -y, +x
        {0, 3, 5},  # v2: -z, +y, +x
        {0, 3, 4},  # v3: -z, +y, -x
        {1, 2, 4},  # v4: +z, -y, -x
        {1, 2, 5},  # v5: +z, -y, +x
        {1, 3, 5},  # v6: +z, +y, +x
        {1, 3, 4},  # v7: +z, +y, -x
    ]

    result = np.zeros(8, dtype=np.int64)
    for corner_idx, target_faces in enumerate(corner_faces):
        found = False
        for v, vfaces in vertex_to_faces.items():
            if vfaces == target_faces:
                result[corner_idx] = np.int64(v)
                found = True
                break
        if not found:
            raise ValueError(
                f"Cannot identify GMSH corner {corner_idx} for cell {e}. "
                f"Vertex->faces: {vertex_to_faces}. "
                f"Target faces: {target_faces}."
            )

    return result


def compute_gll_geometry(
    topology: TopologyData, N: int
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Compute GLL geometry for all elements.

    Returns coords, jacobian (det), dxi_dx (flattened 9), mass.
    """
    n_cell = topology.n_cell
    NGLL = N + 1

    pts = gll_quadrature_points(N)
    w = gll_weights(pts, N)

    verts = topology.vertex_to_coord
    c2s = topology.cell_to_surface
    s2e = topology.surface_to_edge
    e2v = topology.edge_to_vertex

    coords = np.zeros((n_cell, NGLL, NGLL, NGLL, 3), dtype=np.float64)
    jacobian = np.zeros((n_cell, NGLL, NGLL, NGLL), dtype=np.float64)
    dxi_dx = np.zeros((n_cell, NGLL, NGLL, NGLL, 9), dtype=np.float64)
    mass = np.zeros((n_cell, NGLL, NGLL, NGLL), dtype=np.float64)

    for e in range(n_cell):
        vid = _get_cell_vertex_ids(e, c2s, s2e, e2v)
        cv = verts[vid - 1]  # [8, 3] physical corners

        for i in range(NGLL):
            xi = pts[i]
            for j in range(NGLL):
                eta = pts[j]
                for k in range(NGLL):
                    zeta = pts[k]

                    S, dS = _linear_shape_derivs(xi, eta, zeta)
                    x_phys = S @ cv  # (3,)
                    J = dS.T @ cv  # (3,3), J[m,n] = dx_m/dξ_n

                    coords[e, i, j, k] = x_phys
                    detJ = np.linalg.det(J)
                    jacobian[e, i, j, k] = detJ

                    if detJ > 0:
                        dxi_dx[e, i, j, k] = np.linalg.inv(J).ravel()
                    else:
                        dxi_dx[e, i, j, k] = 0.0

                    mass[e, i, j, k] = detJ * w[i] * w[j] * w[k]

    return coords, jacobian, dxi_dx, mass
