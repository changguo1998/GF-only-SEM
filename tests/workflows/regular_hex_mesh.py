"""Generate a regular hexahedral mesh and write mesh.h5 topology.

This is a lightweight alternative to GMSH — creates a rectilinear grid
of hex elements programmatically and writes the standard mesh.h5 format.
Used by workflow integration tests.
"""

import os
import sys
import numpy as np
import meshio
import h5py

# Ensure tools/ is importable (gmsh_to_hdf5)
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_tools_dir = os.path.join(_project_root, "tools")
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from gmsh_to_hdf5 import extract_topology, write_topology  # type: ignore[import-untyped]


def create_regular_hex_mesh(nx, ny, nz, lx, ly, lz):
    """Create a regular hexahedral mesh (rectilinear grid).

    Parameters
    ----------
    nx, ny, nz : int
        Number of elements in x, y, z directions.
    lx, ly, lz : float
        Domain size in x, y, z directions (meters).

    Returns
    -------
    meshio.Mesh
    """
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    nvert_x = nx + 1
    nvert_y = ny + 1
    nvert_z = nz + 1

    # Generate vertex coordinates
    vertices = []
    for iz in range(nvert_z):
        for iy in range(nvert_y):
            for ix in range(nvert_x):
                vertices.append([ix * dx, iy * dy, iz * dz])

    vertices = np.array(vertices, dtype=np.float64)

    # Generate hex cells (GMSH-ordering)
    # v0(0,0,0) v1(1,0,0) v2(1,1,0) v3(0,1,0) — bottom
    # v4(0,0,1) v5(1,0,1) v6(1,1,1) v7(0,1,1) — top
    hex_cells = []
    for ez in range(nz):
        for ey in range(ny):
            for ex in range(nx):
                # Vertex indices (0-based)
                v000 = ez * nvert_y * nvert_x + ey * nvert_x + ex
                v100 = ez * nvert_y * nvert_x + ey * nvert_x + ex + 1
                v110 = ez * nvert_y * nvert_x + (ey + 1) * nvert_x + ex + 1
                v010 = ez * nvert_y * nvert_x + (ey + 1) * nvert_x + ex
                v001 = (ez + 1) * nvert_y * nvert_x + ey * nvert_x + ex
                v101 = (ez + 1) * nvert_y * nvert_x + ey * nvert_x + ex + 1
                v111 = (ez + 1) * nvert_y * nvert_x + (ey + 1) * nvert_x + ex + 1
                v011 = (ez + 1) * nvert_y * nvert_x + (ey + 1) * nvert_x + ex
                hex_cells.append([v000, v100, v110, v010, v001, v101, v111, v011])

    hex_cells = np.array(hex_cells, dtype=np.int64)
    cells: list[tuple[str, np.ndarray]] = [("hexahedron", hex_cells)]

    return meshio.Mesh(vertices, cells)  # type: ignore[arg-type]


def create_halfspace_mesh(
    output_path, nx=6, ny=6, nz=4, lx=4000.0, ly=4000.0, lz=2000.0
):
    """Create mesh.h5 for a half-space model.

    Half-space: free surface at z=0, absorbing boundaries on
    the other 5 sides. z positive downward.

    Parameters
    ----------
    output_path : str
        Path for mesh.h5 output.
    nx, ny, nz : int
        Number of elements in x, y, z.
    lx, ly, lz : float
        Domain size in meters.

    Returns
    -------
    dict
        Domain bounds: {xmin, xmax, ymin, ymax, zmin, zmax}.
    """
    mesh = create_regular_hex_mesh(nx, ny, nz, lx, ly, lz)
    topology = extract_topology(mesh)
    write_topology(output_path, topology)

    return {
        "xmin": 0.0,
        "xmax": lx,
        "ymin": 0.0,
        "ymax": ly,
        "zmin": 0.0,
        "zmax": lz,
    }
