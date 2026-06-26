#!/usr/bin/env python3
"""Generate a regular hexahedral mesh and write mesh.h5.

Creates a rectilinear grid of hex elements and writes the standard
mesh.h5 topology format used by the preprocessor.

Mesh dimensions (nx_elements, ny_elements, nz_elements, lx, ly, lz) are read
from config.py in the same directory.

Usage:
    python examples/halfspace/mesh_gen.py
"""

import os
import sys

import meshio
import numpy as np

# Ensure project root is importable
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.gmsh_to_hdf5 import extract_topology, write_topology

# Import mesh parameters from config.py
_example_dir = os.path.dirname(os.path.abspath(__file__))
if _example_dir not in sys.path:
    sys.path.insert(0, _example_dir)
import config  # type: ignore[import]


def create_regular_hex_mesh(nx: int, ny: int, nz: int,
                            lx: float, ly: float, lz: float) -> meshio.Mesh:
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
                v000 = ez * nvert_y * nvert_x + ey * nvert_x + ex
                v100 = ez * nvert_y * nvert_x + ey * nvert_x + ex + 1
                v110 = ez * nvert_y * nvert_x + (ey + 1) * nvert_x + ex + 1
                v010 = ez * nvert_y * nvert_x + (ey + 1) * nvert_x + ex
                v001 = (ez + 1) * nvert_y * nvert_x + ey * nvert_x + ex
                v101 = (ez + 1) * nvert_y * nvert_x + ey * nvert_x + ex + 1
                v111 = (ez + 1) * nvert_y * nvert_x + (ey + 1) * nvert_x + ex + 1
                v011 = (ez + 1) * nvert_y * nvert_x + (ey + 1) * nvert_x + ex
                hex_cells.append([v000, v100, v110, v010,
                                  v001, v101, v111, v011])

    hex_cells = np.array(hex_cells, dtype=np.int64)
    cells: list[tuple[str, np.ndarray]] = [("hexahedron", hex_cells)]

    return meshio.Mesh(vertices, cells)  # type: ignore[arg-type]


def main() -> None:
    nx = config.nx_elements
    ny = config.ny_elements
    nz = config.nz_elements
    lx = config.lx
    ly = config.ly
    lz = config.lz

    mesh = create_regular_hex_mesh(nx, ny, nz, lx, ly, lz)
    topology = extract_topology(mesh)
    write_topology("mesh.h5", topology)
    print(f"[mesh_gen] Wrote mesh.h5")
    print(f"            Elements: {nx * ny * nz} "
          f"({nx}×{ny}×{nz})")
    print(f"            Vertices: {topology['vertex_to_coord'].shape[0]}")
    print(f"            Domain:   {lx}×{ly}×{lz} m")


if __name__ == "__main__":
    main()