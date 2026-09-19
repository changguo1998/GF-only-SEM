#!/usr/bin/env python3
"""Generate a layer-aligned hexahedral mesh and write model.h5.

Mesh dimensions are read from config.py. The first z element ends at the
material interface; the remaining z elements evenly cover the lower layer.

Usage:
    python examples/layer-shallow-source/mesh_gen.py
"""

from __future__ import annotations

import os
import sys

import meshio
import numpy as np

# Ensure project root is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.gmsh_to_hdf5 import extract_topology, write_topology  # noqa: E402

# Import mesh parameters from config.py
_example_dir = os.path.dirname(os.path.abspath(__file__))
if _example_dir not in sys.path:
    sys.path.insert(0, _example_dir)
import config  # noqa: E402  # type: ignore[import]


def create_layer_aligned_hex_mesh(
    nx: int, ny: int, nz: int, lx: float, ly: float, lz: float, interface_z_m: float
) -> meshio.Mesh:
    """Create a rectilinear mesh with the first z boundary at the interface.

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
    if nz < 2:
        raise ValueError("layer-aligned mesh requires at least two z elements")
    if not 0.0 < interface_z_m < lz:
        raise ValueError("layer interface must lie strictly inside the domain")

    x_coords = np.linspace(0.0, lx, nx + 1)
    y_coords = np.linspace(0.0, ly, ny + 1)
    z_coords = np.concatenate(([0.0], np.linspace(interface_z_m, lz, nz, dtype=np.float64)))

    nvert_x = nx + 1
    nvert_y = ny + 1
    nvert_z = nz + 1

    # Generate vertex coordinates
    vertices = []
    for iz in range(nvert_z):
        for iy in range(nvert_y):
            for ix in range(nvert_x):
                vertices.append([x_coords[ix], y_coords[iy], z_coords[iz]])

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
                hex_cells.append([v000, v100, v110, v010, v001, v101, v111, v011])

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

    mesh = create_layer_aligned_hex_mesh(nx, ny, nz, lx, ly, lz, config.LAYER_INTERFACE_DEPTH_M)
    topology = extract_topology(mesh)
    write_topology("model.h5", topology)
    print("[mesh_gen] Wrote model.h5")
    print(f"            Elements: {nx * ny * nz} ({nx}×{ny}×{nz})")
    print(f"            Vertices: {topology['vertex_to_coord'].shape[0]}")
    print(f"            Domain:   {lx}×{ly}×{lz} m")
    print(
        f"            Layer interface at z={config.LAYER_INTERFACE_DEPTH_M:g} m "
        "aligns with element boundary"
    )


if __name__ == "__main__":
    main()
