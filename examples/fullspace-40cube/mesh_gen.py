#!/usr/bin/env python3
"""Generate the regular hexahedral mesh for the 40-cube full-space case."""

from pathlib import Path
import sys

import meshio
import numpy as np


CASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(CASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from tools.gmsh_to_hdf5 import extract_topology, write_topology  # noqa: E402


def main() -> None:
    """Write a rectilinear 40-cube topology to model.h5."""
    nx = config.nx_elements
    ny = config.ny_elements
    nz = config.nz_elements
    x_coordinates = np.linspace(0.0, config.lx, nx + 1)
    y_coordinates = np.linspace(0.0, config.ly, ny + 1)
    z_coordinates = np.linspace(0.0, config.lz, nz + 1)

    vertices = np.array(
        [[x, y, z] for z in z_coordinates for y in y_coordinates for x in x_coordinates],
        dtype=np.float64,
    )
    vertex_count_x = nx + 1
    vertex_count_y = ny + 1
    cells = []
    for element_z in range(nz):
        for element_y in range(ny):
            for element_x in range(nx):
                vertex_000 = (
                    element_z * vertex_count_y * vertex_count_x
                    + element_y * vertex_count_x
                    + element_x
                )
                vertex_100 = vertex_000 + 1
                vertex_010 = vertex_000 + vertex_count_x
                vertex_110 = vertex_010 + 1
                vertex_001 = vertex_000 + vertex_count_y * vertex_count_x
                vertex_101 = vertex_001 + 1
                vertex_011 = vertex_001 + vertex_count_x
                vertex_111 = vertex_011 + 1
                cells.append(
                    [
                        vertex_000,
                        vertex_100,
                        vertex_110,
                        vertex_010,
                        vertex_001,
                        vertex_101,
                        vertex_111,
                        vertex_011,
                    ]
                )

    mesh = meshio.Mesh(vertices, [("hexahedron", np.asarray(cells, dtype=np.int64))])
    topology = extract_topology(mesh)
    write_topology("model.h5", topology)
    print(f"[mesh_gen] Elements: {nx * ny * nz} ({nx}x{ny}x{nz})")
    print(f"[mesh_gen] Domain: {config.lx}x{config.ly}x{config.lz} m")


if __name__ == "__main__":
    main()
