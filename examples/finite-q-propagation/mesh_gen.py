#!/usr/bin/env python3
"""Generate the regular hexahedral finite-Q propagation mesh."""

import config
import meshio
import numpy as np

from tools.gmsh_to_hdf5 import extract_topology, write_topology


def main():
    nx = config.nx_elements
    ny = config.ny_elements
    nz = config.nz_elements
    axes = [
        np.linspace(0.0, length, count + 1)
        for length, count in ((config.lx, nx), (config.ly, ny), (config.lz, nz))
    ]
    points = np.array([[x, y, z] for z in axes[2] for y in axes[1] for x in axes[0]])

    cells = []
    nx_vertex = nx + 1
    ny_vertex = ny + 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                base = (k * ny_vertex + j) * nx_vertex + i
                upper = base + ny_vertex * nx_vertex
                cells.append(
                    [
                        base,
                        base + 1,
                        base + nx_vertex + 1,
                        base + nx_vertex,
                        upper,
                        upper + 1,
                        upper + nx_vertex + 1,
                        upper + nx_vertex,
                    ]
                )

    topology = extract_topology(meshio.Mesh(points, [("hexahedron", np.asarray(cells))]))
    write_topology("model.h5", topology)
    print(f"[mesh_gen] Wrote {nx * ny * nz} elements to model.h5")


if __name__ == "__main__":
    main()
