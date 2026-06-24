#!/usr/bin/env python3
"""Generate a regular hexahedral mesh and write mesh.h5.

Creates a rectilinear grid of hex elements and writes the standard
mesh.h5 topology format used by the preprocessor.

Usage:
    python examples/halfspace/mesh_gen.py -o mesh.h5 [--aux mesh_auxiliary.h5]
"""

import argparse
import os
import sys

import h5py
import meshio
import numpy as np

# Ensure project root is importable
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.gmsh_to_hdf5 import extract_topology, write_topology, write_auxiliary


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
    parser = argparse.ArgumentParser(
        description="Generate regular hex mesh for half-space example"
    )
    parser.add_argument("-o", "--output", default="mesh.h5",
                        help="Output mesh.h5 path")
    parser.add_argument("--aux", help="Optional auxiliary CSR file")
    parser.add_argument("--nx", type=int, default=10,
                        help="Elements in x (default: 10)")
    parser.add_argument("--ny", type=int, default=10,
                        help="Elements in y (default: 10)")
    parser.add_argument("--nz", type=int, default=5,
                        help="Elements in z (default: 5)")
    parser.add_argument("--lx", type=float, default=10000.0,
                        help="Domain length x [m] (default: 10000)")
    parser.add_argument("--ly", type=float, default=10000.0,
                        help="Domain length y [m] (default: 10000)")
    parser.add_argument("--lz", type=float, default=5000.0,
                        help="Domain length z [m] (default: 5000)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    mesh = create_regular_hex_mesh(args.nx, args.ny, args.nz,
                                   args.lx, args.ly, args.lz)
    topology = extract_topology(mesh)
    write_topology(args.output, topology)
    print(f"[mesh_gen] Wrote mesh.h5: {args.output}")
    print(f"            Elements: {args.nx * args.ny * args.nz} "
          f"({args.nx}×{args.ny}×{args.nz})")
    print(f"            Vertices: {topology['vertex_to_coord'].shape[0]}")
    print(f"            Domain:   {args.lx}×{args.ly}×{args.lz} m")

    if args.aux:
        write_auxiliary(args.aux, topology)
        print(f"[mesh_gen] Wrote auxiliary: {args.aux}")


if __name__ == "__main__":
    main()