"""End-to-end integration tests: GMSH .msh → model.h5 → verify HDF5 schema."""

import os
import sys
import tempfile

import h5py
import meshio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.gmsh_to_hdf5 import extract_topology, write_auxiliary, write_topology


def make_2x2x1_mesh():
    """Build a 4-hex mesh: 2(x) x 2(y) x 1(z) = 4 cells, 18 vertices."""
    # Vertices: (x, y, z) grid
    pts = []
    grid = {}
    for iz in range(2):
        for iy in range(3):
            for ix in range(3):
                idx = len(pts)
                pts.append([float(ix), float(iy), float(iz)])
                grid[(ix, iy, iz)] = idx

    pts = np.array(pts, dtype=float)

    # 4 hexes in xy plane at z=0 layer
    cells = [
        # Cell 0: bottom-left (0,0)-(1,1)
        [
            grid[(0, 0, 0)],
            grid[(1, 0, 0)],
            grid[(1, 1, 0)],
            grid[(0, 1, 0)],
            grid[(0, 0, 1)],
            grid[(1, 0, 1)],
            grid[(1, 1, 1)],
            grid[(0, 1, 1)],
        ],
        # Cell 1: bottom-right (1,0)-(2,1)
        [
            grid[(1, 0, 0)],
            grid[(2, 0, 0)],
            grid[(2, 1, 0)],
            grid[(1, 1, 0)],
            grid[(1, 0, 1)],
            grid[(2, 0, 1)],
            grid[(2, 1, 1)],
            grid[(1, 1, 1)],
        ],
        # Cell 2: top-left (0,1)-(1,2)
        [
            grid[(0, 1, 0)],
            grid[(1, 1, 0)],
            grid[(1, 2, 0)],
            grid[(0, 2, 0)],
            grid[(0, 1, 1)],
            grid[(1, 1, 1)],
            grid[(1, 2, 1)],
            grid[(0, 2, 1)],
        ],
        # Cell 3: top-right (1,1)-(2,2)
        [
            grid[(1, 1, 0)],
            grid[(2, 1, 0)],
            grid[(2, 2, 0)],
            grid[(1, 2, 0)],
            grid[(1, 1, 1)],
            grid[(2, 1, 1)],
            grid[(2, 2, 1)],
            grid[(1, 2, 1)],
        ],
    ]

    return meshio.Mesh(pts, [("hexahedron", np.array(cells, dtype=np.int64))])


class TestFullPipeline:
    """Run full GMSH .msh → model.h5 pipeline and verify outputs."""

    def test_4hex_mesh_hdf5_schema(self):
        """Generate 4-hex mesh, convert to HDF5, verify groups and attributes."""
        mesh = make_2x2x1_mesh()
        topo = extract_topology(mesh)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.h5")
            write_topology(path, topo)

            with h5py.File(path, "r") as f:
                assert "topology" in f, "Missing /topology/ group"
                g = f["topology"]

                # Attributes
                assert int(g.attrs["n_vertex"]) == 18
                assert int(g.attrs["n_cell"]) == 4
                assert g.attrs["n_edge"] > 0
                assert g.attrs["n_surface"] > 0

                # Dataset shapes
                assert g["vertex_to_coord"].shape == (18, 3)
                assert g["vertex_to_coord"].dtype == np.float64
                assert g["edge_to_vertex"].dtype == np.int64
                assert g["surface_to_edge"].dtype == np.int64
                assert g["cell_to_surface"].shape == (4, 6)
                assert g["cell_to_surface"].dtype == np.int64

    def test_interior_face_signs(self):
        """Each interior surface appears exactly once as + and once as - across cells."""
        mesh = make_2x2x1_mesh()
        topo = extract_topology(mesh)
        c2s = topo["cell_to_surface"]
        n_surface = topo["surface_to_edge"].shape[0]

        # Count + and - for each surface ID
        positive_count = np.zeros(n_surface, dtype=int)
        negative_count = np.zeros(n_surface, dtype=int)

        for sid_signed in c2s.flatten():
            sid = int(abs(sid_signed))
            if sid_signed > 0:
                positive_count[sid - 1] += 1
            else:
                negative_count[sid - 1] += 1

        # Interior surfaces: exactly +1 and -1
        # Boundary surfaces: exactly +1 and -0
        for sid in range(n_surface):
            p, n = positive_count[sid], negative_count[sid]
            total = p + n
            assert total == 1 or total == 2, (
                f"Surface {sid + 1}: expected 1 or 2 references, got {total}"
            )
            if total == 2:
                assert p == 1 and n == 1, (
                    f"Interior surface {sid + 1}: expected +1/-1, got +{p}/-{n}"
                )
            else:
                assert p == 1 and n == 0, (
                    f"Boundary surface {sid + 1}: expected +1/-0, got +{p}/-{n}"
                )

    def test_auxiliary_file(self):
        """Generate auxiliary CSR file and verify structure."""
        mesh = make_2x2x1_mesh()
        topo = extract_topology(mesh)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model_auxiliary.h5")
            write_auxiliary(path, topo)

            with h5py.File(path, "r") as f:
                assert "auxiliary" in f
                a = f["auxiliary"]

                # surface_to_cell: 2 entries for interior, 1 for boundary
                s2c = a["surface_to_cell"][:]
                n_surface = topo["surface_to_edge"].shape[0]
                assert s2c.shape == (n_surface, 2)

                # CSR groups
                for name in [
                    "vertex_to_edge",
                    "vertex_to_surface",
                    "vertex_to_cell",
                    "edge_to_surface",
                    "edge_to_cell",
                ]:
                    assert name in a, f"Missing {name}"
                    g = a[name]
                    assert "indptr" in g
                    assert "indices" in g
                    assert g["indptr"][0] == 0


class TestGmshRoundtrip:
    """Write .msh file, read it back, convert — test full file I/O path."""

    def test_msh_file_roundtrip(self):
        """meshio write .msh → meshio read → extract_topology → verify."""
        mesh = make_2x2x1_mesh()

        with tempfile.TemporaryDirectory() as td:
            msh_path = os.path.join(td, "test.msh")
            h5_path = os.path.join(td, "model.h5")

            meshio.write(msh_path, mesh, file_format="gmsh")
            mesh_back = meshio.read(msh_path)
            topo = extract_topology(mesh_back)
            write_topology(h5_path, topo)

            with h5py.File(h5_path, "r") as f:
                g = f["topology"]
                assert int(g.attrs["n_cell"]) == 4
                assert g["cell_to_surface"].shape == (4, 6)
