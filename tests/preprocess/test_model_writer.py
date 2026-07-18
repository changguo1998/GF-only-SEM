import os
import sys
import tempfile

import h5py
import numpy as np

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

from preprocess.model_writer import write_model
from preprocess.topology_reader import TopologyData


def _make_unit_cube_topo():
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=np.float64,
    )
    e2v = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 5],
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 8],
        ],
        dtype=np.int64,
    )
    s2e = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 10, -5, -9],
            [3, 12, -7, -11],
            [-4, 12, -8, -9],
            [2, 11, -6, -10],
        ],
        dtype=np.int64,
    )
    c2s = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    return TopologyData(verts, e2v, s2e, c2s, 8, 12, 6, 1)


def _make_model_h5(path):
    topo = _make_unit_cube_topo()
    with h5py.File(path, "w") as f:
        grp = f.create_group("topology")
        grp.attrs["n_vertex"] = topo.n_vertex
        grp.attrs["n_edge"] = topo.n_edge
        grp.attrs["n_surface"] = topo.n_surface
        grp.attrs["n_cell"] = topo.n_cell
        grp.create_dataset("vertex_to_coord", data=topo.vertex_to_coord)
        grp.create_dataset("edge_to_vertex", data=topo.edge_to_vertex)
        grp.create_dataset("surface_to_edge", data=topo.surface_to_edge)
        grp.create_dataset("cell_to_surface", data=topo.cell_to_surface)
    return topo


def _make_synthetic_fields(n_cell=1, ngll=4):
    shape_scalar = (n_cell, ngll, ngll, ngll)
    shape_vec = (n_cell, ngll, ngll, ngll, 3)
    shape_tensor = (n_cell, ngll, ngll, ngll, 9)

    coords = np.random.randn(*shape_vec).astype(np.float64)
    jacobian = np.abs(np.random.randn(*shape_scalar)).astype(np.float64) + 0.1
    dxi_dx = np.random.randn(*shape_tensor).astype(np.float64)
    mass = np.abs(np.random.randn(*shape_scalar)).astype(np.float64)
    vp = np.full(shape_scalar, 3000.0, dtype=np.float64)
    vs = np.full(shape_scalar, 1500.0, dtype=np.float64)
    density = np.full(shape_scalar, 2500.0, dtype=np.float64)
    is_pml = np.array([False] * n_cell, dtype=np.bool_)
    damping = np.zeros(shape_scalar, dtype=np.float64)

    return {
        "coords": coords,
        "jacobian": jacobian,
        "dxi_dx": dxi_dx,
        "mass": mass,
        "vp": vp,
        "vs": vs,
        "density": density,
        "is_pml": is_pml,
        "damping": damping,
    }


class TestModelWriter:
    def test_model_h5_extension(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model.h5")
            topo = _make_model_h5(model_path)
            fields = _make_synthetic_fields(n_cell=1, ngll=4)
            boundary_tag = np.array([1, 2, 2, 2, 2, 2], dtype=np.int64)
            domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

            write_model(model_path, topo, fields, boundary_tag, domain_bounds)

            with h5py.File(model_path, "r") as f:
                assert "topology" in f
                assert "field" in f
                assert "domain" in f

                felem = f["field/cell"]
                assert "coords" in felem
                assert "dxi_dx" in felem
                assert "jacobian" in felem
                assert "is_pml" in felem

                fsurf = f["field/surface"]
                assert "boundary_tag" in fsurf

                assert np.array_equal(fsurf["boundary_tag"][:], boundary_tag)

                assert f["domain"].attrs["xmin"] == 0.0
                assert f["domain"].attrs["xmax"] == 1.0

    def test_model_h5_preserves_topology(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model.h5")
            topo = _make_model_h5(model_path)
            fields = _make_synthetic_fields(n_cell=1, ngll=4)
            boundary_tag = np.array([1, 2, 2, 2, 2, 2], dtype=np.int64)
            domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

            write_model(model_path, topo, fields, boundary_tag, domain_bounds)

            with h5py.File(model_path, "r") as f:
                topo_grp = f["topology"]
                assert np.array_equal(topo_grp["vertex_to_coord"][:], topo.vertex_to_coord)
                assert np.array_equal(topo_grp["cell_to_surface"][:], topo.cell_to_surface)

    def test_partition_files_created(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model.h5")
            topo = _make_model_h5(model_path)
            fields = _make_synthetic_fields(n_cell=1, ngll=4)
            boundary_tag = np.array([1, 2, 2, 2, 2, 2], dtype=np.int64)
            domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

            partition_result = {
                "element_to_rank": np.array([0], dtype=np.int32),
                "n_ranks": 1,
                "per_rank": {
                    0: {
                        "local_cell_ids": np.array([0], dtype=np.int64),
                        "ghost_cell_ids": np.array([], dtype=np.int64),
                        "ghost_owners": np.array([], dtype=np.int32),
                        "exchange": {},
                    }
                },
            }

            write_model(model_path, topo, fields, boundary_tag, domain_bounds, partition_result)

            part_dir = os.path.join(td, "partitions")
            assert os.path.isdir(part_dir)

            part_path = os.path.join(part_dir, "partition_0.h5")
            assert os.path.isfile(part_path)

            with h5py.File(part_path, "r") as f:
                assert "field/cell" in f
                assert "field/surface" in f
                assert "partition" in f

                felem = f["field/cell"]
                assert "coords" in felem
                assert "mass" in felem
                assert "vp" in felem
                assert "vs" in felem
                assert "density" in felem
                assert "damping" in felem

                part = f["partition"]
                assert part.attrs["n_ranks"] == 1
                assert np.array_equal(part["local_cell_ids"][:], np.array([0], dtype=np.int64))

    def test_partition_multi_rank(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model.h5")
            topo = _make_model_h5(model_path)
            fields = _make_synthetic_fields(n_cell=1, ngll=4)
            boundary_tag = np.array([1, 2, 2, 2, 2, 2], dtype=np.int64)
            domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

            partition_result = {
                "element_to_rank": np.array([0], dtype=np.int32),
                "n_ranks": 1,
                "per_rank": {
                    0: {
                        "local_cell_ids": np.array([0], dtype=np.int64),
                        "ghost_cell_ids": np.array([], dtype=np.int64),
                        "ghost_owners": np.array([], dtype=np.int32),
                        "exchange": {1: {"send_dof": [1, 2, 3], "recv_dof": [1, 2, 3]}},
                    }
                },
            }

            write_model(model_path, topo, fields, boundary_tag, domain_bounds, partition_result)

            part_path = os.path.join(td, "partitions", "partition_0.h5")
            with h5py.File(part_path, "r") as f:
                exch = f["partition/exchange"]
                assert "neighbor_1" in exch
                ng = exch["neighbor_1"]
                assert "send_dof" in ng
                assert "recv_dof" in ng

    def test_is_pml_int8_written(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model.h5")
            topo = _make_model_h5(model_path)
            fields = _make_synthetic_fields(n_cell=1, ngll=4)
            fields["is_pml"] = np.array([True], dtype=np.bool_)
            boundary_tag = np.array([1, 2, 2, 2, 2, 2], dtype=np.int64)
            domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

            write_model(model_path, topo, fields, boundary_tag, domain_bounds)

            with h5py.File(model_path, "r") as f:
                is_pml_data = f["field/cell/is_pml"][:]
                assert is_pml_data.dtype == np.int8
                assert is_pml_data[0] == 1

    def test_no_partition_no_partition_dir(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model.h5")
            topo = _make_model_h5(model_path)
            fields = _make_synthetic_fields(n_cell=1, ngll=4)
            boundary_tag = np.array([1, 2, 2, 2, 2, 2], dtype=np.int64)
            domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

            write_model(
                model_path, topo, fields, boundary_tag, domain_bounds, partition_result=None
            )

            part_dir = os.path.join(td, "partitions")
            assert not os.path.isdir(part_dir)
