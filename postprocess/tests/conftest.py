"""Shared fixtures and synthetic data helpers for gf_post tests."""

import h5py
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures: synthetic mesh.h5
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_mesh_path(tmp_path):
    """Create a minimal mesh.h5 with one element (unit cube, N=2)."""
    ngll = 3  # N=2
    n_cell = 1

    path = tmp_path / "mesh.h5"
    with h5py.File(path, "w") as f:
        # Topology
        topo = f.create_group("topology")
        topo.create_dataset(
            "vertex_to_coord",
            data=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        topo.create_dataset("cell_to_surface", data=np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64))

        # /field/element
        elem = f.create_group("field/element")
        # GLL coords for unit cube — linear interpolation at GLL points
        xi_1d = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        coords = np.zeros((n_cell, ngll, ngll, ngll, 3), dtype=np.float64)
        for k in range(ngll):
            for j in range(ngll):
                for i in range(ngll):
                    x = 0.5 * (xi_1d[i] + 1.0)
                    y = 0.5 * (xi_1d[j] + 1.0)
                    z = 0.5 * (xi_1d[k] + 1.0)
                    coords[0, k, j, i, :] = [x, y, z]
        elem.create_dataset("coords", data=coords)

        # dxi_dx — for unit cube mapped to [-1,1]^3, dxi/dx = diag(2,2,2)
        dxi_dx = np.zeros((n_cell, ngll, ngll, ngll, 3, 3), dtype=np.float64)
        dxi_dx[0, :, :, :, 0, 0] = 2.0
        dxi_dx[0, :, :, :, 1, 1] = 2.0
        dxi_dx[0, :, :, :, 2, 2] = 2.0
        elem.create_dataset("dxi_dx", data=dxi_dx)

        # is_pml
        elem.create_dataset("is_pml", data=np.zeros(n_cell, dtype=np.int8))

        # jacobian (constant for unit cube)
        jacobian = np.full((n_cell, ngll, ngll, ngll), 0.125, dtype=np.float64)
        elem.create_dataset("jacobian", data=jacobian)

        # /field/surface
        surf = f.create_group("field/surface")
        surf.create_dataset("boundary_tag", data=np.ones(6, dtype=np.int8))

        # /partition
        part = f.create_group("partition")
        part.attrs["n_ranks"] = 1

    return path


@pytest.fixture
def synthetic_mesh_2elem_path(tmp_path):
    """Create mesh.h5 with two elements side by side in x-dir (N=2)."""
    ngll = 3
    n_cell = 2

    path = tmp_path / "mesh_2elem.h5"
    with h5py.File(path, "w") as f:
        # Topology (2 cubes)
        topo = f.create_group("topology")
        topo.create_dataset(
            "vertex_to_coord",
            data=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        topo.create_dataset(
            "cell_to_surface",
            data=np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.int64),
        )

        elem = f.create_group("field/element")
        xi_1d = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        coords = np.zeros((n_cell, ngll, ngll, ngll, 3), dtype=np.float64)
        for e in range(2):
            x_offset = float(e)
            for k in range(ngll):
                for j in range(ngll):
                    for i in range(ngll):
                        x = x_offset + 0.5 * (xi_1d[i] + 1.0)
                        y = 0.5 * (xi_1d[j] + 1.0)
                        z = 0.5 * (xi_1d[k] + 1.0)
                        coords[e, k, j, i, :] = [x, y, z]
        elem.create_dataset("coords", data=coords)

        dxi_dx = np.zeros((n_cell, ngll, ngll, ngll, 3, 3), dtype=np.float64)
        dxi_dx[:, :, :, :, 0, 0] = 2.0
        dxi_dx[:, :, :, :, 1, 1] = 2.0
        dxi_dx[:, :, :, :, 2, 2] = 2.0
        elem.create_dataset("dxi_dx", data=dxi_dx)
        elem.create_dataset("is_pml", data=np.zeros(n_cell, dtype=np.int8))

        surf = f.create_group("field/surface")
        surf.create_dataset("boundary_tag", data=np.ones(12, dtype=np.int8))

        part = f.create_group("partition")
        part.attrs["n_ranks"] = 1

    return path


# ---------------------------------------------------------------------------
# Fixtures: synthetic record file(s)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_record_path(tmp_path):
    """Create a minimal single-rank record.h5 with 2 timesteps, N=2, 1 element."""
    ngll = 3
    n_elem_local = 1
    n_checkpoints = 2

    path = tmp_path / "record_0.h5"
    with h5py.File(path, "w") as f:
        f.attrs["rank"] = 0
        f.attrs["source_direction"] = 0  # x
        f.attrs["dt"] = 0.01
        f.attrs["checkpoint_interval"] = 1
        f.attrs["nsteps"] = 2

        f.create_dataset("local_element_ids", data=np.array([1], dtype=np.int64))

        strain = np.zeros((n_checkpoints, n_elem_local, ngll, ngll, ngll, 6), dtype=np.float64)
        for t in range(n_checkpoints):
            strain[t, 0, :, :, :, 0] = float(t) + 1.0  # xx
            strain[t, 0, :, :, :, 1] = 2.0 * (float(t) + 1.0)  # yy
            strain[t, 0, :, :, :, 2] = 3.0 * (float(t) + 1.0)  # zz
            strain[t, 0, :, :, :, 3] = 0.0  # xy
            strain[t, 0, :, :, :, 4] = 0.0  # xz
            strain[t, 0, :, :, :, 5] = 0.0  # yz
        f.create_dataset("strain", data=strain, maxshape=(None, n_elem_local, ngll, ngll, ngll, 6))

    return path


@pytest.fixture
def synthetic_multirank_records(tmp_path, synthetic_mesh_path):
    """Create 2 rank files for a 2-element mesh, one element each."""
    ngll = 3
    n_checkpoints = 2
    paths = []

    for rank in range(2):
        path = tmp_path / f"record_{rank}.h5"
        with h5py.File(path, "w") as f:
            f.attrs["rank"] = rank
            f.attrs["source_direction"] = 0
            f.attrs["dt"] = 0.01
            f.attrs["checkpoint_interval"] = 1
            f.attrs["nsteps"] = 2

            f.create_dataset("local_element_ids", data=np.array([rank + 1], dtype=np.int64))

            strain = np.zeros((n_checkpoints, 1, ngll, ngll, ngll, 6), dtype=np.float64)
            for t in range(n_checkpoints):
                val = float(t) + 1.0 + float(rank) * 10.0
                strain[t, 0, :, :, :, 0] = val
            f.create_dataset("strain", data=strain, maxshape=(None, 1, ngll, ngll, ngll, 6))
        paths.append(path)

    return paths
