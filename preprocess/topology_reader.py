"""Topology reader — read mesh.h5 /topology/ datasets."""

from dataclasses import dataclass

import h5py
import numpy as np
import numpy.typing as npt


@dataclass
class TopologyData:
    vertex_to_coord: npt.NDArray[np.float64]
    edge_to_vertex: npt.NDArray[np.int64]
    surface_to_edge: npt.NDArray[np.int64]
    cell_to_surface: npt.NDArray[np.int64]
    n_vertex: int
    n_edge: int
    n_surface: int
    n_cell: int


def read_topology(path: str) -> TopologyData:
    with h5py.File(path, "r") as f:
        if "topology" not in f:
            raise ValueError(f"File '{path}' missing /topology/ group")
        topo = f["topology"]

        vertex_to_coord = topo["vertex_to_coord"][:].astype(np.float64)
        edge_to_vertex = topo["edge_to_vertex"][:].astype(np.int64)
        surface_to_edge = topo["surface_to_edge"][:].astype(np.int64)
        cell_to_surface = topo["cell_to_surface"][:].astype(np.int64)

        n_vertex = int(topo.attrs["n_vertex"])
        n_edge = int(topo.attrs["n_edge"])
        n_surface = int(topo.attrs["n_surface"])
        n_cell = int(topo.attrs["n_cell"])

    return TopologyData(
        vertex_to_coord=vertex_to_coord,
        edge_to_vertex=edge_to_vertex,
        surface_to_edge=surface_to_edge,
        cell_to_surface=cell_to_surface,
        n_vertex=n_vertex,
        n_edge=n_edge,
        n_surface=n_surface,
        n_cell=n_cell,
    )
