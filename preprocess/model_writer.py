"""Model writer — extend mesh.h5 and write partition_{r}.h5 files."""

import os
import h5py
import numpy as np
import numpy.typing as npt

from preprocess.topology_reader import TopologyData


def write_model(
    mesh_path: str,
    topology: TopologyData,
    fields: dict[str, npt.NDArray],
    boundary_tag: npt.NDArray[np.int64],
    domain_bounds: dict[str, float],
    partition_result: dict | None = None,
) -> None:
    """Extend mesh.h5 with field data and write partition files.

    mesh.h5 is extended (append mode) with:
      /field/element/coords, /field/element/dxi_dx, /field/element/jacobian
      /field/element/is_pml
      /field/surface/boundary_tag

    When partition_result is provided, per-rank partition_{r}.h5 files are
    written to the partitions/ subdirectory with local element field data
    and partition metadata.

    Args:
        mesh_path: Path to mesh.h5 (extended in-place).
        topology: Mesh topology (used for partition files).
        fields: Dict with keys: coords, jacobian, dxi_dx, mass, vp, vs,
                density, is_pml, damping.
        boundary_tag: Surface boundary tags [n_surface] int64.
        domain_bounds: Dict with xmin, xmax, ymin, ymax, zmin, zmax.
        partition_result: Optional partition output from partition().
    """
    _extend_mesh_h5(mesh_path, fields, boundary_tag, domain_bounds)

    if partition_result is not None:
        _write_partition_files(mesh_path, topology, fields, boundary_tag,
                               partition_result)


def _extend_mesh_h5(
    mesh_path: str,
    fields: dict[str, npt.NDArray],
    boundary_tag: npt.NDArray[np.int64],
    domain_bounds: dict[str, float],
) -> None:
    with h5py.File(mesh_path, "a") as f:
        fld = f.require_group("field")
        felem = fld.require_group("element")

        _write_dataset(felem, "coords", fields.get("coords"), dtype="float64")
        _write_dataset(felem, "dxi_dx", fields.get("dxi_dx"), dtype="float64")
        _write_dataset(felem, "jacobian", fields.get("jacobian"), dtype="float64")

        is_pml = fields.get("is_pml", np.array([], dtype=np.bool_))
        _write_dataset(felem, "is_pml", is_pml.astype(np.int8), dtype="int8")

        fsurf = fld.require_group("surface")
        if boundary_tag is not None:
            _write_dataset(fsurf, "boundary_tag", boundary_tag, dtype="int64")

        domain = f.require_group("domain")
        for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
            domain.attrs[key] = float(domain_bounds[key])


def _write_partition_files(
    mesh_path: str,
    topology: TopologyData,
    fields: dict[str, npt.NDArray],
    boundary_tag: npt.NDArray[np.int64],
    partition_result: dict,
) -> None:
    mesh_dir = os.path.dirname(os.path.abspath(mesh_path))
    parts_dir = os.path.join(mesh_dir, "partitions")
    os.makedirs(parts_dir, exist_ok=True)

    element_to_rank = partition_result.get("element_to_rank")
    n_ranks = partition_result.get("n_ranks", 1)

    if element_to_rank is None:
        element_to_rank = np.zeros(topology.n_cell, dtype=np.int32)

    per_rank = partition_result.get("per_rank", {})

    field_keys = ["coords", "dxi_dx", "jacobian", "mass", "vp", "vs",
                   "density", "is_pml", "damping"]

    for r in range(n_ranks):
        local_ids = per_rank.get(r, {}).get(
            "local_element_ids",
            np.where(element_to_rank == r)[0] + 1,
        )
        ghost_ids = per_rank.get(r, {}).get(
            "ghost_element_ids",
            np.array([], dtype=np.int64),
        )
        ghost_owners = per_rank.get(r, {}).get(
            "ghost_owners",
            np.array([], dtype=np.int32),
        )

        local_zero = np.asarray(local_ids, dtype=np.int64) - 1
        n_local = len(local_zero)

        part_path = os.path.join(parts_dir, f"partition_{r}.h5")
        with h5py.File(part_path, "w") as f:
            fld_grp = f.create_group("field")
            felem_grp = fld_grp.create_group("element")

            for key in field_keys:
                arr = fields.get(key)
                if arr is None:
                    continue
                local_data = arr[local_zero] if n_local > 0 else np.array([])
                _write_dataset(felem_grp, key, local_data, compression=True)

            fsurf_grp = fld_grp.create_group("surface")
            if n_local > 0:
                # map surface boundary tags: identify surfaces belonging to
                # local cells and assign boundary_tag values
                c2s = topology.cell_to_surface
                n_surf_local = topology.n_surface
                _write_dataset(fsurf_grp, "boundary_tag", boundary_tag,
                               dtype="int64")

            part_grp = f.create_group("partition")
            part_grp.attrs["n_ranks"] = n_ranks
            _write_dataset(part_grp, "element_to_rank", element_to_rank,
                           dtype="int32")

            _write_dataset(part_grp, "local_element_ids",
                           np.asarray(local_ids, dtype=np.int64), dtype="int64")
            _write_dataset(part_grp, "ghost_element_ids",
                           np.asarray(ghost_ids, dtype=np.int64), dtype="int64")
            _write_dataset(part_grp, "ghost_owners",
                           np.asarray(ghost_owners, dtype=np.int32),
                           dtype="int32")

            exchange = per_rank.get(r, {}).get("exchange", {})
            if exchange:
                exch_grp = part_grp.create_group("exchange")
                for neighbor, pairs in exchange.items():
                    _write_dataset(exch_grp, f"neighbor_{neighbor}",
                                   np.asarray(pairs, dtype=np.int64),
                                   dtype="int64")


def _write_dataset(
    group: h5py.Group,
    name: str,
    data: npt.NDArray | None,
    *,
    dtype: str | None = None,
    compression: bool = False,
) -> None:
    if name in group:
        del group[name]
    if data is None or data.size == 0:
        return
    kwargs = {}
    if compression:
        kwargs["compression"] = "gzip"
        kwargs["compression_opts"] = 4
    dset = group.create_dataset(name, data=data, dtype=dtype, **kwargs)