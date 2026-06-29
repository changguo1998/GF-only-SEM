"""Model writer — extend mesh.h5 and write partition_{r}.h5 files."""

import os

import h5py
import numpy as np
import numpy.typing as npt

from preprocess.topology_reader import TopologyData


def compute_element_tile_index(
    n_cell: int,
    nx_elements: int,
    ny_elements: int,
    pml_xmin: int,
    pml_xmax: int,
    pml_ymin: int,
    pml_ymax: int,
    tilex_elements: list[int],
    tiley_elements: list[int],
) -> npt.NDArray[np.int64]:
    """Compute tile index for every element in a structured hex mesh.

    Elements are numbered row-major: x fastest, then y, then z.
    PML elements at mesh boundaries get tile_index = -1.
    Non-PML interior elements are partitioned into horizontal tiles
    following tilex_elements / tiley_elements.

    Returns:
        [n_cell] int64 array: tile_index (0..n_tiles-1), with -1 for PML.
    """
    n_tilex = len(tilex_elements)
    n_tiley = len(tiley_elements)
    if n_tilex == 0 or n_tiley == 0:
        return np.full(n_cell, -1, dtype=np.int64)

    # Cumulative tile boundaries in interior-element space
    tilex_cumul = [0]
    for sz in tilex_elements:
        tilex_cumul.append(tilex_cumul[-1] + sz)
    tiley_cumul = [0]
    for sz in tiley_elements:
        tiley_cumul.append(tiley_cumul[-1] + sz)

    n_interior_x = nx_elements - pml_xmin - pml_xmax
    n_interior_y = ny_elements - pml_ymin - pml_ymax

    tile_index = np.full(n_cell, -1, dtype=np.int64)

    for elem_idx in range(n_cell):
        # Row-major: x fastest, then y, then z
        i = elem_idx % nx_elements
        j = (elem_idx // nx_elements) % ny_elements

        # Interior element index (non-PML region)
        interior_i = i - pml_xmin
        interior_j = j - pml_ymin

        if interior_i < 0 or interior_i >= n_interior_x:
            continue  # PML in x
        if interior_j < 0 or interior_j >= n_interior_y:
            continue  # PML in y

        # Find tile in x
        tile_x = -1
        for tx in range(n_tilex):
            if tilex_cumul[tx] <= interior_i < tilex_cumul[tx + 1]:
                tile_x = tx
                break
        if tile_x < 0:
            continue

        # Find tile in y
        tile_y = -1
        for ty in range(n_tiley):
            if tiley_cumul[ty] <= interior_j < tiley_cumul[ty + 1]:
                tile_y = ty
                break
        if tile_y < 0:
            continue

        tile_index[elem_idx] = tile_y * n_tilex + tile_x

    return tile_index


def write_model(
    mesh_path: str,
    topology: TopologyData,
    fields: dict[str, npt.NDArray],
    boundary_tag: npt.NDArray[np.int64],
    domain_bounds: dict[str, float],
    partition_result: dict | None = None,
    recording_map: dict | None = None,
    tile_config: dict | None = None,
) -> None:
    """Extend mesh.h5 with field data and write partition files.

    mesh.h5 is extended (append mode) with:
      /field/element/coords, /field/element/dxi_dx, /field/element/jacobian
      /field/element/is_pml
      /field/element/tile_index
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
        tile_config: Optional dict with nx_elements, ny_elements,
                     pml_xmin, pml_xmax, pml_ymin, pml_ymax,
                     tilex_elements, tiley_elements.
    """
    _extend_mesh_h5(mesh_path, fields, boundary_tag, domain_bounds, tile_config=tile_config)

    if partition_result is not None:
        _write_partition_files(
            mesh_path,
            topology,
            fields,
            boundary_tag,
            partition_result,
            recording_map=recording_map,
            tile_config=tile_config,
        )


def _extend_mesh_h5(
    mesh_path: str,
    fields: dict[str, npt.NDArray],
    boundary_tag: npt.NDArray[np.int64],
    domain_bounds: dict[str, float],
    tile_config: dict | None = None,
) -> None:
    with h5py.File(mesh_path, "a") as f:
        fld = f.require_group("field")
        felem = fld.require_group("element")

        _write_dataset(felem, "coords", fields.get("coords"), dtype="float64")
        _write_dataset(felem, "dxi_dx", fields.get("dxi_dx"), dtype="float64")
        _write_dataset(felem, "jacobian", fields.get("jacobian"), dtype="float64")

        is_pml = fields.get("is_pml", np.array([], dtype=np.bool_))
        _write_dataset(felem, "is_pml", is_pml.astype(np.int8), dtype="int8")

        # Write tile_index to mesh.h5 if tile config is available
        if tile_config is not None:
            n_cell = int(f["topology"].attrs["n_cell"])
            tile_idx = compute_element_tile_index(
                n_cell,
                tile_config.get("nx_elements", 0),
                tile_config.get("ny_elements", 0),
                tile_config.get("pml_xmin", 0),
                tile_config.get("pml_xmax", 0),
                tile_config.get("pml_ymin", 0),
                tile_config.get("pml_ymax", 0),
                tile_config.get("tilex_elements", []),
                tile_config.get("tiley_elements", []),
            )
            _write_dataset(felem, "tile_index", tile_idx, dtype="int64")

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
    recording_map: dict | None = None,
    tile_config: dict | None = None,
) -> None:
    mesh_dir = os.path.dirname(os.path.abspath(mesh_path))
    parts_dir = os.path.join(mesh_dir, "partitions")
    os.makedirs(parts_dir, exist_ok=True)

    element_to_rank = partition_result.get("element_to_rank")
    n_ranks = partition_result.get("n_ranks", 1)
    per_rank = partition_result.get("per_rank", {})  # dict rank → dict

    if element_to_rank is None:
        element_to_rank = np.zeros(topology.n_cell, dtype=np.int32)

    field_keys = ["coords", "dxi_dx", "jacobian", "mass", "vp", "vs", "density", "damping"]

    # Precompute global tile_index if tile config provided
    global_tile_index = None
    if tile_config is not None:
        global_tile_index = compute_element_tile_index(
            topology.n_cell,
            tile_config.get("nx_elements", 0),
            tile_config.get("ny_elements", 0),
            tile_config.get("pml_xmin", 0),
            tile_config.get("pml_xmax", 0),
            tile_config.get("pml_ymin", 0),
            tile_config.get("pml_ymax", 0),
            tile_config.get("tilex_elements", []),
            tile_config.get("tiley_elements", []),
        )

    for r in range(n_ranks):
        rk = per_rank.get(r, {})
        local_ids = np.asarray(rk.get("local_element_ids", []), dtype=np.int64)
        ghost_ids = np.asarray(rk.get("ghost_element_ids", []), dtype=np.int64)
        ghost_owners = np.asarray(rk.get("ghost_owners", []), dtype=np.int32)

        local_zero = local_ids  # already 0-based from partition.py
        n_local = len(local_zero)

        part_path = os.path.join(parts_dir, f"partition_{r}.h5")
        with h5py.File(part_path, "w") as f:
            fld_grp = f.create_group("field")
            felem_grp = fld_grp.create_group("element")

            for key in field_keys:
                arr = fields.get(key)
                if arr is None:
                    continue
                local_data = arr[local_zero] if n_local > 0 else np.array([], dtype=arr.dtype)
                _write_dataset(felem_grp, key, local_data, compression=True)

            # Write tile_index to partition file
            if global_tile_index is not None:
                local_tile = (
                    global_tile_index[local_zero]
                    if n_local > 0
                    else np.array([], dtype=np.int64)
                )
                _write_dataset(felem_grp, "tile_index", local_tile, dtype="int64")

            fsurf_grp = fld_grp.create_group("surface")
            _write_dataset(fsurf_grp, "boundary_tag", boundary_tag, dtype="int64")

            part_grp = f.create_group("partition")
            part_grp.attrs["n_ranks"] = n_ranks
            _write_dataset(part_grp, "element_to_rank", element_to_rank, dtype="int32")
            _write_dataset(part_grp, "local_element_ids", local_ids, dtype="int64")
            _write_dataset(part_grp, "ghost_element_ids", ghost_ids, dtype="int64")
            _write_dataset(part_grp, "ghost_owners", ghost_owners, dtype="int32")

            exchange = rk.get("exchange", {})
            if exchange:
                exch_grp = part_grp.create_group("exchange")
                for neighbor, dof_dict in exchange.items():
                    ng = exch_grp.create_group(f"neighbor_{neighbor}")
                    send_arr = np.asarray(dof_dict.get("send_dof", []), dtype=np.int32)
                    recv_arr = np.asarray(dof_dict.get("recv_dof", []), dtype=np.int32)
                    _write_dataset(ng, "send_dof", send_arr, dtype="int32")
                    _write_dataset(ng, "recv_dof", recv_arr, dtype="int32")
            # Write recording map if present
            if recording_map is not None:
                per_rank_rec = recording_map.get("per_rank_recording", {}).get(r)
                if per_rank_rec is not None and len(per_rank_rec.get("vertex_ids", [])) > 0:
                    rec_grp = f.create_group("recording")
                    rec_grp.attrs["basis"] = "mesh_vertices"
                    rec_grp.attrs["record_depth_max_m"] = recording_map.get(
                        "record_depth_actual_m", 0.0
                    )
                    rec_grp.attrs["record_depth_actual_m"] = recording_map.get(
                        "record_depth_actual_m", 0.0
                    )

                    rec_grp.attrs["excludes_pml"] = True
                    _write_dataset(
                        rec_grp,
                        "save_element_mask",
                        np.array(per_rank_rec["save_element_mask"], dtype=bool),
                        dtype="bool",
                    )
                    _write_dataset(
                        rec_grp,
                        "vertex_ids",
                        np.array(per_rank_rec["vertex_ids"], dtype=np.int64),
                        dtype="int64",
                    )
                    _write_dataset(
                        rec_grp,
                        "source_element_local_index",
                        np.array(per_rank_rec["source_element_local_index"], dtype=np.int32),
                        dtype="int32",
                    )
                    _write_dataset(
                        rec_grp,
                        "source_corner_index",
                        np.array(per_rank_rec["source_corner_index"], dtype=np.int32),
                        dtype="int32",
                    )


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