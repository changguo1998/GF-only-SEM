"""Config writer — write config.h5."""

import os

import h5py
import numpy as np
import numpy.typing as npt


def write_config(
    config_path: str,
    config_module,
    domain_bounds: dict[str, float],
    stf_t: npt.NDArray[np.float64],
    stf_values: npt.NDArray[np.float64],
    source_xyz: npt.NDArray[np.float64] | None = None,
    source_loc_result: dict | None = None,
    *,
    solver_dt: float | None = None,
    snapshot_stride: int | None = None,
    nsteps: int | None = None,
    recording_map: dict | None = None,
) -> None:
    """Write config.h5 with simulation, domain, and source data.

    Args:
        config_path: Path to the target file (parent dir created if needed).
        config_module: Loaded config module with simulation parameters.
        domain_bounds: Dict with xmin, xmax, ymin, ymax, zmin, zmax.
        stf_t: Time array [nsteps] float64.
        stf_values: STF amplitude array [nsteps] float64.
        source_xyz: Optional source position [x, y, z] float64.
        source_loc_result: Optional dict from locate_source() with
            element_ids, xi, eta, zeta, weights, n_src_elem.
        solver_dt: Auto-computed CFL timestep. Falls back to config_module.output_dt_s.
        snapshot_stride: Solver steps per snapshot. Falls back to 1.
        nsteps: Total solver steps. Falls back to config_module.nsteps.
    """
    if solver_dt is None:
        solver_dt = float(config_module.output_dt_s)
    if snapshot_stride is None:
        snapshot_stride = 1
    if nsteps is None:
        nsteps = int(config_module.nsteps)

    parent_dir = os.path.dirname(os.path.abspath(config_path))
    os.makedirs(parent_dir, exist_ok=True)

    with h5py.File(config_path, "w") as f:
        _write_simulation(
            f, config_module, solver_dt, snapshot_stride, nsteps, recording_map=recording_map
        )
        _write_domain(f, domain_bounds)
        _write_source(f, stf_t, stf_values, source_xyz, source_loc_result)


def _write_simulation(
    f: h5py.File,
    config_module,
    solver_dt: float,
    snapshot_stride: int,
    nsteps: int,
    recording_map: dict | None = None,
) -> None:
    grp = f.create_group("simulation")
    grp.attrs["title"] = config_module.title

    grp.attrs["polynomial_order"] = int(config_module.polynomial_order)
    grp.attrs["solver_dt"] = solver_dt
    grp.attrs["output_dt_s"] = float(config_module.output_dt_s)
    grp.attrs["snapshot_stride"] = snapshot_stride
    grp.attrs["nsteps"] = nsteps
    grp.attrs["cfl_safety"] = float(config_module.cfl_safety)
    grp.attrs["snapshot_precision"] = config_module.snapshot_precision
    grp.attrs["storage_limit_gb"] = float(config_module.storage_limit_gb)
    grp.attrs["record_depth_max_m"] = float(config_module.record_depth_max_m)
    # Mesh grid dimensions
    grp.attrs["nx_elements"] = int(getattr(config_module, "nx_elements", 0))
    grp.attrs["ny_elements"] = int(getattr(config_module, "ny_elements", 0))
    grp.attrs["nz_elements"] = int(getattr(config_module, "nz_elements", 0))
    # PML thickness (elements)
    pml = getattr(config_module, "pml_thickness", {})
    grp.attrs["pml_xmin"] = int(pml.get("xmin", 0))
    grp.attrs["pml_xmax"] = int(pml.get("xmax", 0))
    grp.attrs["pml_ymin"] = int(pml.get("ymin", 0))
    grp.attrs["pml_ymax"] = int(pml.get("ymax", 0))
    grp.attrs["pml_zmin"] = int(pml.get("zmin", 0))
    grp.attrs["pml_zmax"] = int(pml.get("zmax", 0))
    grp.attrs["log_stride"] = int(getattr(config_module, "log_stride", 1))
    # Tile sizes (element counts)
    tilex = getattr(config_module, "tilex_elements", [])
    tiley = getattr(config_module, "tiley_elements", [])
    import numpy as np

    if tilex:
        grp.create_dataset("tilex_elements", data=np.array(tilex, dtype=np.int64))
    if tiley:
        grp.create_dataset("tiley_elements", data=np.array(tiley, dtype=np.int64))
    if recording_map is not None:
        grp.attrs["record_depth_actual_m"] = recording_map.get(
            "record_depth_actual_m", float(config_module.record_depth_max_m)
        )


def _write_domain(f: h5py.File, domain_bounds: dict[str, float]) -> None:
    grp = f.create_group("domain")
    for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        grp.attrs[key] = float(domain_bounds[key])

    # Store pml_thickness if available from attrs of parent
    # (caller may add this as a separate group attribute if needed)


def _write_source(
    f: h5py.File,
    stf_t: npt.NDArray[np.float64],
    stf_values: npt.NDArray[np.float64],
    source_xyz: npt.NDArray[np.float64] | None,
    source_loc_result: dict | None = None,
) -> None:
    grp = f.create_group("source")

    grp.create_dataset("stf_t", data=stf_t, dtype="float64")
    grp.create_dataset("stf_values", data=stf_values, dtype="float64")

    if source_xyz is not None:
        grp.attrs["x"] = float(source_xyz[0])
        grp.attrs["y"] = float(source_xyz[1])
        grp.attrs["z"] = float(source_xyz[2])

    # Write precomputed source element list + Lagrange weights
    if source_loc_result is not None:
        n_src = source_loc_result.get("n_src_elem", 0)
        grp.attrs["n_src_elements"] = n_src
        if n_src > 0:
            elem_grp = grp.create_group("elements")
            elem_grp.create_dataset(
                "element_ids", data=source_loc_result["element_ids"], dtype="int64"
            )
            elem_grp.create_dataset("xi", data=source_loc_result["xi"], dtype="float64")
            elem_grp.create_dataset("eta", data=source_loc_result["eta"], dtype="float64")
            elem_grp.create_dataset("zeta", data=source_loc_result["zeta"], dtype="float64")
            elem_grp.create_dataset("weights", data=source_loc_result["weights"], dtype="float64")
