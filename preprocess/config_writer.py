"""Config writer — write configs/config.h5."""

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
) -> None:
    """Write configs/config.h5 with simulation, domain, and source data.

    Args:
        config_path: Path to the target file (parent dir created if needed).
        config_module: Loaded config module with simulation parameters.
        domain_bounds: Dict with xmin, xmax, ymin, ymax, zmin, zmax.
        stf_t: Time array [nsteps] float64.
        stf_values: STF amplitude array [nsteps] float64.
        source_xyz: Optional source position [x, y, z] float64.
    """
    parent_dir = os.path.dirname(os.path.abspath(config_path))
    os.makedirs(parent_dir, exist_ok=True)

    with h5py.File(config_path, "w") as f:
        _write_simulation(f, config_module)
        _write_domain(f, domain_bounds)
        _write_source(f, stf_t, stf_values, source_xyz)


def _write_simulation(f: h5py.File, config_module) -> None:
    grp = f.create_group("simulation")
    grp.attrs["title"] = config_module.title

    grp.attrs["polynomial_order"] = int(config_module.polynomial_order)
    grp.attrs["dt"] = float(config_module.output_dt)
    grp.attrs["nsteps"] = int(config_module.nsteps)
    grp.attrs["cfl_safety"] = float(config_module.cfl_safety)
    grp.attrs["cfl_threshold"] = float(config_module.cfl_threshold)
    grp.attrs["checkpoint_interval"] = int(config_module.checkpoint_interval)
    grp.attrs["checkpoint_precision"] = config_module.checkpoint_precision
    grp.attrs["storage_limit_gb"] = float(config_module.storage_limit_gb)


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
) -> None:
    grp = f.create_group("source")

    grp.create_dataset("stf_t", data=stf_t, dtype="float64")
    grp.create_dataset("stf_values", data=stf_values, dtype="float64")

    if source_xyz is not None:
        grp.attrs["x"] = float(source_xyz[0])
        grp.attrs["y"] = float(source_xyz[1])
        grp.attrs["z"] = float(source_xyz[2])