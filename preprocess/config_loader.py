"""Config loader — import config.py via importlib, validate required fields."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REQUIRED_KEYS = {
    "title": (str,),
    "polynomial_order": (int,),
    "output_dt_s": (int, float),
    "total_duration_s": (int, float),
    "cfl_safety": (int, float),
    "snapshot_precision": (str,),
    "storage_limit_gb": (int, float),
    "n_ranks": (int,),
    "pml_thickness": (dict,),
    "source_x_m": (int, float),
    "source_y_m": (int, float),
    "record_depth_max_m": (int, float),
    "tilex_elements": (list,),
    "tiley_elements": (list,),
    "log_stride": (int,),
}

REQUIRED_CALLABLES = ["stf_func", "vp_m_s", "vs_m_s", "density_kg_m3"]


class ConfigValidationError(Exception):
    """Raised when config validation fails."""

    pass


def load_config(config_path: str, validate: bool = True) -> ModuleType:
    """Load a Python config script as a module and optionally validate.

    Args:
        config_path: Path to config.py file.
        validate: If True, validate all required fields.

    Returns:
        The loaded config module.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise ConfigValidationError(f"Config file not found: {config_path}")

    # Derive unique module name from path
    module_name = f"_config_{path.stem}_{id(path)}"

    # Add parent dir to sys.path temporarily
    parent_dir = str(path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ConfigValidationError(f"Could not load config file: {config_path}")

    mod = importlib.util.module_from_spec(spec)
    # Store original sys.path to restore later
    old_path = sys.path[:]

    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise ConfigValidationError(f"Error executing config file: {e}") from e
    finally:
        sys.path = old_path

    if validate:
        validate_config(mod)

    return mod


def validate_config(mod: ModuleType) -> None:
    """Validate a loaded config module.

    Checks required attributes, their types, and valid ranges.

    Raises:
        ConfigValidationError: If any validation fails.
    """
    # Check required scalar attributes
    for key, expected_types in REQUIRED_KEYS.items():
        if not hasattr(mod, key):
            raise ConfigValidationError(f"Missing required config field: '{key}'")
        val = getattr(mod, key)
        if not isinstance(val, expected_types):
            raise ConfigValidationError(
                f"Config field '{key}' has wrong type. Expected {expected_types}, got {type(val)}"
            )

    # Check required callables
    for name in REQUIRED_CALLABLES:
        if not hasattr(mod, name):
            raise ConfigValidationError(f"Missing required config callable: '{name}'")
        fn = getattr(mod, name)
        if not callable(fn):
            raise ConfigValidationError(f"Config field '{name}' must be callable, got {type(fn)}")

    # Validate ranges
    _validate_range(mod)
    _validate_pml_thickness(mod.pml_thickness)
    _validate_snapshot_precision(mod.snapshot_precision)
    _validate_tile_sizes(mod)


def _validate_range(mod: ModuleType) -> None:
    """Validate numeric ranges for config fields."""
    errors = []

    if mod.polynomial_order < 1:
        errors.append(f"polynomial_order must be >= 1, got {mod.polynomial_order}")

    if mod.output_dt_s <= 0:
        errors.append(f"output_dt_s must be > 0, got {mod.output_dt_s}")

    if mod.total_duration_s <= 0:
        errors.append(f"total_duration_s must be > 0, got {mod.total_duration_s}")

    if not (0 < mod.cfl_safety < 1):
        errors.append(f"cfl_safety must be between 0 and 1, got {mod.cfl_safety}")

    if mod.storage_limit_gb <= 0:
        errors.append(f"storage_limit_gb must be > 0, got {mod.storage_limit_gb}")

    if mod.n_ranks < 1:
        errors.append(f"n_ranks must be >= 1, got {mod.n_ranks}")

    if mod.log_stride < 1:
        errors.append(f"log_stride must be >= 1, got {mod.log_stride}")

    if errors:
        raise ConfigValidationError("; ".join(errors))


def _validate_pml_thickness(pml: dict) -> None:
    """Validate pml_thickness dict."""
    required_keys = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
    if set(pml.keys()) != required_keys:
        raise ConfigValidationError(
            f"pml_thickness must have keys {required_keys}, got {set(pml.keys())}"
        )
    for key, val in pml.items():
        if not isinstance(val, int) or val < 0:
            raise ConfigValidationError(f"pml_thickness['{key}'] must be >= 0 integer, got {val}")


def _validate_tile_sizes(mod: ModuleType) -> None:
    """Validate tilex_elements and tiley_elements against mesh dims and PML."""
    tilex = getattr(mod, "tilex_elements", [])
    tiley = getattr(mod, "tiley_elements", [])
    pml = getattr(mod, "pml_thickness", {})
    nx = getattr(mod, "nx_elements", 0)
    ny = getattr(mod, "ny_elements", 0)

    if not isinstance(tilex, list) or not tilex:
        raise ConfigValidationError("tilex_elements must be a non-empty list")
    if not isinstance(tiley, list) or not tiley:
        raise ConfigValidationError("tiley_elements must be a non-empty list")

    for val in tilex:
        if not isinstance(val, int) or val < 1:
            raise ConfigValidationError(f"tilex_elements values must be positive ints, got {val}")
    for val in tiley:
        if not isinstance(val, int) or val < 1:
            raise ConfigValidationError(f"tiley_elements values must be positive ints, got {val}")

    nx_interior = nx - pml.get("xmin", 0) - pml.get("xmax", 0)
    ny_interior = ny - pml.get("ymin", 0) - pml.get("ymax", 0)

    if sum(tilex) != nx_interior:
        raise ConfigValidationError(
            f"sum(tilex_elements)={sum(tilex)} != nx_interior={nx_interior} "
            f"(nx={nx} - pml_xmin={pml.get('xmin', 0)} - pml_xmax={pml.get('xmax', 0)})"
        )
    if sum(tiley) != ny_interior:
        raise ConfigValidationError(
            f"sum(tiley_elements)={sum(tiley)} != ny_interior={ny_interior} "
            f"(ny={ny} - pml_ymin={pml.get('ymin', 0)} - pml_ymax={pml.get('ymax', 0)})"
        )


def _validate_snapshot_precision(precision: str) -> None:
    """Validate snapshot_precision."""
    if precision not in ("float32", "float64"):
        raise ConfigValidationError(
            f"snapshot_precision must be 'float32' or 'float64', got '{precision}'"
        )
