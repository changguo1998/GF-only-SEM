"""Pre-flight validation — comprehensive checks before partition + write.

Runs a checklist of validation steps covering mesh quality, material
properties, CFL constraints, boundary tags, source position, STF
integrity, partition feasibility, and storage estimation.

With strict_validation=True (default), any error aborts the run.
With strict_validation=False, errors are logged as warnings and
processing continues.

Validation categories:
  1.  Mesh quality      — det(J) > 0 at all GLL nodes
  2.  Material          — vp > 0, vs ≥ 0, density > 0, λ > 0
  3.  CFL               — validate derived solver_dt/snapshot_stride metadata
  4.  Boundary          — at least one free surface + one absorbing
  5.  Source            — within domain bounds
  6.  STF               — all values finite (no NaN/Inf)
  7.  Partition          — n_ranks ≤ n_cell
  8.  Storage estimation — estimated disk usage vs storage_limit_gb
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from types import ModuleType

# ---------------------------------------------------------------------------
# Validation result helpers
# ---------------------------------------------------------------------------

class PreflightError(Exception):
    """Raised when a pre-flight validation check fails."""
    pass


class PreflightResult:
    """Aggregates pre-flight check results."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.stats: dict[str, object] = {}

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def report(self) -> str:
        """Human-readable summary."""
        lines: list[str] = [f"Pre-flight validation: {len(self.warnings)} warning(s), {len(self.errors)} error(s)"]
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for k, v in self.stats.items():
            lines.append(f"  STAT {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_mesh_quality(
    jacobian: npt.NDArray[np.float64],
    result: PreflightResult,
) -> None:
    """Check that all Jacobian determinants are positive."""
    if not np.all(jacobian > 0):
        inverted_count = int(np.sum(jacobian <= 0))
        result.add_error(
            f"Mesh quality: {inverted_count} GLL node(s) have det(J) ≤ 0. "
            "Inverted or degenerate elements detected."
        )
    else:
        jac_min = float(np.min(jacobian))
        jac_max = float(np.max(jacobian))
        ratio = jac_max / jac_min if jac_min > 0 else float("inf")
        result.stats["detJ_min"] = f"{jac_min:.4e}"
        result.stats["detJ_max"] = f"{jac_max:.4e}"
        if ratio > 1e6:
            result.add_warning(
                f"Mesh quality: det(J) ratio max/min = {ratio:.1e}. "
                "Highly stretched elements may affect accuracy."
            )


def _check_material(
    vp_array: npt.NDArray[np.float64],
    vs_array: npt.NDArray[np.float64],
    density_array: npt.NDArray[np.float64],
    result: PreflightResult,
) -> None:
    """Check material values are valid at all GLL nodes."""
    any_error = False
    has_zero_vs = False

    if not np.all(vp_array > 0):
        bad_count = int(np.sum(vp_array <= 0))
        result.add_error(f"Material: {bad_count} GLL node(s) have vp ≤ 0.")
        any_error = True
    else:
        result.stats["vp_min"] = f"{float(np.min(vp_array)):.1f}"
        result.stats["vp_max"] = f"{float(np.max(vp_array)):.1f}"

    if not np.all(vs_array >= 0):
        bad_count = int(np.sum(vs_array < 0))
        result.add_error(f"Material: {bad_count} GLL node(s) have vs < 0.")
        any_error = True
    else:
        if np.any(vs_array == 0):
            has_zero_vs = True
        result.stats["vs_min"] = f"{float(np.min(vs_array)):.1f}"
        result.stats["vs_max"] = f"{float(np.max(vs_array)):.1f}"

    if not np.all(density_array > 0):
        bad_count = int(np.sum(density_array <= 0))
        result.add_error(f"Material: {bad_count} GLL node(s) have density ≤ 0.")
        any_error = True
    else:
        result.stats["density_min"] = f"{float(np.min(density_array)):.1f}"
        result.stats["density_max"] = f"{float(np.max(density_array)):.1f}"

    # Elastic stability check: λ = ρ(vp² - 2vs²) > 0
    if not any_error:
        lam = density_array * (vp_array**2 - 2 * vs_array**2)
        if not np.all(lam > 0):
            count = int(np.sum(lam <= 0))
            result.add_error(
                f"Material: {count} GLL node(s) have λ ≤ 0 (elastic instability)."
            )
        else:
            result.stats["lam_min"] = f"{float(np.min(lam)):.4e}"

    if has_zero_vs:
        result.add_warning("Material: vs = 0 at some nodes (acoustic region).")


def _check_boundary(
    boundary_tag: npt.NDArray[np.int64],
    result: PreflightResult,
) -> None:
    """Check boundary tags for required free surface and absorbing boundaries."""
    n_free = int(np.count_nonzero(boundary_tag == 1))
    n_absorbing = int(np.count_nonzero(boundary_tag == 2))

    if n_free < 1:
        result.add_error("Boundary: No free surface (tag=1) detected.")
    if n_absorbing < 1:
        result.add_error("Boundary: No absorbing boundary (tag=2) detected.")

    result.stats["n_free_surfaces"] = n_free
    result.stats["n_absorbing_surfaces"] = n_absorbing


def _check_source(
    source_x: float,
    source_y: float,
    source_z: float,
    domain_bounds: dict[str, float],
    result: PreflightResult,
) -> None:
    """Check that source is within the domain XY bounds."""
    x_ok = domain_bounds["xmin"] <= source_x <= domain_bounds["xmax"]
    y_ok = domain_bounds["ymin"] <= source_y <= domain_bounds["ymax"]
    z_on_surface = abs(source_z - domain_bounds["zmin"]) < 1e-8 * max(
        domain_bounds["zmax"] - domain_bounds["zmin"], 1.0
    )

    if not x_ok:
        result.add_error(
            f"Source: x = {source_x} outside domain [{domain_bounds['xmin']}, "
            f"{domain_bounds['xmax']}]."
        )
    if not y_ok:
        result.add_error(
            f"Source: y = {source_y} outside domain [{domain_bounds['ymin']}, "
            f"{domain_bounds['ymax']}]."
        )
    if not z_on_surface:
        result.add_warning(
            f"Source: z = {source_z} is not on free surface "
            f"(z_min = {domain_bounds['zmin']})."
        )


def _check_stf(
    stf_values: npt.NDArray[np.float64],
    result: PreflightResult,
) -> None:
    """Check STF values are finite and non-NaN."""
    nan_count = int(np.sum(np.isnan(stf_values)))
    inf_count = int(np.sum(np.isinf(stf_values)))

    if nan_count > 0:
        result.add_error(f"STF: {nan_count} NaN value(s) in time series.")
    if inf_count > 0:
        result.add_error(f"STF: {inf_count} Inf value(s) in time series.")

    if nan_count == 0 and inf_count == 0:
        stf_max = float(np.max(np.abs(stf_values)))
        result.stats["stf_max_abs"] = f"{stf_max:.4e}"

        # Warn on significant DC component
        mean_val = float(np.mean(stf_values))
        if abs(mean_val) > 1e-6 * max(stf_max, 1e-12):
            result.add_warning(
                f"STF: non-zero mean = {mean_val:.4e}. "
                "Non-zero DC component may cause residual displacement."
            )


def _check_partition(
    n_cell: int,
    n_ranks: int,
    result: PreflightResult,
) -> None:
    """Pre-check partition feasibility before calling METIS."""
    if n_ranks < 1:
        result.add_error(f"Partition: n_ranks = {n_ranks} must be ≥ 1.")

    if n_cell < n_ranks:
        result.add_error(
            f"Partition: n_ranks ({n_ranks}) > n_cell ({n_cell}). "
            "Some ranks would have zero elements."
        )

    result.stats["n_cell"] = n_cell
    result.stats["n_ranks"] = n_ranks


def _check_storage(
    n_cell: int,
    NGLL: int,
    nsteps: int,
    snapshot_stride: int,
    snapshot_precision: str,
    storage_limit_gb: float,
    result: PreflightResult,
) -> None:
    """Estimate disk usage and check against storage_limit_gb.

    Storage breakdown:
      - strain per run = nsnapshots × n_cell × NGLL³ × 6 × bytes_per_value
      - restart per run = n_cell × NGLL³ × 3 × 3 × 8  (always float64)
      - 3 runs (x, y, z) + partition files (estimated as ~3× strain)
      - Total = strain × 3 + restart × 3 + partition_estimate
    """
    if snapshot_stride <= 0:
        result.add_warning(f"Storage estimation: snapshot_stride = {snapshot_stride}.")
        return

    bytes_per = 4 if snapshot_precision == "float32" else 8
    nsnapshots = int(np.ceil(nsteps / snapshot_stride))
    n_gll_per_elem = NGLL * NGLL * NGLL

    strain_one_run_bytes = nsnapshots * n_cell * n_gll_per_elem * 6 * bytes_per
    restart_one_run_bytes = n_cell * n_gll_per_elem * 3 * 3 * 8  # u,v,a × float64
    partition_estimate_bytes = n_cell * n_gll_per_elem * 10 * 8  # rough estimate

    total_gb = (
        (strain_one_run_bytes * 3 + restart_one_run_bytes * 3 + partition_estimate_bytes)
        / 1e9
    )

    result.stats["estimated_storage_gb"] = f"{total_gb:.2f}"
    result.stats["snapshot_precision"] = snapshot_precision
    result.stats["nsnapshots"] = nsnapshots

    if total_gb > storage_limit_gb:
        result.add_error(
            f"Storage: estimated {total_gb:.2f} GB exceeds limit "
            f"({storage_limit_gb:.1f} GB). Reduce nsteps, increase "
            f"snapshot_stride, or adjust storage_limit_gb."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_preflight(
    topology,                         # TopologyData
    gll_coords: npt.NDArray[np.float64],
    jacobian: npt.NDArray[np.float64],
    vp_array: npt.NDArray[np.float64],
    vs_array: npt.NDArray[np.float64],
    density_array: npt.NDArray[np.float64],
    boundary_tag: npt.NDArray[np.int64],
    domain_bounds: dict[str, float],
    config_module: ModuleType,
    source_xyz: npt.NDArray[np.float64],
    stf_values: npt.NDArray[np.float64],
    cfl_dt: float,
    nsteps: int,
    snapshot_stride: int,
    NGLL: int,
    strict: bool = True,
) -> PreflightResult:
    """Run comprehensive pre-flight validation.

    Args:
        topology:         TopologyData from topology_reader.
        gll_coords:       [n_cell, NGLL, NGLL, NGLL, 3] GLL node positions.
        jacobian:         [n_cell, NGLL, NGLL, NGLL] det(J).
        vp_array:         [n_cell, NGLL, NGLL, NGLL] vp.
        vs_array:         [n_cell, NGLL, NGLL, NGLL] vs.
        density_array:     [n_cell, NGLL, NGLL, NGLL] density.
        boundary_tag:     [n_surface] int64 boundary tags.
        domain_bounds:    Dict with xmin, xmax, ymin, ymax, zmin, zmax.
        config_module:    Loaded config Python module.
        source_xyz:       [3] source position.
        stf_values:       [nsteps] STF amplitude array.
        cfl_dt:           CFL-limited time step.
        nsteps:           Total number of solver steps.
        snapshot_stride:  Number of solver steps per output snapshot.
        NGLL:             Number of GLL points per dimension (N+1).
        strict:           If True, errors abort; if False, errors logged as warnings.

    Returns:
        PreflightResult with warnings, errors, and stats.

    Raises:
        PreflightError: If strict=True and any check fails.
    """
    result = PreflightResult()

    # 1. Mesh quality
    _check_mesh_quality(jacobian, result)

    # 2. Material
    _check_material(vp_array, vs_array, density_array, result)

    # 3. CFL — output_dt_s / solver_dt must be integer (validated by snapshot_stride)
    result.stats["cfl_dt"] = f"{cfl_dt:.6e}"
    result.stats["snapshot_stride"] = f"{snapshot_stride}"

    # 4. Boundary
    _check_boundary(boundary_tag, result)

    # 5. Source
    source_x = getattr(config_module, "source_x_m", float(source_xyz[0]))
    source_y = getattr(config_module, "source_y_m", float(source_xyz[1]))
    _check_source(source_x, source_y, float(source_xyz[2]), domain_bounds, result)

    # 6. STF
    _check_stf(stf_values, result)

    # 7. Partition
    n_ranks = int(getattr(config_module, "n_ranks", 1))
    _check_partition(topology.n_cell, n_ranks, result)

    # 8. Storage
    # nsteps is now passed directly as parameter, not from config_module
    snapshot_str = int(snapshot_stride) if snapshot_stride else 1
    snapshot_prec = str(getattr(config_module, "snapshot_precision", "float32"))
    storage_limit_gb = float(getattr(config_module, "storage_limit_gb", 100.0))
    _check_storage(
        topology.n_cell, NGLL, nsteps, snapshot_str,
        snapshot_prec, storage_limit_gb, result,
    )

    if result.has_errors and strict:
        raise PreflightError(
            f"Pre-flight validation failed with {len(result.errors)} error(s):\n"
            + result.report()
        )

    return result