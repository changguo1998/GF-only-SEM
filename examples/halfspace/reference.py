#!/usr/bin/env python3
"""
Green's function for Lamb's problem — Johnson (1974) formulation.

Computes the 3×3 displacement Green tensor G^H (Heaviside step response)
for a point force in a homogeneous elastic half-space, and provides a CLI
to generate an analytic reference waveform for SEM comparison.

References
----------
Johnson, L. R. (1974). Green's function for Lamb's problem.
    Geophysical Journal International, 37(1), 99-131.

Usage
-----
::

    python examples/halfspace/reference.py \\
        <library_root> --source X Y Z --receiver X Y Z --output <path>

Coordinates follow the GreenFunctionLibrary reciprocity convention:

* ``--receiver`` is the physical receiver and must match a saved SEM
  source coordinate within ``--receiver-tolerance-m``.
* ``--source`` is the physical source location (the PyFK receiver).

The Lamb solution places the source **below** the free surface by a
configurable offset (``--source-depth-m``) because the analytic
formulation requires z > 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from numpy.polynomial.legendre import leggauss

# ---------------------------------------------------------------------------
# Make the project root importable so that ``greenfun`` can be found
# without setting PYTHONPATH externally.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = (_SCRIPT_DIR / "../..").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from greenfun.index_cache import load_or_rebuild_index  # noqa: E402

# ===================================================================
#  Lamb Green's function — core computation
# ===================================================================


def _gauss_quad(n_pts: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return Gauss-Legendre nodes and weights on [-1, 1]."""
    return leggauss(n_pts)


def _integrate_segments(
    integrand: callable,
    a: float,
    b: float,
    nodes: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    args: tuple,
    n_seg: int = 20,
) -> npt.NDArray[np.float64]:
    """Composite Gauss-Legendre quadrature over [a, b]."""
    if b <= a:
        return np.zeros((3, 3), dtype=np.float64)
    edges = np.linspace(a, b, n_seg + 1)
    result = np.zeros((3, 3), dtype=np.float64)
    for i_seg in range(n_seg):
        lo = edges[i_seg]
        hi = edges[i_seg + 1]
        mid = (hi + lo) * 0.5
        half = (hi - lo) * 0.5
        x = nodes * half + mid
        for k in range(len(nodes)):
            result += integrand(x[k], args) * weights[k] * half
    return result


def _build_green_matrices(
    p: float, q: complex, phi: float, vp: float, vs: float
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], complex, complex, complex]:
    """P-wave (mat_p) and S-wave (mat_s) matrices (Johnson 1974, Appendix B)."""
    eta_a = np.sqrt(1.0 / vp**2 + p**2 - q**2 + 0.0j)
    eta_b = np.sqrt(1.0 / vs**2 + p**2 - q**2 + 0.0j)
    gamma = eta_b**2 + p**2 - q**2
    sigma = gamma**2 + 4.0 * eta_b * eta_a * (q**2 - p**2)

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    p2q2 = p**2 + q**2

    mat_p = np.zeros((3, 3), dtype=np.complex128)
    mat_p[0, 0] = 2.0 * eta_b * (p2q2 * cphi**2 - p**2)
    mat_p[0, 1] = 2.0 * eta_b * p2q2 * sphi * cphi
    mat_p[0, 2] = 2.0 * q * eta_b * eta_a * cphi
    mat_p[1, 0] = mat_p[0, 1]
    mat_p[1, 1] = 2.0 * eta_b * (p2q2 * sphi**2 - p**2)
    mat_p[1, 2] = 2.0 * q * eta_a * eta_b * sphi
    mat_p[2, 0] = q * gamma * cphi
    mat_p[2, 1] = q * gamma * sphi
    mat_p[2, 2] = eta_a * gamma

    g4ab = gamma - 4.0 * eta_a * eta_b
    mat_s = np.zeros((3, 3), dtype=np.complex128)
    mat_s[0, 0] = (eta_b**2 * gamma - g4ab * (p2q2 * sphi**2 - p**2)) / eta_b
    mat_s[0, 1] = p2q2 * g4ab * sphi * cphi / eta_b
    mat_s[0, 2] = -q * gamma * cphi
    mat_s[1, 0] = mat_s[0, 1]
    mat_s[1, 1] = (eta_b**2 * gamma - g4ab * (p2q2 * cphi**2 - p**2)) / eta_b
    mat_s[1, 2] = -q * gamma * sphi
    mat_s[2, 0] = -2.0 * q * eta_a * eta_b * cphi
    mat_s[2, 1] = -2.0 * q * eta_a * eta_b * sphi
    mat_s[2, 2] = 2.0 * eta_a * (q**2 - p**2)

    return mat_p, mat_s, sigma, eta_a, eta_b


def _integrand_p(p: float, args: tuple) -> npt.NDArray[np.float64]:
    """P-wave contribution to the Cagniard-de Hoop integral."""
    theta, phi, r, vp, vs, rho, t = args
    result = np.zeros((3, 3), dtype=np.float64)
    if t <= r * np.sqrt(1.0 / vp**2 + p**2):
        return result
    mu = rho * vs**2
    sq = np.sqrt((t / r) ** 2 - 1.0 / vp**2 - p**2 + 0.0j)
    q = -(t / r) * np.sin(theta) + 1j * sq * np.cos(theta)
    mat_p, _, sigma, eta_a, _ = _build_green_matrices(p, q, phi, vp, vs)
    denom = np.sqrt((t / r) ** 2 - 1.0 / vp**2 - p**2 + 0.0j)
    sp = eta_a / sigma / denom * mat_p
    return sp.real / (np.pi**2 * mu * r)


def _integrand_s_cagniard(p: float, args: tuple) -> npt.NDArray[np.float64]:
    """S-wave contribution (Cagniard contour)."""
    theta, phi, r, vp, vs, rho, t = args
    result = np.zeros((3, 3), dtype=np.float64)
    if t <= r * np.sqrt(1.0 / vs**2 + p**2):
        return result
    mu = rho * vs**2
    sq = np.sqrt((t / r) ** 2 - 1.0 / vs**2 - p**2 + 0.0j)
    q = -(t / r) * np.sin(theta) + 1j * sq * np.cos(theta)
    _, mat_s, sigma, _, eta_b = _build_green_matrices(p, q, phi, vp, vs)
    denom = np.sqrt((t / r) ** 2 - 1.0 / vs**2 - p**2 + 0.0j)
    sp = eta_b / sigma / denom * mat_s
    return sp.real / (np.pi**2 * mu * r)


def _integrand_s_branch(p: float, args: tuple) -> npt.NDArray[np.float64]:
    """S-wave contribution (branch-cut correction)."""
    theta, phi, r, vp, vs, rho, t = args
    result = np.zeros((3, 3), dtype=np.float64)
    if t >= r * np.sqrt(1.0 / vs**2 + p**2):
        return result
    mu = rho * vs**2
    branch = np.sqrt(1.0 / vs**2 + p**2 - (t / r) ** 2 + 0.0j)
    q_real = -(t / r) * np.sin(theta) + branch.real * np.cos(theta)
    _, mat_s, sigma, _, eta_b = _build_green_matrices(p, q_real, phi, vp, vs)
    denom = np.sqrt(-((t / r) ** 2) + 1.0 / vs**2 + p**2 + 0.0j)
    sp = eta_b / sigma / denom * mat_s
    return sp.imag / (np.pi**2 * mu * r)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_step_response(
    vp: float,
    vs: float,
    density: float,
    source_xyz: npt.ArrayLike,
    receiver_xyz: npt.ArrayLike,
    time: npt.ArrayLike,
    n_quad_pts: int = 7,
    n_seg: int = 20,
) -> npt.NDArray[np.float64]:
    r"""Displacement Green tensor G^H for a Heavisite step source.

    Parameters
    ----------
    vp, vs : float
        P- and S-wave velocities [m/s].
    density : float
        Mass density [kg/m^3].
    source_xyz : (3,) array
        Source location ``[x, y, z]`` **below** the free surface (z > 0) [m].
    receiver_xyz : (3,) array
        Receiver location [m].
    time : (nt,) array
        Time samples [s].
    n_quad_pts : int
        Gauss-Legendre quadrature points per segment (default 7).
    n_seg : int
        Number of integration segments (default 20).

    Returns
    -------
    g_fn : (nt, 3, 3) array
        Step-response Green tensor G^H in **m/N**.
    """
    src = np.asarray(source_xyz, dtype=np.float64)
    recv = np.asarray(receiver_xyz, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    nt = t.shape[0]

    dr = recv - src
    r = float(np.linalg.norm(dr))
    if r <= 0.0:
        raise ValueError("source and receiver must be separated")
    r_horiz = float(np.linalg.norm(dr[:2]))
    theta = np.arctan2(r_horiz, src[2])
    phi = np.arctan2(dr[1], dr[0])

    nodes, weights = _gauss_quad(n_quad_pts)
    g_fn = np.zeros((nt, 3, 3), dtype=np.float64)

    for i in range(nt):
        t0 = t[i]
        pp = (t0 / r) ** 2 - (1.0 / vp) ** 2
        pb = (t0 / r) ** 2 - (1.0 / vs) ** 2
        args = (theta, phi, r, vp, vs, density, t0)

        if pp > 0.0:
            p_max = np.sqrt(pp)
            g_fn[i] += _integrate_segments(_integrand_p, 0.0, p_max, nodes, weights, args, n_seg)

        if pb > 0.0:
            p_max = np.sqrt(pb)
            g_fn[i] += _integrate_segments(
                _integrand_s_cagniard, 0.0, p_max, nodes, weights, args, n_seg
            )

        # Branch-cut correction for S-wave (beyond critical angle)
        if np.sin(theta) > vs / vp:
            numer = t0 / r - np.sqrt(1.0 / vs**2 - 1.0 / vp**2) * np.cos(theta)
            p2 = (numer / np.sin(theta)) ** 2 - 1.0 / vp**2
            if p2 > 0.0:
                p2_max = np.sqrt(p2)
                if t0 < r / vs:
                    g_fn[i] -= _integrate_segments(
                        _integrand_s_branch, 0.0, p2_max, nodes, weights, args, n_seg
                    )
                elif pb > 0.0:
                    g_fn[i] -= _integrate_segments(
                        _integrand_s_branch, np.sqrt(pb), p2_max, nodes, weights, args, n_seg
                    )

    return g_fn


def impulse_response_from_step(
    g_step: npt.NDArray[np.float64], dt: float
) -> npt.NDArray[np.float64]:
    """Numerical derivative: impulse response g_fn = dG^H/dt (2nd-order central differences)."""
    nt = g_step.shape[0]
    g_fn = np.zeros_like(g_step)
    if nt < 3:
        return g_fn
    g_fn[0] = (g_step[1] - g_step[0]) / dt
    g_fn[1:-1] = (g_step[2:] - g_step[:-2]) / (2.0 * dt)
    g_fn[-1] = (g_step[-1] - g_step[-2]) / dt
    return g_fn


def convolve_with_stf(
    g_step: npt.NDArray[np.float64], source_values: npt.NDArray[np.float64], dt: float
) -> npt.NDArray[np.float64]:
    """Displacement for an arbitrary source time function.

    Computes  u(t) = d/dt[G^H(t) * s(t)]  =  G(t) * s(t)  where
    s(t) is the source time function and G = dG^H/dt.
    """
    g_fn = impulse_response_from_step(g_step, dt)
    nt = g_fn.shape[0]
    u = np.zeros_like(g_fn)
    for i in range(3):
        for j in range(3):
            # mode='full' then trim to first nt samples (causal: output at t
            # depends on inputs at tau <= t).  mode='same' would truncate the
            # convolution peak for early-arriving impulse responses.
            full = np.convolve(g_fn[:, i, j], source_values, mode="full") * dt
            u[:, i, j] = full[:nt]
    return u


# ===================================================================
#  CLI — generate reference waveform from SEM config
# ===================================================================

_QUANTITIES = ("displacement",)


def _as_vector(values: list[float], name: str) -> npt.NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    return vector


def _infer_work_dir(library_root: Path) -> Path:
    root = library_root.resolve()
    for candidate in (root.parent, root):
        if (candidate / "config.h5").exists() and (candidate / "model.h5").exists():
            return candidate
    raise FileNotFoundError(
        "Could not infer work directory.  Expected config.h5 and model.h5 "
        f"next to or inside {library_root}."
    )


def _match_saved_receiver(
    library_root: Path, receiver_xyz_m: npt.NDArray[np.float64], tolerance_m: float
) -> npt.NDArray[np.float64]:
    index = load_or_rebuild_index(library_root)
    if not index.sources:
        raise ValueError(f"No saved SEM source coordinates found in {library_root}")
    saved = np.asarray([e.source_xyz_m for e in index.sources], dtype=np.float64)
    distances = np.linalg.norm(saved - receiver_xyz_m[None, :], axis=1)
    nearest_idx = int(np.argmin(distances))
    nearest_dist = float(distances[nearest_idx])
    if nearest_dist > tolerance_m:
        raise ValueError(
            f"receiver_xyz {receiver_xyz_m} is {nearest_dist:.3f} m from nearest "
            f"SEM source coordinate {saved[nearest_idx]}; tolerance is {tolerance_m:.3f} m"
        )
    return saved[nearest_idx].copy()


def _read_settings(work_dir: Path, relative_tolerance: float) -> dict:
    """Read vp, vs, density and output time/STF from the SEM config."""
    with h5py.File(work_dir / "config.h5", "r") as cfg:
        sim = cfg["simulation"].attrs
        stf_t = np.asarray(cfg["source/stf_t"], dtype=np.float64)
        stf_v = np.asarray(cfg["source/stf_values"], dtype=np.float64)
        stride = int(sim["snapshot_stride"])
        output_dt_s = float(sim["output_dt_s"])
        output_time_s = stf_t[::stride]
        source_values = np.interp(output_time_s, stf_t, stf_v)

    with h5py.File(work_dir / "model.h5", "r") as mod:
        is_pml = np.asarray(mod["field/element/is_pml"], dtype=bool)
        interior = ~is_pml
        vp = np.asarray(mod["field/element/vp"][interior], dtype=np.float64)
        vs = np.asarray(mod["field/element/vs"][interior], dtype=np.float64)
        rho = np.asarray(mod["field/element/density"][interior], dtype=np.float64)

    def _uniform(values: npt.NDArray[np.float64], name: str) -> float:
        center = float(np.median(values))
        scale = max(abs(center), 1.0)
        dev = float(np.max(np.abs(values - center)))
        if dev > relative_tolerance * scale:
            raise ValueError(
                f"Lamb benchmark requires a homogeneous model; {name} varies by "
                f"{dev:.6e}, tolerance is {relative_tolerance * scale:.6e}"
            )
        return center

    return {
        "vp_m_s": _uniform(vp, "vp"),
        "vs_m_s": _uniform(vs, "vs"),
        "density_kg_m3": _uniform(rho, "density"),
        "output_time_s": output_time_s,
        "output_dt_s": output_dt_s,
        "source_values": source_values,
    }


def compute_reference_result(
    library_root: Path,
    source_xyz_m: npt.NDArray[np.float64],
    receiver_xyz_m: npt.NDArray[np.float64],
    quantity: str,
    receiver_tolerance_m: float,
    model_relative_tolerance: float,
    source_depth_m: float,
    n_quad_pts: int,
    n_seg: int,
) -> dict[str, npt.NDArray[np.float64]]:
    if quantity != "displacement":
        raise ValueError("Lamb reference currently supports quantity='displacement' only")

    library_root = library_root.resolve()
    work_dir = _infer_work_dir(library_root)
    sem_source_xyz_m = _match_saved_receiver(library_root, receiver_xyz_m, receiver_tolerance_m)
    settings = _read_settings(work_dir, model_relative_tolerance)

    # Place the analytic source below the free surface.
    analytic_source_xyz = sem_source_xyz_m.copy()
    analytic_source_xyz[2] = max(analytic_source_xyz[2], source_depth_m)

    g_step = compute_step_response(
        vp=settings["vp_m_s"],
        vs=settings["vs_m_s"],
        density=settings["density_kg_m3"],
        source_xyz=analytic_source_xyz,
        receiver_xyz=source_xyz_m,
        time=settings["output_time_s"],
        n_quad_pts=n_quad_pts,
        n_seg=n_seg,
    )

    displacement = convolve_with_stf(g_step, settings["source_values"], settings["output_dt_s"])

    return {
        "time": np.asarray(settings["output_time_s"], dtype=np.float64),
        "source_xyz_m": source_xyz_m.astype(np.float64),
        "receiver_xyz_m": receiver_xyz_m.astype(np.float64),
        "sem_source_xyz_m": sem_source_xyz_m.astype(np.float64),
        "analytic_source_xyz_m": analytic_source_xyz.astype(np.float64),
        "displacement": displacement.astype(np.float64),
        "vp_m_s": np.array(settings["vp_m_s"], dtype=np.float64),
        "vs_m_s": np.array(settings["vs_m_s"], dtype=np.float64),
        "density_kg_m3": np.array(settings["density_kg_m3"], dtype=np.float64),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate analytic Lamb half-space reference waveform for SEM comparison."
    )
    p.add_argument("library_root", type=Path, help="Path to greenfun library root")
    p.add_argument("--source", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--receiver", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--quantity", choices=_QUANTITIES, default="displacement")
    p.add_argument("--output", type=Path, required=True, help="Output .npz path")
    p.add_argument(
        "--receiver-tolerance-m",
        type=float,
        default=100.0,
        help="Max distance to a saved SEM source coordinate [m]",
    )
    p.add_argument(
        "--model-relative-tolerance",
        type=float,
        default=1.0e-8,
        help="Allowed relative variation of non-PML vp/vs/density",
    )
    p.add_argument(
        "--source-depth-m",
        type=float,
        default=10.0,
        help="Depth of analytic source below free surface [m] (Johnson formulation requires z>0)",
    )
    p.add_argument(
        "--n-quad-pts", type=int, default=7, help="Gauss-Legendre quadrature points per segment"
    )
    p.add_argument("--n-seg", type=int, default=20, help="Number of integration segments")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_xyz_m = _as_vector(args.source, "source")
    receiver_xyz_m = _as_vector(args.receiver, "receiver")
    result = compute_reference_result(
        args.library_root,
        source_xyz_m,
        receiver_xyz_m,
        args.quantity,
        args.receiver_tolerance_m,
        args.model_relative_tolerance,
        args.source_depth_m,
        args.n_quad_pts,
        args.n_seg,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"Wrote {args.output}")
    print(f"source_xyz_m        : {result['source_xyz_m']}")
    print(f"receiver_xyz_m      : {result['receiver_xyz_m']}")
    print(f"sem_source_xyz_m    : {result['sem_source_xyz_m']}")
    print(f"analytic_src_xyz_m  : {result['analytic_source_xyz_m']}")
    print(f"time                : {result['time'].shape}")
    print(f"displacement        : {result['displacement'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
