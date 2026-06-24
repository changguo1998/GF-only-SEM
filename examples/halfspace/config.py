"""Half-space simulation configuration.

This config defines a homogeneous elastic half-space:
  - Free surface at z=0 (source here)
  - Absorbing boundaries on the 5 other sides (perfectly matched layers)
  - A Ricker wavelet (second derivative of Gaussian) point force at domain center

Domain: 10 km × 10 km × 5 km (x, y, z)
Material: Vp=5000 m/s, Vs=3000 m/s, density=2700 kg/m³ (granite-like)
Mesh: regular hexahedral, configured in mesh_gen.py (default 500 elements)

Run with:
    python -m preprocess mesh.h5 examples/halfspace/config.py
"""

import numpy as np

# ── Simulation identity ──────────────────────────────────────────────
title = "halfspace_example"

# ── SEM discretization ───────────────────────────────────────────────
polynomial_order = 4  # GLL quadrature order (N=4 → 5 GLL nodes/axis)

# ── Time stepping ────────────────────────────────────────────────────
output_dt_s = 0.01  # Desired snapshot interval [s]
total_duration_s = 5.0  # Total simulation duration [s]
cfl_safety = 0.5  # CFL safety factor (0 < cfl_safety < 1)

# ── I/O ───────────────────────────────────────────────────────────────
snapshot_precision = "float32"  # "float32" or "float64" for strain snapshots
storage_limit_gb = 5.0  # Warn if estimated output exceeds this

# ── Parallelism ──────────────────────────────────────────────────────
n_ranks = 16  # Number of MPI ranks (METIS partition)

# ── Boundary conditions ──────────────────────────────────────────────
# PML thickness in elements on each face. zmin=0: free surface, zmax=3: PML
pml_thickness = {
    "xmin": 3,
    "xmax": 3,
    "ymin": 3,
    "ymax": 3,
    "zmin": 0,
    "zmax": 3,  # free surface at z=0
}

# ── Source ───────────────────────────────────────────────────────────
# Point force at center of free surface (z=0)
source_x_m = 5000.0
source_y_m = 5000.0


# ── Source time function (callable) ──────────────────────────────────
def stf_func(t_s):
    """Ricker wavelet (second derivative of Gaussian).

    Peak frequency f0=2 Hz, peak time t0=1.0 s.
    """
    f0_hz = 2.0
    t0_s = 1.0
    a = np.pi * f0_hz * (t_s - t0_s)
    return (1.0 - 2.0 * a**2) * np.exp(-(a**2))


# ── Material model (callables) ───────────────────────────────────────
def vp_m_s(x_m, y_m, z_m):
    """P-wave velocity [m/s]."""
    return 5000.0


def vs_m_s(x_m, y_m, z_m):
    """S-wave velocity [m/s]."""
    return 3000.0


def density_kg_m3(x_m, y_m, z_m):
    """Density [kg/m³]."""
    return 2700.0

