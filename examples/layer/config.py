"""Layered half-space simulation configuration.

This config defines a two-layer elastic half-space:
  - Soft surface layer (500 m) over a stiffer half-space
  - Free surface at z=0
  - Absorbing boundaries on the 5 other sides (perfectly matched layers)
  - A Ricker wavelet point force buried at 250 m depth

Material properties are depth-dependent piecewise functions matching
the PyFK LAYER_MODEL for consistent comparison.

Domain: 10 km × 10 km × 5 km (x, y, z)
Layer 1 (z < 500 m):  Vp=2500 m/s, Vs=1500 m/s, density=2200 kg/m³
Layer 2 (z >= 500 m): Vp=5000 m/s, Vs=3000 m/s, density=2700 kg/m³
Mesh: regular hexahedral, 18×18×10 = 3240 elements
      (z-element boundary at 500 m aligns with layer interface)

Run with:
    cp this_file /path/to/workdir/config.py
    cd /path/to/workdir
    python -m preprocess
"""

from __future__ import annotations

import numpy as np

# ── Simulation identity ───────
title = "layer_example"

# ── Mesh dimensions ───────
nx_elements = 18  # Elements in x
ny_elements = 18  # Elements in y
nz_elements = 10  # Elements in z (each 500 m thick)
lx = 10000.0  # Domain length x [m]
ly = 10000.0  # Domain length y [m]
lz = 5000.0  # Domain length z [m]

# ── SEM discretization ───
polynomial_order = 4  # GLL quadrature order (N=4 → 5 GLL nodes/axis)

# ── Time stepping ───────
output_dt_s = 0.01  # Desired snapshot interval [s]
total_duration_s = 5.0  # Total simulation duration [s]
cfl_safety = 0.5  # CFL safety factor (0 < cfl_safety < 1)
log_stride = 100  # Progress-report interval in solver steps
restart_dt_s = 0.5  # Restart checkpoint interval [s] (0 = disable)

# ── I/O ───
snapshot_precision = "float32"  # "float32" or "float64" for strain snapshots
storage_limit_gb = 10.0  # Warn if estimated output exceeds this
record_depth_max_m = 2000.0  # Record strain at vertices within this depth of free surface [m]
tilex_elements = [
    4,
    4,
    4,
]  # Horizontal x tile sizes in elements (nx = sum(tilex) + pml_xmin + pml_xmax)
tiley_elements = [
    4,
    4,
    4,
]  # Horizontal y tile sizes in elements (ny = sum(tiley) + pml_ymin + pml_ymax)
green_tile_size_m = (
    2000.0  # Optional spatial tile size [m] (overrides tilex_elements/tiley_elements when set)
)

# ── Parallelism ───
n_ranks = 16  # Number of MPI ranks (METIS partition)

# ── Boundary conditions ───
# PML thickness in elements on each face. zmin=0: free surface, zmax=3: PML
pml_thickness = {
    "xmin": 3,
    "xmax": 3,
    "ymin": 3,
    "ymax": 3,
    "zmin": 0,
    "zmax": 3,  # free surface at z=0
}

# ── Source ───
# Point force at element interior (not on a shared grid node) to avoid the
# 4-element shared-node source-inflation bug (see commit e1ac709, deferred.md §6).
# Element (9,9,0) center: x=y=5278 m is 9.5*dx, safely inside element [5000,5556].
source_x_m = 5278.0
source_y_m = 5278.0
source_z_m = 250.0  # buried at 250m depth, middle of layer 1 (0-500m)


# ── Source time function (callable) ───
def stf_func(t_s):
    """Ricker wavelet (second derivative of Gaussian).

    Peak frequency f0=2 Hz, peak time t0=1.0 s.
    """
    f0_hz = 2.0
    t0_s = 1.0
    a = np.pi * f0_hz * (t_s - t0_s)
    return (1.0 - 2.0 * a**2) * np.exp(-(a**2))


# ── Material model — depth-dependent piecewise functions ───

_INTERFACE_Z_M = 500.0  # Layer interface depth [m]

# Layer 1 (soft surface layer)
_VP1 = 2500.0
_VS1 = 1500.0
_RHO1 = 2200.0

# Layer 2 (stiff half-space)
_VP2 = 5000.0
_VS2 = 3000.0
_RHO2 = 2700.0


def vp_m_s(x_m, y_m, z_m):
    """P-wave velocity [m/s] — piecewise by depth."""
    if np.ndim(z_m) == 0:
        return _VP1 if float(z_m) < _INTERFACE_Z_M else _VP2
    return np.where(z_m < _INTERFACE_Z_M, _VP1, _VP2)


def vs_m_s(x_m, y_m, z_m):
    """S-wave velocity [m/s] — piecewise by depth."""
    if np.ndim(z_m) == 0:
        return _VS1 if float(z_m) < _INTERFACE_Z_M else _VS2
    return np.where(z_m < _INTERFACE_Z_M, _VS1, _VS2)


def density_kg_m3(x_m, y_m, z_m):
    """Density [kg/m³] — piecewise by depth."""
    if np.ndim(z_m) == 0:
        return _RHO1 if float(z_m) < _INTERFACE_Z_M else _RHO2
    return np.where(z_m < _INTERFACE_Z_M, _RHO1, _RHO2)


# ===================================================================
# PyFK reference parameters (kept for backward compatibility)
# ===================================================================
# Column format: [thickness_km, vs_km_s, vp_km_s, density_g_cm3, Qs, Qp]
# Last row must have thickness = 0 (bottom half-space).
# PyFK units: km, km/s, g/cm³.
LAYER_MODEL = np.array(
    [
        [0.5, 1.5, 2.5, 2.2, 100.0, 200.0],  # Surface layer
        [0.0, 3.0, 5.0, 2.7, 500.0, 1000.0],  # Bottom half-space
    ],
    dtype=np.float64,
)

SOURCE_XYZ_M = (0.0, 0.0, 490.0)  # (x, y, z) in m — slightly above layer interface
RECEIVER_XYZ_M = (5000.0, 0.0, 0.0)  # (x, y, z) in m
DT_S = 0.01
N_TIME = 1000  # 10 s total

# PyFK solver parameters
SAMPLES_BEFORE_FIRST_ARRIVAL = 100
FORCE_AMPLITUDE = 1e5  # 1 N = 1e5 dyne (PyFK uses CGS internally; sf source m0 = amp * 1e-15)
QS_DEFAULT = 100.0
QP_DEFAULT = 200.0
DK = 0.3
SMTH = 1.0
PMIN = 0.0
PMAX = 1.0
KMAX = 15.0
