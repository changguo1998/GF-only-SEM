"""Mesh-size study case: fullspace20 — homogeneous elastic full-space.

Part of examples/meshsize: four grids (18/20/22/24 elements per axis) with
IDENTICAL physics except the grid. Domain 18 km³, Ricker f0=1 Hz (t0=2 s),
8 s at dt=0.01, N=4, vp=5000 / vs=3000 / rho=2700, point force source at
(9375, 9375, 9375) m in every case. All 6 faces PML (~1.0 λp in meters);
recorded interior is partitioned into 16 tiles.

Mesh: regular hexahedral, 20×20×20 = 8000 elements, 900 m (3.3 elem/λs, 5.6 elem/λp), PML 6 = 5.40 km
Source: point force at (9375, 9375, 9375) m (identical in all four cases)
"""

import numpy as np

# ── Simulation identity ───────
title = "fullspace_meshsize_20"

# ── Mesh dimensions ───────
nx_elements = 20
ny_elements = 20
nz_elements = 20
lx = 18000.0
ly = 18000.0
lz = 18000.0

# ── SEM discretization ───
polynomial_order = 4  # N=4 → 5 GLL nodes/axis

# ── Time stepping ───────
output_dt_s = 0.01
total_duration_s = 8.0
cfl_safety = 0.5
log_stride = 100
restart_dt_s = 0.5

# ── I/O ───
snapshot_precision = "float32"
storage_limit_gb = 20.0
record_depth_max_m = 18000.0  # record ALL vertices
tilex_elements = [2, 2, 2, 2]  # sums to 8 = nx - pml_xmin - pml_xmax
tiley_elements = [2, 2, 2, 2]  # sums to 8 = ny - pml_ymin - pml_ymax

# ── Parallelism ───
n_ranks = 16

# ── Boundary conditions ───
# All 6 faces have PML → absorbing on all sides (full-space)
pml_thickness = {
    "xmin": 6,
    "xmax": 6,
    "ymin": 6,
    "ymax": 6,
    "zmin": 6,  # PML on bottom (no free surface!)
    "zmax": 6,  # PML on top
}

# ── Source (identical in every meshsize case) ───
# (9375, 9375, 9375) m — inside every element (never on a GLL node) of every
# of the 18³/20³/22³/24³ grids, so the source is bit-identical across cases.
source_x_m = 9375.0
source_y_m = 9375.0
source_z_m = 9375.0

source_force_amplitude_n = 1.0e20
f0_for_pml_hz = 1.0


# ── Source time function (callable) ───
def stf_func(t_s):
    """Ricker wavelet (second derivative of Gaussian).

    f0=1.0 Hz, t0=2.0 s.
    """
    f0_hz = 1.0
    t0_s = 2.0
    a = np.pi * f0_hz * (t_s - t0_s)
    return source_force_amplitude_n * (1.0 - 2.0 * a**2) * np.exp(-(a**2))


# ── Material model (callables) ───
def vp_m_s(x_m, y_m, z_m):
    return 5000.0


def vs_m_s(x_m, y_m, z_m):
    return 3000.0


def density_kg_m3(x_m, y_m, z_m):
    return 2700.0


# ── SLS attenuation (viscoelastic parameters) ───
# The preprocessor auto-injects these into model.h5 (field/cell/tau_*).
# Q→∞ (elastic limit): visco output is bit-identical to elastic.
q_mu = 1.0e9
q_kappa = 1.0e9
n_sls = 3
