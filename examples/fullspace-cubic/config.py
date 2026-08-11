"""Full-space cubic simulation configuration.

This config defines a homogeneous elastic full-space:
  - Cubic domain with PML absorbing boundaries on all 6 faces
  - Point force source at the domain center
  - No free surface → the full-space Stokes solution is the exact reference

The all-PML setup makes this the ideal test case for analytical Green's
function verification: the SEM solution should match the full-space Stokes
solution exactly (within discretization error).

Domain: 18 km × 18 km × 18 km (x, y, z)
Material: Vp=5000 m/s, Vs=3000 m/s, density=2700 kg/m³ (granite-like)
Mesh: regular hexahedral, 24×24×24 = 13824 elements, 750 m (4.0 elem/λs, 6.7 elem/λp)
Source: point force at element (12,12,12)
center (9375, 9375, 9375) m
"""

import numpy as np

# ── Simulation identity ───────
title = "fullspace_cubic_example"

# ── Mesh dimensions ───────
nx_elements = 24
ny_elements = 24
nz_elements = 24
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
tilex_elements = [2, 2, 3, 3]  # sums to 10 = nx - pml_xmin - pml_xmax
tiley_elements = [2, 2, 3, 3]  # sums to 10 = ny - pml_ymin - pml_ymax

# ── Parallelism ───
n_ranks = 16

# ── Boundary conditions ───
# All 6 faces have PML → absorbing on all sides (full-space)
pml_thickness = {
    "xmin": 7,
    "xmax": 7,
    "ymin": 7,
    "ymax": 7,
    "zmin": 7,  # PML on bottom (no free surface!)
    "zmax": 7,  # PML on top
}

# ── Source ───
# Point force at cell center nearest the domain centre:
#   element 12 of 24 spans [9000, 9750] m
#   → center 9375 m
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
