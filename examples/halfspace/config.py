"""Half-space simulation configuration.

This config defines a homogeneous elastic half-space:
  - Free surface at z=0
  - Absorbing boundaries on the 5 other sides (perfectly matched layers)
  - A Ricker wavelet (second derivative of Gaussian) point force buried at 278 m

Domain: 10 km × 10 km × 5 km (x, y, z)
Material: Vp=5000 m/s, Vs=3000 m/s, density=2700 kg/m³ (granite-like)
Mesh: regular hexahedral, 18×18×9 = 2916 elements (read by mesh_gen.py via import)

Run with:
    cp this_file /path/to/workdir/config.py
    cd /path/to/workdir
    python -m preprocess

(Preprocess reads model.h5 + config.py from the current working directory.)
"""

import numpy as np

# ── Simulation identity ───────
title = "halfspace_example"

# ── Mesh dimensions ───────
nx_elements = 18  # Elements in x
ny_elements = 18  # Elements in y
nz_elements = 9  # Elements in z
lx = 10000.0  # Domain length x [m]
ly = 10000.0  # Domain length y [m]
lz = 5000.0  # Domain length z [m]

# ── SEM discretization ───
polynomial_order = 4  # GLL quadrature order (N=4 → 5 GLL nodes/axis)

# ── Time stepping ───────
output_dt_s = 0.01  # Desired snapshot interval [s]
total_duration_s = 5.0  # Total simulation duration [s]
cfl_safety = 0.5  # CFL safety factor (0 < cfl_safety < 1)
log_stride = 100  # Progress-report interval in solver steps (1 = every step)
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
# Point force at center of domain
# source_z_m=None -> free surface (zmin); float -> buried source
# Source at element interior (not on shared edge) to test radiation pattern.
# Element (9,9,0) center: xi=eta=zeta=0, all 125 GLL nodes get non-zero weights.
source_x_m = 5278.0
source_y_m = 5278.0
source_z_m = 278.0  # buried at 278 m depth (element center)

# Source force amplitude [N]. STF returns force in Newtons; multiply the
# dimensionless Ricker wavelet by this amplitude. Larger amplitude lifts
# the Green's function values away from the float32 denormal zone (<1e-38),
# reducing truncation error in float32 snapshot storage.
source_force_amplitude_n = 1.0e20


# ── Source time function (callable) ───
def stf_func(t_s):
    """Ricker wavelet (second derivative of Gaussian) scaled to source force [N].

    Peak frequency f0=2 Hz, peak time t0=1.0 s.
    Returns force amplitude in Newtons (dimensionless Ricker × source_force_amplitude_n).
    """
    f0_hz = 2.0
    t0_s = 1.0
    a = np.pi * f0_hz * (t_s - t0_s)
    return source_force_amplitude_n * (1.0 - 2.0 * a**2) * np.exp(-(a**2))


# ── Material model (callables) ───
def vp_m_s(x_m, y_m, z_m):
    """P-wave velocity [m/s]."""
    return 5000.0


def vs_m_s(x_m, y_m, z_m):
    """S-wave velocity [m/s]."""
    return 3000.0


def density_kg_m3(x_m, y_m, z_m):
    """Density [kg/m³]."""
    return 2700.0
