"""Expanded 40 km homogeneous full-space diagnostic model.

The mesh keeps the 1 km element size of the verified 28-cube expanded model
while moving the five-element C-PML farther from the source. Debug snapshots
use a 0.05 s output interval to keep the full-domain four-field records within
the local disk budget; the solver timestep remains CFL-controlled.
"""

import numpy as np

title = "fullspace_expanded_40cube_debug"

nx_elements = 40
ny_elements = 40
nz_elements = 40
lx = 40000.0
ly = 40000.0
lz = 40000.0

polynomial_order = 4

output_dt_s = 0.05
total_duration_s = 8.0
cfl_safety = 0.5
log_stride = 100
restart_dt_s = 0.5

snapshot_precision = "float32"
storage_limit_gb = 300.0
record_depth_max_m = 40000.0
tilex_elements = [8, 8, 7, 7]
tiley_elements = [8, 8, 7, 7]

n_ranks = 16

pml_thickness = {"xmin": 5, "xmax": 5, "ymin": 5, "ymax": 5, "zmin": 5, "zmax": 5}

# Element [20, 21] km contains the domain centre; use its centre to avoid an
# element interface while preserving cubic symmetry.
source_x_m = 20500.0
source_y_m = 20500.0
source_z_m = 20500.0

source_force_amplitude_n = 1.0e20
f0_for_pml_hz = 1.0


def stf_func(t_s):
    """Return the 1 Hz Ricker source time function."""
    f0_hz = 1.0
    t0_s = 2.0
    phase = np.pi * f0_hz * (t_s - t0_s)
    return source_force_amplitude_n * (1.0 - 2.0 * phase**2) * np.exp(-(phase**2))


def vp_m_s(x_m, y_m, z_m):
    return 5000.0


def vs_m_s(x_m, y_m, z_m):
    return 3000.0


def density_kg_m3(x_m, y_m, z_m):
    return 2700.0


q_mu = 1.0e9
q_kappa = 1.0e9
n_sls = 3
