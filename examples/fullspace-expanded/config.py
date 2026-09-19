"""Coarsened expanded homogeneous full-space diagnostic model.

The 28 km domain uses 22 elements per axis and a 1.273 km element size. At the
1 Hz source frequency this gives 2.36 elements per S wavelength and 3.93
elements per P wavelength. Five PML elements form a 6.364 km layer; the source
is placed at the nearest element centre to the domain centre.
"""

import numpy as np

title = "fullspace_expanded_22cube"

nx_elements = 22
ny_elements = 22
nz_elements = 22
lx = 28000.0
ly = 28000.0
lz = 28000.0

polynomial_order = 4

output_dt_s = 0.01
total_duration_s = 8.0
cfl_safety = 0.5
log_stride = 100
restart_dt_s = 0.5

snapshot_precision = "float32"
storage_limit_gb = 20.0
record_depth_max_m = 28000.0
tilex_elements = [3, 3, 3, 3]
tiley_elements = [3, 3, 3, 3]

n_ranks = 16

pml_thickness = {"xmin": 5, "xmax": 5, "ymin": 5, "ymax": 5, "zmin": 5, "zmax": 5}

# Half an element from the domain centre avoids an element interface.
source_x_m = 14636.363636363636
source_y_m = 14636.363636363636
source_z_m = 14636.363636363636

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
