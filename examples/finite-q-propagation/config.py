"""Finite-Q analytical propagation validation configuration.

The transverse y-force is sampled on the +x ray at one and two S wavelengths.
Q_mu=20 makes attenuation and dispersion measurable in a 4.4 s run, while
Q_kappa uses the exact elastic sentinel to isolate the shear response. Lengths
and times are scaled by two from the former 2 Hz case. The transverse section
is widened to 14 elements so the non-PML width exceeds one S wavelength while
preserving the 1 km element size and source-grid alignment.
"""

import numpy as np

title = "finite_q_propagation"

nx_elements = 20
ny_elements = 14
nz_elements = 14
lx = 20000.0
ly = 14000.0
lz = 14000.0
polynomial_order = 4

output_dt_s = 0.08
total_duration_s = 4.4
cfl_safety = 0.5
log_stride = 100
restart_dt_s = 0.0

snapshot_precision = "float32"
storage_limit_gb = 1.0
record_depth_max_m = 7000.0
tilex_elements = [5, 5]
tiley_elements = [2, 2]
n_ranks = 2

pml_thickness = {"xmin": 5, "xmax": 5, "ymin": 5, "ymax": 5, "zmin": 5, "zmax": 5}

source_x_m = 6500.0
source_y_m = 7000.0
source_z_m = 7000.0
source_force_amplitude_n = 1.0e18
f0_for_pml_hz = 1.0

q_mu = 20.0
q_kappa = 1.0e9
n_sls = 3


def stf_func(time_s):
    """One-hertz Ricker point-force history centered at 1.2 s."""
    frequency_hz = 1.0
    peak_time_s = 1.2
    phase = np.pi * frequency_hz * (time_s - peak_time_s)
    return source_force_amplitude_n * (1.0 - 2.0 * phase**2) * np.exp(-(phase**2))


def vp_m_s(x_m, y_m, z_m):
    return 5000.0


def vs_m_s(x_m, y_m, z_m):
    return 3000.0


def density_kg_m3(x_m, y_m, z_m):
    return 2700.0
