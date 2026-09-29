"""Large homogeneous half-space validation model.

The 50 x 50 x 25 km mesh uses 1 km elements. Five-element C-PML layers on
the four sides and bottom leave a 40 x 40 x 20 km physical half-space. The
source is offset to one third of the physical x extent so the selected
far-side tile contains long, pre-PML-return body-wave paths.
"""

import numpy as np

title = "halfspace_50x50x25_far_tile"

nx_elements = 50
ny_elements = 50
nz_elements = 25
lx = 50000.0
ly = 50000.0
lz = 25000.0

polynomial_order = 4

output_dt_s = 0.05
total_duration_s = 7.5
cfl_safety = 0.5
log_stride = 100
restart_dt_s = 0.0

snapshot_precision = "float32"
storage_limit_gb = 260.0
record_depth_max_m = 2000.0

# Interior bounds are x/y=[5, 45] km. Tiles x1-x3/y2 cover x=[15, 34] km
# and y=[23, 28] km, including the selected 2-14 km validation offsets.
tilex_elements = [10, 10, 4, 5, 11]
tiley_elements = [10, 8, 5, 7, 10]
postprocess_tiles = [(1, 2), (2, 2), (3, 2)]

validation_offsets_m = [2000.0, 5000.0, 8000.0, 11000.0, 14000.0]
validation_depths_m = [0.0, 1000.0, 2000.0]
validation_end_times_s = [3.5, 4.5, 5.5, 6.5, 7.3]

n_ranks = 16

pml_thickness = {"xmin": 5, "xmax": 5, "ymin": 5, "ymax": 5, "zmin": 0, "zmax": 5}

# The physical x interval starts at 5 km. Its exact one-third point is
# 18.333 km; 18.5 km is the nearest element centre and avoids an interface.
source_x_m = 18500.0
source_y_m = 25500.0
source_z_m = None

source_force_amplitude_n = 1.0e20
f0_for_pml_hz = 1.0


def stf_func(t_s):
    """Return a 1 Hz Ricker point-force history centered at 1 s."""
    frequency_hz = 1.0
    peak_time_s = 1.0
    phase = np.pi * frequency_hz * (t_s - peak_time_s)
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
