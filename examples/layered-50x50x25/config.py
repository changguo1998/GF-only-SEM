"""Large layered half-space mesh and waveform configuration."""

import os

import numpy as np

title = "layered_50x50x25_source_depth"

nx_elements = 50
ny_elements = 50
nz_elements = 25
lx = 50000.0
ly = 50000.0
lz = 25000.0

polynomial_order = 4
output_dt_s = 0.05
total_duration_s = 10.0
cfl_safety = 0.5
log_stride = 100
restart_dt_s = 0.0

snapshot_precision = "float32"
storage_limit_gb = 320.0
record_depth_max_m = 2000.0

tilex_elements = [10, 10, 4, 5, 11]
tiley_elements = [10, 8, 5, 7, 10]
postprocess_tiles = [(1, 2), (2, 2), (3, 2)]

validation_offsets_m = [2000.0, 5000.0, 8000.0, 11000.0, 14000.0]
validation_depths_m = [0.0]
validation_end_times_s = [3.5, 4.5, 5.5, 6.5, 7.3]

n_ranks = 16
pml_thickness = {"xmin": 5, "xmax": 5, "ymin": 5, "ymax": 5, "zmin": 0, "zmax": 5}

source_x_m = 18500.0
source_y_m = 25500.0
source_depth_m = float(os.environ.get("GF_SOURCE_DEPTH_M", "100.0"))
source_z_m = None if source_depth_m < 0.0 else source_depth_m

source_force_amplitude_n = 1.0e20
f0_for_pml_hz = 1.0

frequency_hz = 1.0
peak_time_s = 1.0


def stf_func(time_s):
    phase = np.pi * frequency_hz * (time_s - peak_time_s)
    return source_force_amplitude_n * (1.0 - 2.0 * phase**2) * np.exp(-(phase**2))


LAYER_INTERFACE_DEPTH_M = 1000.0
INTERFACE_TOLERANCE_M = 1.0e-6

VP_TOP_M_S = 4000.0
VS_TOP_M_S = 2500.0
DENSITY_TOP_KG_M3 = 2500.0
VP_BOTTOM_M_S = 5000.0
VS_BOTTOM_M_S = 3000.0
DENSITY_BOTTOM_KG_M3 = 2700.0


def vp_m_s(x_m, y_m, z_m):
    return np.where(
        z_m <= LAYER_INTERFACE_DEPTH_M + INTERFACE_TOLERANCE_M, VP_TOP_M_S, VP_BOTTOM_M_S
    )


def vs_m_s(x_m, y_m, z_m):
    return np.where(
        z_m <= LAYER_INTERFACE_DEPTH_M + INTERFACE_TOLERANCE_M, VS_TOP_M_S, VS_BOTTOM_M_S
    )


def density_kg_m3(x_m, y_m, z_m):
    return np.where(
        z_m <= LAYER_INTERFACE_DEPTH_M + INTERFACE_TOLERANCE_M,
        DENSITY_TOP_KG_M3,
        DENSITY_BOTTOM_KG_M3,
    )


# PyFK uses [thickness_km, vs_km_s, vp_km_s, density_g_cm3, Qs, Qp].
LAYER_MODEL = np.array(
    [
        [1.0, VS_TOP_M_S / 1000, VP_TOP_M_S / 1000, DENSITY_TOP_KG_M3 / 1000, 1.0e9, 1.0e9],
        [
            0.0,
            VS_BOTTOM_M_S / 1000,
            VP_BOTTOM_M_S / 1000,
            DENSITY_BOTTOM_KG_M3 / 1000,
            1.0e9,
            1.0e9,
        ],
    ],
    dtype=np.float64,
)

SOURCE_XYZ_M = (0.0, 0.0, 100.0)
RECEIVER_XYZ_M = (5000.0, 0.0, 0.0)
FORCE_AMPLITUDE = 1.0e5
SAMPLES_BEFORE_FIRST_ARRIVAL = 100
DK = 0.3
SMTH = 1.0
PMIN = 0.0
PMAX = 1.0
KMAX = 15.0
