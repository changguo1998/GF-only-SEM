#!/usr/bin/env python3
"""Generate the four meshsize fullspace case dirs (18/20/22/24) + fixed receivers.

Single source of truth for examples/meshsize/fullspace{18,20,22,24}:
everything except the grid (element size, PML layer count, tile split) is
IDENTICAL across the four cases:

  * domain 18 km³, N=4 GLL order, vp=5000 / vs=3000 / rho=2700
  * Ricker f0=1 Hz, t0=2 s, 8 s duration, dt=0.01, force 1e20 N
  * SOURCE at exactly (9375, 9375, 9375) m in all four — inside every element
    (never on a GLL node) of every grid: 18³ nodes at 9000/10000 m, 20³ at
    9000/9800, 22³ at 9000/9818.2, 24³ at 9000/9750
  * PML thickness held ~constant in METERS (5.00/5.40/4.91/5.25 km ≈ 1.0 λp):
    18³→5, 20³→6, 22³→6, 24³→7
  * tiles partition the interior: 18/20 → [2,2,2,2]², 22/24 → [2,2,3,3]²

Fixed receivers: 64 points, seed-0 uniform draw in [5500,12500]³, saved to
examples/meshsize/receivers_fixed.npy. Every grid evaluates the analytical
solution at the NEAREST recorded GLL node's true coordinates, so the
comparison has no h-dependent position error.

Usage:  python3 gen_grids.py            # writes all four case dirs + receivers
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

GRIDS = {
    18: dict(
        pml=5,
        tiles=(2, 2, 2, 2),
        desc="18×18×18 = 5832 elements, 1000 m (3.0 elem/λs, 5.0 elem/λp), PML 5 = 5.00 km",
    ),
    20: dict(
        pml=6,
        tiles=(2, 2, 2, 2),
        desc="20×20×20 = 8000 elements, 900 m (3.3 elem/λs, 5.6 elem/λp), PML 6 = 5.40 km",
    ),
    22: dict(
        pml=6,
        tiles=(2, 2, 3, 3),
        desc="22×22×22 = 10648 elements, 818 m (3.7 elem/λs, 6.1 elem/λp), PML 6 = 4.91 km",
    ),
    24: dict(
        pml=7,
        tiles=(2, 2, 3, 3),
        desc="24×24×24 = 13824 elements, 750 m (4.0 elem/λs, 6.7 elem/λp), PML 7 = 5.25 km",
    ),
    28: dict(
        pml=8,
        tiles=(3, 3, 3, 3),
        desc="28×28×28 = 21952 elements, 643 m (4.7 elem/λs, 7.8 elem/λp), PML 8 = 5.14 km",  # coarsest ceiling — no denser grids
    ),
}

SOURCE_M = 9375.0  # identical everywhere; inside every element of every grid
BOX_LO, BOX_HI = 5500.0, 12500.0  # fixed receiver region (interior of all grids)
N_FIXED = 64
SEED = 0


def render_config(n: int, g: dict) -> str:
    src = SOURCE_M
    tiles = ", ".join(str(t) for t in g["tiles"])
    interior = n - 2 * g["pml"]
    p = g["pml"]
    return f'''"""Mesh-size study case: fullspace{n} — homogeneous elastic full-space.

Part of examples/meshsize: four grids (18/20/22/24 elements per axis) with
IDENTICAL physics except the grid. Domain 18 km³, Ricker f0=1 Hz (t0=2 s),
8 s at dt=0.01, N=4, vp=5000 / vs=3000 / rho=2700, point force source at
(9375, 9375, 9375) m in every case. All 6 faces PML (~1.0 λp in meters);
recorded interior is partitioned into 16 tiles.

Mesh: regular hexahedral, {g["desc"]}
Source: point force at (9375, 9375, 9375) m (identical in all four cases)
"""

import numpy as np

# ── Simulation identity ───────
title = "fullspace_meshsize_{n}"

# ── Mesh dimensions ───────
nx_elements = {n}
ny_elements = {n}
nz_elements = {n}
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
tilex_elements = [{tiles}]  # sums to {interior} = nx - pml_xmin - pml_xmax
tiley_elements = [{tiles}]  # sums to {interior} = ny - pml_ymin - pml_ymax

# ── Parallelism ───
n_ranks = 16

# ── Boundary conditions ───
# All 6 faces have PML → absorbing on all sides (full-space)
pml_thickness = {{
    "xmin": {p},
    "xmax": {p},
    "ymin": {p},
    "ymax": {p},
    "zmin": {p},  # PML on bottom (no free surface!)
    "zmax": {p},  # PML on top
}}

# ── Source (identical in every meshsize case) ───
# (9375, 9375, 9375) m — inside every element (never on a GLL node) of every
# of the 18³/20³/22³/24³ grids, so the source is bit-identical across cases.
source_x_m = {src}
source_y_m = {src}
source_z_m = {src}

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
'''


def render_cpp(n: int, g: dict) -> str:
    src = SOURCE_M
    tiles = ", ".join(str(t) for t in g["tiles"])
    p = g["pml"]
    return f"""/// config_user_fullspace{n}.cpp — meshsize study case fullspace{n}
///
/// Build: cmake -B build -DGF_USER_CONFIG={os.path.basename(os.path.dirname(HERE))}/fullspace{n}/config_user_fullspace.cpp
/// (examples/meshsize/fullspace{n}/config_user_fullspace.cpp via CASE_DIR)

#include <cmath>
#include <vector>

#include "gf_config.h"

namespace gf {{

Config get_config() {{
    Config c;
    c.title = "fullspace_meshsize_{n}";
    c.nx_elements = {n};
    c.ny_elements = {n};
    c.lx_m = 18000.0;
    c.ly_m = 18000.0;
    c.lz_m = 18000.0;
    c.polynomial_order = 4;
    c.output_dt_s = 0.01;
    c.total_duration_s = 8.0;
    c.cfl_safety = 0.5;
    c.log_stride = 100;
    c.restart_dt_s = 0.5;
    c.snapshot_precision_bytes = 4;
    c.storage_limit_gb = 20.0;
    c.record_depth_max_m = 18000.0;
    c.tilex_elements = {{{tiles}}};
    c.tiley_elements = {{{tiles}}};
    c.n_ranks = 16;  // default only — the CLI overrides from config.py:n_ranks
    c.pml_xmin = {p};
    c.pml_xmax = {p};
    c.pml_ymin = {p};
    c.pml_ymax = {p};
    c.pml_zmin = {p};  // PML on bottom — no free surface
    c.pml_zmax = {p};  // PML on top
    c.source_x_m = {src};
    c.source_y_m = {src};
    c.source_z_m = {src};
    c.source_force_amplitude_n = 1.0e20;
    c.f0_for_pml_hz = 1.0;
    return c;
}}

double stf_func(double t_s) {{
    double f0_hz = 1.0;
    double t0_s = 2.0;
    double a = M_PI * f0_hz * (t_s - t0_s);
    return 1.0e20 * (1.0 - 2.0 * a * a) * std::exp(-(a * a));
}}

void evaluate_stf_array(double dt, int nsteps, std::vector<double>& times,
                        std::vector<double>& values) {{
    times.resize(nsteps);
    values.resize(nsteps);
    for (int i = 0; i < nsteps; ++i) {{
        double t = i * dt;
        times[i] = t;
        values[i] = stf_func(t);
    }}
}}

void evaluate_vp(std::size_t n, const double* /*x*/, const double* /*y*/, const double* /*z*/,
                 double* result) {{
    for (std::size_t i = 0; i < n; ++i)
        result[i] = 5000.0;
}}

void evaluate_vs(std::size_t n, const double* /*x*/, const double* /*y*/, const double* /*z*/,
                 double* result) {{
    for (std::size_t i = 0; i < n; ++i)
        result[i] = 3000.0;
}}

void evaluate_density(std::size_t n, const double* /*x*/, const double* /*y*/, const double* /*z*/,
                      double* result) {{
    for (std::size_t i = 0; i < n; ++i)
        result[i] = 2700.0;
}}

}}  // namespace gf
"""


def write_fixed_receivers():
    rng = np.random.default_rng(SEED)
    points = rng.uniform(BOX_LO, BOX_HI, size=(N_FIXED, 3))
    out = os.path.join(HERE, "receivers_fixed.npy")
    np.save(out, points)
    print(f"wrote {out} ({N_FIXED} points in [{BOX_LO},{BOX_HI}]³ m, seed {SEED})")


def main():
    if "--receivers-only" in sys.argv:
        write_fixed_receivers()
        return
    for n, g in GRIDS.items():
        case_dir = os.path.join(HERE, f"fullspace{n}")
        os.makedirs(case_dir, exist_ok=True)
        with open(os.path.join(case_dir, "config.py"), "w") as f:
            f.write(render_config(n, g))
        with open(os.path.join(case_dir, "config_user_fullspace.cpp"), "w") as f:
            f.write(render_cpp(n, g))
        print(
            f"wrote fullspace{n}/config.py + config_user_fullspace.cpp "
            f"(pml={g['pml']} -> {g['pml'] * 18000.0 / n:.2f} km, "
            f"tiles=[{', '.join(map(str, g['tiles']))}]², src={SOURCE_M:.0f})"
        )
    write_fixed_receivers()


if __name__ == "__main__":
    main()
