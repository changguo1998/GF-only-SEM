/// config_user_fullspace28.cpp — meshsize study case fullspace28
///
/// Build: cmake -B build -DGF_USER_CONFIG=examples/fullspace28/config_user_fullspace.cpp
/// (examples/meshsize/fullspace28/config_user_fullspace.cpp via CASE_DIR)

#include <cmath>
#include <vector>

#include "gf_config.h"

namespace gf {

Config get_config() {
    Config c;
    c.title = "fullspace_meshsize_28";
    c.nx_elements = 28;
    c.ny_elements = 28;
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
    c.tilex_elements = {3, 3, 3, 3};
    c.tiley_elements = {3, 3, 3, 3};
    c.n_ranks = 16;  // default only — the CLI overrides from config.py:n_ranks
    c.pml_xmin = 8;
    c.pml_xmax = 8;
    c.pml_ymin = 8;
    c.pml_ymax = 8;
    c.pml_zmin = 8;  // PML on bottom — no free surface
    c.pml_zmax = 8;  // PML on top
    c.source_x_m = 9375.0;
    c.source_y_m = 9375.0;
    c.source_z_m = 9375.0;
    c.source_force_amplitude_n = 1.0e20;
    c.f0_for_pml_hz = 1.0;
    return c;
}

double stf_func(double t_s) {
    double f0_hz = 1.0;
    double t0_s = 2.0;
    double a = M_PI * f0_hz * (t_s - t0_s);
    return 1.0e20 * (1.0 - 2.0 * a * a) * std::exp(-(a * a));
}

void evaluate_stf_array(double dt, int nsteps, std::vector<double>& times,
                        std::vector<double>& values) {
    times.resize(nsteps);
    values.resize(nsteps);
    for (int i = 0; i < nsteps; ++i) {
        double t = i * dt;
        times[i] = t;
        values[i] = stf_func(t);
    }
}

void evaluate_vp(std::size_t n, const double* /*x*/, const double* /*y*/, const double* /*z*/,
                 double* result) {
    for (std::size_t i = 0; i < n; ++i)
        result[i] = 5000.0;
}

void evaluate_vs(std::size_t n, const double* /*x*/, const double* /*y*/, const double* /*z*/,
                 double* result) {
    for (std::size_t i = 0; i < n; ++i)
        result[i] = 3000.0;
}

void evaluate_density(std::size_t n, const double* /*x*/, const double* /*y*/, const double* /*z*/,
                      double* result) {
    for (std::size_t i = 0; i < n; ++i)
        result[i] = 2700.0;
}

}  // namespace gf
