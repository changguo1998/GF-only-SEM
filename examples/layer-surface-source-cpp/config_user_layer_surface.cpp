/// Pure C++ layered half-space example with a surface point force.

#include <cmath>
#include <vector>

#include "gf_config.h"

namespace gf {

Config get_config() {
    Config config;
    config.title = "layer_surface_source_cpp";
    config.nx_elements = 22;
    config.ny_elements = 22;
    config.lx_m = 10000.0;
    config.ly_m = 10000.0;
    config.lz_m = 5000.0;
    config.polynomial_order = 4;
    config.output_dt_s = 0.01;
    config.total_duration_s = 5.0;
    config.cfl_safety = 0.5;
    config.log_stride = 100;
    config.restart_dt_s = 0.5;
    config.snapshot_precision_bytes = 4;
    config.storage_limit_gb = 10.0;
    config.record_depth_max_m = 2000.0;
    config.tilex_elements = {4, 4, 4, 4};
    config.tiley_elements = {4, 4, 4, 4};
    config.n_ranks = 16;
    config.pml_xmin = 3;
    config.pml_xmax = 3;
    config.pml_ymin = 3;
    config.pml_ymax = 3;
    config.pml_zmin = 0;
    config.pml_zmax = 3;
    config.source_x_m = 5278.0;
    config.source_y_m = 5278.0;
    config.source_z_m = -1.0;  // Negative selects the z-min free surface.
    config.source_force_amplitude_n = 1.0e20;
    config.f0_for_pml_hz = 1.0;
    return config;
}

double stf_func(double time_s) {
    constexpr double frequency_hz = 1.0;
    constexpr double peak_time_s = 1.0;
    double phase = M_PI * frequency_hz * (time_s - peak_time_s);
    return 1.0e20 * (1.0 - 2.0 * phase * phase) * std::exp(-(phase * phase));
}

void evaluate_stf_array(double solver_dt, int nsteps, std::vector<double>& times,
                        std::vector<double>& values) {
    times.resize(nsteps);
    values.resize(nsteps);
    for (int i = 0; i < nsteps; ++i) {
        times[i] = i * solver_dt;
        values[i] = stf_func(times[i]);
    }
}

void evaluate_vp(std::size_t count, const double*, const double*, const double* depth_m,
                 double* result) {
    for (std::size_t i = 0; i < count; ++i)
        result[i] = depth_m[i] <= 500.0 + 1.0e-6 ? 2500.0 : 5000.0;
}

void evaluate_vs(std::size_t count, const double*, const double*, const double* depth_m,
                 double* result) {
    for (std::size_t i = 0; i < count; ++i)
        result[i] = depth_m[i] <= 500.0 + 1.0e-6 ? 1500.0 : 3000.0;
}

void evaluate_density(std::size_t count, const double*, const double*, const double* depth_m,
                      double* result) {
    for (std::size_t i = 0; i < count; ++i)
        result[i] = depth_m[i] <= 500.0 + 1.0e-6 ? 2200.0 : 2700.0;
}

}  // namespace gf
