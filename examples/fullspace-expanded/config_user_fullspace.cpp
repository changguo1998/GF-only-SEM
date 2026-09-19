/// Coarsened 28 km homogeneous full-space diagnostic configuration.

#include <cmath>
#include <vector>

#include "gf_config.h"

namespace gf {

Config get_config() {
    Config config;
    config.title = "fullspace_expanded_22cube";
    config.nx_elements = 22;
    config.ny_elements = 22;
    config.lx_m = 28000.0;
    config.ly_m = 28000.0;
    config.lz_m = 28000.0;
    config.polynomial_order = 4;
    config.output_dt_s = 0.01;
    config.total_duration_s = 8.0;
    config.cfl_safety = 0.5;
    config.log_stride = 100;
    config.restart_dt_s = 0.5;
    config.snapshot_precision_bytes = 4;
    config.storage_limit_gb = 20.0;
    config.record_depth_max_m = 28000.0;
    config.tilex_elements = {3, 3, 3, 3};
    config.tiley_elements = {3, 3, 3, 3};
    config.n_ranks = 16;
    config.pml_xmin = 5;
    config.pml_xmax = 5;
    config.pml_ymin = 5;
    config.pml_ymax = 5;
    config.pml_zmin = 5;
    config.pml_zmax = 5;
    config.source_x_m = 14636.363636363636;
    config.source_y_m = 14636.363636363636;
    config.source_z_m = 14636.363636363636;
    config.source_force_amplitude_n = 1.0e20;
    config.f0_for_pml_hz = 1.0;
    return config;
}

double stf_func(double time_s) {
    constexpr double f0_hz = 1.0;
    constexpr double t0_s = 2.0;
    double phase = M_PI * f0_hz * (time_s - t0_s);
    return 1.0e20 * (1.0 - 2.0 * phase * phase) * std::exp(-(phase * phase));
}

void evaluate_stf_array(double solver_dt, int nsteps, std::vector<double>& times,
                        std::vector<double>& values) {
    times.resize(nsteps);
    values.resize(nsteps);
    for (int i = 0; i < nsteps; ++i) {
        double time_s = i * solver_dt;
        times[i] = time_s;
        values[i] = stf_func(time_s);
    }
}

void evaluate_vp(std::size_t count, const double*, const double*, const double*, double* result) {
    for (std::size_t i = 0; i < count; ++i)
        result[i] = 5000.0;
}

void evaluate_vs(std::size_t count, const double*, const double*, const double*, double* result) {
    for (std::size_t i = 0; i < count; ++i)
        result[i] = 3000.0;
}

void evaluate_density(std::size_t count, const double*, const double*, const double*,
                      double* result) {
    for (std::size_t i = 0; i < count; ++i)
        result[i] = 2700.0;
}

}  // namespace gf
