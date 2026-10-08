/// C++ preprocessing configuration for the large half-space validation case.

#include <cmath>
#include <cstdlib>
#include <vector>

#include "gf_config.h"

namespace gf {

Config get_config() {
    Config config;
    config.title = "halfspace_50x50x25_shallow_source";
    config.nx_elements = 50;
    config.ny_elements = 50;
    config.lx_m = 50000.0;
    config.ly_m = 50000.0;
    config.lz_m = 25000.0;
    config.polynomial_order = 4;
    config.output_dt_s = 0.05;
    config.total_duration_s = 7.5;
    config.cfl_safety = 0.5;
    config.log_stride = 100;
    config.restart_dt_s = 0.0;
    config.snapshot_precision_bytes = 4;
    config.storage_limit_gb = 260.0;
    config.record_depth_max_m = 2000.0;
    config.tilex_elements = {10, 10, 4, 5, 11};
    config.tiley_elements = {10, 8, 5, 7, 10};
    config.n_ranks = 16;
    config.pml_xmin = 5;
    config.pml_xmax = 5;
    config.pml_ymin = 5;
    config.pml_ymax = 5;
    config.pml_zmax = 5;
    config.source_x_m = 18500.0;
    config.source_y_m = 25500.0;
    const char* source_depth = std::getenv("GF_SOURCE_DEPTH_M");
    config.source_z_m = source_depth == nullptr ? 1000.0 : std::strtod(source_depth, nullptr);
    config.source_force_amplitude_n = 1.0e20;
    config.f0_for_pml_hz = 1.0;
    return config;
}

double stf_func(double time_s) {
    const double frequency_hz = 1.0;
    const double peak_time_s = 1.0;
    const double phase = M_PI * frequency_hz * (time_s - peak_time_s);
    return 1.0e20 * (1.0 - 2.0 * phase * phase) * std::exp(-(phase * phase));
}

void evaluate_stf_array(double dt, int nsteps, std::vector<double>& times,
                        std::vector<double>& values) {
    times.resize(nsteps);
    values.resize(nsteps);
    for (int step = 0; step < nsteps; ++step) {
        times[step] = step * dt;
        values[step] = stf_func(times[step]);
    }
}

void evaluate_vp(std::size_t count, const double*, const double*, const double*, double* result) {
    for (std::size_t index = 0; index < count; ++index)
        result[index] = 5000.0;
}

void evaluate_vs(std::size_t count, const double*, const double*, const double*, double* result) {
    for (std::size_t index = 0; index < count; ++index)
        result[index] = 3000.0;
}

void evaluate_density(std::size_t count, const double*, const double*, const double*,
                      double* result) {
    for (std::size_t index = 0; index < count; ++index)
        result[index] = 2700.0;
}

}  // namespace gf
