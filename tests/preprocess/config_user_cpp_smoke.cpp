/// Small compile-time configuration for the pure C++ preprocess smoke test.

#include <cmath>
#include <vector>

#include "gf_config.h"

namespace gf {

Config get_config() {
    Config config;
    config.title = "cpp_preprocess_smoke";
    config.nx_elements = 4;
    config.ny_elements = 4;
    config.lx_m = 4000.0;
    config.ly_m = 4000.0;
    config.lz_m = 4000.0;
    config.polynomial_order = 2;
    config.output_dt_s = 0.02;
    config.total_duration_s = 2.0;
    config.cfl_safety = 0.5;
    config.log_stride = 1;
    config.restart_dt_s = 0.0;
    config.snapshot_precision_bytes = 8;
    config.storage_limit_gb = 1.0;
    config.record_depth_max_m = 2000.0;
    config.tilex_elements = {2};
    config.tiley_elements = {2};
    config.n_ranks = 2;
    config.pml_xmin = 1;
    config.pml_xmax = 1;
    config.pml_ymin = 1;
    config.pml_ymax = 1;
    config.pml_zmin = 1;
    config.pml_zmax = 1;
    config.source_x_m = 2000.0;
    config.source_y_m = 2000.0;
    config.source_z_m = 2000.0;
    config.source_force_amplitude_n = 1.0;
    config.f0_for_pml_hz = 5.0;
    return config;
}

double stf_func(double time_s) {
    constexpr double frequency_hz = 5.0;
    constexpr double center_time_s = 0.02;
    double phase = M_PI * frequency_hz * (time_s - center_time_s);
    return (1.0 - 2.0 * phase * phase) * std::exp(-phase * phase);
}

void evaluate_stf_array(double solver_dt, int nsteps, std::vector<double>& times,
                        std::vector<double>& values) {
    times.resize(nsteps);
    values.resize(nsteps);
    for (int step = 0; step < nsteps; ++step) {
        times[step] = step * solver_dt;
        values[step] = stf_func(times[step]);
    }
}

void evaluate_vp(std::size_t count, const double*, const double*, const double*, double* result) {
    for (std::size_t index = 0; index < count; ++index)
        result[index] = 3000.0;
}

void evaluate_vs(std::size_t count, const double*, const double*, const double*, double* result) {
    for (std::size_t index = 0; index < count; ++index)
        result[index] = 1700.0;
}

void evaluate_density(std::size_t count, const double*, const double*, const double*,
                      double* result) {
    for (std::size_t index = 0; index < count; ++index)
        result[index] = 2500.0;
}

}  // namespace gf
