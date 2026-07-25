/// config_user_layer.cpp — layer example user config for static linking
///
/// Build: cmake -B build -DGF_USER_CONFIG=preprocess/cpp/config_user_layer.cpp

#include <cmath>
#include <vector>

#include "gf_config.h"

namespace gf {

Config get_config() {
    Config c;
    c.title = "layer_example";
    c.nx_elements = 18;
    c.ny_elements = 18;
    // nz_elements not in Config struct — derived from mesh topology
    c.lx_m = 10000.0;
    c.ly_m = 10000.0;
    c.lz_m = 5000.0;
    c.polynomial_order = 4;
    c.output_dt_s = 0.01;
    c.total_duration_s = 5.0;
    c.cfl_safety = 0.5;
    c.log_stride = 100;
    c.restart_dt_s = 0.5;
    c.snapshot_precision_bytes = 4;
    c.storage_limit_gb = 10.0;
    c.record_depth_max_m = 2000.0;
    c.tilex_elements = {4, 4, 4};
    c.tiley_elements = {4, 4, 4};
    c.n_ranks = 16;
    c.pml_xmin = 3;
    c.pml_xmax = 3;
    c.pml_ymin = 3;
    c.pml_ymax = 3;
    c.pml_zmin = 0;
    c.pml_zmax = 3;
    c.source_x_m = 5278.0;
    c.source_y_m = 5278.0;
    c.source_z_m = 250.0;
    c.source_force_amplitude_n = 1.0e20;
    c.f0_for_pml_hz = 2.0;
    return c;
}

double stf_func(double t_s) {
    double f0_hz = 2.0;
    double t0_s = 1.0;
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

// Two-layer model: interface at z = 500 m
//   Layer 1 (z < 500 m):  Vp=2500 m/s, Vs=1500 m/s, ρ=2200 kg/m³
//   Layer 2 (z >= 500 m): Vp=5000 m/s, Vs=3000 m/s, ρ=2700 kg/m³
namespace {
constexpr double k_interface_z_m = 500.0;
}  // namespace

void evaluate_vp(std::size_t n, const double* /*x*/, const double* /*y*/, const double* z,
                 double* result) {
    for (std::size_t i = 0; i < n; ++i)
        result[i] = (z[i] < k_interface_z_m) ? 2500.0 : 5000.0;
}

void evaluate_vs(std::size_t n, const double* /*x*/, const double* /*y*/, const double* z,
                 double* result) {
    for (std::size_t i = 0; i < n; ++i)
        result[i] = (z[i] < k_interface_z_m) ? 1500.0 : 3000.0;
}

void evaluate_density(std::size_t n, const double* /*x*/, const double* /*y*/, const double* z,
                      double* result) {
    for (std::size_t i = 0; i < n; ++i)
        result[i] = (z[i] < k_interface_z_m) ? 2200.0 : 2700.0;
}

}  // namespace gf