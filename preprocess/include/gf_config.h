#pragma once
/// gf_config.h — user-defined simulation configuration and material model
///
/// Users create a single config.cpp that implements all functions declared
/// here.  The build system links it statically into gf_preprocess:
///
///   cmake -B build -DGF_USER_CONFIG=my_config.cpp
///   cmake --build build
///
/// The resulting gf_preprocess binary is fully self-contained — no Python
/// needed at runtime.

#include <cstddef>
#include <string>
#include <vector>

namespace gf {

// ═══════════════════════════════════════════════════════════════════════════
//  Simulation configuration
// ═══════════════════════════════════════════════════════════════════════════

struct Config {
    // Mesh
    int nx_elements = 0;
    int ny_elements = 0;
    double lx_m = 0.0;
    double ly_m = 0.0;
    double lz_m = 0.0;
    int polynomial_order = 4;

    // Time stepping
    double output_dt_s = 0.01;
    double total_duration_s = 5.0;
    double cfl_safety = 0.5;
    int log_stride = 100;
    double restart_dt_s = 0.5;

    // I/O
    int snapshot_precision_bytes = 4;  // 4 = float32, 8 = float64
    double storage_limit_gb = 10.0;
    double record_depth_max_m = 0.0;
    std::vector<int> tilex_elements;
    std::vector<int> tiley_elements;

    // Parallelism
    int n_ranks = 1;

    // Boundary conditions
    int pml_xmin = 0, pml_xmax = 0;
    int pml_ymin = 0, pml_ymax = 0;
    int pml_zmin = 0, pml_zmax = 0;

    // Source
    double source_x_m = 0.0;
    double source_y_m = 0.0;
    double source_z_m = -1.0;  // negative = free surface
    double source_force_amplitude_n = 1.0e20;
    double f0_for_pml_hz = 2.0;

    std::string title = "gf_calculation";
};

// ── User must implement ────────────────────────────────────────────────────

Config get_config();
double stf_func(double t_s);
void evaluate_vp(std::size_t n, const double* x, const double* y, const double* z, double* result);
void evaluate_vs(std::size_t n, const double* x, const double* y, const double* z, double* result);
void evaluate_density(std::size_t n, const double* x, const double* y, const double* z,
                      double* result);

// ── Framework utilities (provided, users do not implement) ─────────────────

void evaluate_stf_array(double dt, int nsteps, std::vector<double>& times,
                        std::vector<double>& values);

}  // namespace gf