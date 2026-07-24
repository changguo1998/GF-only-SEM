#pragma once
/// gf_config.h — user-defined simulation configuration and material model

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gf {

// ═══════════════════════════════════════════════════════════════════════════
//  Simulation configuration — returned by get_config()
// ═══════════════════════════════════════════════════════════════════════════

struct Config {
    int nx_elements = 0;
    int ny_elements = 0;
    double lx_m = 0.0, ly_m = 0.0, lz_m = 0.0;
    int polynomial_order = 4;

    double output_dt_s = 0.01;
    double total_duration_s = 5.0;
    double cfl_safety = 0.5;
    int log_stride = 100;
    double restart_dt_s = 0.5;

    int snapshot_precision_bytes = 4;
    double storage_limit_gb = 10.0;
    double record_depth_max_m = 0.0;
    std::vector<int> tilex_elements;
    std::vector<int> tiley_elements;

    int n_ranks = 1;

    int pml_xmin = 0, pml_xmax = 0;
    int pml_ymin = 0, pml_ymax = 0;
    int pml_zmin = 0, pml_zmax = 0;

    double source_x_m = 0.0, source_y_m = 0.0;
    double source_z_m = -1.0;  // negative = free surface
    double source_force_amplitude_n = 1.0e20;
    double f0_for_pml_hz = 2.0;

    std::string title = "gf_calculation";
};

// ═══════════════════════════════════════════════════════════════════════════
//  Source location result
// ═══════════════════════════════════════════════════════════════════════════

struct SourceResult {
    std::vector<int> cell_ids;
    std::vector<double> xi, eta, zeta;
    std::vector<std::vector<double>> weights;  // per-cell, flat [ngll^3]
    int n_src_cell = 0;
};

// ═══════════════════════════════════════════════════════════════════════════
//  User must implement
// ═══════════════════════════════════════════════════════════════════════════

Config get_config();
double stf_func(double t_s);
void evaluate_vp(std::size_t n, const double* x, const double* y, const double* z, double* result);
void evaluate_vs(std::size_t n, const double* x, const double* y, const double* z, double* result);
void evaluate_density(std::size_t n, const double* x, const double* y, const double* z,
                      double* result);

// ═══════════════════════════════════════════════════════════════════════════
//  Framework utilities (provided, users do not implement)
// ═══════════════════════════════════════════════════════════════════════════

void evaluate_stf_array(double dt, int nsteps, std::vector<double>& times,
                        std::vector<double>& values);

SourceResult locate_source(const Config& cfg, const double* gll_coords_flat, int n_cell, int ngll,
                           const int64_t* cell_to_surface, int n_surface,
                           const int64_t* boundary_tag, const bool* is_pml);

}  // namespace gf