#pragma once
/// gf_config.h — user-defined simulation configuration and material model

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gf {

struct Config {
    int nx_elements = 0, ny_elements = 0;
    double lx_m = 0.0, ly_m = 0.0, lz_m = 0.0;
    int polynomial_order = 4;
    double output_dt_s = 0.01, total_duration_s = 5.0, cfl_safety = 0.5;
    int log_stride = 100;
    double restart_dt_s = 0.5;
    int snapshot_precision_bytes = 4;
    double storage_limit_gb = 10.0, record_depth_max_m = 0.0;
    std::vector<int> tilex_elements, tiley_elements;
    int n_ranks = 1;
    int pml_xmin = 0, pml_xmax = 0, pml_ymin = 0, pml_ymax = 0;
    int pml_zmin = 0, pml_zmax = 0;
    double source_x_m = 0.0, source_y_m = 0.0, source_z_m = -1.0;
    double source_force_amplitude_n = 1.0e20, f0_for_pml_hz = 2.0;
    std::string title = "gf_calculation";
};

struct SourceResult {
    std::vector<int> cell_ids;
    std::vector<double> xi, eta, zeta;
    std::vector<std::vector<double>> weights;
    int n_src_cell = 0;
};

// ── User must implement ────────────────────────────────────────────────────

Config get_config();
double stf_func(double t_s);
void evaluate_vp(std::size_t n, const double* x, const double* y, const double* z, double* r);
void evaluate_vs(std::size_t n, const double* x, const double* y, const double* z, double* r);
void evaluate_density(std::size_t n, const double* x, const double* y, const double* z, double* r);

// ── Framework utilities ─────────────────────────────────────────────────────

void evaluate_stf_array(double dt, int nsteps, std::vector<double>& times,
                        std::vector<double>& values);
SourceResult locate_source(const Config& cfg, const double* gll_coords_flat, int n_cell, int ngll,
                           const int64_t* cell_to_surface, int n_surface,
                           const int64_t* boundary_tag, const int* is_pml);
void compute_cpml_profiles(const double* gll_coords_flat, int n_cell, int ngll, const int* is_pml,
                           const int* pml_regions, const double* domain_bounds,
                           const double* pml_widths, const double* vp_flat, double f0_hz,
                           std::vector<double>& K_store, std::vector<double>& d_store,
                           std::vector<double>& alpha_store);

// ── METIS partition + global node numbering ──
void partition_metis(const char* model_path, int n_ranks);
void compute_global_node_ids(const char* model_path, int ngll);

// ── config.h5 writer ──
void write_config_h5(const char* config_path, const Config& cfg, double solver_dt,
                     int snapshot_stride, int nsteps, const std::vector<double>& stf_t,
                     const std::vector<double>& stf_values, const std::vector<double>& source_xyz,
                     const SourceResult& src_result, double record_depth_actual_m,
                     const std::vector<int32_t>& element_to_rank, int n_ranks, double log_dt_s);

}  // namespace gf