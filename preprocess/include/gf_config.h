#pragma once
/// gf_config.h — user-defined simulation configuration and material model
///
/// Users create a single config.cpp that implements all functions declared
/// here.  The build system links it statically into gf_preprocess:
///
///   cmake -B build -DGF_USER_CONFIG=my_config.cpp
///   cmake --build build
///
/// The resulting gf_preprocess binary is fully self-contained — no Python,
/// no external config files needed at runtime.  Just run:
///
///   gf_preprocess model.h5
///
/// Every function in this header has a default weak implementation (provided
/// by the framework) so users only need to implement what they need.  At a
/// minimum, implement get_config() and the three evaluate_* material functions.

#include <cstddef>
#include <string>
#include <vector>

namespace gf {

// ═══════════════════════════════════════════════════════════════════════════
//  Simulation configuration — returned by get_config()
// ═══════════════════════════════════════════════════════════════════════════

struct Config {
    // ── Mesh ──────────────────────────────────────────────────────────
    int nx_elements = 0;       ///< Elements in X direction
    int ny_elements = 0;       ///< Elements in Y direction
    double lx_m = 0.0;         ///< Domain size X [m]
    double ly_m = 0.0;         ///< Domain size Y [m]
    double lz_m = 0.0;         ///< Domain size Z [m]
    int polynomial_order = 4;  ///< GLL quadrature order (N)

    // ── Time stepping ─────────────────────────────────────────────────
    double output_dt_s = 0.01;      ///< Snapshot interval [s]
    double total_duration_s = 5.0;  ///< Total simulation duration [s]
    double cfl_safety = 0.5;        ///< CFL safety factor (0, 1)
    int log_stride = 100;           ///< Progress-log interval in steps
    double restart_dt_s = 0.5;      ///< Restart checkpoint interval [s] (0 = off)

    // ── I/O ───────────────────────────────────────────────────────────
    int snapshot_precision_bytes = 4;  ///< 4 = float32, 8 = float64
    double storage_limit_gb = 10.0;    ///< Warn if estimated output exceeds this
    double record_depth_max_m = 0.0;   ///< Max recording depth below free surface [m]
    std::vector<int> tilex_elements;   ///< Tile sizes in X (elements)
    std::vector<int> tiley_elements;   ///< Tile sizes in Y (elements)

    // ── Parallelism ───────────────────────────────────────────────────
    int n_ranks = 1;  ///< Number of MPI ranks (METIS partition)

    // ── Boundary conditions ───────────────────────────────────────────
    int pml_xmin = 0;  ///< PML thickness in elements, -X face
    int pml_xmax = 0;  ///< PML thickness in elements, +X face
    int pml_ymin = 0;
    int pml_ymax = 0;
    int pml_zmin = 0;
    int pml_zmax = 0;

    // ── Source ────────────────────────────────────────────────────────
    double source_x_m = 0.0;                   ///< Source X coordinate [m]
    double source_y_m = 0.0;                   ///< Source Y coordinate [m]
    double source_z_m = -1.0;                  ///< Source Z [m]; negative = free surface
    double source_force_amplitude_n = 1.0e20;  ///< Force amplitude [N]
    double f0_for_pml_hz = 2.0;                ///< Dominant frequency for C-PML tuning [Hz]

    /// Title string written into config.h5 (informational only)
    std::string title = "gf_calculation";
};

/// User must implement: return the simulation configuration.
/// Called once at startup by gf_preprocess.
Config get_config();

// ═══════════════════════════════════════════════════════════════════════════
//  Source time function (STF)
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate the source time function at time @p t_s [s].
/// Must be callable with scalar or array t_s (vectorized).
/// Return force in Newtons.
double stf_func(double t_s);

// ═══════════════════════════════════════════════════════════════════════════
//  Material model — evaluated at every GLL node
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate P-wave velocity [m/s] at @p n coordinate points.
/// @param n       Number of evaluation points.
/// @param x,y,z   Flat arrays of coordinates [m], length n.
/// @param result  Output array, length n, pre-allocated by caller.
void evaluate_vp(std::size_t n, const double* x, const double* y, const double* z, double* result);

/// Evaluate S-wave velocity [m/s].
void evaluate_vs(std::size_t n, const double* x, const double* y, const double* z, double* result);

/// Evaluate density [kg/m³].
void evaluate_density(std::size_t n, const double* x, const double* y, const double* z,
                      double* result);

}  // namespace gf