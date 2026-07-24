/// config_default.cpp — default implementations of gf::Config functions
///
/// These are compiled into gf_preprocess when no user config is provided
/// (GF_USER_CONFIG not set).  They provide sensible defaults that users
/// can override by supplying their own config.cpp.

#include <cmath>

#include "gf_config.h"

namespace gf {

// ── Config ─────────────────────────────────────────────────────────────────

Config get_config() {
    Config cfg;
    cfg.title = "gf_calculation (default config — user should override)";
    return cfg;
}

// ── STF ────────────────────────────────────────────────────────────────────

double stf_func(double t_s) {
    // Ricker wavelet (second derivative of Gaussian), f0=2 Hz, t0=1.0 s
    double f0_hz = 2.0;
    double t0_s = 1.0;
    double a = M_PI * f0_hz * (t_s - t0_s);
    return 1.0e20 * (1.0 - 2.0 * a * a) * std::exp(-(a * a));
}

// ── Material ───────────────────────────────────────────────────────────────

void evaluate_vp(std::size_t n, const double* /*x*/, const double* /*y*/, const double* /*z*/,
                 double* result) {
    for (std::size_t i = 0; i < n; ++i)
        result[i] = 3000.0;
}

void evaluate_vs(std::size_t n, const double* /*x*/, const double* /*y*/, const double* /*z*/,
                 double* result) {
    for (std::size_t i = 0; i < n; ++i)
        result[i] = 1500.0;
}

void evaluate_density(std::size_t n, const double* /*x*/, const double* /*y*/, const double* /*z*/,
                      double* result) {
    for (std::size_t i = 0; i < n; ++i)
        result[i] = 2500.0;
}

}  // namespace gf