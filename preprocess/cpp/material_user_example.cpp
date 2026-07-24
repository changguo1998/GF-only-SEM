/// material_user_example.cpp — template for user-defined material models
///
/// Copy this file, implement the three evaluate_* functions, then build:
///
///   cmake -B build -DGF_MATERIAL_USER_SOURCE=my_material.cpp
///   cmake --build build
///
/// The functions receive ALL GLL node coordinates in one call (flat arrays).
/// Use simple loops or OpenMP for parallelism.

#include <cmath>

#include "gf_material_user.h"

namespace gf {

void evaluate_vp(std::size_t n, const double* x, const double* y, const double* z,
                 double* result) {
    // Example: homogeneous half-space, Vp = 5000 m/s
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = 5000.0;
    }
}

void evaluate_vs(std::size_t n, const double* x, const double* y, const double* z,
                 double* result) {
    // Example: Vs = 3000 m/s
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = 3000.0;
    }
}

void evaluate_density(std::size_t n, const double* x, const double* y, const double* z,
                      double* result) {
    // Example: density = 2700 kg/m³
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = 2700.0;
    }
}

}  // namespace gf