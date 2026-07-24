#pragma once
/// gf_material_user.h — user-defined material model interface
///
/// Users implement these three functions in a single .cpp file, then link
/// against the pre-built gf_preprocess library via CMake:
///
///   cmake -B build -DGF_MATERIAL_USER_SOURCE=my_material.cpp
///   cmake --build build
///
/// The resulting gf_preprocess binary calls these functions directly
/// (no HDF5 round-trip, no Python dependency) between stage1 and stage2.
///
/// When GF_MATERIAL_USER_SOURCE is NOT provided, gf_preprocess falls back
/// to reading pre-computed vp/vs/density arrays from HDF5 (written by
/// the Python cli.py orchestrator).

#include <cstddef>

namespace gf {

/// Evaluate P-wave velocity [m/s] at @p n coordinate points.
///
/// @param n       Number of evaluation points.
/// @param x       Flat array of x-coordinates [m], length n.
/// @param y       Flat array of y-coordinates [m], length n.
/// @param z       Flat array of z-coordinates [m], length n.
/// @param result  Output array, length n, pre-allocated by caller.
void evaluate_vp(std::size_t n, const double* x, const double* y, const double* z, double* result);

/// Evaluate S-wave velocity [m/s] at @p n coordinate points.
void evaluate_vs(std::size_t n, const double* x, const double* y, const double* z, double* result);

/// Evaluate density [kg/m³] at @p n coordinate points.
void evaluate_density(std::size_t n, const double* x, const double* y, const double* z,
                      double* result);

}  // namespace gf