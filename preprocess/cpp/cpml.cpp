/// cpml.cpp — C-PML (Convolutional PML) damping profile computation
///
/// Port of preprocess/pml_cpml.py.
/// Computes κ/d/α profiles, SPECFEM3D parameter separation, recursive
/// convolution coefficients, acceleration coefficients, and strain coefficients.
///
/// Reference: Wang et al. (2006), SPECFEM3D pml_compute_memory_variables.f90

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "gf_config.h"

namespace gf {
namespace {

// ── Constants (SPECFEM3D defaults) ──────────────────────────────────────────

constexpr double K_MAX_PML = 1.0;  // no stretching (K=1 everywhere)
constexpr double K_MIN_PML = 1.0;
constexpr int NPOWER = 2;
constexpr double R_COEF = 1e-5;  // target reflection coefficient
constexpr double DIST_EPSILON = 1e-3;
constexpr double THETA = 1.0 / 8.0;  // Wang et al. second-order convolution
constexpr double MIN_DISTANCE = 1e-12;
constexpr double MIN_DISTANCE_FACTOR = 1.0 / 8.0;
constexpr double COEF_SAFETY_CLAMP = 3.0;

// ── PML region codes ───────────────────────────────────────────────────────

constexpr int CPML_X_ONLY = 1, CPML_Y_ONLY = 2, CPML_Z_ONLY = 3;
constexpr int CPML_XY_ONLY = 4, CPML_XZ_ONLY = 5, CPML_YZ_ONLY = 6;
constexpr int CPML_XYZ = 7;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Check if axis is active for a given PML region.
bool is_axis_active(int region, int axis) {
    if (region == 0)
        return false;  // interior
    switch (axis) {
        case 0:
            return region == CPML_X_ONLY || region == CPML_XY_ONLY || region == CPML_XZ_ONLY ||
                   region == CPML_XYZ;
        case 1:
            return region == CPML_Y_ONLY || region == CPML_XY_ONLY || region == CPML_YZ_ONLY ||
                   region == CPML_XYZ;
        case 2:
            return region == CPML_Z_ONLY || region == CPML_XZ_ONLY || region == CPML_YZ_ONLY ||
                   region == CPML_XYZ;
        default:
            return false;
    }
}

/// Compute damping profile d(x) along one direction.
/// d = -(NPOWER+1) * vp * ln(R) / (2 * width) * dist^(1.2*NPOWER)
inline double damping_value(double dist, double vp, double pml_width) {
    if (pml_width <= 0.0)
        return 0.0;
    double exponent = 1.2 * NPOWER;
    return -(NPOWER + 1.0) * vp * std::log(R_COEF) / (2.0 * pml_width) * std::pow(dist, exponent);
}

inline double safe_denominator(double value) {
    return std::abs(value) < MIN_DISTANCE ? MIN_DISTANCE : value;
}

inline double clamp_coefficient(double value) {
    return std::clamp(value, -COEF_SAFETY_CLAMP, COEF_SAFETY_CLAMP);
}

void separate_two_changeable(double& first, double& second, double separation_twice) {
    if (first >= second)
        first = second + separation_twice;
    else
        second = first + separation_twice;
}

void separate_one_changeable(double& changeable, double fixed, double separation_twice,
                             double separation_fourfold) {
    changeable = changeable >= fixed ? fixed + separation_twice : fixed + separation_fourfold;
}

void separate_edge_node(double& first_alpha, double& second_alpha, double first_k, double second_k,
                        double& first_d, double& second_d, double minimum_separation,
                        double separation_twice, double separation_fourfold) {
    if (std::abs(first_alpha - second_alpha) < minimum_separation)
        separate_two_changeable(first_alpha, second_alpha, separation_twice);

    double first_beta = first_alpha + first_d / std::max(first_k, 1.0);
    double second_beta = second_alpha + second_d / std::max(second_k, 1.0);
    if (std::abs(first_beta - second_alpha) < minimum_separation)
        separate_one_changeable(first_beta, second_alpha, separation_twice, separation_fourfold);
    if (std::abs(second_beta - first_alpha) < minimum_separation)
        separate_one_changeable(second_beta, first_alpha, separation_twice, separation_fourfold);

    first_d = (first_beta - first_alpha) * std::max(first_k, 1.0);
    second_d = (second_beta - second_alpha) * std::max(second_k, 1.0);
}

void separate_xyz_node(double& alpha_x, double& alpha_y, double& alpha_z, double k_x, double k_y,
                       double k_z, double& d_x, double& d_y, double& d_z,
                       double minimum_separation, double separation_twice,
                       double separation_fourfold) {
    // Separate the three alpha values in the same staged order as SPECFEM3D.
    if (std::abs(alpha_x - alpha_y) < minimum_separation) {
        if (alpha_x > alpha_y)
            alpha_x = alpha_y + separation_twice;
        else
            alpha_y = alpha_x + separation_twice;
        double maximum = std::max(alpha_x, alpha_y);
        double minimum = std::min(alpha_x, alpha_y);
        if (alpha_z > maximum) {
            if (std::abs(alpha_z - maximum) < minimum_separation)
                alpha_z = maximum + separation_twice;
        } else if (alpha_z < minimum) {
            if (std::abs(alpha_z - minimum) < minimum_separation) {
                if (alpha_x > alpha_y) {
                    alpha_x = alpha_z + separation_fourfold;
                    alpha_y = alpha_z + separation_twice;
                } else {
                    alpha_y = alpha_z + separation_fourfold;
                    alpha_x = alpha_z + separation_twice;
                }
            }
        } else if (alpha_x > alpha_y) {
            alpha_x = alpha_y + separation_fourfold;
            alpha_z = alpha_y + separation_twice;
        } else {
            alpha_y = alpha_x + separation_fourfold;
            alpha_z = alpha_x + separation_twice;
        }
    }

    if (std::abs(alpha_x - alpha_z) < minimum_separation) {
        if (alpha_x > alpha_z)
            alpha_x = alpha_z + separation_twice;
        else
            alpha_z = alpha_x + separation_twice;
        double maximum = std::max(alpha_x, alpha_z);
        double minimum = std::min(alpha_x, alpha_z);
        if (alpha_y > maximum) {
            if (std::abs(alpha_y - maximum) < minimum_separation)
                alpha_y = maximum + separation_twice;
        } else if (alpha_y < minimum) {
            if (std::abs(alpha_y - minimum) < minimum_separation) {
                if (alpha_x > alpha_z) {
                    alpha_x = alpha_y + separation_fourfold;
                    alpha_z = alpha_y + separation_twice;
                } else {
                    alpha_z = alpha_y + separation_fourfold;
                    alpha_x = alpha_y + separation_twice;
                }
            }
        } else if (alpha_x > alpha_z) {
            alpha_x = alpha_z + separation_fourfold;
            alpha_y = alpha_z + separation_twice;
        } else {
            alpha_z = alpha_x + separation_fourfold;
            alpha_y = alpha_x + separation_twice;
        }
    }

    if (std::abs(alpha_y - alpha_z) < minimum_separation) {
        if (alpha_y > alpha_z)
            alpha_y = alpha_z + separation_twice;
        else
            alpha_z = alpha_y + separation_twice;
        double maximum = std::max(alpha_y, alpha_z);
        double minimum = std::min(alpha_y, alpha_z);
        if (alpha_x > maximum) {
            if (std::abs(alpha_x - maximum) < minimum_separation)
                alpha_x = maximum + separation_twice;
        } else if (alpha_x < minimum) {
            if (std::abs(alpha_x - minimum) < minimum_separation) {
                if (alpha_y > alpha_z) {
                    alpha_y = alpha_x + separation_fourfold;
                    alpha_z = alpha_x + separation_twice;
                } else {
                    alpha_z = alpha_x + separation_fourfold;
                    alpha_y = alpha_x + separation_twice;
                }
            }
        } else if (alpha_y > alpha_z) {
            alpha_y = alpha_z + separation_fourfold;
            alpha_x = alpha_z + separation_twice;
        } else {
            alpha_z = alpha_y + separation_fourfold;
            alpha_x = alpha_y + separation_twice;
        }
    }

    // Separate every beta from the two alpha values belonging to the other axes.
    auto separate_beta = [&](double beta, double first_other_alpha, double second_other_alpha) {
        double maximum = std::max(first_other_alpha, second_other_alpha);
        double minimum = std::min(first_other_alpha, second_other_alpha);
        if (beta > maximum) {
            if (std::abs(beta - maximum) < minimum_separation)
                beta = maximum + separation_twice;
        } else if (beta < minimum) {
            if (std::abs(beta - minimum) < minimum_separation)
                beta = maximum + separation_twice;
        } else {
            if (std::abs(beta - maximum) < minimum_separation)
                beta = maximum + separation_twice;
            if (std::abs(beta - minimum) < minimum_separation) {
                beta = minimum + separation_twice;
                if (std::abs(beta - maximum) < minimum_separation)
                    beta = maximum + separation_twice;
            }
        }
        return beta;
    };

    double beta_x = separate_beta(alpha_x + d_x / std::max(k_x, 1.0), alpha_y, alpha_z);
    double beta_y = separate_beta(alpha_y + d_y / std::max(k_y, 1.0), alpha_x, alpha_z);
    double beta_z = separate_beta(alpha_z + d_z / std::max(k_z, 1.0), alpha_x, alpha_y);
    d_x = (beta_x - alpha_x) * std::max(k_x, 1.0);
    d_y = (beta_y - alpha_y) * std::max(k_y, 1.0);
    d_z = (beta_z - alpha_z) * std::max(k_z, 1.0);
}

double minimum_axis_squared_distance(const double* coordinates, int ngll, int logical_axis) {
    double minimum_squared_distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < ngll; ++i) {
        for (int j = 0; j < ngll; ++j) {
            for (int k = 0; k < ngll; ++k) {
                int previous_i = i, previous_j = j, previous_k = k;
                if (logical_axis == 0) {
                    if (i == 0)
                        continue;
                    previous_i = i - 1;
                } else if (logical_axis == 1) {
                    if (j == 0)
                        continue;
                    previous_j = j - 1;
                } else {
                    if (k == 0)
                        continue;
                    previous_k = k - 1;
                }
                int current = ((i * ngll + j) * ngll + k) * 3;
                int previous = ((previous_i * ngll + previous_j) * ngll + previous_k) * 3;
                double squared_distance = 0.0;
                for (int component = 0; component < 3; ++component) {
                    double difference =
                        coordinates[current + component] - coordinates[previous + component];
                    squared_distance += difference * difference;
                }
                if (squared_distance > 0.0)
                    minimum_squared_distance =
                        std::min(minimum_squared_distance, squared_distance);
            }
        }
    }
    return minimum_squared_distance;
}

void separate_pml_parameters(const double* coordinates, int n_cell, int ngll,
                             const int* pml_regions, const double* pml_widths,
                             std::vector<double>& k_store, std::vector<double>& d_store,
                             std::vector<double>& alpha_store) {
    int nodes_per_cell = ngll * ngll * ngll;
    bool has_damped_cell = false;
    for (double damping : d_store)
        has_damped_cell = has_damped_cell || damping > 0.0;

    double minimum_squared_distance = std::numeric_limits<double>::infinity();
    for (int cell = 0; cell < n_cell; ++cell) {
        bool include = !has_damped_cell;
        for (int node = 0; node < nodes_per_cell && !include; ++node) {
            int base = (cell * nodes_per_cell + node) * 3;
            include = d_store[base] > 0.0 || d_store[base + 1] > 0.0 || d_store[base + 2] > 0.0 ||
                      k_store[base] > 1.0;
        }
        if (!include)
            continue;
        const double* cell_coordinates = coordinates + cell * nodes_per_cell * 3;
        for (int axis = 0; axis < 3; ++axis) {
            double squared_distance = minimum_axis_squared_distance(cell_coordinates, ngll, axis);
            if (std::isfinite(squared_distance) && squared_distance > 0.0)
                minimum_squared_distance = std::min(minimum_squared_distance, squared_distance);
        }
    }
    if (!std::isfinite(minimum_squared_distance) || minimum_squared_distance <= 0.0)
        return;

    double maximum_alpha = *std::max_element(alpha_store.begin(), alpha_store.end());
    if (maximum_alpha <= 0.0)
        maximum_alpha = M_PI * 10.0 * 1.1;
    double maximum_pml_width = *std::max_element(pml_widths, pml_widths + 6);
    if (maximum_pml_width <= 0.0)
        return;
    double minimum_separation = maximum_alpha * std::sqrt(minimum_squared_distance) /
                                maximum_pml_width * MIN_DISTANCE_FACTOR;
    if (minimum_separation <= 0.0)
        return;
    double separation_twice = minimum_separation * 2.0;
    double separation_fourfold = minimum_separation * 4.0;

    for (int cell = 0; cell < n_cell; ++cell) {
        int region = pml_regions ? pml_regions[cell] : 0;
        if (region == 0)
            continue;
        for (int node = 0; node < nodes_per_cell; ++node) {
            int base = (cell * nodes_per_cell + node) * 3;
            double& d_x = d_store[base];
            double& d_y = d_store[base + 1];
            double& d_z = d_store[base + 2];
            if (d_x == 0.0 && d_y == 0.0 && d_z == 0.0)
                continue;
            double& alpha_x = alpha_store[base];
            double& alpha_y = alpha_store[base + 1];
            double& alpha_z = alpha_store[base + 2];
            double k_x = k_store[base];
            double k_y = k_store[base + 1];
            double k_z = k_store[base + 2];
            if (region == CPML_XY_ONLY)
                separate_edge_node(alpha_x, alpha_y, k_x, k_y, d_x, d_y, minimum_separation,
                                   separation_twice, separation_fourfold);
            else if (region == CPML_XZ_ONLY)
                separate_edge_node(alpha_x, alpha_z, k_x, k_z, d_x, d_z, minimum_separation,
                                   separation_twice, separation_fourfold);
            else if (region == CPML_YZ_ONLY)
                separate_edge_node(alpha_y, alpha_z, k_y, k_z, d_y, d_z, minimum_separation,
                                   separation_twice, separation_fourfold);
            else if (region == CPML_XYZ)
                separate_xyz_node(alpha_x, alpha_y, alpha_z, k_x, k_y, k_z, d_x, d_y, d_z,
                                  minimum_separation, separation_twice, separation_fourfold);
        }
    }
}

std::array<double, 3> convolution_coefficients(double damping_parameter, double solver_dt) {
    double half_step_decay = std::exp(-0.5 * damping_parameter * solver_dt);
    double coefficient_zero = half_step_decay * half_step_decay;
    double coefficient_one = 0.0;
    double coefficient_two = 0.0;
    if (std::abs(damping_parameter) >= MIN_DISTANCE) {
        coefficient_one = (1.0 - half_step_decay) / damping_parameter;
        coefficient_two = coefficient_one * half_step_decay;
    } else {
        double solver_dt_squared = solver_dt * solver_dt;
        double solver_dt_cubed = solver_dt_squared * solver_dt;
        double solver_dt_fourth = solver_dt_cubed * solver_dt;
        double damping_squared = damping_parameter * damping_parameter;
        double damping_cubed = damping_squared * damping_parameter;
        coefficient_one = solver_dt * 0.5 - 0.125 * solver_dt_squared * damping_parameter +
                          solver_dt_cubed * damping_squared / 48.0 -
                          solver_dt_fourth * damping_cubed / 384.0;
        coefficient_two = solver_dt * 0.5 - 0.375 * solver_dt_squared * damping_parameter +
                          7.0 * solver_dt_cubed * damping_squared / 48.0 -
                          5.0 * solver_dt_fourth * damping_cubed / 128.0;
    }
    return {coefficient_zero, coefficient_one, coefficient_two};
}

std::array<double, 5> acceleration_coefficients(int region, double k_x, double d_x, double alpha_x,
                                                double k_y, double d_y, double alpha_y, double k_z,
                                                double d_z, double alpha_z) {
    double beta_x = alpha_x + d_x / std::max(k_x, 1.0);
    double beta_y = alpha_y + d_y / std::max(k_y, 1.0);
    double beta_z = alpha_z + d_z / std::max(k_z, 1.0);
    double coefficient_one = 0.0, coefficient_two = 0.0, coefficient_three = 0.0;
    double coefficient_four = 0.0, coefficient_five = 0.0;

    if (region == CPML_XYZ) {
        double coefficient_zero = k_x * k_y * k_z;
        coefficient_one =
            coefficient_zero * (beta_x + beta_y + beta_z - alpha_x - alpha_y - alpha_z);
        coefficient_two = coefficient_zero * (beta_x - alpha_x) * (beta_y - alpha_y - alpha_x) +
                          coefficient_zero * (beta_y - alpha_y) * (beta_z - alpha_z - alpha_y) +
                          coefficient_zero * (beta_z - alpha_z) * (beta_x - alpha_x - alpha_z);
        coefficient_three =
            coefficient_zero * alpha_x * alpha_x * (beta_x - alpha_x) * (beta_y - alpha_x) *
            (beta_z - alpha_x) /
            (safe_denominator(alpha_y - alpha_x) * safe_denominator(alpha_z - alpha_x));
        coefficient_four =
            coefficient_zero * alpha_y * alpha_y * (beta_x - alpha_y) * (beta_y - alpha_y) *
            (beta_z - alpha_y) /
            (safe_denominator(alpha_x - alpha_y) * safe_denominator(alpha_z - alpha_y));
        coefficient_five =
            coefficient_zero * alpha_z * alpha_z * (beta_x - alpha_z) * (beta_y - alpha_z) *
            (beta_z - alpha_z) /
            (safe_denominator(alpha_y - alpha_z) * safe_denominator(alpha_z - alpha_x));
    } else if (region == CPML_XY_ONLY) {
        double coefficient_zero = k_x * k_y;
        coefficient_one = coefficient_zero * (beta_x + beta_y - alpha_x - alpha_y);
        coefficient_two = coefficient_zero * (beta_x - alpha_x) * (beta_y - alpha_y - alpha_x) -
                          coefficient_zero * (beta_y - alpha_y) * alpha_y;
        coefficient_three = coefficient_zero * alpha_x * alpha_x * (beta_x - alpha_x) *
                            (beta_y - alpha_x) / safe_denominator(alpha_y - alpha_x);
        coefficient_four = coefficient_zero * alpha_y * alpha_y * (beta_x - alpha_y) *
                           (beta_y - alpha_y) / safe_denominator(alpha_x - alpha_y);
    } else if (region == CPML_XZ_ONLY) {
        double coefficient_zero = k_x * k_z;
        coefficient_one = coefficient_zero * (beta_x + beta_z - alpha_x - alpha_z);
        coefficient_two = coefficient_zero * (beta_x - alpha_x) * (-alpha_x) +
                          coefficient_zero * (beta_z - alpha_z) * (beta_x - alpha_x - alpha_z);
        coefficient_three = coefficient_zero * alpha_x * alpha_x * (beta_x - alpha_x) *
                            (beta_z - alpha_x) / safe_denominator(alpha_z - alpha_x);
        coefficient_five = coefficient_zero * alpha_z * alpha_z * (beta_x - alpha_z) *
                           (beta_z - alpha_z) / safe_denominator(alpha_x - alpha_z);
    } else if (region == CPML_YZ_ONLY) {
        double coefficient_zero = k_y * k_z;
        coefficient_one = coefficient_zero * (beta_y + beta_z - alpha_y - alpha_z);
        coefficient_two = coefficient_zero * (beta_y - alpha_y) * (beta_z - alpha_z - alpha_y) -
                          coefficient_zero * (beta_z - alpha_z) * alpha_z;
        coefficient_four = coefficient_zero * alpha_y * alpha_y * (beta_y - alpha_y) *
                           (beta_z - alpha_y) / safe_denominator(alpha_z - alpha_y);
        coefficient_five = coefficient_zero * alpha_z * alpha_z * (beta_y - alpha_z) *
                           (beta_z - alpha_z) / safe_denominator(alpha_y - alpha_z);
    } else if (region == CPML_X_ONLY) {
        double difference = beta_x - alpha_x;
        coefficient_one = k_x * difference;
        coefficient_two = -k_x * alpha_x * difference;
        coefficient_three = k_x * alpha_x * alpha_x * difference;
    } else if (region == CPML_Y_ONLY) {
        double difference = beta_y - alpha_y;
        coefficient_one = k_y * difference;
        coefficient_two = -k_y * alpha_y * difference;
        coefficient_four = k_y * alpha_y * alpha_y * difference;
    } else if (region == CPML_Z_ONLY) {
        double difference = beta_z - alpha_z;
        coefficient_one = k_z * difference;
        coefficient_two = -k_z * alpha_z * difference;
        coefficient_five = k_z * alpha_z * alpha_z * difference;
    }
    return {coefficient_one, coefficient_two, coefficient_three, coefficient_four,
            coefficient_five};
}

std::array<double, 4> strain_mixed_coefficients(int region, double k_x, double d_x, double alpha_x,
                                                double k_y, double d_y, double alpha_y, double k_z,
                                                double d_z, double alpha_z) {
    double beta_x = alpha_x + d_x / std::max(k_x, 1.0);
    double beta_y = alpha_y + d_y / std::max(k_y, 1.0);
    double beta_z = alpha_z + d_z / std::max(k_z, 1.0);
    double coefficient_zero = 0.0, coefficient_one = 0.0, coefficient_two = 0.0;
    double coefficient_three = 0.0;

    if (region == CPML_XYZ) {
        coefficient_zero = k_x * k_y / std::max(k_z, 1.0);
        coefficient_one =
            -coefficient_zero * (alpha_x - alpha_z) * (alpha_x - beta_x) * (alpha_x - beta_y) /
            (safe_denominator(alpha_x - alpha_y) * safe_denominator(alpha_x - beta_z));
        coefficient_two =
            -coefficient_zero * (alpha_y - alpha_z) * (alpha_y - beta_x) * (alpha_y - beta_y) /
            (safe_denominator(alpha_y - alpha_x) * safe_denominator(alpha_y - beta_z));
        coefficient_three =
            -coefficient_zero * (beta_z - alpha_z) * (beta_z - beta_x) * (beta_z - beta_y) /
            (safe_denominator(beta_z - alpha_x) * safe_denominator(beta_z - alpha_y));
    } else if (region == CPML_X_ONLY) {
        coefficient_zero = k_x;
        coefficient_one = -coefficient_zero * (alpha_x - beta_x);
    } else if (region == CPML_Y_ONLY) {
        coefficient_zero = k_y;
        coefficient_two = -coefficient_zero * (alpha_y - beta_y);
    } else if (region == CPML_Z_ONLY) {
        coefficient_zero = 1.0 / std::max(k_z, 1.0);
        coefficient_three = -coefficient_zero * (beta_z - alpha_z);
    } else if (region == CPML_XY_ONLY) {
        coefficient_zero = k_x * k_y;
        coefficient_one = -coefficient_zero * (alpha_x - beta_x) * (alpha_x - beta_y) /
                          safe_denominator(alpha_x - alpha_y);
        coefficient_two = -coefficient_zero * (alpha_y - beta_x) * (alpha_y - beta_y) /
                          safe_denominator(alpha_y - alpha_x);
    } else if (region == CPML_XZ_ONLY) {
        coefficient_zero = k_x / std::max(k_z, 1.0);
        coefficient_one = -coefficient_zero * (alpha_x - alpha_z) * (alpha_x - beta_x) /
                          safe_denominator(alpha_x - beta_z);
        coefficient_three = -coefficient_zero * (beta_z - alpha_z) * (beta_z - beta_x) /
                            safe_denominator(beta_z - alpha_x);
    } else if (region == CPML_YZ_ONLY) {
        coefficient_zero = k_y / std::max(k_z, 1.0);
        coefficient_two = -coefficient_zero * (alpha_y - alpha_z) * (alpha_y - beta_y) /
                          safe_denominator(alpha_y - beta_z);
        coefficient_three = -coefficient_zero * (beta_z - alpha_z) * (beta_z - beta_y) /
                            safe_denominator(beta_z - alpha_y);
    }
    return {coefficient_zero, coefficient_one, coefficient_two, coefficient_three};
}

std::array<double, 2> strain_direction_coefficients(int region, int axis, double k_value,
                                                    double d_value, double alpha_value) {
    if (!is_axis_active(region, axis))
        return {1.0, 0.0};
    double beta_value = alpha_value + d_value / std::max(k_value, 1.0);
    return {k_value, -k_value * (alpha_value - beta_value)};
}

}  // anonymous namespace

// ── Public API ──────────────────────────────────────────────────────────────

void compute_cpml_profiles(const double* gll_coords_flat,  // [n_cell * ngll³ * 3]
                           int n_cell, int ngll,
                           const int* is_pml,            // [n_cell]
                           const int* pml_regions,       // [n_cell], 0=interior, 1-7=PML
                           const double* domain_bounds,  // [6]: xmin,xmax,ymin,ymax,zmin,zmax
                           const double* pml_widths,     // [6]: corresponding PML widths
                           const double* vp_flat,        // [n_cell * ngll³]
                           double f0_hz,
                           // Output: each [n_cell * ngll³ * 3], row-major per node per axis(0,1,2)
                           std::vector<double>& K_store, std::vector<double>& d_store,
                           std::vector<double>& alpha_store) {
    int n_node = ngll * ngll * ngll;
    int total_nodes = n_cell * n_node;
    int stride_coords = n_node * 3;
    int stride_scalar = n_node;

    // Initialize: K=1, d=0, alpha=0
    K_store.assign(total_nodes * 3, 1.0);
    d_store.assign(total_nodes * 3, 0.0);
    alpha_store.assign(total_nodes * 3, 0.0);

    // alpha_max per axis: π·f0 scaled (SPECFEM3D convention)
    double alpha_max[3] = {
        M_PI * f0_hz * 0.9,  // x
        M_PI * f0_hz * 1.0,  // y
        M_PI * f0_hz * 1.1   // z
    };

    // Face definitions: (axis, boundary_value, pml_width_index, sign)
    // sign: -1 for min face, +1 for max face (distance = |coord - boundary|)
    struct Face {
        int axis;
        double boundary;
        int wi;
        int direction_sign;
    };
    Face faces[6] = {
        {0, domain_bounds[0], 0, -1},  // xmin
        {0, domain_bounds[1], 1, +1},  // xmax
        {1, domain_bounds[2], 2, -1},  // ymin
        {1, domain_bounds[3], 3, +1},  // ymax
        {2, domain_bounds[4], 4, -1},  // zmin
        {2, domain_bounds[5], 5, +1},  // zmax
    };

    for (int e = 0; e < n_cell; ++e) {
        if (!is_pml[e])
            continue;

        int region = pml_regions ? pml_regions[e] : 0;
        if (region == 0)
            continue;

        const double* coords = gll_coords_flat + e * stride_coords;
        const double* vp_e = vp_flat + e * stride_scalar;

        for (const auto& face : faces) {
            double width = pml_widths[face.wi];
            if (width <= 0.0)
                continue;
            if (!is_axis_active(region, face.axis))
                continue;

            double boundary = face.boundary;
            double center = 0.0;
            for (int node = 0; node < n_node; ++node)
                center += coords[node * 3 + face.axis];
            center /= n_node;
            double tolerance = 1.0e-6 * width;
            if (face.direction_sign < 0 && center >= boundary + width + tolerance)
                continue;
            if (face.direction_sign > 0 && center <= boundary - width - tolerance)
                continue;

            for (int node = 0; node < n_node; ++node) {
                // Normalized depth from the PML interior interface toward the boundary.
                double coord = coords[node * 3 + face.axis];
                double dist = 1.0 - std::abs(coord - boundary) / width;
                if (dist < 0.0)
                    dist = 0.0;
                if (dist > 1.0 - DIST_EPSILON)
                    dist = 1.0 - DIST_EPSILON;

                double vp_val = vp_e[node];

                // K (stretching) — currently K_MAX=1, so K stays 1
                double K_val = K_MIN_PML + (K_MAX_PML - 1.0) * dist;
                // d (damping)
                double d_val = damping_value(dist, vp_val, width);
                // alpha (frequency shift)
                double alpha_val = alpha_max[face.axis] * (1.0 - dist);

                // Clamp
                if (K_val < 1.0)
                    K_val = 1.0;
                if (d_val < 0.0)
                    d_val = 0.0;
                if (alpha_val < 0.0)
                    alpha_val = 0.0;

                int out_idx = (e * n_node + node) * 3 + face.axis;
                K_store[out_idx] = K_val;
                d_store[out_idx] = d_val;
                alpha_store[out_idx] = alpha_val;
            }
        }
    }

    separate_pml_parameters(gll_coords_flat, n_cell, ngll, pml_regions, pml_widths, K_store,
                            d_store, alpha_store);
}

void compute_cpml_coefficients(int n_cell, int ngll, const int* pml_regions, double solver_dt,
                               const std::vector<double>& k_store,
                               const std::vector<double>& d_store,
                               const std::vector<double>& alpha_store,
                               std::vector<double>& coefficient_alpha,
                               std::vector<double>& coefficient_beta,
                               std::vector<double>& coefficient_acceleration,
                               std::vector<double>& coefficient_strain) {
    int nodes_per_cell = ngll * ngll * ngll;
    int total_nodes = n_cell * nodes_per_cell;
    coefficient_alpha.assign(total_nodes * 9, 0.0);
    coefficient_beta.assign(total_nodes * 9, 0.0);
    coefficient_acceleration.assign(total_nodes * 5, 0.0);
    coefficient_strain.assign(total_nodes * 18, 0.0);

    for (int global_node = 0; global_node < total_nodes; ++global_node) {
        int profile_base = global_node * 3;
        int convolution_base = global_node * 9;
        for (int axis = 0; axis < 3; ++axis) {
            double alpha = alpha_store[profile_base + axis];
            double beta =
                alpha + d_store[profile_base + axis] / std::max(k_store[profile_base + axis], 1.0);
            std::array<double, 3> alpha_values = convolution_coefficients(alpha, solver_dt);
            std::array<double, 3> beta_values = convolution_coefficients(beta, solver_dt);
            for (int coefficient = 0; coefficient < 3; ++coefficient) {
                coefficient_alpha[convolution_base + axis * 3 + coefficient] =
                    alpha_values[coefficient];
                coefficient_beta[convolution_base + axis * 3 + coefficient] =
                    beta_values[coefficient];
            }
        }

        int cell = global_node / nodes_per_cell;
        int region = pml_regions ? pml_regions[cell] : 0;
        if (region == 0)
            continue;

        double k_x = k_store[profile_base], k_y = k_store[profile_base + 1];
        double k_z = k_store[profile_base + 2];
        double d_x = d_store[profile_base], d_y = d_store[profile_base + 1];
        double d_z = d_store[profile_base + 2];
        double alpha_x = alpha_store[profile_base], alpha_y = alpha_store[profile_base + 1];
        double alpha_z = alpha_store[profile_base + 2];

        std::array<double, 5> acceleration = acceleration_coefficients(
            region, k_x, d_x, alpha_x, k_y, d_y, alpha_y, k_z, d_z, alpha_z);
        int acceleration_base = global_node * 5;
        for (int coefficient = 0; coefficient < 5; ++coefficient)
            coefficient_acceleration[acceleration_base + coefficient] =
                clamp_coefficient(acceleration[coefficient]);

        int strain_base = global_node * 18;
        std::array<double, 4> strain_x = strain_mixed_coefficients(
            region, k_z, d_z, alpha_z, k_y, d_y, alpha_y, k_x, d_x, alpha_x);
        std::array<double, 4> strain_y = strain_mixed_coefficients(
            region, k_x, d_x, alpha_x, k_z, d_z, alpha_z, k_y, d_y, alpha_y);
        std::array<double, 4> strain_z = strain_mixed_coefficients(
            region, k_x, d_x, alpha_x, k_y, d_y, alpha_y, k_z, d_z, alpha_z);
        std::array<double, 2> strain_direction_x =
            strain_direction_coefficients(region, 0, k_x, d_x, alpha_x);
        std::array<double, 2> strain_direction_y =
            strain_direction_coefficients(region, 1, k_y, d_y, alpha_y);
        std::array<double, 2> strain_direction_z =
            strain_direction_coefficients(region, 2, k_z, d_z, alpha_z);
        for (int coefficient = 0; coefficient < 4; ++coefficient) {
            coefficient_strain[strain_base + coefficient] =
                clamp_coefficient(strain_x[coefficient]);
            coefficient_strain[strain_base + 4 + coefficient] =
                clamp_coefficient(strain_y[coefficient]);
            coefficient_strain[strain_base + 8 + coefficient] =
                clamp_coefficient(strain_z[coefficient]);
        }
        for (int coefficient = 0; coefficient < 2; ++coefficient) {
            coefficient_strain[strain_base + 12 + coefficient] =
                clamp_coefficient(strain_direction_x[coefficient]);
            coefficient_strain[strain_base + 14 + coefficient] =
                clamp_coefficient(strain_direction_y[coefficient]);
            coefficient_strain[strain_base + 16 + coefficient] =
                clamp_coefficient(strain_direction_z[coefficient]);
        }
    }
}

}  // namespace gf
