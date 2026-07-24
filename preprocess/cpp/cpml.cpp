/// cpml.cpp — C-PML (Convolutional PML) damping profile computation
///
/// Port of preprocess/pml_cpml.py.
/// Computes κ (stretch), d (damping), α (shift) profiles per GLL node
/// per direction for all PML elements.
///
/// Reference: Wang et al. (2006), SPECFEM3D pml_compute_memory_variables.f90

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

}  // anonymous namespace

// ── Public API ──────────────────────────────────────────────────────────────

void compute_cpml_profiles(const double* gll_coords_flat,  // [n_cell * ngll³ * 3]
                           int n_cell, int ngll,
                           const bool* is_pml,           // [n_cell]
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
    };
    Face faces[6] = {
        {0, domain_bounds[0], 0},  // xmin
        {0, domain_bounds[1], 1},  // xmax
        {1, domain_bounds[2], 2},  // ymin
        {1, domain_bounds[3], 3},  // ymax
        {2, domain_bounds[4], 4},  // zmin
        {2, domain_bounds[5], 5},  // zmax
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

            for (int node = 0; node < n_node; ++node) {
                // Distance from PML interior boundary, normalized to [0, 1]
                double coord = coords[node * 3 + face.axis];
                double dist = std::abs(coord - boundary) / width;
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

    // ── SPECFEM3D parameter separation for corner/edge PML regions ─────
    // Adjust alpha at nodes where multiple axes are active so that
    // partial-fraction denominators (α_x - α_y, α_x - β_z, etc.) do not
    // approach zero.  This follows pml_set_local_dampingcoeff.f90.
    //
    // Strategy: ensure α profiles for different axes are separated by at
    // least a minimum threshold.  When two α values at the same node are
    // too close, shift one of them slightly.
    double alpha_sep_threshold = M_PI * f0_hz * 0.1;  // 10% separaration

    for (int e = 0; e < n_cell; ++e) {
        if (!is_pml[e])
            continue;
        int region = pml_regions ? pml_regions[e] : 0;

        // Only apply separation in multi-axis regions
        if (region != CPML_XY_ONLY && region != CPML_XZ_ONLY && region != CPML_YZ_ONLY &&
            region != CPML_XYZ)
            continue;

        for (int node = 0; node < n_node; ++node) {
            int base = (e * n_node + node) * 3;
            double ax = alpha_store[base + 0];
            double ay = alpha_store[base + 1];
            double az = alpha_store[base + 2];

            // Pairwise separation: ensure each pair differs by at least threshold
            auto separate_pair = [&](double& a, double& b) {
                if (a > 0.0 && b > 0.0 && std::abs(a - b) < alpha_sep_threshold) {
                    // Shift the larger one up slightly
                    if (a >= b)
                        a = b + alpha_sep_threshold;
                    else
                        b = a + alpha_sep_threshold;
                }
            };
            separate_pair(ax, ay);
            separate_pair(ax, az);
            separate_pair(ay, az);

            alpha_store[base + 0] = ax;
            alpha_store[base + 1] = ay;
            alpha_store[base + 2] = az;
        }
    }
}

}  // namespace gf