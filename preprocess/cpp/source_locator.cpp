/// source_locator.cpp — find source element(s) and compute Lagrange weights
///
/// Port of preprocess/source_locator.py.
/// Uses Eigen for small-matrix linear algebra (already a project dependency).

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gf_config.h"

namespace gf {
namespace {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Arr8x3 = Eigen::Matrix<double, 8, 3>;

// ── GLL quadrature points (1-D) ────────────────────────────────────────────

/// Compute Gauss-Lobatto-Legendre nodes in [-1, 1].
/// Uses Newton iteration on the GLL derivative polynomial.
std::vector<double> gll_points(int N) {
    // Gauss-Lobatto-Legendre nodes in [-1, 1].
    // Uses Newton iteration to find the zeros of P_N'(x) (interior nodes).
    int n = N + 1;
    std::vector<double> xi(n);
    if (n == 1) {
        xi[0] = 0.0;
        return xi;
    }
    if (n == 2) {
        xi[0] = -1.0;
        xi[1] = 1.0;
        return xi;
    }
    xi[0] = -1.0;
    xi[n - 1] = 1.0;

    // Chebyshev initial guess for interior points
    for (int i = 1; i < n - 1; ++i) {
        xi[i] = -std::cos(M_PI * i / N);
    }

    // Newton iteration: find zeros of P_N'(x)
    // P_N'(x) = N * (x*P_N(x) - P_{N-1}(x)) / (x^2 - 1)
    // P_N''(x) = (2*x*P_N'(x) - N*(N+1)*P_N(x)) / (1 - x^2)
    for (int iter = 0; iter < 30; ++iter) {
        double max_delta = 0.0;
        for (int i = 1; i < n - 1; ++i) {
            double x = xi[i];

            // Evaluate P_{N-1}(x) and P_N(x) via recurrence
            double pn = 1.0;  // P_0
            double pn1 = x;   // P_1
            for (int k = 2; k <= N; ++k) {
                double pk = ((2.0 * k - 1.0) * x * pn1 - (k - 1.0) * pn) / k;
                pn = pn1;
                pn1 = pk;
            }
            // pn = P_{N-1}(x), pn1 = P_N(x)

            double x2 = x * x;
            double pprime = N * (x * pn1 - pn) / (x2 - 1.0);
            double p2prime = (2.0 * x * pprime - N * (N + 1.0) * pn1) / (1.0 - x2);
            double delta = pprime / p2prime;

            xi[i] -= delta;
            max_delta = std::max(max_delta, std::abs(delta));
        }
        if (max_delta < 1e-15)
            break;
    }
    return xi;
}

// ── Hex reference corners (GMSH ordering) ──────────────────────────────────

const Arr8x3& hex_corners() {
    static const Arr8x3 c = (Arr8x3() << -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1,
                             -1, 1, 1, 1, 1, -1, 1, 1)
                                .finished();
    return c;
}

// ── Newton iteration: find (xi,eta,zeta) for a physical point ─────────────

Vec3 newton_find_xi(const Vec3& target, const Arr8x3& corners, int max_iter = 50,
                    double tol = 1e-12) {
    Vec3 xi(0.0, 0.0, 0.0);  // start at element center

    for (int iter = 0; iter < max_iter; ++iter) {
        // Linear shape functions and derivatives at xi
        Eigen::Matrix<double, 8, 1> N;
        Eigen::Matrix<double, 8, 3> dN;
        const auto& c = hex_corners();
        for (int a = 0; a < 8; ++a) {
            double ca = c(a, 0), cb = c(a, 1), cc = c(a, 2);
            N(a) = 0.125 * (1 + ca * xi[0]) * (1 + cb * xi[1]) * (1 + cc * xi[2]);
            dN(a, 0) = 0.125 * ca * (1 + cb * xi[1]) * (1 + cc * xi[2]);
            dN(a, 1) = 0.125 * (1 + ca * xi[0]) * cb * (1 + cc * xi[2]);
            dN(a, 2) = 0.125 * (1 + ca * xi[0]) * (1 + cb * xi[1]) * cc;
        }

        Vec3 x_current = corners.transpose() * N;
        Mat3 J = corners.transpose() * dN;  // dx/dxi

        Vec3 residual = target - x_current;
        Vec3 dxi = J.colPivHouseholderQr().solve(residual);
        xi += dxi;
        if (dxi.norm() < tol)
            break;
    }

    return xi;
}

/// Check if xi is inside [-1,1]^3 within margin
bool inside_element(const Vec3& xi, double margin = 1e-8) {
    return xi[0] >= -(1.0 + margin) && xi[0] <= 1.0 + margin && xi[1] >= -(1.0 + margin) &&
           xi[1] <= 1.0 + margin && xi[2] >= -(1.0 + margin) && xi[2] <= 1.0 + margin;
}

// ── Lagrange basis at a point ──────────────────────────────────────────────

void lagrange_basis_at(double xi_eval, const std::vector<double>& nodes, std::vector<double>& w) {
    int n = static_cast<int>(nodes.size());
    w.assign(n, 1.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                w[i] *= (xi_eval - nodes[j]) / (nodes[i] - nodes[j]);
            }
        }
    }
}

/// Compute 3-D Lagrange interpolation weights at (xi, eta, zeta)
void compute_source_weights_3d(const Vec3& xi_vec, const std::vector<double>& gll_pts,
                               std::vector<double>& weights, int& ngll) {
    ngll = static_cast<int>(gll_pts.size());
    std::vector<double> lx, ly, lz;
    lagrange_basis_at(xi_vec[0], gll_pts, lx);
    lagrange_basis_at(xi_vec[1], gll_pts, ly);
    lagrange_basis_at(xi_vec[2], gll_pts, lz);

    weights.resize(ngll * ngll * ngll);
    for (int i = 0; i < ngll; ++i)
        for (int j = 0; j < ngll; ++j)
            for (int k = 0; k < ngll; ++k)
                weights[(i * ngll + j) * ngll + k] = lx[i] * ly[j] * lz[k];
}

}  // anonymous namespace

// ── Public API ──────────────────────────────────────────────────────────────

SourceResult locate_source(const Config& cfg, const double* gll_coords_flat, int n_cell, int ngll,
                           const int64_t* cell_to_surface, int n_surface,
                           const int64_t* boundary_tag, const int* is_pml) {
    SourceResult result;

    // User-specified source coordinates in meters.
    Vec3 source_pt(cfg.source_x_m, cfg.source_y_m, cfg.source_z_m >= 0 ? cfg.source_z_m : 0.0);
    bool is_buried = cfg.source_z_m >= 0;

    // Determine candidate elements
    std::vector<int> candidates;
    if (is_buried && is_pml) {
        // Buried mode: AABB test on non-PML elements
        for (int e = 0; e < n_cell; ++e) {
            if (is_pml[e])
                continue;
            // Compute AABB from GLL coords
            double xmin = 1e30, xmax = -1e30, ymin = 1e30, ymax = -1e30, zmin = 1e30, zmax = -1e30;
            int stride = ngll * ngll * ngll * 3;
            const double* ec = gll_coords_flat + e * stride;
            for (int i = 0; i < ngll * ngll * ngll; ++i) {
                xmin = std::min(xmin, ec[i * 3 + 0]);
                xmax = std::max(xmax, ec[i * 3 + 0]);
                ymin = std::min(ymin, ec[i * 3 + 1]);
                ymax = std::max(ymax, ec[i * 3 + 1]);
                zmin = std::min(zmin, ec[i * 3 + 2]);
                zmax = std::max(zmax, ec[i * 3 + 2]);
            }
            double margin = 1e-10;
            if (source_pt[0] >= xmin - margin && source_pt[0] <= xmax + margin &&
                source_pt[1] >= ymin - margin && source_pt[1] <= ymax + margin &&
                source_pt[2] >= zmin - margin && source_pt[2] <= zmax + margin)
                candidates.push_back(e);
        }
    } else {
        // Surface mode: search free-surface elements
        for (int e = 0; e < n_cell; ++e) {
            for (int fi = 0; fi < 6; ++fi) {
                int sid = std::abs(static_cast<int>(cell_to_surface[e * 6 + fi])) - 1;
                if (sid >= 0 && sid < n_surface && boundary_tag[sid] == 1) {
                    candidates.push_back(e);
                    break;
                }
            }
        }
    }
    if (candidates.empty()) {
        fprintf(stderr, "ERROR: source not found in any candidate element\n");
        std::exit(1);
    }

    // GLL points
    auto gll_pts = gll_points(ngll - 1);

    // Newton iteration for each candidate
    int stride = ngll * ngll * ngll * 3;
    for (int e : candidates) {
        // Extract corners from GLL coords
        Arr8x3 corners;
        int idx = ngll - 1;
        const double* ec = gll_coords_flat + e * stride;
        auto get_node = [&](int i, int j, int k) -> Vec3 {
            int off = (i * ngll * ngll + j * ngll + k) * 3;
            return Vec3(ec[off], ec[off + 1], ec[off + 2]);
        };
        corners.row(0) = get_node(0, 0, 0);
        corners.row(1) = get_node(idx, 0, 0);
        corners.row(2) = get_node(idx, idx, 0);
        corners.row(3) = get_node(0, idx, 0);
        corners.row(4) = get_node(0, 0, idx);
        corners.row(5) = get_node(idx, 0, idx);
        corners.row(6) = get_node(idx, idx, idx);
        corners.row(7) = get_node(0, idx, idx);

        Vec3 xi = newton_find_xi(source_pt, corners);
        if (!inside_element(xi))
            continue;

        int w_ngll;
        std::vector<double> w;
        compute_source_weights_3d(xi, gll_pts, w, w_ngll);

        double w_sum = 0.0;
        for (double v : w)
            w_sum += v;
        if (w_sum < 1e-12)
            continue;

        // Normalize
        for (double& v : w)
            v /= w_sum;

        result.cell_ids.push_back(e);
        result.xi.push_back(xi[0]);
        result.eta.push_back(xi[1]);
        result.zeta.push_back(xi[2]);
        result.weights.push_back(w);
    }

    if (result.cell_ids.empty()) {
        fprintf(stderr, "ERROR: source at (%g,%g,%g) not contained in any element\n", source_pt[0],
                source_pt[1], source_pt[2]);
        std::exit(1);
    }

    result.n_src_cell = static_cast<int>(result.cell_ids.size());
    return result;
}

}  // namespace gf