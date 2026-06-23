#pragma once

#include <cmath>
#include <vector>
#include <stdexcept>
#include "gf/types.hpp"

namespace gf {

namespace detail {

// Legendre polynomial P_N(x) via Bonnet's recurrence
inline double legendre_p(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    double p0 = 1.0, p1 = x, pn = 0.0;
    for (int k = 1; k < n; ++k) {
        pn = ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0);
        p0 = p1;
        p1 = pn;
    }
    return pn;
}

// Derivative of Legendre polynomial P'_N(x)
inline double legendre_p_prime(int n, double x) {
    // P'_n(x) = n*(x*P_n(x) - P_{n-1}(x)) / (x^2 - 1)
    if (n == 0) return 0.0;
    if (std::abs(x - 1.0) < 1e-14) return 0.5 * n * (n + 1.0);
    if (std::abs(x + 1.0) < 1e-14) return (n % 2 == 0 ? -1.0 : 1.0) * 0.5 * n * (n + 1.0);
    double pn = legendre_p(n, x);
    double pnm1 = legendre_p(n - 1, x);
    return n * (x * pn - pnm1) / (x * x - 1.0);
}

} // namespace detail

// -----------------------------------------------------------------------
// Compute GLL nodes (roots of P'_N(x)) via Newton iteration
// Endpoints always -1 and +1; interior nodes from cos-based initial guess.
// -----------------------------------------------------------------------
inline std::vector<double> gll_nodes(int N) {
    if (N < 1) throw std::invalid_argument("GLL order N must be >= 1");
    int ngl = N + 1;
    std::vector<double> nodes(ngl);
    nodes[0] = -1.0;
    nodes[N] = 1.0;

    const double eps = 1.0e-15;
    const int max_iter = 100;

    for (int i = 1; i < N; ++i) {
        // Initial guess: cos((N-i)*pi/N) — good approximation for GLL nodes
        double xi = -std::cos(M_PI * i / N);

        // Newton iteration to find root of P'_N(x) = 0
        for (int iter = 0; iter < max_iter; ++iter) {
            double f  = detail::legendre_p_prime(N, xi);
            // Legendre ODE: (1-x²)P''_N - 2xP'_N + N(N+1)P_N = 0
            // so P''_N = (2xP'_N - N(N+1)P_N)/(1-x²).
            double df = (2.0 * xi * f
                         - N * (N + 1.0) * detail::legendre_p(N, xi))
                        / (1.0 - xi * xi);
            if (std::abs(df) < 1.0e-14) {
                throw std::runtime_error("GLL Newton iteration encountered near-zero derivative");
            }
            double dx = -f / df;
            xi += dx;
            if (std::abs(dx) < eps * (1.0 + std::abs(xi))) break;
        }
        nodes[i] = xi;
    }
    return nodes;
}

// -----------------------------------------------------------------------
// GLL weights: 2/(N*(N+1)) for endpoints,
//              2/(N*(N+1)*P_N(xi)^2) for interior nodes
// -----------------------------------------------------------------------
inline std::vector<double> gll_weights(int N, const std::vector<double>& nodes) {
    if (N < 1) throw std::invalid_argument("GLL order N must be >= 1");
    int ngl = N + 1;
    std::vector<double> w(ngl);
    for (int i = 0; i < ngl; ++i) {
        double pn = detail::legendre_p(N, nodes[i]);
        w[i] = 2.0 / (N * (N + 1.0) * pn * pn);
    }
    return w;
}

// -----------------------------------------------------------------------
// Derivative matrix D[i][j] = d(ell_j)/d(xi) at node xi_i
// D flattened as (N+1)x(N+1) row-major
// -----------------------------------------------------------------------
inline std::vector<double> gll_derivative_matrix(int N, const std::vector<double>& nodes) {
    if (N < 1) throw std::invalid_argument("GLL order N must be >= 1");
    int ngl = N + 1;
    std::vector<double> D(ngl * ngl, 0.0);
    // Precompute Legendre polynomial values at each node
    std::vector<double> PN(ngl);
    for (int i = 0; i < ngl; ++i) {
        PN[i] = detail::legendre_p(N, nodes[i]);
    }
    for (int i = 0; i < ngl; ++i) {
        for (int j = 0; j < ngl; ++j) {
            if (i == j) {
                // Diagonal: D_ii = -N(N+1)/4 for endpoint, 0 for interior
                if (i == 0) {
                    D[i * ngl + j] = -0.25 * N * (N + 1.0);
                } else if (i == N) {
                    D[i * ngl + j] =  0.25 * N * (N + 1.0);
                } else {
                    D[i * ngl + j] = 0.0;
                }
            } else {
                // Off-diagonal: D_ij = P_N(xi_i) / (P_N(xi_j) * (xi_i - xi_j))
                D[i * ngl + j] = PN[i] / (PN[j] * (nodes[i] - nodes[j]));
            }
        }
    }
    return D;
}

// -----------------------------------------------------------------------
// Evaluate all N+1 Lagrange basis polynomials at arbitrary xi in [-1,1]
// -----------------------------------------------------------------------
inline std::vector<double> lagrange_basis(double xi, const std::vector<double>& nodes) {
    int ngl = static_cast<int>(nodes.size());
    std::vector<double> ell(ngl, 1.0);
    for (int j = 0; j < ngl; ++j) {
        for (int k = 0; k < ngl; ++k) {
            if (k == j) continue;
            ell[j] *= (xi - nodes[k]) / (nodes[j] - nodes[k]);
        }
    }
    return ell;
}

// -----------------------------------------------------------------------
// Compute full GLLQuad from order N
// -----------------------------------------------------------------------
inline GLLQuad make_gll_quad(int N) {
    GLLQuad quad;
    quad.N = N;
    quad.points     = gll_nodes(N);
    quad.weights    = gll_weights(N, quad.points);
    quad.derivatives = gll_derivative_matrix(N, quad.points);
    return quad;
}

} // namespace gf