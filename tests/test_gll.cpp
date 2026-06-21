// tests/test_gll.cpp — GLL quadrature unit tests
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "gf/gll.hpp"
#include <cmath>

using namespace gf;
using Catch::Matchers::WithinAbs;

TEST_CASE("GLL nodes: N=1 returns endpoints", "[gll]") {
    auto nodes = gll_nodes(1);
    REQUIRE(nodes.size() == 2);
    REQUIRE_THAT(nodes[0], WithinAbs(-1.0, 1e-15));
    REQUIRE_THAT(nodes[1], WithinAbs( 1.0, 1e-15));
}

TEST_CASE("GLL nodes: N=2 includes zero", "[gll]") {
    auto nodes = gll_nodes(2);
    REQUIRE(nodes.size() == 3);
    REQUIRE_THAT(nodes[0], WithinAbs(-1.0, 1e-15));
    REQUIRE_THAT(nodes[1], WithinAbs( 0.0, 1e-15));
    REQUIRE_THAT(nodes[2], WithinAbs( 1.0, 1e-15));
}

TEST_CASE("GLL nodes: N=3 are symmetric", "[gll]") {
    auto nodes = gll_nodes(3);
    REQUIRE(nodes.size() == 4);
    // GLL nodes are symmetric about 0
    REQUIRE_THAT(nodes[0] + nodes[3], WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(nodes[1] + nodes[2], WithinAbs(0.0, 1e-15));
    // Monotonic
    REQUIRE(nodes[0] < nodes[1]);
    REQUIRE(nodes[1] < nodes[2]);
    REQUIRE(nodes[2] < nodes[3]);
}

TEST_CASE("GLL nodes: N=5 known values", "[gll]") {
    auto nodes = gll_nodes(5);
    REQUIRE(nodes.size() == 6);
    // Endpoints
    REQUIRE_THAT(nodes[0], WithinAbs(-1.0, 1e-15));
    REQUIRE_THAT(nodes[5], WithinAbs( 1.0, 1e-15));
    // Symmetry
    for (int i = 0; i < 3; ++i) {
        REQUIRE_THAT(nodes[i] + nodes[5 - i], WithinAbs(0.0, 1e-14));
    }
}

TEST_CASE("GLL weights: N=2 are correct", "[gll]") {
    auto nodes = gll_nodes(2);
    auto w = gll_weights(2, nodes);
    REQUIRE(w.size() == 3);
    // Weights for N=2: w[0]=w[2]=1/3, w[1]=4/3
    REQUIRE_THAT(w[0], WithinAbs(1.0/3.0, 1e-15));
    REQUIRE_THAT(w[1], WithinAbs(4.0/3.0, 1e-15));
    REQUIRE_THAT(w[2], WithinAbs(1.0/3.0, 1e-15));
}

TEST_CASE("GLL weights: sum to 2", "[gll]") {
    for (int N : {1, 2, 3, 4, 5}) {
        auto nodes = gll_nodes(N);
        auto w = gll_weights(N, nodes);
        double sum = 0.0;
        for (double wi : w) sum += wi;
        REQUIRE_THAT(sum, WithinAbs(2.0, 1e-14));
    }
}

TEST_CASE("GLL derivative matrix: sum of each row is zero", "[gll]") {
    // The derivative matrix D satisfies Σ_j D[i][j] = 0 for all i
    for (int N : {2, 3, 4}) {
        auto nodes = gll_nodes(N);
        auto D = gll_derivative_matrix(N, nodes);
        int ngl = N + 1;
        for (int i = 0; i < ngl; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < ngl; ++j) {
                row_sum += D[i * ngl + j];
            }
            REQUIRE_THAT(row_sum, WithinAbs(0.0, 1e-14));
        }
    }
}

TEST_CASE("GLL derivative matrix: differentiates polynomial exactly", "[gll]") {
    // For f(x) = x^2 at N=3, the derivative matrix should give exact derivative 2*x
    const int N = 3;
    auto nodes = gll_nodes(N);
    auto D = gll_derivative_matrix(N, nodes);
    int ngl = N + 1;

    for (int i = 0; i < ngl; ++i) {
        double xi = nodes[i];
        // df/dx at xi: sum over j of D[i][j] * f[j]
        double df_approx = 0.0;
        for (int j = 0; j < ngl; ++j) {
            df_approx += D[i * ngl + j] * (nodes[j] * nodes[j]);
        }
        double df_exact = 2.0 * xi;
        REQUIRE_THAT(df_approx, WithinAbs(df_exact, 1e-12));
    }
}

TEST_CASE("Lagrange basis: partition of unity", "[gll]") {
    // Sum of all Lagrange basis polynomials at any point is 1
    for (int N : {2, 3}) {
        auto nodes = gll_nodes(N);
        int ngl = N + 1;
        for (int k = 0; k < 10; ++k) {
            double xi = -1.0 + 2.0 * k / 9.0; // sample points in [-1, 1]
            auto ell = lagrange_basis(xi, nodes);
            double sum = 0.0;
            for (double l : ell) sum += l;
            REQUIRE_THAT(sum, WithinAbs(1.0, 1e-14));
        }
    }
}

TEST_CASE("Lagrange basis: interpolation property", "[gll]") {
    // ell_i(xi_j) = delta_{ij}
    for (int N : {2, 3, 4}) {
        auto nodes = gll_nodes(N);
        int ngl = N + 1;
        for (int i = 0; i < ngl; ++i) {
            auto ell = lagrange_basis(nodes[i], nodes);
            for (int j = 0; j < ngl; ++j) {
                if (i == j) {
                    REQUIRE_THAT(ell[j], WithinAbs(1.0, 1e-15));
                } else {
                    REQUIRE_THAT(ell[j], WithinAbs(0.0, 1e-15));
                }
            }
        }
    }
}

TEST_CASE("make_gll_quad: consistent GLLQuad", "[gll]") {
    auto q = make_gll_quad(3);
    REQUIRE(q.N == 3);
    REQUIRE(q.points.size() == 4);
    REQUIRE(q.weights.size() == 4);
    REQUIRE(q.derivatives.size() == 16);
    // Weights sum to 2
    double wsum = 0.0;
    for (double w : q.weights) wsum += w;
    REQUIRE_THAT(wsum, WithinAbs(2.0, 1e-14));
}