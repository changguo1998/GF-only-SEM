// tests/test_sls.cpp — SLS attenuation constants and helpers
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "gf/attenuation.hpp"

TEST_CASE("SLS compile-time constants", "[sls][constants]") {
    REQUIRE(SLS::N_SLS == 3);
    REQUIRE(SLS::NDIM == 3);
    REQUIRE(SLS::VOIGT_COMPONENTS == 6);
    REQUIRE(SLS::MEMORY_PER_NODE == 18);
    REQUIRE(SLS::TAU_PER_NODE == 9);
}

TEST_CASE("SLS voigt_index maps symmetrically", "[sls][voigt]") {
    // Diagonal
    REQUIRE(SLS::voigt_index(0, 0) == 0);
    REQUIRE(SLS::voigt_index(1, 1) == 1);
    REQUIRE(SLS::voigt_index(2, 2) == 2);
    // Off-diagonal (symmetric)
    REQUIRE(SLS::voigt_index(0, 1) == 3);
    REQUIRE(SLS::voigt_index(1, 0) == 3);
    REQUIRE(SLS::voigt_index(0, 2) == 4);
    REQUIRE(SLS::voigt_index(2, 0) == 4);
    REQUIRE(SLS::voigt_index(1, 2) == 5);
    REQUIRE(SLS::voigt_index(2, 1) == 5);
}

TEST_CASE("SLS offset functions compute correct flat indices", "[sls][offsets]") {
    constexpr size_t node = 5;
    constexpr int mechanism = 2;
    constexpr int voigt = 4;

    // sls_memory_offset
    {
        size_t expected = node * SLS::MEMORY_PER_NODE +
                          static_cast<size_t>(mechanism) * SLS::VOIGT_COMPONENTS +
                          static_cast<size_t>(voigt);
        REQUIRE(SLS::sls_memory_offset(node, mechanism, voigt) == expected);
    }

    // tau_offset
    {
        size_t expected = node * SLS::N_SLS + static_cast<size_t>(mechanism);
        REQUIRE(SLS::tau_offset(node, mechanism) == expected);
        REQUIRE(SLS::coef_offset(node, mechanism) == expected);
    }

    // strain_offset
    {
        size_t expected = node * SLS::VOIGT_COMPONENTS + static_cast<size_t>(voigt);
        REQUIRE(SLS::strain_offset(node, voigt) == expected);
    }
}

TEST_CASE("SLS coefficient precomputation", "[sls][coefficients]") {
    constexpr int n_node = 10;
    double solver_dt = 0.001;
    double tau_s = 0.01;
    double tau_e = 0.015;

    // Allocate input arrays
    std::vector<double> tau_sigma(n_node * SLS::N_SLS);
    std::vector<double> tau_epsilon(n_node * SLS::N_SLS);
    std::vector<double> decay(n_node * SLS::N_SLS);
    std::vector<double> forcing_mu(n_node * SLS::N_SLS * SLS::FORCING_WEIGHTS);
    std::vector<double> forcing_kappa(n_node * SLS::N_SLS * SLS::FORCING_WEIGHTS);

    for (int node = 0; node < n_node; ++node) {
        for (int l = 0; l < SLS::N_SLS; ++l) {
            tau_sigma[SLS::tau_offset(node, l)] = tau_s;
            tau_epsilon[SLS::tau_offset(node, l)] = tau_e;
        }
    }

    SLS::precompute_sls_coefficients(tau_sigma.data(), tau_epsilon.data(), tau_epsilon.data(),
                                     n_node, solver_dt, decay.data(), forcing_mu.data(),
                                     forcing_kappa.data());

    double expected_a = std::exp(-solver_dt / tau_s);
    const double step_ratio = solver_dt / tau_s;
    const double previous_time_weight = (1.0 - expected_a) / step_ratio - expected_a;
    const double current_time_weight = 1.0 - (1.0 - expected_a) / step_ratio;
    const double normalized_weight = (tau_e / tau_s - 1.0) / (SLS::N_SLS * tau_e / tau_s);

    for (int node = 0; node < n_node; ++node) {
        for (int l = 0; l < SLS::N_SLS; ++l) {
            double a = decay[SLS::coef_offset(node, l)];
            double previous = forcing_mu[SLS::forcing_offset(node, l, 0)];
            double current = forcing_mu[SLS::forcing_offset(node, l, 1)];
            REQUIRE_THAT(a, Catch::Matchers::WithinRel(expected_a, 1e-12));
            REQUIRE_THAT(previous, Catch::Matchers::WithinRel(
                                       normalized_weight * previous_time_weight, 1e-12));
            REQUIRE_THAT(current, Catch::Matchers::WithinRel(
                                      normalized_weight * current_time_weight, 1e-12));
            REQUIRE_THAT(forcing_kappa[SLS::forcing_offset(node, l, 0)],
                         Catch::Matchers::WithinRel(previous, 1e-12));
            REQUIRE_THAT(forcing_kappa[SLS::forcing_offset(node, l, 1)],
                         Catch::Matchers::WithinRel(current, 1e-12));
        }
    }
}

TEST_CASE("SLS no-attenuation limit (tau_e == tau_s)", "[sls][limit]") {
    double solver_dt = 0.001;
    double tau_s = 0.01;
    double tau_e = 0.01;  // equal → no attenuation

    std::vector<double> tau_sigma(1 * SLS::N_SLS, tau_s);
    std::vector<double> tau_epsilon(1 * SLS::N_SLS, tau_e);
    std::vector<double> decay(1 * SLS::N_SLS);
    std::vector<double> forcing_mu(1 * SLS::N_SLS * SLS::FORCING_WEIGHTS);
    std::vector<double> forcing_kappa(1 * SLS::N_SLS * SLS::FORCING_WEIGHTS);

    SLS::precompute_sls_coefficients(tau_sigma.data(), tau_epsilon.data(), tau_epsilon.data(), 1,
                                     solver_dt, decay.data(), forcing_mu.data(),
                                     forcing_kappa.data());

    for (int l = 0; l < SLS::N_SLS; ++l) {
        REQUIRE_THAT(forcing_mu[SLS::forcing_offset(0, l, 0)],
                     Catch::Matchers::WithinAbs(0.0, 1e-15));
        REQUIRE_THAT(forcing_mu[SLS::forcing_offset(0, l, 1)],
                     Catch::Matchers::WithinAbs(0.0, 1e-15));
    }
}

TEST_CASE("SLS coefficient bounds", "[sls][bounds]") {
    double solver_dt = 0.001;
    double tau_s = 0.01;
    double tau_e = 0.02;

    std::vector<double> tau_sigma(1 * SLS::N_SLS, tau_s);
    std::vector<double> tau_epsilon(1 * SLS::N_SLS, tau_e);
    std::vector<double> decay(1 * SLS::N_SLS);
    std::vector<double> forcing_mu(1 * SLS::N_SLS * SLS::FORCING_WEIGHTS);
    std::vector<double> forcing_kappa(1 * SLS::N_SLS * SLS::FORCING_WEIGHTS);

    SLS::precompute_sls_coefficients(tau_sigma.data(), tau_epsilon.data(), tau_epsilon.data(), 1,
                                     solver_dt, decay.data(), forcing_mu.data(),
                                     forcing_kappa.data());

    for (int l = 0; l < SLS::N_SLS; ++l) {
        double a = decay[SLS::coef_offset(0, l)];
        double previous = forcing_mu[SLS::forcing_offset(0, l, 0)];
        double current = forcing_mu[SLS::forcing_offset(0, l, 1)];
        // a = exp(-dt/tau_s) ∈ (0, 1)
        REQUIRE(a > 0.0);
        REQUIRE(a < 1.0);
        REQUIRE(previous > 0.0);
        REQUIRE(previous < 1.0);
        REQUIRE(current > 0.0);
        REQUIRE(current < 1.0);
    }
}
