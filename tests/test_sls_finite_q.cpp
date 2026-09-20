// tests/test_sls_finite_q.cpp — finite-Q constitutive response of the viscoelastic kernel

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <vector>

#include "gf/attenuation.hpp"
#include "gf/element.hpp"
#include "gf/gll.hpp"

namespace {

struct UnitCubeElement {
    int ngll;
    int n_node;
    std::vector<double> coordinates;
    std::vector<double> inverse_jacobian;
    std::vector<double> jacobian;
    std::vector<double> lambda;
    std::vector<double> mu;
    std::vector<double> derivative_matrix;
    std::vector<double> weights;

    explicit UnitCubeElement(int polynomial_order)
        : ngll(polynomial_order + 1), n_node(ngll * ngll * ngll) {
        const std::vector<double> nodes = gf::gll_nodes(polynomial_order);
        weights = gf::gll_weights(polynomial_order, nodes);
        derivative_matrix = gf::gll_derivative_matrix(polynomial_order, nodes);

        coordinates.resize(n_node * 3);
        inverse_jacobian.assign(n_node * 9, 0.0);
        jacobian.assign(n_node, 0.125);
        lambda.assign(n_node, 1.125e10);
        mu.assign(n_node, 5.625e9);

        for (int i = 0; i < ngll; ++i) {
            for (int j = 0; j < ngll; ++j) {
                for (int k = 0; k < ngll; ++k) {
                    const int node = (i * ngll + j) * ngll + k;
                    coordinates[node * 3] = 0.5 * (nodes[i] + 1.0);
                    coordinates[node * 3 + 1] = 0.5 * (nodes[j] + 1.0);
                    coordinates[node * 3 + 2] = 0.5 * (nodes[k] + 1.0);
                    inverse_jacobian[node * 9] = 2.0;
                    inverse_jacobian[node * 9 + 4] = 2.0;
                    inverse_jacobian[node * 9 + 8] = 2.0;
                }
            }
        }
    }
};

double projection(const std::vector<double>& values, const std::vector<double>& basis) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t index = 0; index < values.size(); ++index) {
        numerator += values[index] * basis[index];
        denominator += basis[index] * basis[index];
    }
    return numerator / denominator;
}

enum class StrainMode { Shear, Volumetric };

constexpr std::array<double, SLS::N_SLS> TAU_SIGMA_VALUES_S = {
    0.186873539567019, 0.02491998701168405, 0.003323133676931235};

std::complex<double> harmonic_modulus_ratio(
    StrainMode mode, double solver_dt, double frequency_hz,
    const std::array<double, SLS::N_SLS>& tau_epsilon_values_s) {
    constexpr int polynomial_order = 3;
    constexpr int n_cycles = 20;
    constexpr int discarded_cycles = 5;
    const double angular_frequency = 2.0 * std::acos(-1.0) * frequency_hz;
    const int steps_per_cycle = static_cast<int>(std::lround(1.0 / (frequency_hz * solver_dt)));

    UnitCubeElement element(polynomial_order);
    std::vector<double> unit_displacement(element.n_node * 3, 0.0);
    for (int node = 0; node < element.n_node; ++node) {
        if (mode == StrainMode::Shear) {
            unit_displacement[node * 3 + 1] = element.coordinates[node * 3];
        } else {
            for (int component = 0; component < 3; ++component) {
                unit_displacement[node * 3 + component] =
                    element.coordinates[node * 3 + component];
            }
        }
    }

    std::vector<double> unit_residual(element.n_node * 3, 0.0);
    gf::compute_element_residual(1, element.inverse_jacobian.data(), element.jacobian.data(),
                                 element.lambda.data(), element.mu.data(),
                                 element.derivative_matrix.data(), element.weights.data(),
                                 element.ngll, unit_displacement.data(), unit_residual.data());

    std::vector<double> tau_sigma(element.n_node * SLS::N_SLS);
    std::vector<double> tau_epsilon_mu(element.n_node * SLS::N_SLS);
    std::vector<double> tau_epsilon_kappa(element.n_node * SLS::N_SLS);
    for (int node = 0; node < element.n_node; ++node) {
        for (int mechanism = 0; mechanism < SLS::N_SLS; ++mechanism) {
            const size_t offset = SLS::tau_offset(node, mechanism);
            tau_sigma[offset] = TAU_SIGMA_VALUES_S[mechanism];
            tau_epsilon_mu[offset] = mode == StrainMode::Shear ? tau_epsilon_values_s[mechanism]
                                                               : TAU_SIGMA_VALUES_S[mechanism];
            tau_epsilon_kappa[offset] = mode == StrainMode::Volumetric
                                            ? tau_epsilon_values_s[mechanism]
                                            : TAU_SIGMA_VALUES_S[mechanism];
        }
    }

    std::vector<double> decay(element.n_node * SLS::N_SLS);
    std::vector<double> forcing_mu(element.n_node * SLS::N_SLS * SLS::FORCING_WEIGHTS);
    std::vector<double> forcing_kappa(element.n_node * SLS::N_SLS * SLS::FORCING_WEIGHTS);
    SLS::precompute_sls_coefficients(tau_sigma.data(), tau_epsilon_mu.data(),
                                     tau_epsilon_kappa.data(), element.n_node, solver_dt,
                                     decay.data(), forcing_mu.data(), forcing_kappa.data());

    std::vector<double> displacement(element.n_node * 3, 0.0);
    std::vector<double> residual(element.n_node * 3, 0.0);
    std::vector<double> memory(element.n_node * SLS::MEMORY_PER_NODE, 0.0);
    std::vector<double> previous_strain(element.n_node * SLS::VOIGT_COMPONENTS, 0.0);
    std::complex<double> displacement_spectrum = 0.0;
    std::complex<double> residual_spectrum = 0.0;

    for (int step = 0; step < n_cycles * steps_per_cycle; ++step) {
        const double time_s = step * solver_dt;
        const double strain_amplitude = std::sin(angular_frequency * time_s);
        for (size_t index = 0; index < displacement.size(); ++index) {
            displacement[index] = strain_amplitude * unit_displacement[index];
        }
        std::fill(residual.begin(), residual.end(), 0.0);
        gf::compute_element_residual(1, element.inverse_jacobian.data(), element.jacobian.data(),
                                     element.lambda.data(), element.mu.data(),
                                     element.derivative_matrix.data(), element.weights.data(),
                                     element.ngll, displacement.data(), residual.data(), nullptr,
                                     nullptr, nullptr, memory.data(), previous_strain.data(),
                                     decay.data(), forcing_mu.data(), forcing_kappa.data(), true);

        if (step >= discarded_cycles * steps_per_cycle) {
            const std::complex<double> fourier_weight =
                std::exp(std::complex<double>(0.0, -angular_frequency * time_s));
            displacement_spectrum += strain_amplitude * fourier_weight;
            residual_spectrum += projection(residual, unit_residual) * fourier_weight;
        }
    }

    return residual_spectrum / displacement_spectrum;
}

void require_expected_modulus(const std::complex<double>& measured_modulus_ratio,
                              const std::complex<double>& expected_modulus_ratio) {
    const double measured_effective_q =
        std::abs(measured_modulus_ratio.real() / measured_modulus_ratio.imag());
    const double expected_effective_q =
        std::abs(expected_modulus_ratio.real() / expected_modulus_ratio.imag());

    INFO("measured modulus ratio = " << measured_modulus_ratio);
    INFO("expected modulus ratio = " << expected_modulus_ratio);
    INFO("measured effective Q = " << measured_effective_q);
    INFO("expected effective Q = " << expected_effective_q);
    REQUIRE(std::abs(measured_modulus_ratio - expected_modulus_ratio) < 0.01);
}

}  // namespace

TEST_CASE("Finite-Q shear response matches SPECFEM constitutive law", "[sls][finite-q]") {
    constexpr std::array<double, SLS::N_SLS> tau_epsilon_mu_values_s = {
        0.233016592750913, 0.02994444382282767, 0.004283862487455020};
    constexpr std::complex<double> expected_modulus_ratio(0.9260452145900969,
                                                          0.045591241967354604);

    const std::complex<double> measured_modulus_ratio =
        harmonic_modulus_ratio(StrainMode::Shear, 0.0005, 18.0, tau_epsilon_mu_values_s);
    require_expected_modulus(measured_modulus_ratio, expected_modulus_ratio);
}

TEST_CASE("Finite-Q bulk response matches SPECFEM constitutive law", "[sls][finite-q]") {
    constexpr std::array<double, SLS::N_SLS> tau_epsilon_kappa_values_s = {
        0.281966668348107, 0.03607809663879578, 0.005638875613224546};
    constexpr std::complex<double> expected_modulus_ratio(0.8577820965570316, 0.08480056515766467);

    const std::complex<double> measured_modulus_ratio =
        harmonic_modulus_ratio(StrainMode::Volumetric, 0.0005, 18.0, tau_epsilon_kappa_values_s);
    require_expected_modulus(measured_modulus_ratio, expected_modulus_ratio);
}

TEST_CASE("Finite-Q response is stable under timestep refinement", "[sls][finite-q]") {
    constexpr std::array<double, SLS::N_SLS> tau_epsilon_mu_values_s = {
        0.233016592750913, 0.02994444382282767, 0.004283862487455020};
    constexpr std::complex<double> expected_modulus_ratio(0.9260452145900969,
                                                          0.045591241967354604);

    const std::complex<double> coarse =
        harmonic_modulus_ratio(StrainMode::Shear, 0.0005, 18.0, tau_epsilon_mu_values_s);
    const std::complex<double> refined =
        harmonic_modulus_ratio(StrainMode::Shear, 0.00025, 18.0, tau_epsilon_mu_values_s);

    INFO("coarse modulus ratio = " << coarse);
    INFO("refined modulus ratio = " << refined);
    REQUIRE(std::abs(refined - coarse) < 0.003);
    REQUIRE(std::abs(refined - expected_modulus_ratio) < 0.005);
}
