// tests/test_newmark.cpp — Newmark explicit predictor-corrector tests
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>
#include <cmath>
#include "gf/newmark.hpp"
#include "gf/types.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Newmark predictor constant acceleration", "[newmark]") {
    const int n_dof = 9;  // 3 nodes x 3 dof
    NewmarkParams params;
    params.dt = 0.001;
    params.beta = 0.0;
    params.gamma = 0.5;

    // Initial state: zero displacement, zero velocity, constant acceleration = 2.0
    std::vector<double> u(n_dof, 0.0);
    std::vector<double> v(n_dof, 0.0);
    std::vector<double> a(n_dof, 2.0);
    std::vector<double> u_tilde(n_dof, 0.0);
    std::vector<double> v_tilde(n_dof, 0.0);

    newmark_predictor(params, u, v, a, u_tilde, v_tilde);

    // For constant acceleration a=2:
    // u_tilde = u + dt*v + 0.5*dt^2*a = 0 + 0 + 0.5*0.001^2*2 = 1e-6
    // v_tilde = v + 0.5*dt*a = 0 + 0.5*0.001*2 = 0.001
    double dt = params.dt;
    double expected_u = 0.5 * dt * dt * 2.0;
    double expected_v = 0.5 * dt * 2.0;
    for (int i = 0; i < n_dof; ++i) {
        REQUIRE_THAT(u_tilde[i], WithinAbs(expected_u, 1e-15));
        REQUIRE_THAT(v_tilde[i], WithinAbs(expected_v, 1e-15));
    }
    // u and v unchanged
    for (int i = 0; i < n_dof; ++i) {
        REQUIRE_THAT(u[i], WithinAbs(0.0, 1e-15));
        REQUIRE_THAT(v[i], WithinAbs(0.0, 1e-15));
    }
}

TEST_CASE("Newmark corrector with lumped mass", "[newmark]") {
    const int n_dof = 6;  // 2 nodes x 3 dof
    NewmarkParams params;
    params.dt = 0.001;
    params.beta = 0.0;
    params.gamma = 0.5;

    // Mass at each node (lumped, 3 dof each)
    std::vector<double> mass(n_dof, 10.0);
    // Residual (force)
    std::vector<double> residual(n_dof, 2.0);
    // Predicted state
    std::vector<double> u(n_dof, 0.5);
    std::vector<double> v(n_dof, 0.1);
    std::vector<double> a(n_dof, 0.0);

    newmark_corrector(params, mass, residual, u, v, a);

    // a_new[i] = residual[i] / mass[i] = 2.0 / 10.0 = 0.2
    // v_new[i] = v_tilde[i] + 0.5*dt*a_new[i] (but v_tilde = v + 0.5*dt*a = v since a=0)
    // so v_new = 0.1 + 0.5*0.001*0.2 = 0.1 + 0.0001 = 0.1001
    double expected_a = 2.0 / 10.0;
    double expected_v = 0.1 + 0.5 * params.dt * expected_a;
    for (int i = 0; i < n_dof; ++i) {
        REQUIRE_THAT(a[i], WithinAbs(expected_a, 1e-15));
        REQUIRE_THAT(v[i], WithinAbs(expected_v, 1e-15));
        // u unchanged when beta=0
        REQUIRE_THAT(u[i], WithinAbs(0.5, 1e-15));
    }
}

TEST_CASE("Newmark predictor-corrector energy conservation", "[newmark]") {
    // Verifies that undamped free vibration conserves total energy
    // Spring-mass system: M*a + K*u = 0
    // Simple test: one DOF with mass=1, spring constant k, initial displacement u0
    const int n_dof = 1;
    NewmarkParams params;
    params.dt = 0.001;
    params.beta = 0.0;
    params.gamma = 0.5;

    std::vector<double> u = {1.0};   // initial displacement
    std::vector<double> v = {0.0};   // initial velocity
    std::vector<double> a = {0.0};
    std::vector<double> mass = {1.0}; // unit mass

    double k_spring = 10.0;  // spring stiffness

    // Run a few time steps and check that total mechanical energy is approx conserved
    int nsteps = 100;
    double initial_energy = 0.5 * (mass[0] * v[0] * v[0] + k_spring * u[0] * u[0]);

    for (int step = 0; step < nsteps; ++step) {
        std::vector<double> u_tilde(n_dof, 0.0);
        std::vector<double> v_tilde(n_dof, 0.0);

        // Predict
        newmark_predictor(params, u, v, a, u_tilde, v_tilde);

        // Compute residual: r = -K*u_tilde (elastic force = -k*u)
        std::vector<double> residual(n_dof, 0.0);
        residual[0] = -k_spring * u_tilde[0];

        // Correct
        newmark_corrector(params, mass, residual, u, v, a);
    }

    double final_energy = 0.5 * (mass[0] * v[0] * v[0] + k_spring * u[0] * u[0]);

    // Energy should be approximately conserved (symplectic integrator)
    // For explicit Newmark (central difference), energy grows slowly.
    // Expect energy ratio close to 1 for small dt.
    REQUIRE_THAT(final_energy, WithinRel(initial_energy, 0.01));
}