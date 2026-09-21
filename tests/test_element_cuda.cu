/**
 * @file test_element_cuda.cu
 * @brief CUDA element residual tests — compare CUDA result against CPU reference.
 *
 * Builds only when GF_WITH_CUDA is enabled.
 * Each test generates random input, runs the CUDA backend and compares against
 * an inline CPU reference implementation.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "gf/backend.hpp"
#include "gf/cuda_step.hpp"
#include "gf/element.hpp"
#include "gf/gll.hpp"
#include "gf/kernel_helpers.hpp"
#include "gf/pml.hpp"
#include "gf/types.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

namespace {

/// Fill a vector with random uniform values in [-1, 1].
void random_fill(std::vector<double>& v, unsigned seed = 42) {
    std::srand(seed);
    for (auto& x : v) {
        x = 2.0 * static_cast<double>(std::rand()) / RAND_MAX - 1.0;
    }
}

/// Inline CPU reference for element residual (elastic, no SLS/PML).
///
/// Duplicates the elastic CPU kernel algorithm so that CUDA-only builds
/// (where libgf_cuda_nompi provides only the CUDA implementation of
/// compute_element_residual) can still cross-validate the result.
void reference_element_residual(int n_elem, const double* dxi_dx, const double* jacobian,
                                const double* lambda_, const double* mu_, const double* D,
                                const double* weights, int NGLL, const double* u, double* r) {
    const int n_node = NGLL * NGLL * NGLL;
    for (int elem = 0; elem < n_elem; ++elem) {
        const double* dd_all = dxi_dx + elem * n_node * 9;
        const double* elem_lambda = lambda_ + elem * n_node;
        const double* elem_mu = mu_ + elem * n_node;
        const double* elem_u = u + elem * n_node * 3;
        double* elem_r = r + elem * n_node * 3;

        for (int i = 0; i < NGLL; ++i) {
            for (int j = 0; j < NGLL; ++j) {
                for (int k = 0; k < NGLL; ++k) {
                    const int n = (i * NGLL + j) * NGLL + k;
                    const double lambda = elem_lambda[n];
                    const double mu = elem_mu[n];
                    if (mu <= 0.0)
                        continue;
                    const double* dd = &dd_all[9 * n];

                    double dudxi[3], dudeta[3], dudzeta[3];
                    compute_reference_gradient(i, j, k, NGLL, D, elem_u, dudxi, dudeta, dudzeta);

                    double du_dx[3][3];
                    transform_to_physical(dudxi, dudeta, dudzeta, dd, du_dx);

                    double eps[3][3];
                    compute_strain_tensor(du_dx, eps);
                    double eps_kk = eps[0][0] + eps[1][1] + eps[2][2];

                    double sigma[3][3];
                    for (int l = 0; l < 3; ++l) {
                        for (int m = 0; m < 3; ++m) {
                            sigma[l][m] = 2.0 * mu * eps[l][m];
                        }
                        sigma[l][l] += lambda * eps_kk;
                    }
                    scatter_residual(i, j, k, NGLL, sigma, dd, D, weights,
                                     jacobian[elem * n_node + n], elem_r);
                }
            }
        }
    }
}

/// Build a random element at polynomial order N.
struct RandomElement {
    int ngll;
    int n_node;
    std::vector<double> dxi_dx;
    std::vector<double> jacobian;
    std::vector<double> vp;
    std::vector<double> vs;
    std::vector<double> lambda_;
    std::vector<double> mu_;
    std::vector<double> density;
    std::vector<double> D;
    std::vector<double> w;
    std::vector<double> nodes;

    RandomElement(int N, unsigned seed = 123) : ngll(N + 1), n_node((N + 1) * (N + 1) * (N + 1)) {
        nodes = gll_nodes(N);
        w = gll_weights(N, nodes);
        D = gll_derivative_matrix(N, nodes);

        dxi_dx.resize(n_node * 9);
        jacobian.resize(n_node);
        vp.resize(n_node);
        vs.resize(n_node);
        density.resize(n_node);

        // Random dxi_dx with positive diagonal (physically plausible)
        random_fill(dxi_dx, seed);
        // Ensure diagonal dominance for invertibility
        for (int n = 0; n < n_node; ++n) {
            dxi_dx[n * 9 + 0] += 3.0;  // dξ/dx
            dxi_dx[n * 9 + 4] += 3.0;  // dη/dy
            dxi_dx[n * 9 + 8] += 3.0;  // dζ/dz
        }

        random_fill(jacobian, seed + 1);
        for (auto& j : jacobian)
            j = std::abs(j) + 0.01;  // ensure positive

        random_fill(vp, seed + 2);
        for (auto& v : vp)
            v = std::abs(v) + 1000.0;  // vp > 1000

        random_fill(vs, seed + 3);
        for (auto& v : vs)
            v = std::abs(v) + 500.0;  // vs > 500

        random_fill(density, seed + 4);
        for (auto& d_mag : density)
            d_mag = std::abs(d_mag) + 1000.0;  // density > 1000

        // Precompute elastic coefficients from Vp, Vs, density
        lambda_.resize(n_node);
        mu_.resize(n_node);
        for (int i = 0; i < n_node; ++i) {
            double vs2 = vs[i] * vs[i];
            double vp2 = vp[i] * vp[i];
            mu_[i] = density[i] * vs2;
            lambda_[i] = density[i] * (vp2 - 2.0 * vs2);
        }
    }
};

}  // anonymous namespace

TEST_CASE("CUDA element residual matches CPU reference — N=3 random", "[element][cuda]") {
    int N = 3;
    RandomElement elem(N, 42);

    std::vector<double> u(elem.n_node * 3, 0.0);
    random_fill(u, 99);

    std::vector<double> r_cpu(elem.n_node * 3, 0.0);
    std::vector<double> r_cuda(elem.n_node * 3, 0.0);

    // CPU reference (inline)
    reference_element_residual(1, elem.dxi_dx.data(), elem.jacobian.data(), elem.lambda_.data(),
                               elem.mu_.data(), elem.D.data(), elem.w.data(), elem.ngll, u.data(),
                               r_cpu.data());

    // CUDA result
    compute_element_residual(1, elem.dxi_dx.data(), elem.jacobian.data(), elem.lambda_.data(),
                             elem.mu_.data(), elem.D.data(), elem.w.data(), elem.ngll, u.data(),
                             r_cuda.data());

    // Compare with relative tolerance
    // GPU uses atomicAdd which changes summation order vs CPU sequential loop.
    // This introduces machine-epsilon-level differences (~4e-16 relative).
    double max_rel_diff = 0.0;
    size_t max_idx = 0;
    for (size_t i = 0; i < r_cpu.size(); ++i) {
        double denom = std::max(std::abs(r_cpu[i]), 1.0e-14);
        double rel_diff = std::abs(r_cpu[i] - r_cuda[i]) / denom;
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
            max_idx = i;
        }
    }
    // Relative tolerance: 1e-12 allows machine-epsilon differences from
    // non-deterministic atomic summation order.
    REQUIRE(max_rel_diff < 1.0e-12);
}

TEST_CASE("CUDA element residual matches CPU reference — N=5 random", "[element][cuda]") {
    int N = 5;
    RandomElement elem(N, 123);

    std::vector<double> u(elem.n_node * 3, 0.0);
    random_fill(u, 456);

    std::vector<double> r_cpu(elem.n_node * 3, 0.0);
    std::vector<double> r_cuda(elem.n_node * 3, 0.0);

    reference_element_residual(1, elem.dxi_dx.data(), elem.jacobian.data(), elem.lambda_.data(),
                               elem.mu_.data(), elem.D.data(), elem.w.data(), elem.ngll, u.data(),
                               r_cpu.data());

    compute_element_residual(1, elem.dxi_dx.data(), elem.jacobian.data(), elem.lambda_.data(),
                             elem.mu_.data(), elem.D.data(), elem.w.data(), elem.ngll, u.data(),
                             r_cuda.data());

    // Compare with relative tolerance
    double max_rel_diff = 0.0;
    for (size_t i = 0; i < r_cpu.size(); ++i) {
        double denom = std::max(std::abs(r_cpu[i]), 1.0e-14);
        double rel_diff = std::abs(r_cpu[i] - r_cuda[i]) / denom;
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }
    }
    REQUIRE(max_rel_diff < 1.0e-12);
}

TEST_CASE("CUDA element residual — rigid body translation zero", "[element][cuda]") {
    int N = 3;
    RandomElement elem(N, 77);

    // Uniform translation
    std::vector<double> u(elem.n_node * 3, 0.0);
    for (int i = 0; i < elem.n_node; ++i) {
        u[i * 3 + 0] = 1.0;
        u[i * 3 + 1] = 2.0;
        u[i * 3 + 2] = 3.0;
    }

    std::vector<double> r(elem.n_node * 3, 0.0);
    compute_element_residual(1, elem.dxi_dx.data(), elem.jacobian.data(), elem.lambda_.data(),
                             elem.mu_.data(), elem.D.data(), elem.w.data(), elem.ngll, u.data(),
                             r.data());

    for (size_t i = 0; i < r.size(); ++i) {
        REQUIRE_THAT(r[i], WithinAbs(0.0, 1e-5));
    }
}

TEST_CASE("CUDA C-PML timestep updates match CPU reference", "[pml][cpml][cuda]") {
    constexpr int polynomial_order = 2;
    constexpr int ngll = polynomial_order + 1;
    constexpr int n_node = ngll * ngll * ngll;
    constexpr int n_dof = n_node * 3;
    constexpr double solver_dt = 0.002;

    std::vector<double> nodes = gll_nodes(polynomial_order);
    std::vector<double> weights = gll_weights(polynomial_order, nodes);
    std::vector<double> derivative = gll_derivative_matrix(polynomial_order, nodes);
    std::vector<double> mass(n_node, 1.0);
    std::vector<double> damping(n_node, 0.0);
    std::vector<double> dxi_dx(n_node * 9, 0.0);
    std::vector<double> jacobian(n_node, 1.25);
    std::vector<double> lambda(n_node, 2.0);
    std::vector<double> mu(n_node, 1.0);
    std::vector<double> density(n_node, 2.5);
    std::vector<int32_t> local_cell2rank_node(n_node);
    for (int node_index = 0; node_index < n_node; ++node_index) {
        dxi_dx[node_index * 9 + 0] = 1.0;
        dxi_dx[node_index * 9 + 4] = 1.0;
        dxi_dx[node_index * 9 + 8] = 1.0;
        local_cell2rank_node[node_index] = node_index;
    }

    RankData cpu_part;
    cpu_part.has_cpml = true;
    cpu_part.n_local_cell = 1;
    cpu_part.ngll = ngll;
    cpu_part.local_cell2rank_node = local_cell2rank_node;
    cpu_part.pml_region.assign(1, 1);
    cpu_part.pml_coef_alpha.resize(n_node * 9);
    cpu_part.pml_coef_beta.resize(n_node * 9);
    cpu_part.pml_coef_abar.resize(n_node * 5);
    cpu_part.pml_coef_strain.resize(n_node * 18, 0.0);
    cpu_part.density = density;
    cpu_part.jacobian = jacobian;
    cpu_part.dxi_dx = dxi_dx;
    cpu_part.mass = mass;
    for (int node_index = 0; node_index < n_node; ++node_index) {
        for (int direction = 0; direction < 3; ++direction) {
            int coefficient_offset = node_index * 9 + direction * 3;
            cpu_part.pml_coef_alpha[coefficient_offset + 0] = 0.91;
            cpu_part.pml_coef_alpha[coefficient_offset + 1] = 0.17;
            cpu_part.pml_coef_alpha[coefficient_offset + 2] = -0.08;
            cpu_part.pml_coef_beta[coefficient_offset + 0] = 0.87;
            cpu_part.pml_coef_beta[coefficient_offset + 1] = 0.13;
            cpu_part.pml_coef_beta[coefficient_offset + 2] = -0.04;
        }
        for (int coefficient = 0; coefficient < 5; ++coefficient) {
            cpu_part.pml_coef_abar[node_index * 5 + coefficient] =
                0.01 * static_cast<double>(coefficient + 1);
        }
    }
    cpml_initialize(cpu_part, n_node);
    RankData gpu_part = cpu_part;

    std::vector<double> displacement(n_dof);
    std::vector<double> velocity(n_dof);
    std::vector<double> acceleration(n_dof);
    std::vector<double> displacement_tilde(n_dof);
    for (int dof_index = 0; dof_index < n_dof; ++dof_index) {
        displacement[dof_index] = 0.01 * static_cast<double>(dof_index + 1);
        velocity[dof_index] = -0.02 * static_cast<double>((dof_index % 7) + 1);
        acceleration[dof_index] = 0.03 * static_cast<double>((dof_index % 5) - 2);
        displacement_tilde[dof_index] = displacement[dof_index] + solver_dt * velocity[dof_index] +
                                        0.5 * solver_dt * solver_dt * acceleration[dof_index];
    }

    cpml_save_displ_old(cpu_part, displacement, velocity, acceleration, solver_dt, n_node);
    cpml_save_displ_new(cpu_part, displacement_tilde, velocity, acceleration, solver_dt, n_node);
    cpml_update_displ_memory(cpu_part, n_node);
    cpml_update_strain_memory(cpu_part, derivative.data(), weights.data(), ngll);
    std::vector<double> cpu_residual(n_dof, 0.0);
    cpml_accel_contribution(cpu_part, displacement_tilde, velocity, acceleration, solver_dt,
                            local_cell2rank_node, weights, cpu_residual, 1, n_node);

    ConfigData config;
    config.n_src_cell = 1;
    CudaDeviceState gpu_state = cuda_allocate_state(
        1, ngll, mass, damping, dxi_dx, jacobian, lambda, mu, derivative.data(), weights.data(),
        config, n_dof, local_cell2rank_node, n_node, mass, damping);
    cuda_upload_cpml_data(gpu_state, gpu_part, n_node);
    cuda_copy_state_from_host(gpu_state, displacement, velocity, acceleration);
    cuda_newmark_predict(gpu_state, solver_dt, 0.0);
    cuda_cpml_update_displ_fields(gpu_state, solver_dt, n_node);
    cuda_cpml_update_displ_memory(gpu_state, n_node);
    cuda_cpml_update_strain_memory(gpu_state, ngll, n_node);
    cuda_zero_residual(gpu_state);
    cuda_cpml_accel_contribution(gpu_state, solver_dt, ngll, n_node);

    std::vector<double> gpu_displ_old(cpu_part.pml_displ_old.size());
    std::vector<double> gpu_displ_new(cpu_part.pml_displ_new.size());
    std::vector<double> gpu_rmemory_displ(cpu_part.rmemory_displ.size());
    std::vector<double> gpu_rmemory_strain(cpu_part.rmemory_strain.size());
    std::vector<double> gpu_residual(n_dof);
    cudaMemcpy(gpu_displ_old.data(), gpu_state.d_pml_displ_old,
               gpu_displ_old.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_displ_new.data(), gpu_state.d_pml_displ_new,
               gpu_displ_new.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_rmemory_displ.data(), gpu_state.d_rmemory_displ,
               gpu_rmemory_displ.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_rmemory_strain.data(), gpu_state.d_rmemory_strain,
               gpu_rmemory_strain.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_residual.data(), gpu_state.d_local_cell_residual,
               gpu_residual.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cuda_free_state(gpu_state);

    auto require_vectors_equal = [](const std::vector<double>& cpu_values,
                                    const std::vector<double>& gpu_values) {
        REQUIRE(cpu_values.size() == gpu_values.size());
        for (size_t value_index = 0; value_index < cpu_values.size(); ++value_index) {
            REQUIRE_THAT(gpu_values[value_index], WithinAbs(cpu_values[value_index], 1.0e-12));
        }
    };
    require_vectors_equal(cpu_part.pml_displ_old, gpu_displ_old);
    require_vectors_equal(cpu_part.pml_displ_new, gpu_displ_new);
    require_vectors_equal(cpu_part.rmemory_displ, gpu_rmemory_displ);
    require_vectors_equal(cpu_part.rmemory_strain, gpu_rmemory_strain);
    require_vectors_equal(cpu_residual, gpu_residual);
}
