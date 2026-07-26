// forward/share/src/pml.cpp
#include "gf/pml.hpp"

#include <cassert>
#include <cmath>

namespace gf {

// ---------------------------------------------------------------------------
// Legacy linear-ramp PML damping (backward compatibility)
// ---------------------------------------------------------------------------

void apply_pml_damping(const std::vector<double>& damping_profile,
                       const std::vector<double>& /*u*/, std::vector<double>& v, int n_dof) {
    // n_node_per_elem = NGLL^3
    // n_dof = n_elem * n_node_per_elem * 3
    // Each node has 3 DOFs sharing the same damping coefficient
    const size_t n_total_dof = static_cast<size_t>(n_dof);

    for (size_t i = 0; i < n_total_dof; ++i) {
        // damping_profile index: node = i / 3
        const size_t node = i / 3;
        const double d = damping_profile[node];

        if (d > 0.0) {
            v[i] -= d * v[i];
        }
    }
}

// ---------------------------------------------------------------------------
// C-PML constants (Wang et al. 2006)
// ---------------------------------------------------------------------------

static constexpr double THETA = 1.0 / 8.0;

// ---------------------------------------------------------------------------
// C-PML implementation
// ---------------------------------------------------------------------------

void cpml_initialize(RankData& part, int n_node) {
    if (!part.has_cpml)
        return;

    int n_local_cell = part.n_local_cell;
    size_t n_pml_node = static_cast<size_t>(n_local_cell) * n_node;

    // Allocate and zero-initialize memory state
    part.pml_displ_old.assign(n_pml_node * 3, 0.0);
    part.pml_displ_new.assign(n_pml_node * 3, 0.0);
    part.rmemory_displ.assign(n_pml_node * 9, 0.0);  // 3 components × 3 directions
    part.rmemory_strain.assign(n_pml_node * CpmlStrain::MEMORY_PER_NODE,
                               0.0);  // 27 lijk + 12 lx/ly/lz
}

void cpml_save_displ_old(RankData& part, const std::vector<double>& displacement,
                         const std::vector<double>& velocity,
                         const std::vector<double>& acceleration, double dt, int n_node) {
    if (!part.has_cpml)
        return;

    const int n_local_cell = part.n_local_cell;
    const double c1 = (1.0 - 2.0 * THETA) * 0.5 * dt;  // (1-2θ)/2 * dt
    const double c2 = (1.0 - THETA) * 0.5 * dt * dt;   // (1-θ)/2 * dt²

    // PML_displ_old = u + c1 * v + c2 * a  (fresh computation, BEFORE predictor)
    // Matches SPECFEM3D update_displ_elastic_PML called before predictor
    // (update_displacement_scheme.f90:296)
    for (int e = 0; e < n_local_cell; ++e) {
        int region = (e < static_cast<int>(part.pml_region.size())) ? part.pml_region[e] : 0;
        if (region == 0)
            continue;

        int elem_off = e * n_node * 3;
        for (int n = 0; n < n_node; ++n) {
            int node_local = elem_off + n * 3;
            int rank_node = part.local_cell2rank_node[e * n_node + n];
            int rank_dof = rank_node * 3;

            for (int d = 0; d < 3; ++d) {
                part.pml_displ_old[node_local + d] = displacement[rank_dof + d] +
                                                     c1 * velocity[rank_dof + d] +
                                                     c2 * acceleration[rank_dof + d];
            }
        }
    }
}

void cpml_save_displ_new(RankData& part, const std::vector<double>& displacement_tilde,
                         const std::vector<double>& velocity,
                         const std::vector<double>& acceleration, double dt, int n_node) {
    if (!part.has_cpml)
        return;

    const int n_local_cell = part.n_local_cell;
    const double c1 = (1.0 - 2.0 * THETA) * 0.5 * dt;  // (1-2θ)/2 * dt
    const double half_dt = 0.5 * dt;                   // dt/2 for predicted velocity

    // PML_displ_new = u_tilde + c1 * v_pred
    // where v_pred = v + dt/2 * a (predicted velocity, matching SPECFEM3D's
    // in-place predictor veloc += dt/2*accel)
    // No c2*a term because accel = 0 after predictor.
    // Matches SPECFEM3D update_displ_elastic_PML called after predictor
    // (update_displacement_scheme.f90:305)
    for (int e = 0; e < n_local_cell; ++e) {
        int region = (e < static_cast<int>(part.pml_region.size())) ? part.pml_region[e] : 0;
        if (region == 0)
            continue;

        int elem_off = e * n_node * 3;
        for (int n = 0; n < n_node; ++n) {
            int node_local = elem_off + n * 3;
            int rank_node = part.local_cell2rank_node[e * n_node + n];
            int rank_dof = rank_node * 3;

            for (int d = 0; d < 3; ++d) {
                double v_pred = velocity[rank_dof + d] + half_dt * acceleration[rank_dof + d];
                part.pml_displ_new[node_local + d] =
                    displacement_tilde[rank_dof + d] + c1 * v_pred;
            }
        }
    }
}

/// Update C-PML displacement memory variables (host-side).
void cpml_update_displ_memory(RankData& part, int n_node) {
    if (!part.has_cpml)
        return;

    const int n_local_cell = part.n_local_cell;

    for (int e = 0; e < n_local_cell; ++e) {
        int region = (e < static_cast<int>(part.pml_region.size())) ? part.pml_region[e] : 0;
        if (region == 0)
            continue;

        int elem_off = e * n_node;
        for (int n = 0; n < n_node; ++n) {
            int node_off = (elem_off + n) * 9;   // rmemory_displ: 9 per node
            int coef_off = (elem_off + n) * 9;   // pml_coef_alpha: 9 per node
            int displ_off = (elem_off + n) * 3;  // pml_displ: 3 per node

            // For each direction d ∈ {x, y, z}:
            //   rmemory[comp*3 + d] = coef0[d] * rmemory[comp*3 + d]
            //                        + coef1[d] * PML_displ_new[comp]
            //                        + coef2[d] * PML_displ_old[comp]
            for (int comp = 0; comp < 3; ++comp) {
                double new_val = part.pml_displ_new[displ_off + comp];
                double old_val = part.pml_displ_old[displ_off + comp];

                for (int d = 0; d < 3; ++d) {
                    int mem_idx = node_off + comp * 3 + d;
                    int coef_idx = coef_off + d * 3;  // {coef0, coef1, coef2} for direction d

                    part.rmemory_displ[mem_idx] =
                        part.pml_coef_alpha[coef_idx + 0] * part.rmemory_displ[mem_idx] +
                        part.pml_coef_alpha[coef_idx + 1] * new_val +
                        part.pml_coef_alpha[coef_idx + 2] * old_val;
                }
            }
        }
    }
}

void cpml_accel_contribution(const RankData& part, const std::vector<double>& displacement_tilde,
                             const std::vector<double>& velocity,
                             const std::vector<double>& acceleration, double dt,
                             const std::vector<int32_t>& local_cell2rank_node,
                             const std::vector<double>& gll_weights, std::vector<double>& residual,
                             int n_local_cell, int n_node) {
    if (!part.has_cpml)
        return;

    const int ngll = part.ngll;
    const double half_dt = 0.5 * dt;  // dt/2 for predicted velocity

    for (int e = 0; e < n_local_cell; ++e) {
        int region = (e < static_cast<int>(part.pml_region.size())) ? part.pml_region[e] : 0;
        if (region == 0)
            continue;

        int elem_off = e * n_node;
        int elem_resid_off = elem_off * 3;  // residual: 3 per node

        for (int n = 0; n < n_node; ++n) {
            // GLL weight product w_i * w_j * w_k
            int nk = n % ngll;
            int nj = (n / ngll) % ngll;
            int ni = n / (ngll * ngll);
            double wgll = gll_weights[ni] * gll_weights[nj] * gll_weights[nk];

            int rank_node = local_cell2rank_node[elem_off + n];
            int rank_dof = rank_node * 3;

            int node_coef_off = (elem_off + n) * 5;  // abar: 5 per node
            int node_mem_off = (elem_off + n) * 9;   // rmemory: 9 per node
            int node_field_off = part.mass.size() > 0
                                     ? static_cast<int>(part.mass.size()) / part.n_local_cell
                                     : n_node;
            // Material properties at this GLL node
            int mat_off = elem_off + n;
            double rho = part.density[mat_off];
            double jac = part.jacobian[mat_off];
            // Scale factor: w * ρ * J (matches SPECFEM3D pml_compute_accel_contribution)
            // The residual is later divided by mass (∝ ρ * w * J), so the net
            // acceleration contribution is just the PML term itself.
            double rho_val = (rho > 0.0) ? rho : 0.0;

            // Coefficients Ā₁…Ā₅
            double A1 = part.pml_coef_abar[node_coef_off + 0];
            double A2 = part.pml_coef_abar[node_coef_off + 1];
            double A3 = part.pml_coef_abar[node_coef_off + 2];
            double A4 = part.pml_coef_abar[node_coef_off + 3];
            double A5 = part.pml_coef_abar[node_coef_off + 4];

            // Scale factor: w * ρ * J
            double scale = wgll * rho_val * jac;

            for (int comp = 0; comp < 3; ++comp) {
                // Use PREDICTED displacement and velocity (matching SPECFEM3D
                // which modifies displ/veloc in-place during predictor)
                double u_val = displacement_tilde[rank_dof + comp];
                double v_val = velocity[rank_dof + comp] + half_dt * acceleration[rank_dof + comp];

                // Memory variables for this component: mem[x], mem[y], mem[z]
                double mem_x = part.rmemory_displ[node_mem_off + comp * 3 + 0];
                double mem_y = part.rmemory_displ[node_mem_off + comp * 3 + 1];
                double mem_z = part.rmemory_displ[node_mem_off + comp * 3 + 2];

                double accel_pml =
                    scale * (A1 * v_val + A2 * u_val + A3 * mem_x + A4 * mem_y + A5 * mem_z);

                // SPECFEM3D sign convention: accel -= PML_contribution.
                // Our residual already has a negative sign (scatter_residual
                // uses r -= sigma:gradN), so we must also SUBTRACT the PML
                // contribution to match SPECFEM3D's accel -= (force + PML).
                residual[elem_resid_off + n * 3 + comp] -= accel_pml;
            }
        }
    }
}

void cpml_update_strain_memory(RankData& part, const double* D, const double* /*weights*/,
                               int NGLL) {
    if (!part.has_cpml)
        return;

    const int n_local_cell = part.n_local_cell;
    const int n_node = NGLL * NGLL * NGLL;

    for (int e = 0; e < n_local_cell; ++e) {
        int region = (e < static_cast<int>(part.pml_region.size())) ? part.pml_region[e] : 0;
        if (region == 0)
            continue;

        const int elem_off = e * n_node;

        for (int n = 0; n < n_node; ++n) {
            const int i = n / (NGLL * NGLL);
            const int j = (n / NGLL) % NGLL;
            const int k = n % NGLL;

            // --- 1. Compute reference-space gradients of PML displ fields ---
            double new_dudxi[3] = {0, 0, 0};
            double new_dudeta[3] = {0, 0, 0};
            double new_dudzeta[3] = {0, 0, 0};
            double old_dudxi[3] = {0, 0, 0};
            double old_dudeta[3] = {0, 0, 0};
            double old_dudzeta[3] = {0, 0, 0};

            for (int s = 0; s < NGLL; ++s) {
                const double Di_s = D[i * NGLL + s];
                const double Dj_s = D[j * NGLL + s];
                const double Dk_s = D[k * NGLL + s];

                const int n_sjk = (s * NGLL + j) * NGLL + k;
                const int n_isk = (i * NGLL + s) * NGLL + k;
                const int n_ijs = (i * NGLL + j) * NGLL + s;

                const int new_sjk = (elem_off + n_sjk) * 3;
                const int new_isk = (elem_off + n_isk) * 3;
                const int new_ijs = (elem_off + n_ijs) * 3;
                const int old_sjk = (elem_off + n_sjk) * 3;
                const int old_isk = (elem_off + n_isk) * 3;
                const int old_ijs = (elem_off + n_ijs) * 3;

                for (int dir = 0; dir < 3; ++dir) {
                    new_dudxi[dir] += Di_s * part.pml_displ_new[new_sjk + dir];
                    new_dudeta[dir] += Dj_s * part.pml_displ_new[new_isk + dir];
                    new_dudzeta[dir] += Dk_s * part.pml_displ_new[new_ijs + dir];
                    old_dudxi[dir] += Di_s * part.pml_displ_old[old_sjk + dir];
                    old_dudeta[dir] += Dj_s * part.pml_displ_old[old_isk + dir];
                    old_dudzeta[dir] += Dk_s * part.pml_displ_old[old_ijs + dir];
                }
            }

            // --- 2. Transform to physical gradients ---
            const double* dd = &part.dxi_dx[(elem_off + n) * 9];

            double new_phys_grad[9];
            double old_phys_grad[9];
            for (int comp = 0; comp < 3; ++comp) {
                new_phys_grad[comp * 3 + 0] =
                    new_dudxi[comp] * dd[0] + new_dudeta[comp] * dd[1] + new_dudzeta[comp] * dd[2];
                new_phys_grad[comp * 3 + 1] =
                    new_dudxi[comp] * dd[3] + new_dudeta[comp] * dd[4] + new_dudzeta[comp] * dd[5];
                new_phys_grad[comp * 3 + 2] =
                    new_dudxi[comp] * dd[6] + new_dudeta[comp] * dd[7] + new_dudzeta[comp] * dd[8];
                old_phys_grad[comp * 3 + 0] =
                    old_dudxi[comp] * dd[0] + old_dudeta[comp] * dd[1] + old_dudzeta[comp] * dd[2];
                old_phys_grad[comp * 3 + 1] =
                    old_dudxi[comp] * dd[3] + old_dudeta[comp] * dd[4] + old_dudzeta[comp] * dd[5];
                old_phys_grad[comp * 3 + 2] =
                    old_dudxi[comp] * dd[6] + old_dudeta[comp] * dd[7] + old_dudzeta[comp] * dd[8];
            }

            // --- 3. Update strain memory with β convolution ---
            using namespace CpmlStrain;
            for (int grad = 0; grad < NUM_GRADIENT_COMPS; ++grad) {
                double new_grad = new_phys_grad[grad];
                double old_grad = old_phys_grad[grad];
                for (int conv_dir = 0; conv_dir < NUM_CONV_DIRECTIONS; ++conv_dir) {
                    /// Compute flat memory offset for C-PML strain memory (node, gradient,
                    /// component).
                    size_t mem_off = strain_memory_offset(elem_off + n, grad, conv_dir);
                    int beta_off =
                        (elem_off + n) * BETA_COEFS_PER_NODE + conv_dir * BETA_COEFS_PER_DIR;
                    part.rmemory_strain[mem_off] =
                        part.pml_coef_beta[beta_off + BETA_COEF0] * part.rmemory_strain[mem_off] +
                        part.pml_coef_beta[beta_off + BETA_COEF1] * new_grad +
                        part.pml_coef_beta[beta_off + BETA_COEF2] * old_grad;
                }
            }

            // --- 3b. Update LX/LY/LZ memory with α convolution ---
            // α coefficients from pml_coef_alpha (same as for displacement memory)
            constexpr int ALPHA_COEFS_PER_DIR = 3;
            const double* alpha_base = &part.pml_coef_alpha[(elem_off + n) * 9];
            size_t base_node = elem_off + n;

            // LX: alpha_x (CONV_X), gradients 4,5,7,8
            int alpha_x_off = CONV_X * ALPHA_COEFS_PER_DIR;
            double ax0 = alpha_base[alpha_x_off + 0];
            double ax1 = alpha_base[alpha_x_off + 1];
            double ax2 = alpha_base[alpha_x_off + 2];
            for (auto grad : {DUY_DY, DUY_DZ, DUZ_DY, DUZ_DZ}) {
                int slot = lx_slot_for_grad(grad);
                size_t mem_off = lx_memory_offset(base_node, slot);
                part.rmemory_strain[mem_off] = ax0 * part.rmemory_strain[mem_off] +
                                               ax1 * new_phys_grad[grad] +
                                               ax2 * old_phys_grad[grad];
            }

            // LY: alpha_y (CONV_Y), gradients 0,2,6,8
            int alpha_y_off = CONV_Y * ALPHA_COEFS_PER_DIR;
            double ay0 = alpha_base[alpha_y_off + 0];
            double ay1 = alpha_base[alpha_y_off + 1];
            double ay2 = alpha_base[alpha_y_off + 2];
            for (auto grad : {DUX_DX, DUX_DZ, DUZ_DX, DUZ_DZ}) {
                int slot = ly_slot_for_grad(grad);
                size_t mem_off = ly_memory_offset(base_node, slot);
                part.rmemory_strain[mem_off] = ay0 * part.rmemory_strain[mem_off] +
                                               ay1 * new_phys_grad[grad] +
                                               ay2 * old_phys_grad[grad];
            }

            // LZ: alpha_z (CONV_Z), gradients 0,1,3,4
            int alpha_z_off = CONV_Z * ALPHA_COEFS_PER_DIR;
            double az0 = alpha_base[alpha_z_off + 0];
            double az1 = alpha_base[alpha_z_off + 1];
            double az2 = alpha_base[alpha_z_off + 2];
            for (auto grad : {DUX_DX, DUX_DY, DUY_DX, DUY_DY}) {
                int slot = lz_slot_for_grad(grad);
                size_t mem_off = lz_memory_offset(base_node, slot);
                part.rmemory_strain[mem_off] = az0 * part.rmemory_strain[mem_off] +
                                               az1 * new_phys_grad[grad] +
                                               az2 * old_phys_grad[grad];
            }
        }
    }
}
}  // namespace gf
