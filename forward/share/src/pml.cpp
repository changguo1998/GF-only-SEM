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
    part.rmemory_displ.assign(n_pml_node * 9, 0.0);    // 3 components × 3 directions
    part.rmemory_strain.assign(n_pml_node * 27, 0.0);  // 9 gradients × 3 directions
}

void cpml_update_displ_fields(RankData& part, const std::vector<double>& displacement,
                              const std::vector<double>& velocity,
                              const std::vector<double>& acceleration, double dt, int n_node) {
    if (!part.has_cpml)
        return;

    const int n_local_cell = part.n_local_cell;
    const int ngll = part.ngll;
    const double c1 = (1.0 - 2.0 * THETA) * 0.5 * dt;  // (1-2θ)/2 * dt
    const double c2 = (1.0 - THETA) * 0.5 * dt * dt;   // (1-θ)/2 * dt²

    // Swap: old <- new (previous step's "new" becomes this step's "old")
    std::swap(part.pml_displ_old, part.pml_displ_new);

    // Compute new PML_displ_new = u + c1 * v
    // (acceleration term c2 * a is only in PML_displ_old, set at the end of
    //  the previous step; here we compute the new "new" field)
    for (int e = 0; e < n_local_cell; ++e) {
        int region = (e < static_cast<int>(part.pml_region.size())) ? part.pml_region[e] : 0;
        if (region == 0)
            continue;

        int elem_off = e * n_node * 3;
        for (int n = 0; n < n_node; ++n) {
            int node_local = elem_off + n * 3;
            // Get rank-level node index
            int rank_node = part.local_cell2rank_node[e * n_node + n];
            int rank_dof = rank_node * 3;

            for (int d = 0; d < 3; ++d) {
                part.pml_displ_new[node_local + d] =
                    displacement[rank_dof + d] + c1 * velocity[rank_dof + d];
            }
        }
    }

    // Update PML_displ_old: add c2 * a to the swapped (previous new) field
    // PML_displ_old was the previous step's PML_displ_new (= u_prev + c1*v_prev)
    // Now add the acceleration term: += c2 * a_prev
    // Note: acceleration at this point is from the previous timestep (not yet updated)
    for (int e = 0; e < n_local_cell; ++e) {
        int region = (e < static_cast<int>(part.pml_region.size())) ? part.pml_region[e] : 0;
        if (region == 0)
            continue;

        int elem_off = e * n_node * 3;
        for (int n = 0; n < n_node; ++n) {
            int rank_node = part.local_cell2rank_node[e * n_node + n];
            int rank_dof = rank_node * 3;

            for (int d = 0; d < 3; ++d) {
                part.pml_displ_old[elem_off + n * 3 + d] += c2 * acceleration[rank_dof + d];
            }
        }
    }
}

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

void cpml_accel_contribution(const RankData& part, const std::vector<double>& displacement,
                             const std::vector<double>& velocity,
                             const std::vector<int32_t>& local_cell2rank_node,
                             const std::vector<double>& gll_weights, std::vector<double>& residual,
                             int n_local_cell, int n_node) {
    if (!part.has_cpml)
        return;

    const int ngll = part.ngll;

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
            double rho_inv = (rho > 0.0) ? 1.0 / rho : 0.0;

            // Coefficients Ā₁…Ā₅
            double A1 = part.pml_coef_abar[node_coef_off + 0];
            double A2 = part.pml_coef_abar[node_coef_off + 1];
            double A3 = part.pml_coef_abar[node_coef_off + 2];
            double A4 = part.pml_coef_abar[node_coef_off + 3];
            double A5 = part.pml_coef_abar[node_coef_off + 4];

            // Scale factor: w * (1/ρ) * J
            double scale = wgll * rho_inv * jac;

            for (int comp = 0; comp < 3; ++comp) {
                double u_val = displacement[rank_dof + comp];
                double v_val = velocity[rank_dof + comp];

                // Memory variables for this component: mem[x], mem[y], mem[z]
                double mem_x = part.rmemory_displ[node_mem_off + comp * 3 + 0];
                double mem_y = part.rmemory_displ[node_mem_off + comp * 3 + 1];
                double mem_z = part.rmemory_displ[node_mem_off + comp * 3 + 2];

                double accel_pml =
                    scale * (A1 * v_val + A2 * u_val + A3 * mem_x + A4 * mem_y + A5 * mem_z);

                residual[elem_resid_off + n * 3 + comp] += accel_pml;
            }
        }
    }
}

}  // namespace gf
