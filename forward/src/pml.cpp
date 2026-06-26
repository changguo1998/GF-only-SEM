// forward/src/pml.cpp
#include "gf/pml.hpp"

#include <cassert>

namespace gf {

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

}  // namespace gf