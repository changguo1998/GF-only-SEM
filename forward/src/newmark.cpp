// forward/src/newmark.cpp
#include "gf/newmark.hpp"

namespace gf {

void newmark_predictor(
    const NewmarkParams& params,
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& a,
    std::vector<double>& u_tilde,
    std::vector<double>& v_tilde
) {
    const double dt    = params.dt;
    const double dt2   = dt * dt;
    const double half  = 0.5;
    const size_t n_dof = u.size();

    for (size_t i = 0; i < n_dof; ++i) {
        // u_tilde = u + dt*v + 0.5*dt^2*a
        u_tilde[i] = u[i] + dt * v[i] + half * dt2 * a[i];

        // v_tilde = v + 0.5*dt*a
        v_tilde[i] = v[i] + half * dt * a[i];
    }
}

void newmark_corrector(
    const NewmarkParams& params,
    const std::vector<double>& mass,
    const std::vector<double>& residual,
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& a
) {
    const double dt    = params.dt;
    const double half  = 0.5;
    const size_t n_dof = u.size();

    for (size_t i = 0; i < n_dof; ++i) {
        // a_new = residual / mass
        a[i] = residual[i] / mass[i];

        // v_new = v_tilde + 0.5*dt*a_new
        // v currently holds v_tilde (from predictor)
        v[i] += half * dt * a[i];

        // u unchanged when beta=0
        // (dsplacement is already at u_tilde from predictor, which is
        //  the final displacement for beta=0)
    }
}

} // namespace gf