#include "gf/element.hpp"
#include "gf/gll.hpp"

namespace gf {

// Inline helper: 1D flat index from (i, j, k) in element
static inline int idx(int i, int j, int k, int NGLL) {
    return (i * NGLL + j) * NGLL + k;
}

void compute_element_residual(
    const double* dxi_dx,
    const double* jacobian,
    const double* vp,
    const double* vs,
    const double* density,
    const double* D,
    const double* weights,
    int NGLL,
    const double* u,
    double* r
) {
    const int n_node = NGLL * NGLL * NGLL;

    for (int i = 0; i < NGLL; ++i) {
        for (int j = 0; j < NGLL; ++j) {
            for (int k = 0; k < NGLL; ++k) {
                const int n = idx(i, j, k, NGLL);

                // --- Material properties at this GLL node ---
                const double rho = density[n];
                if (rho <= 0.0) continue;
                const double vp2 = vp[n] * vp[n];
                const double vs2 = vs[n] * vs[n];
                const double lambda = rho * (vp2 - 2.0 * vs2);
                const double mu     = rho * vs2;
                const double l2mu   = lambda + 2.0 * mu;

                // --- Inverse Jacobian at this node ---
                const double* dd = &dxi_dx[9 * n];
                // dd[0]=dξ/dx, dd[1]=dη/dx, dd[2]=dζ/dx
                // dd[3]=dξ/dy, dd[4]=dη/dy, dd[5]=dζ/dy
                // dd[6]=dξ/dz, dd[7]=dη/dz, dd[8]=dζ/dz

                // --- Compute displacement gradient in reference space ---
                double dudxi[3]  = {0.0, 0.0, 0.0};
                double dudeta[3] = {0.0, 0.0, 0.0};
                double dudzeta[3]= {0.0, 0.0, 0.0};

                for (int s = 0; s < NGLL; ++s) {
                    const double Di_s = D[i * NGLL + s];
                    const double Dj_s = D[j * NGLL + s];
                    const double Dk_s = D[k * NGLL + s];

                    const int n_sjk  = idx(s, j, k, NGLL);
                    const int n_isk  = idx(i, s, k, NGLL);
                    const int n_ijs  = idx(i, j, s, NGLL);

                    for (int dir = 0; dir < 3; ++dir) {
                        dudxi[dir]   += Di_s * u[3 * n_sjk  + dir];
                        dudeta[dir]  += Dj_s * u[3 * n_isk  + dir];
                        dudzeta[dir] += Dk_s * u[3 * n_ijs  + dir];
                    }
                }

                // --- Transform to physical gradient ---
                double du_dx[3][3];
                for (int comp = 0; comp < 3; ++comp) {
                    du_dx[comp][0] = dudxi[comp]*dd[0] + dudeta[comp]*dd[1] + dudzeta[comp]*dd[2];
                    du_dx[comp][1] = dudxi[comp]*dd[3] + dudeta[comp]*dd[4] + dudzeta[comp]*dd[5];
                    du_dx[comp][2] = dudxi[comp]*dd[6] + dudeta[comp]*dd[7] + dudzeta[comp]*dd[8];
                }

                // --- Symmetric strain tensor ---
                double eps[3][3];
                for (int l = 0; l < 3; ++l) {
                    for (int m = 0; m < 3; ++m) {
                        eps[l][m] = 0.5 * (du_dx[l][m] + du_dx[m][l]);
                    }
                }

                // --- Isotropic stress ---
                double eps_kk = eps[0][0] + eps[1][1] + eps[2][2];
                double sigma[3][3];
                for (int l = 0; l < 3; ++l) {
                    for (int m = 0; m < 3; ++m) {
                        sigma[l][m] = 2.0 * mu * eps[l][m];
                    }
                    sigma[l][l] += lambda * eps_kk;
                }

                // --- Quadrature weight factor ---
                const double factor = jacobian[n] * weights[i] * weights[j] * weights[k];

                // --- Accumulate residual contributions ---
                // ξ-direction contributions to nodes (s, j, k)
                for (int s = 0; s < NGLL; ++s) {
                    const double Dis = D[i * NGLL + s];
                    const double gradN[3] = {Dis*dd[0], Dis*dd[3], Dis*dd[6]};
                    const int n_s = idx(s, j, k, NGLL);
                    r[3*n_s + 0] -= (sigma[0][0]*gradN[0] + sigma[0][1]*gradN[1] + sigma[0][2]*gradN[2]) * factor;
                    r[3*n_s + 1] -= (sigma[1][0]*gradN[0] + sigma[1][1]*gradN[1] + sigma[1][2]*gradN[2]) * factor;
                    r[3*n_s + 2] -= (sigma[2][0]*gradN[0] + sigma[2][1]*gradN[1] + sigma[2][2]*gradN[2]) * factor;
                }

                // η-direction contributions to nodes (i, s, k)
                for (int s = 0; s < NGLL; ++s) {
                    const double Djs = D[j * NGLL + s];
                    const double gradN[3] = {Djs*dd[1], Djs*dd[4], Djs*dd[7]};
                    const int n_s = idx(i, s, k, NGLL);
                    r[3*n_s + 0] -= (sigma[0][0]*gradN[0] + sigma[0][1]*gradN[1] + sigma[0][2]*gradN[2]) * factor;
                    r[3*n_s + 1] -= (sigma[1][0]*gradN[0] + sigma[1][1]*gradN[1] + sigma[1][2]*gradN[2]) * factor;
                    r[3*n_s + 2] -= (sigma[2][0]*gradN[0] + sigma[2][1]*gradN[1] + sigma[2][2]*gradN[2]) * factor;
                }

                // ζ-direction contributions to nodes (i, j, s)
                for (int s = 0; s < NGLL; ++s) {
                    const double Dks = D[k * NGLL + s];
                    const double gradN[3] = {Dks*dd[2], Dks*dd[5], Dks*dd[8]};
                    const int n_s = idx(i, j, s, NGLL);
                    r[3*n_s + 0] -= (sigma[0][0]*gradN[0] + sigma[0][1]*gradN[1] + sigma[0][2]*gradN[2]) * factor;
                    r[3*n_s + 1] -= (sigma[1][0]*gradN[0] + sigma[1][1]*gradN[1] + sigma[1][2]*gradN[2]) * factor;
                    r[3*n_s + 2] -= (sigma[2][0]*gradN[0] + sigma[2][1]*gradN[1] + sigma[2][2]*gradN[2]) * factor;
                }
            }
        }
    }
}

} // namespace gf