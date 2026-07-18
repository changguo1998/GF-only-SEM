// forward/share/src/source.cpp
#include "gf/source.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "gf/gll.hpp"

namespace gf {

static inline int idx3(int i, int j, int k, int ngll) {
    return (i * ngll + j) * ngll + k;
}

bool PointForceSource::locate(double src_x, double src_y, double src_z,
                              const std::vector<double>& coords, const std::vector<double>& dxi_dx,
                              int n_local_cell, int ngll) {
    const int n_node = ngll * ngll * ngll;
    const int stride3 = n_node * 3;
    const int stride9 = n_node * 9;
    const std::vector<double>& nodes = gll_nodes(ngll - 1);

    const int max_iter = 20;
    const double tol = 1.0e-10;

    for (int e = 0; e < n_local_cell; ++e) {
        int base3 = e * stride3;
        int base9 = e * stride9;

        double xi = 0.0, eta = 0.0, zeta = 0.0;

        for (int iter = 0; iter < max_iter; ++iter) {
            std::vector<double> lx = lagrange_basis(xi, nodes);
            std::vector<double> ly = lagrange_basis(eta, nodes);
            std::vector<double> lz = lagrange_basis(zeta, nodes);

            double xv = 0, yv = 0, zv = 0;
            for (int i = 0; i < ngll; ++i) {
                double wxi = lx[i];
                for (int j = 0; j < ngll; ++j) {
                    double wxy = wxi * ly[j];
                    for (int k = 0; k < ngll; ++k) {
                        double w = wxy * lz[k];
                        int n = idx3(i, j, k, ngll) * 3;
                        xv += w * coords[base3 + n + 0];
                        yv += w * coords[base3 + n + 1];
                        zv += w * coords[base3 + n + 2];
                    }
                }
            }

            double rx = xv - src_x, ry = yv - src_y, rz = zv - src_z;
            if (std::sqrt(rx * rx + ry * ry + rz * rz) < tol)
                break;

            double Ji[9] = {0};
            for (int i = 0; i < ngll; ++i) {
                double wxi = lx[i];
                for (int j = 0; j < ngll; ++j) {
                    double wxy = wxi * ly[j];
                    for (int k = 0; k < ngll; ++k) {
                        double w = wxy * lz[k];
                        int n9 = idx3(i, j, k, ngll) * 9;
                        for (int m = 0; m < 9; ++m) {
                            Ji[m] += w * dxi_dx[base9 + n9 + m];
                        }
                    }
                }
            }

            xi += -(Ji[0] * rx + Ji[1] * ry + Ji[2] * rz);
            eta += -(Ji[3] * rx + Ji[4] * ry + Ji[5] * rz);
            zeta += -(Ji[6] * rx + Ji[7] * ry + Ji[8] * rz);

            xi = std::max(-1.0, std::min(1.0, xi));
            eta = std::max(-1.0, std::min(1.0, eta));
            zeta = std::max(-1.0, std::min(1.0, zeta));
        }

        if (std::abs(xi) <= 1.0 + 1e-8 && std::abs(eta) <= 1.0 + 1e-8 &&
            std::abs(zeta) <= 1.0 + 1e-8) {
            std::vector<double> lx = lagrange_basis(xi, nodes);
            std::vector<double> ly = lagrange_basis(eta, nodes);
            std::vector<double> lz = lagrange_basis(zeta, nodes);

            double xv = 0, yv = 0, zv = 0;
            for (int i = 0; i < ngll; ++i) {
                double wxi = lx[i];
                for (int j = 0; j < ngll; ++j) {
                    double wxy = wxi * ly[j];
                    for (int k = 0; k < ngll; ++k) {
                        double w = wxy * lz[k];
                        int n = idx3(i, j, k, ngll) * 3;
                        xv += w * coords[base3 + n + 0];
                        yv += w * coords[base3 + n + 1];
                        zv += w * coords[base3 + n + 2];
                    }
                }
            }
            const double rx = xv - src_x;
            const double ry = yv - src_y;
            const double rz = zv - src_z;
            if (std::sqrt(rx * rx + ry * ry + rz * rz) >= 1.0e-8) {
                continue;
            }

            double md = std::numeric_limits<double>::max();
            for (int i = 0; i < ngll; ++i) {
                for (int j = 0; j < ngll; ++j) {
                    for (int k = 0; k < ngll; ++k) {
                        double d = (xi - nodes[i]) * (xi - nodes[i]) +
                                   (eta - nodes[j]) * (eta - nodes[j]) +
                                   (zeta - nodes[k]) * (zeta - nodes[k]);
                        if (d < md) {
                            md = d;
                            gll_i = i;
                            gll_j = j;
                            gll_k = k;
                        }
                    }
                }
            }

            element_id = e + 1;
            wx = lx[gll_i] * ly[gll_j] * lz[gll_k];
            wy = 0.0;
            wz = 0.0;
            return true;
        }
    }
    return false;
}

void PointForceSource::apply(double force_x, double force_y, double force_z,
                             const RankData& rank_data, std::vector<double>& rhs) const {
    int ngll = rank_data.ngll;
    int n_node = ngll * ngll * ngll;

    int elem_idx = -1;
    for (int e = 0; e < rank_data.n_local_cell; ++e) {
        if (rank_data.local_cell_ids[e] == element_id) {
            elem_idx = e;
            break;
        }
    }
    if (elem_idx < 0)
        return;

    int node_idx = elem_idx * n_node + (gll_i * ngll + gll_j) * ngll + gll_k;
    const double weight = (wy == 0.0 && wz == 0.0) ? wx : (wx * wy * wz);
    rhs[node_idx * 3 + 0] += weight * force_x;
    rhs[node_idx * 3 + 1] += weight * force_y;
    rhs[node_idx * 3 + 2] += weight * force_z;
}

}  // namespace gf