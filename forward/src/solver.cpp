// forward/src/solver.cpp
#include "gf/solver.hpp"

#include <hdf5.h>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gf/CompressionFilter.h"
#include "gf/assembly.hpp"
#include "gf/element.hpp"
#include "gf/exchange.hpp"
#include "gf/gll.hpp"
#include "gf/io.hpp"
#include "gf/pml.hpp"
#include "gf/record.hpp"
#include "gf/types.hpp"

namespace gf {

namespace {

// 1D flat index for (elem, i, j, k) in element-first layout
inline int elem_idx(int e, int i, int j, int k, int n_node, int stride = 1) {
    return (e * n_node + (i * stride + j) * stride + k);
}

// Newmark explicit predict step: ũ = u + dt·v + (dt²/2)·(1-2β)·a
inline void newmark_predict(double dt, double beta, const std::vector<double>& u,
                            const std::vector<double>& v, const std::vector<double>& a,
                            std::vector<double>& u_tilde) {
    for (size_t i = 0; i < u.size(); ++i) {
        u_tilde[i] = u[i] + dt * v[i] + (0.5 * dt * dt * (1.0 - 2.0 * beta)) * a[i];
    }
}

// Newmark correct: a_new = M⁻¹·r, v += dt·γ·a_new, u += dt·v + dt²/2·a_new
inline void newmark_correct(double dt, double /*beta*/, double gamma,
                            const std::vector<double>& mass, std::vector<double>& u,
                            std::vector<double>& v, std::vector<double>& a,
                            std::vector<double>& r) {
    for (size_t i = 0; i < r.size(); ++i) {
        double a_new = r[i] / mass[i / 3];  // same mass for all 3 directions
        u[i] += dt * v[i] + 0.5 * dt * dt * a_new;
        v[i] += dt * gamma * a_new;
        a[i] = a_new;
    }
}

}  // anonymous namespace

int run_forward(const std::string& direction) {
    // All paths relative to CWD
    std::string config_path = "config.h5";
    std::string partition_dir = "partitions";
    std::string output_dir = "wavefields";
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) {
        std::cout << "gf_solver: " << nprocs << " MPI ranks, direction=" << direction << std::endl;
    }

    try {
        // === Read config (all ranks read same file) ===
        ConfigData cfg = read_config(config_path);
        int ngll = cfg.polynomial_order + 1;

        // === Read partition for this rank ===
        std::string partition_path = partition_dir + "/partition_" + std::to_string(rank) + ".h5";
        RankData part = read_partition(partition_path, rank);

        int n_local = part.n_local_elem;
        int n_node = ngll * ngll * ngll;
        int n_local_dof = n_local * n_node * 3;

        // === GLL quadrature ===
        std::vector<double> gll_pts = gll_nodes(cfg.polynomial_order);
        std::vector<double> gll_wts = gll_weights(cfg.polynomial_order, gll_pts);
        std::vector<double> D_mat = gll_derivative_matrix(cfg.polynomial_order, gll_pts);

        // === Allocate displacement, velocity, acceleration ===
        // Flat arrays: [n_local_dof] for owned elements

        std::vector<double> u(n_local_dof, 0.0);
        std::vector<double> v(n_local_dof, 0.0);
        std::vector<double> a(n_local_dof, 0.0);
        std::vector<double> r(n_local_dof, 0.0);        // residual
        std::vector<double> u_tilde(n_local_dof, 0.0);  // predicted displacement

        // === Use precomputed exchange patterns from partition file ===
        // These contain send_dof and recv_dof indices for shared interface nodes.
        // For CG-SEM: send_dof == recv_dof (both point to local element interface DOFs,
        // since each rank accumulates neighbor contributions into its own local DOFs).
        const auto& exchange_patterns = part.exchange_patterns;

        // === Initialize record writer ===
        CompressionConfig comp_cfg;
        comp_cfg.method = CompressionMethod::None;
        if (cfg.snapshot_precision == "lzf")
            comp_cfg.method = CompressionMethod::LZF;
        else if (cfg.snapshot_precision == "zlib")
            comp_cfg.method = CompressionMethod::Zlib;

        bool use_float32 = (cfg.snapshot_precision == "float32");
        RecordWriter record(output_dir, direction, rank, n_local, part.local_element_ids.data(),
                            ngll, comp_cfg, use_float32);

        // === Locate source element ===
        // Search for element containing source; for now use element 0
        int src_elem = 0;
        int src_node = 0;  // target GLL node within element

        // === Newmark parameters ===
        double beta = 0.0;  // explicit central difference
        double gamma = 0.5;
        double dt = cfg.solver_dt;

        // === Timing (rank 0 only) ===
        auto t_start = std::chrono::steady_clock::now();

        if (rank == 0) {
            std::cout << "  snapshots: " << (cfg.nsteps / cfg.snapshot_stride)
                      << " (stride=" << cfg.snapshot_stride << ")"
                      << "  total sim time: " << (cfg.nsteps * cfg.solver_dt) << " s\n"
                      << "  " << std::setw(12) << std::left << "iter/total" << std::setw(16)
                      << "clock" << std::setw(14) << "elapsed" << std::setw(16) << "eta"
                      << "finish\n";
        }

        // === Main time loop ===
        for (int step = 0; step < cfg.nsteps; ++step) {
            // --- Newmark predict ---
            newmark_predict(dt, beta, u, v, a, u_tilde);

            // --- Zero residual ---
            std::fill(r.begin(), r.end(), 0.0);

            // --- Element residual computation (matrix-free) ---
            for (int e = 0; e < n_local; ++e) {
                const double* elem_dxi_dx = part.dxi_dx.data() + e * n_node * 9;
                const double* elem_jac = part.jacobian.data() + e * n_node;
                const double* elem_vp = part.vp.data() + e * n_node;
                const double* elem_vs = part.vs.data() + e * n_node;
                const double* elem_rho = part.density.data() + e * n_node;
                const double* elem_u = u_tilde.data() + e * n_node * 3;
                double* elem_r = r.data() + e * n_node * 3;

                compute_element_residual(elem_dxi_dx, elem_jac, elem_vp, elem_vs, elem_rho,
                                         D_mat.data(), gll_wts.data(), ngll, elem_u, elem_r);
            }

            // --- PML damping ---
            apply_pml_damping(part.pml_damping, u_tilde, v, static_cast<int>(v.size()));

            // --- Source injection ---
            // Distribute STF * source_weights at source_node
            {
                int dir = (direction == "x") ? 0 : ((direction == "y") ? 1 : 2);
                double stf_val = 0.0;
                if (step < static_cast<int>(cfg.stf_t.size())) {
                    stf_val = cfg.stf_values[step];
                }
                int dof_idx = src_elem * n_node * 3 + src_node * 3 + dir;
                if (dof_idx < n_local_dof) {
                    r[dof_idx] += stf_val;
                }
            }

            // --- MPI halo exchange (CG-SEM assembly) ---
            // Exchange residual r at shared interface nodes so that contributions
            // from neighbor ranks are summed (accumulate, not overwrite).
            // After this, r has all contributions at shared nodes across all ranks.
            exchange_halo(exchange_patterns, r, 3);

            // --- Newmark correct ---
            newmark_correct(dt, beta, gamma, part.mass, u, v, a, r);

            // --- Write snapshot (every snapshot_stride solver steps) ---
            if (cfg.snapshot_stride > 0 && step % cfg.snapshot_stride == 0) {
                // Compute strain at all GLL nodes (Voigt order: εxx, εyy, εzz, εxy, εxz, εyz)
                std::vector<double> strain(n_local * n_node * 6, 0.0);

                for (int e = 0; e < n_local; ++e) {
                    for (int i = 0; i < ngll; ++i) {
                        for (int j = 0; j < ngll; ++j) {
                            for (int k = 0; k < ngll; ++k) {
                                const int n = (i * ngll + j) * ngll + k;
                                const double* dd = &part.dxi_dx[e * n_node * 9 + n * 9];
                                const double* uu = &u[e * n_node * 3 + n * 3];

                                // Reference gradient: du_dxi
                                double dudxi[3] = {0.0, 0.0, 0.0};
                                double dudeta[3] = {0.0, 0.0, 0.0};
                                double dudzeta[3] = {0.0, 0.0, 0.0};

                                for (int s = 0; s < ngll; ++s) {
                                    double Di_s = D_mat[i * ngll + s];
                                    double Dj_s = D_mat[j * ngll + s];
                                    double Dk_s = D_mat[k * ngll + s];
                                    int n_sjk = (s * ngll + j) * ngll + k;
                                    int n_isk = (i * ngll + s) * ngll + k;
                                    int n_ijs = (i * ngll + j) * ngll + s;

                                    for (int d = 0; d < 3; ++d) {
                                        dudxi[d] += Di_s * uu[3 * n_sjk + d];
                                        dudeta[d] += Dj_s * uu[3 * n_isk + d];
                                        dudzeta[d] += Dk_s * uu[3 * n_ijs + d];
                                    }
                                }

                                // Transform to physical gradient
                                double du_dx[3][3];
                                for (int comp = 0; comp < 3; ++comp) {
                                    du_dx[comp][0] = dudxi[comp] * dd[0] + dudeta[comp] * dd[1] +
                                                     dudzeta[comp] * dd[2];
                                    du_dx[comp][1] = dudxi[comp] * dd[3] + dudeta[comp] * dd[4] +
                                                     dudzeta[comp] * dd[5];
                                    du_dx[comp][2] = dudxi[comp] * dd[6] + dudeta[comp] * dd[7] +
                                                     dudzeta[comp] * dd[8];
                                }

                                // Symmetric strain (Voigt order)
                                int strain_offset = e * n_node * 6 + n * 6;
                                strain[strain_offset + 0] = du_dx[0][0];  // εxx
                                strain[strain_offset + 1] = du_dx[1][1];  // εyy
                                strain[strain_offset + 2] = du_dx[2][2];  // εzz
                                strain[strain_offset + 3] =
                                    0.5 * (du_dx[0][1] + du_dx[1][0]);  // εxy
                                strain[strain_offset + 4] =
                                    0.5 * (du_dx[0][2] + du_dx[2][0]);  // εxz
                                strain[strain_offset + 5] =
                                    0.5 * (du_dx[1][2] + du_dx[2][1]);  // εyz
                            }
                        }
                    }
                }
                record.write_step(step, strain.data());

                // --- Progress report (rank 0 only) ---
                if (rank == 0) {
                    auto t_now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                    double eta = (step + 1 < cfg.nsteps)
                                     ? elapsed * (cfg.nsteps - step - 1) / (step + 1)
                                     : 0.0;

                    auto fmt_dur = [](double s) -> std::string {
                        int h = static_cast<int>(s) / 3600;
                        int m = (static_cast<int>(s) % 3600) / 60;
                        int sec = static_cast<int>(s) % 60;
                        std::ostringstream os;
                        os << std::setw(2) << std::setfill('0') << h << ":" << std::setw(2)
                           << std::setfill('0') << m << ":" << std::setw(2) << std::setfill('0')
                           << sec;
                        return os.str();
                    };

                    std::time_t now_t = std::time(nullptr);
                    std::time_t finish_t = now_t + static_cast<std::time_t>(eta);
                    char now_buf[32], fin_buf[32];
                    std::strftime(now_buf, sizeof(now_buf), "%m-%d %H:%M:%S",
                                  std::localtime(&now_t));
                    std::strftime(fin_buf, sizeof(fin_buf), "%m-%d %H:%M:%S",
                                  std::localtime(&finish_t));

                    std::cout << "  " << std::setw(12) << std::left
                              << (std::to_string(step + 1) + "/" + std::to_string(cfg.nsteps))
                              << std::setw(16) << now_buf << std::setw(14) << fmt_dur(elapsed)
                              << std::setw(16) << fmt_dur(eta) << fin_buf << std::endl;
                }
            }
        }

        // === Finalize ===
        record.close();

        if (rank == 0) {
            std::cout << "gf_solver: simulation complete, " << cfg.nsteps << " steps completed."
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

}  // namespace gf