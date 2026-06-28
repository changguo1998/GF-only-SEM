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
#include "gf/logger.hpp"
#include "gf/pml.hpp"
#include "gf/record.hpp"
#include "gf/restart.hpp"
#include "gf/types.hpp"

namespace gf {

namespace {

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

int run_forward(const std::string& direction, bool resume_mode) {
    // All paths relative to CWD
    std::string config_path = "config.h5";
    std::string partition_dir = "partitions";
    std::string output_dir = "wavefields";
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    Logger logger(direction, rank);
    logger.info(std::to_string(nprocs) + " MPI ranks, direction=" + direction);

    try {
        // === Read config (all ranks read same file) ===
        auto t_io = std::chrono::steady_clock::now();
        ConfigData cfg = read_config(config_path);
        int ngll = cfg.polynomial_order + 1;
        auto io_elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_io).count();
        logger.debug("  config read: " + std::to_string(io_elapsed) + "s");
        logger.debug("  polynomial_order=" + std::to_string(cfg.polynomial_order) +
                     " ngll=" + std::to_string(ngll));
        logger.debug("  solver_dt=" + std::to_string(cfg.solver_dt) +
                     " nsteps=" + std::to_string(cfg.nsteps));
        logger.debug("  snapshot_stride=" + std::to_string(cfg.snapshot_stride) +
                     " precision=" + cfg.snapshot_precision);

        // === Read partition for this rank ===
        t_io = std::chrono::steady_clock::now();
        std::string partition_path = partition_dir + "/partition_" + std::to_string(rank) + ".h5";
        RankData part = read_partition(partition_path, rank);
        io_elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_io).count();
        logger.debug("  partition read: " + std::to_string(io_elapsed) + "s");

        int n_local = part.n_local_elem;
        int n_node = ngll * ngll * ngll;
        int n_local_dof = n_local * n_node * 3;
        logger.info("  n_local=" + std::to_string(n_local) + " n_gll_per_elem=" +
                    std::to_string(n_node) + " dofs=" + std::to_string(n_local_dof));

        // Memory estimate (8 bytes per double)
        size_t mem_bytes = n_local_dof * 4 * 8;  // u, v, a, r
        mem_bytes += n_local * n_node * 6 * 8;   // strain (allocated per snapshot)
        double mem_mb = static_cast<double>(mem_bytes) / (1024.0 * 1024.0);
        logger.debug("  est memory (state): " + std::to_string(mem_mb) + " MB");

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
        RecordWriter record(output_dir, direction, rank, part.recording, ngll, comp_cfg,
                            use_float32, cfg.record_depth_max_m, cfg.record_depth_actual_m);
        logger.debug("  record vertices: " + std::to_string(record.n_vertices()));

        // === Locate source element ===
        // Search for element containing source; for now use element 0
        int src_elem = 0;
        int src_node = 0;  // target GLL node within element

        // === Newmark parameters ===
        double beta = 0.0;  // explicit central difference
        double gamma = 0.5;
        double dt = cfg.solver_dt;

        // === Timing ===
        auto t_start = std::chrono::steady_clock::now();

        int n_snapshots = cfg.nsteps / cfg.snapshot_stride;
        logger.info("  snapshots: " + std::to_string(n_snapshots) +
                    " (stride=" + std::to_string(cfg.snapshot_stride) + ")" +
                    "  total sim time: " + std::to_string(cfg.nsteps * cfg.solver_dt) + " s");

        // === Initialize restart writer ===
        int restart_stride = 0;
        if (cfg.restart_dt_s > 0.0 && cfg.solver_dt > 0.0) {
            restart_stride = static_cast<int>(std::round(cfg.restart_dt_s / cfg.solver_dt));
            if (restart_stride < 1)
                restart_stride = 1;
        } else if (cfg.restart_stride > 0) {
            restart_stride = cfg.restart_stride;
        }
        bool do_restart = (restart_stride > 0);
        RestartWriter restart_writer(output_dir, direction, rank, n_local, ngll);
        if (do_restart) {
            logger.info("  restart stride: " + std::to_string(restart_stride));
        } else {
            logger.debug("  restart: disabled");
        }

        // === Main time loop ===
        int start_step = 0;
        if (resume_mode) {
            try {
                RestartState rs = read_restart(output_dir, direction, rank);
                if (!rs.displacement.empty() && rs.step > 0 && rs.step < cfg.nsteps) {
                    u = std::move(rs.displacement);
                    v = std::move(rs.velocity);
                    a = std::move(rs.acceleration);
                    if (!rs.pml_damping.empty()) {
                        part.pml_damping = std::move(rs.pml_damping);
                    }
                    start_step = rs.step + 1;
                    logger.info("  resumed at step " + std::to_string(start_step) +
                                " (time_s=" + std::to_string(rs.time_s) + ")");
                }
            } catch (const std::exception& e) {
                logger.error(std::string("  resume failed: ") + e.what() +
                             " — starting from scratch");
            }
        }
        for (int step = start_step; step < cfg.nsteps; ++step) {
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

            // --- Write restart (every restart_stride solver steps) ---
            if (do_restart && step > 0 && step % restart_stride == 0) {
                restart_writer.write(step, step * dt, u, v, a, part.pml_damping);
            }

            // --- Write snapshot (every snapshot_stride solver steps) ---
            if (cfg.snapshot_stride > 0 && step % cfg.snapshot_stride == 0) {
                // Compute strain: if recording map is active, compute only at the
                // specific GLL corner nodes needed (fast path). Otherwise fall back
                // to full-volume GLL strain computation.
                std::vector<double> rec_strain;
                bool has_recording =
                    part.recording.has_recording && part.recording.vertex_ids.size() > 0;

                if (has_recording) {
                    size_t nv = part.recording.vertex_ids.size();
                    rec_strain.resize(nv * 6, 0.0);
                    for (size_t vi = 0; vi < nv; ++vi) {
                        int elem = part.recording.src_elem_local[vi];
                        int corner = part.recording.src_corner[vi];

                        // Corner → GLL (i, j, k) indices
                        int ci = (corner & 1) ? (ngll - 1) : 0;
                        int cj = (corner & 2) ? (ngll - 1) : 0;
                        int ck = (corner & 4) ? (ngll - 1) : 0;

                        const int cn = (ci * ngll + cj) * ngll + ck;
                        const double* dd = &part.dxi_dx[elem * n_node * 9 + cn * 9];
                        const double* uu = &u[elem * n_node * 3 + cn * 3];

                        // Reference gradient at this GLL node
                        double dudxi[3] = {0.0, 0.0, 0.0};
                        double dudeta[3] = {0.0, 0.0, 0.0};
                        double dudzeta[3] = {0.0, 0.0, 0.0};
                        for (int s = 0; s < ngll; ++s) {
                            double Di_s = D_mat[ci * ngll + s];
                            double Dj_s = D_mat[cj * ngll + s];
                            double Dk_s = D_mat[ck * ngll + s];
                            int n_sjk = (s * ngll + cj) * ngll + ck;
                            int n_isk = (ci * ngll + s) * ngll + ck;
                            int n_ijs = (ci * ngll + cj) * ngll + s;
                            for (int d = 0; d < 3; ++d) {
                                dudxi[d] += Di_s * uu[3 * n_sjk + d];
                                dudeta[d] += Dj_s * uu[3 * n_isk + d];
                                dudzeta[d] += Dk_s * uu[3 * n_ijs + d];
                            }
                        }

                        // Transform to physical gradient
                        double du_dx[3][3];
                        for (int comp = 0; comp < 3; ++comp) {
                            du_dx[comp][0] =
                                dudxi[comp] * dd[0] + dudeta[comp] * dd[1] + dudzeta[comp] * dd[2];
                            du_dx[comp][1] =
                                dudxi[comp] * dd[3] + dudeta[comp] * dd[4] + dudzeta[comp] * dd[5];
                            du_dx[comp][2] =
                                dudxi[comp] * dd[6] + dudeta[comp] * dd[7] + dudzeta[comp] * dd[8];
                        }

                        // Symmetric strain (Voigt order)
                        double* out = &rec_strain[vi * 6];
                        out[0] = du_dx[0][0];                        // εxx
                        out[1] = du_dx[1][1];                        // εyy
                        out[2] = du_dx[2][2];                        // εzz
                        out[3] = 0.5 * (du_dx[0][1] + du_dx[1][0]);  // εxy
                        out[4] = 0.5 * (du_dx[0][2] + du_dx[2][0]);  // εxz
                        out[5] = 0.5 * (du_dx[1][2] + du_dx[2][1]);  // εyz
                    }
                } else {
                    // Full GLL strain computation (no recording map)
                    rec_strain.resize(n_local * n_node * 6, 0.0);
                    for (int e = 0; e < n_local; ++e) {
                        for (int i = 0; i < ngll; ++i) {
                            for (int j = 0; j < ngll; ++j) {
                                for (int k = 0; k < ngll; ++k) {
                                    const int n = (i * ngll + j) * ngll + k;
                                    const double* dd = &part.dxi_dx[e * n_node * 9 + n * 9];
                                    const double* uu = &u[e * n_node * 3 + n * 3];
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
                                    double du_dx[3][3];
                                    for (int comp = 0; comp < 3; ++comp) {
                                        du_dx[comp][0] = dudxi[comp] * dd[0] +
                                                         dudeta[comp] * dd[1] +
                                                         dudzeta[comp] * dd[2];
                                        du_dx[comp][1] = dudxi[comp] * dd[3] +
                                                         dudeta[comp] * dd[4] +
                                                         dudzeta[comp] * dd[5];
                                        du_dx[comp][2] = dudxi[comp] * dd[6] +
                                                         dudeta[comp] * dd[7] +
                                                         dudzeta[comp] * dd[8];
                                    }
                                    int strain_offset = e * n_node * 6 + n * 6;
                                    rec_strain[strain_offset + 0] = du_dx[0][0];
                                    rec_strain[strain_offset + 1] = du_dx[1][1];
                                    rec_strain[strain_offset + 2] = du_dx[2][2];
                                    rec_strain[strain_offset + 3] =
                                        0.5 * (du_dx[0][1] + du_dx[1][0]);
                                    rec_strain[strain_offset + 4] =
                                        0.5 * (du_dx[0][2] + du_dx[2][0]);
                                    rec_strain[strain_offset + 5] =
                                        0.5 * (du_dx[1][2] + du_dx[2][1]);
                                }
                            }
                        }
                    }
                }

                record.write_step(step, rec_strain.data());

                // --- Progress report ---
                {
                    auto t_now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                    double eta = (step + 1 < cfg.nsteps)
                                     ? elapsed * (cfg.nsteps - step - 1) / (step + 1)
                                     : 0.0;

                    std::ostringstream prog;
                    prog << std::setw(12) << std::left
                         << (std::to_string(step + 1) + "/" + std::to_string(cfg.nsteps))
                         << " elapsed=" << std::fixed << std::setprecision(1) << elapsed
                         << "s eta=" << eta << "s";
                    logger.raw(prog.str());
                }
            }
        }

        // === Finalize ===
        record.close();
        restart_writer.close();

        auto t_end = std::chrono::steady_clock::now();
        double total_elapsed = std::chrono::duration<double>(t_end - t_start).count();
        logger.info("simulation complete, " + std::to_string(cfg.nsteps) + " steps in " +
                    std::to_string(total_elapsed) + "s");
    } catch (const std::exception& e) {
        logger.error(std::string("Error: ") + e.what());
        return 1;
    }

    return 0;
}

}  // namespace gf