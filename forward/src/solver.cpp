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
inline void newmark_predict(double solver_dt, double beta, const std::vector<double>& displacement,
                            const std::vector<double>& velocity,
                            const std::vector<double>& acceleration,
                            std::vector<double>& displacement_tilde) {
    for (size_t i = 0; i < displacement.size(); ++i) {
        displacement_tilde[i] =
            displacement[i] + solver_dt * velocity[i] +
            (0.5 * solver_dt * solver_dt * (1.0 - 2.0 * beta)) * acceleration[i];
    }
}

// Newmark correct: a_new = M⁻¹·r, v += dt·γ·a_new, u += dt·v + dt²/2·a_new
inline void newmark_correct(double solver_dt, double /*beta*/, double gamma,
                            const std::vector<double>& mass, std::vector<double>& displacement,
                            std::vector<double>& velocity, std::vector<double>& acceleration,
                            std::vector<double>& residual) {
    for (size_t i = 0; i < residual.size(); ++i) {
        double a_new = residual[i] / mass[i / 3];  // same mass for all 3 directions
        displacement[i] += solver_dt * velocity[i] + 0.5 * solver_dt * solver_dt * a_new;
        velocity[i] += solver_dt * gamma * a_new;
        acceleration[i] = a_new;
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
        size_t mem_bytes = n_local_dof * 4 * 8;  // displacement, velocity, acceleration, residual
        mem_bytes += n_local * n_node * 6 * 8;   // strain (allocated per snapshot)
        double mem_mb = static_cast<double>(mem_bytes) / (1024.0 * 1024.0);
        logger.debug("  est memory (state): " + std::to_string(mem_mb) + " MB");

        // === GLL quadrature ===
        std::vector<double> gll_pts = gll_nodes(cfg.polynomial_order);
        std::vector<double> gll_wts = gll_weights(cfg.polynomial_order, gll_pts);
        std::vector<double> D_mat = gll_derivative_matrix(cfg.polynomial_order, gll_pts);

        // === Allocate displacement, velocity, acceleration ===
        // Flat arrays: [n_local_dof] for owned elements

        std::vector<double> displacement(n_local_dof, 0.0);
        std::vector<double> velocity(n_local_dof, 0.0);
        std::vector<double> acceleration(n_local_dof, 0.0);
        std::vector<double> residual(n_local_dof, 0.0);            // residual
        std::vector<double> displacement_tilde(n_local_dof, 0.0);  // predicted displacement

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
        double solver_dt = cfg.solver_dt;

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
                    displacement = std::move(rs.displacement);
                    velocity = std::move(rs.velocity);
                    acceleration = std::move(rs.acceleration);
                    if (!rs.pml_damping.empty()) {
                        part.pml_damping = std::move(rs.pml_damping);
                    }
                    start_step = rs.step + 1;
                    logger.info("  resumed at step " + std::to_string(start_step) +
                                " (time_s=" + std::to_string(rs.time_s) + ")");
                }
            } catch (const std::exception& ex) {
                logger.error(std::string("  resume failed: ") + ex.what() +
                             " — starting from scratch");
            }
        }
        for (int step = start_step; step < cfg.nsteps; ++step) {
            // --- Newmark predict ---
            newmark_predict(solver_dt, beta, displacement, velocity, acceleration,
                            displacement_tilde);

            // --- Zero residual ---
            std::fill(residual.begin(), residual.end(), 0.0);

            // --- Element residual computation (matrix-free) ---
            for (int elem = 0; elem < n_local; ++elem) {
                const double* elem_dxi_dx = part.dxi_dx.data() + elem * n_node * 9;
                const double* elem_jac = part.jacobian.data() + elem * n_node;
                const double* elem_vp = part.vp.data() + elem * n_node;
                const double* elem_vs = part.vs.data() + elem * n_node;
                const double* elem_rho = part.density.data() + elem * n_node;
                const double* elem_u = displacement_tilde.data() + elem * n_node * 3;
                double* elem_r = residual.data() + elem * n_node * 3;

                compute_element_residual(elem_dxi_dx, elem_jac, elem_vp, elem_vs, elem_rho,
                                         D_mat.data(), gll_wts.data(), ngll, elem_u, elem_r);
            }

            // --- PML damping ---
            apply_pml_damping(part.pml_damping, displacement_tilde, velocity,
                              static_cast<int>(velocity.size()));

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
                    residual[dof_idx] += stf_val;
                }
            }

            // --- MPI halo exchange (CG-SEM assembly) ---
            // Exchange residual r at shared interface nodes so that contributions
            // from neighbor ranks are summed (accumulate, not overwrite).
            // After this, r has all contributions at shared nodes across all ranks.
            exchange_halo(exchange_patterns, residual, 3);

            // --- Newmark correct ---
            newmark_correct(solver_dt, beta, gamma, part.mass, displacement, velocity,
                            acceleration, residual);

            // --- Write restart (every restart_stride solver steps) ---
            if (do_restart && step > 0 && step % restart_stride == 0) {
                restart_writer.write(step, step * solver_dt, displacement, velocity, acceleration,
                                     part.pml_damping);
            }

            // --- Write snapshot (every snapshot_stride solver steps) ---
            if (cfg.snapshot_stride > 0 && step % cfg.snapshot_stride == 0) {
                // Compute strain: in normal recording mode, write only shallow
                // mesh-vertex records. Some ranks legitimately have zero recorded
                // vertices; they must write empty record files, not full-GLL data.
                // Full-volume GLL strain is only for legacy runs without recording.
                std::vector<double> rec_strain;
                bool recording_mode = cfg.record_depth_max_m > 0.0;
                bool has_recording =
                    part.recording.has_recording && !part.recording.vertex_ids.empty();

                if (has_recording) {
                    size_t n_vertices = part.recording.vertex_ids.size();
                    rec_strain.resize(n_vertices * 6, 0.0);
                    for (size_t vertex_idx = 0; vertex_idx < n_vertices; ++vertex_idx) {
                        int elem = part.recording.src_elem_local[vertex_idx];
                        int corner = part.recording.src_corner[vertex_idx];

                        // Corner → GLL (i, j, k) indices
                        int corner_i = (corner & 1) ? (ngll - 1) : 0;
                        int corner_j = (corner & 2) ? (ngll - 1) : 0;
                        int corner_k = (corner & 4) ? (ngll - 1) : 0;

                        const int corner_node = (corner_i * ngll + corner_j) * ngll + corner_k;
                        const double* dxi_dx_ptr =
                            &part.dxi_dx[elem * n_node * 9 + corner_node * 9];
                        const double* disp_ptr =
                            &displacement[elem * n_node * 3 + corner_node * 3];

                        // Reference gradient at this GLL node
                        double dudxi[3] = {0.0, 0.0, 0.0};
                        double dudeta[3] = {0.0, 0.0, 0.0};
                        double dudzeta[3] = {0.0, 0.0, 0.0};
                        for (int s = 0; s < ngll; ++s) {
                            double Di_s = D_mat[corner_i * ngll + s];
                            double Dj_s = D_mat[corner_j * ngll + s];
                            double Dk_s = D_mat[corner_k * ngll + s];
                            int node_sjk = (s * ngll + corner_j) * ngll + corner_k;
                            int node_isk = (corner_i * ngll + s) * ngll + corner_k;
                            int node_ijs = (corner_i * ngll + corner_j) * ngll + s;
                            for (int d = 0; d < 3; ++d) {
                                dudxi[d] += Di_s * disp_ptr[3 * node_sjk + d];
                                dudeta[d] += Dj_s * disp_ptr[3 * node_isk + d];
                                dudzeta[d] += Dk_s * disp_ptr[3 * node_ijs + d];
                            }
                        }

                        // Transform to physical gradient
                        double du_dx[3][3];
                        for (int component = 0; component < 3; ++component) {
                            du_dx[component][0] = dudxi[component] * dxi_dx_ptr[0] +
                                                  dudeta[component] * dxi_dx_ptr[1] +
                                                  dudzeta[component] * dxi_dx_ptr[2];
                            du_dx[component][1] = dudxi[component] * dxi_dx_ptr[3] +
                                                  dudeta[component] * dxi_dx_ptr[4] +
                                                  dudzeta[component] * dxi_dx_ptr[5];
                            du_dx[component][2] = dudxi[component] * dxi_dx_ptr[6] +
                                                  dudeta[component] * dxi_dx_ptr[7] +
                                                  dudzeta[component] * dxi_dx_ptr[8];
                        }

                        // Symmetric strain (Voigt order)
                        double* out = &rec_strain[vertex_idx * 6];
                        out[0] = du_dx[0][0];                        // εxx
                        out[1] = du_dx[1][1];                        // εyy
                        out[2] = du_dx[2][2];                        // εzz
                        out[3] = 0.5 * (du_dx[0][1] + du_dx[1][0]);  // εxy
                        out[4] = 0.5 * (du_dx[0][2] + du_dx[2][0]);  // εxz
                        out[5] = 0.5 * (du_dx[1][2] + du_dx[2][1]);  // εyz
                    }
                } else if (!recording_mode) {
                    // Full GLL strain computation (no recording map)
                    rec_strain.resize(n_local * n_node * 6, 0.0);
                    for (int elem = 0; elem < n_local; ++elem) {
                        for (int i = 0; i < ngll; ++i) {
                            for (int j = 0; j < ngll; ++j) {
                                for (int k = 0; k < ngll; ++k) {
                                    const int node_idx = (i * ngll + j) * ngll + k;
                                    const double* dxi_dx_ptr =
                                        &part.dxi_dx[elem * n_node * 9 + node_idx * 9];
                                    const double* disp_ptr =
                                        &displacement[elem * n_node * 3 + node_idx * 3];
                                    double dudxi[3] = {0.0, 0.0, 0.0};
                                    double dudeta[3] = {0.0, 0.0, 0.0};
                                    double dudzeta[3] = {0.0, 0.0, 0.0};
                                    for (int s = 0; s < ngll; ++s) {
                                        double Di_s = D_mat[i * ngll + s];
                                        double Dj_s = D_mat[j * ngll + s];
                                        double Dk_s = D_mat[k * ngll + s];
                                        int node_sjk = (s * ngll + j) * ngll + k;
                                        int node_isk = (i * ngll + s) * ngll + k;
                                        int node_ijs = (i * ngll + j) * ngll + s;
                                        for (int d = 0; d < 3; ++d) {
                                            dudxi[d] += Di_s * disp_ptr[3 * node_sjk + d];
                                            dudeta[d] += Dj_s * disp_ptr[3 * node_isk + d];
                                            dudzeta[d] += Dk_s * disp_ptr[3 * node_ijs + d];
                                        }
                                    }
                                    double du_dx[3][3];
                                    for (int component = 0; component < 3; ++component) {
                                        du_dx[component][0] = dudxi[component] * dxi_dx_ptr[0] +
                                                              dudeta[component] * dxi_dx_ptr[1] +
                                                              dudzeta[component] * dxi_dx_ptr[2];
                                        du_dx[component][1] = dudxi[component] * dxi_dx_ptr[3] +
                                                              dudeta[component] * dxi_dx_ptr[4] +
                                                              dudzeta[component] * dxi_dx_ptr[5];
                                        du_dx[component][2] = dudxi[component] * dxi_dx_ptr[6] +
                                                              dudeta[component] * dxi_dx_ptr[7] +
                                                              dudzeta[component] * dxi_dx_ptr[8];
                                    }
                                    int strain_offset = elem * n_node * 6 + node_idx * 6;
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
                    int pct = (step + 1) * 100 / cfg.nsteps;
                    double eta = (step + 1 < cfg.nsteps)
                                     ? elapsed * (cfg.nsteps - step - 1) / (step + 1)
                                     : 0.0;

                    // Estimated finish time = now + eta
                    auto finish_tp =
                        std::chrono::system_clock::now() +
                        std::chrono::duration_cast<std::chrono::system_clock::duration>(
                            std::chrono::duration<double>(eta));
                    auto finish_t = std::chrono::system_clock::to_time_t(finish_tp);
                    char finish_buf[20];
                    std::strftime(finish_buf, sizeof(finish_buf), "%Y-%m-%d %H:%M:%S",
                                  std::localtime(&finish_t));

                    std::ostringstream prog;
                    prog << std::setw(7) << std::left
                         << (std::to_string(step + 1) + "/" + std::to_string(cfg.nsteps)) << " "
                         << std::setw(4) << pct << "%"
                         << " elapsed=" << std::fixed << std::setprecision(1) << std::setw(6)
                         << elapsed << "s  eta=" << std::setw(6) << eta << "s"
                         << "  finish~" << finish_buf;
                    logger.progress(prog.str());
                }
            }
        }

        logger.progress_done();

        // === Finalize ===
        record.close();
        restart_writer.close();

        auto t_end = std::chrono::steady_clock::now();
        double total_elapsed = std::chrono::duration<double>(t_end - t_start).count();
        logger.info("simulation complete, " + std::to_string(cfg.nsteps) + " steps in " +
                    std::to_string(total_elapsed) + "s");
    } catch (const std::exception& ex) {
        logger.error(std::string("Error: ") + ex.what());
        return 1;
    }

    return 0;
}

}  // namespace gf