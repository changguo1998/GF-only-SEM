// forward/share/src/solver.cpp
#include "gf/solver.hpp"

#ifndef GF_NO_MPI
#include <mpi.h>
#endif

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
#include "gf/backend.hpp"
#ifdef GF_WITH_CUDA
#include "gf/cuda_step.hpp"
#endif
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

// Newmark correct: a_new = M⁻¹·r; commit predictor and update velocity
inline void newmark_correct(double solver_dt, double beta, double gamma,
                            const std::vector<double>& mass, std::vector<double>& displacement,
                            std::vector<double>& velocity, std::vector<double>& acceleration,
                            std::vector<double>& residual) {
    for (size_t i = 0; i < residual.size(); ++i) {
        double m = mass[i / 3];
        if (m <= 0.0) {
            acceleration[i] = 0.0;
            continue;
        }
        double a_old = acceleration[i];
        double a_new = residual[i] / m;
        displacement[i] += solver_dt * velocity[i] +
                           solver_dt * solver_dt * ((0.5 - beta) * a_old + beta * a_new);
        velocity[i] += solver_dt * ((1.0 - gamma) * a_old + gamma * a_new);
        acceleration[i] = a_new;
    }
}

}  // anonymous namespace

int run_forward(const std::string& direction, bool resume_mode, int effective_nprocs) {
    // All paths relative to CWD
    std::string config_path = "config.h5";
    std::string partition_dir = "partitions";
    std::string output_dir = "wavefields";
    int rank = 0;
    int nprocs = 1;
#ifndef GF_NO_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif
    int eff_nprocs = nprocs;
    if (effective_nprocs > 0 && effective_nprocs < nprocs)
        eff_nprocs = effective_nprocs;

    Logger logger(direction, rank);
#ifdef GF_NO_MPI
    logger.info("single process, direction=" + direction);
#else
    if (eff_nprocs < nprocs) {
        logger.info(std::to_string(nprocs) + " MPI ranks (reduced to " +
                    std::to_string(eff_nprocs) + " effective), direction=" + direction);
    } else {
        logger.info(std::to_string(nprocs) + " MPI ranks, direction=" + direction);
    }
#endif

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

        // === Read partition(s) for this rank ===
        t_io = std::chrono::steady_clock::now();
        RankData part;
        if (eff_nprocs == 1) {
            // Single effective rank: merge all partitions into full domain
            part = read_partition_all(partition_dir);
            logger.debug("  merged " + std::to_string(part.n_local_element) +
                         " elements from all partitions");
        } else if (eff_nprocs < nprocs) {
            // Reduced effective ranks: block-distribute partitions
            int eff_rank = rank % eff_nprocs;
            part = read_partition_range(partition_dir, eff_rank, eff_nprocs);
            logger.debug("  effective rank " + std::to_string(eff_rank) + ": " +
                         std::to_string(part.n_local_element) + " elements");
        } else {
            std::string partition_path =
                partition_dir + "/partition_" + std::to_string(rank) + ".h5";
            part = read_partition(partition_path, rank);
        }
        io_elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_io).count();
        logger.debug("  partition read: " + std::to_string(io_elapsed) + "s");

        int n_local_element = part.n_local_element;
        int n_node = ngll * ngll * ngll;
        int n_local_element_dof = n_local_element * n_node * 3;  // element-local DOF for kernel temp arrays

        // Use global DOF numbering if local_element2rank_node is available (CG-SEM assembly).
        // Fall back to element-local DOF (backward compat) otherwise.
        bool use_global_dof = (part.n_rank_node > 0 && !part.local_element2rank_node.empty());
        int n_rank_dof = use_global_dof ? part.n_rank_node * 3 : n_local_element_dof;

        logger.info("  n_local_element=" + std::to_string(n_local_element) + " n_gll_per_elem=" +
                    std::to_string(n_node) +
                    (use_global_dof ? " n_rank_node=" + std::to_string(part.n_rank_node) :
                                       " dofs=" + std::to_string(n_local_element_dof)));

        // Memory estimate (8 bytes per double)
        size_t mem_bytes = n_rank_dof * 4 * 8;  // displacement, velocity, acceleration, residual
        mem_bytes += n_local_element_dof * 2 * 8;           // elem temp arrays (displacement, residual)
        double mem_mb = static_cast<double>(mem_bytes) / (1024.0 * 1024.0);
        logger.debug("  est memory (state): " + std::to_string(mem_mb) + " MB");

        // === GLL quadrature ===
        std::vector<double> gll_pts = gll_nodes(cfg.polynomial_order);
        std::vector<double> gll_wts = gll_weights(cfg.polynomial_order, gll_pts);
        std::vector<double> D_mat = gll_derivative_matrix(cfg.polynomial_order, gll_pts);

        // === Allocate state vectors ===
        // Global-sized arrays for CG-SEM (or element-local for backward compat).
        // Element-local temp arrays for the element kernel.

        std::vector<double> displacement(n_rank_dof, 0.0);
        std::vector<double> velocity(n_rank_dof, 0.0);
        std::vector<double> acceleration(n_rank_dof, 0.0);
        std::vector<double> residual(n_rank_dof, 0.0);            // global residual
        std::vector<double> displacement_tilde(n_rank_dof, 0.0);  // predicted displacement

        // Element-local temp arrays for kernel (always element-local)
        std::vector<double> local_element_displacement(n_local_element_dof, 0.0);
        std::vector<double> local_element_residual(n_local_element_dof, 0.0);
#ifdef GF_WITH_CUDA
        CudaDeviceState gpu_state;
#endif

        // === Use precomputed exchange patterns from partition file ===
        // These contain send_dof and recv_dof indices for shared interface nodes.
        // With local_element2rank_node-based global DOF (Phase 0.3), these are node_id*3+dir indices.
        // Without local_element2rank_node (legacy), these are element-local (elem*n_node+node)*3+dir.
        const auto& exchange_patterns = part.exchange_patterns;

        // === Assemble global mass and damping (one-time, at startup) ===
        // When local_element2rank_node is available, element-local mass/damping values are
        // scattered to global-sized arrays.  Mass accumulates (shared node
        // masses sum); damping assigns (all sharing elements have the same
        // damping profile on shared faces).
        std::vector<double> rank_node_mass(n_rank_dof / 3, 0.0);    // [n_rank_node] — node-sized
        std::vector<double> rank_node_damping(n_rank_dof / 3, 0.0); // [n_rank_node]
        if (use_global_dof) {
            for (int e = 0; e < n_local_element; ++e) {
                for (int n = 0; n < n_node; ++n) {
                    int node_id = part.local_element2rank_node[e * n_node + n];
                    rank_node_mass[node_id] += part.mass[e * n_node + n];
                    rank_node_damping[node_id] = part.pml_damping[e * n_node + n];
                }
            }

            if (!exchange_patterns.empty()) {
                std::vector<double> mass_exchange(n_rank_dof, 0.0);
                for (int node_id = 0; node_id < part.n_rank_node; ++node_id) {
                    double m = rank_node_mass[node_id];
                    mass_exchange[node_id * 3 + 0] = m;
                    mass_exchange[node_id * 3 + 1] = m;
                    mass_exchange[node_id * 3 + 2] = m;
                }
                exchange_halo(exchange_patterns, mass_exchange, 3);
                for (int node_id = 0; node_id < part.n_rank_node; ++node_id) {
                    rank_node_mass[node_id] = mass_exchange[node_id * 3 + 0];
                }
            }
        }

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

        // === Build source element lookup table ===
        // Map precomputed source element IDs (from config.h5) to local element indices.
        // -1 means the source element is not on this rank.
        std::vector<int> src_elem_to_local(cfg.n_src_elements, -1);
        for (int si = 0; si < cfg.n_src_elements; ++si) {
            for (int e = 0; e < n_local_element; ++e) {
                if (part.local_element_ids[e] == cfg.src_element_ids[si]) {
                    src_elem_to_local[si] = e;
                    break;
                }
            }
        }
        if (cfg.n_src_elements > 0) {
            logger.debug("  source elements: " + std::to_string(cfg.n_src_elements));
        }

#ifdef GF_WITH_CUDA
        // === Allocate GPU state (single-GPU native path) ===
        {
            gpu_state =
                cuda_allocate_state(n_local_element, ngll, part.mass, part.pml_damping, part.dxi_dx,
                                    part.jacobian, part.lambda_, part.mu_, D_mat.data(),
                                    gll_wts.data(), cfg, part.recording, n_local_element_dof,
                                    part.local_element2rank_node, part.n_rank_node, rank_node_mass, rank_node_damping);
            // Copy source element offsets to device
            std::vector<int> src_offsets(cfg.n_src_elements, -1);
            for (int si = 0; si < cfg.n_src_elements; ++si) {
                src_offsets[si] = src_elem_to_local[si];
            }
            GF_CUDA_CHECK(cudaMemcpy(gpu_state.d_src_elem_offsets, src_offsets.data(),
                                     cfg.n_src_elements * sizeof(int), cudaMemcpyHostToDevice));
        }
#endif

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
        RestartWriter restart_writer(output_dir, direction, rank, n_local_element, ngll,
                                      use_global_dof, part.n_rank_node);
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
#ifdef GF_WITH_CUDA
            // === GPU-native path (single-GPU, no MPI) ===
            // All state vectors live on device. Only copy for I/O.
            if (gpu_state.use_global_dof) {
                // ---- CG-SEM global assembly on GPU ----
                cuda_newmark_predict(gpu_state, solver_dt, beta);

                // Gather predicted displacement → element-local for kernel
                cuda_gather_predicted(gpu_state);

                cuda_zero_residual(gpu_state);
                cuda_launch_element_residual(gpu_state, ngll, n_local_element);
                cuda_pml_damping(gpu_state);
                {
                    int dir = (direction == "x") ? 0 : ((direction == "y") ? 1 : 2);
                    double stf_val = 0.0;
                    if (step < static_cast<int>(cfg.stf_t.size())) {
                        stf_val = cfg.stf_values[step];
                    }
                    if (stf_val != 0.0) {
                        cuda_source_injection(gpu_state, dir, stf_val, cfg.src_weights.data(),
                                              cfg.n_src_elements);
                    }
                }
                // Scatter element-local → global (atomicAdd at shared nodes)
                cuda_scatter_to_rank(gpu_state);

                // No MPI exchange (single process)
                cuda_newmark_correct(gpu_state, solver_dt, gamma);
            } else {
                // ---- Legacy element-local path ----
                cuda_newmark_predict(gpu_state, solver_dt, beta);
                cuda_zero_residual(gpu_state);
                cuda_launch_element_residual(gpu_state, ngll, n_local_element);
                cuda_pml_damping(gpu_state);
                {
                    int dir = (direction == "x") ? 0 : ((direction == "y") ? 1 : 2);
                    double stf_val = 0.0;
                    if (step < static_cast<int>(cfg.stf_t.size())) {
                        stf_val = cfg.stf_values[step];
                    }
                    if (stf_val != 0.0) {
                        cuda_source_injection(gpu_state, dir, stf_val, cfg.src_weights.data(),
                                              cfg.n_src_elements);
                    }
                }
                cuda_newmark_correct(gpu_state, solver_dt, gamma);
            }

            // --- Write restart (every restart_stride solver steps) ---
            if (do_restart && step > 0 && step % restart_stride == 0) {
                cuda_copy_state_to_host(gpu_state, displacement, velocity, acceleration);
                restart_writer.write(step, step * solver_dt, displacement, velocity, acceleration,
                                     part.pml_damping);
            }

            // --- Write snapshot (every snapshot_stride solver steps) ---
            if (cfg.snapshot_stride > 0 && step % cfg.snapshot_stride == 0) {
                std::vector<double> rec_strain;
                std::vector<double> rec_displacement;
                std::vector<double> rec_velocity;
                std::vector<double> rec_acceleration;
                bool recording_mode = cfg.record_depth_max_m > 0.0;
                bool has_recording =
                    part.recording.has_recording && !part.recording.vertex_ids.empty();

                if (has_recording) {
                    // For global DOF: gather displacement to element-local for strain
                    if (gpu_state.use_global_dof) {
                        cuda_gather_from_rank(gpu_state);
                    }
                    size_t n_vertices = part.recording.vertex_ids.size();
                    rec_strain.resize(n_vertices * 6, 0.0);
                    rec_displacement.resize(n_vertices * 3, 0.0);
                    rec_velocity.resize(n_vertices * 3, 0.0);
                    rec_acceleration.resize(n_vertices * 3, 0.0);

                    cuda_compute_strain(gpu_state, D_mat.data(), ngll, part.dxi_dx);
                    cuda_copy_strain_to_host(gpu_state, rec_strain.data());

                    // Copy full state from device to extract recorded vertex values
                    cuda_copy_state_to_host(gpu_state, displacement, velocity, acceleration);
                    for (size_t vertex_idx = 0; vertex_idx < n_vertices; ++vertex_idx) {
                        int elem = part.recording.src_elem_local[vertex_idx];
                        int corner = part.recording.src_corner[vertex_idx];
                        int corner_i = (corner & 1) ? (ngll - 1) : 0;
                        int corner_j = (corner & 2) ? (ngll - 1) : 0;
                        int corner_k = (corner & 4) ? (ngll - 1) : 0;
                        int corner_node = (corner_i * ngll + corner_j) * ngll + corner_k;
                        if (gpu_state.use_global_dof) {
                            int node_id = part.local_element2rank_node[elem * n_node + corner_node];
                            for (int d = 0; d < 3; ++d) {
                                rec_displacement[vertex_idx * 3 + d] = displacement[node_id * 3 + d];
                                rec_velocity[vertex_idx * 3 + d] = velocity[node_id * 3 + d];
                                rec_acceleration[vertex_idx * 3 + d] = acceleration[node_id * 3 + d];
                            }
                        } else {
                            int dof_base = (elem * n_node + corner_node) * 3;
                            for (int d = 0; d < 3; ++d) {
                                rec_displacement[vertex_idx * 3 + d] = displacement[dof_base + d];
                                rec_velocity[vertex_idx * 3 + d] = velocity[dof_base + d];
                                rec_acceleration[vertex_idx * 3 + d] = acceleration[dof_base + d];
                            }
                        }
                    }
                }
                record.write_step(step, rec_strain.data(), rec_displacement.data(),
                                  rec_velocity.data(), rec_acceleration.data());
            }
#else
            // --- CPU path ---
            if (use_global_dof) {
                // === CG-SEM global assembly path ===

                // 1. Newmark predictor (global arrays)
                newmark_predict(solver_dt, beta, displacement, velocity, acceleration,
                                displacement_tilde);

                // 2. Sync predicted displacement at shared interface nodes.
                //    Each rank's predictor uses its own (u,v,a) which may differ
                //    at shared nodes. Averaging u_tilde before the element kernel
                //    ensures both ranks use the same displacement → consistent
                //    residual → correct assembled acceleration.
                if (!exchange_patterns.empty()) {
                    std::vector<double> ut_avg(displacement_tilde);
                    exchange_halo(exchange_patterns, ut_avg, 3);
                    for (const auto& pat : exchange_patterns) {
                        for (int dof_idx : pat.recv_dof_indices) {
                            displacement_tilde[dof_idx] = 0.5 * ut_avg[dof_idx];
                        }
                    }
                }

                // 3. Gather predicted displacement → element-local for kernel
                gather_from_rank(displacement_tilde, part.local_element2rank_node, n_local_element, n_node,
                                   local_element_displacement);

                // 3. Zero element-local residual, compute element kernel
                std::fill(local_element_residual.begin(), local_element_residual.end(), 0.0);
                compute_element_residual<gf::ActiveBackend>(
                    n_local_element, part.dxi_dx.data(), part.jacobian.data(), part.lambda_.data(),
                    part.mu_.data(), D_mat.data(), gll_wts.data(), ngll, local_element_displacement.data(),
                    local_element_residual.data());

                // 4. PML damping on global velocity (direct — no gather/scatter)
                for (int node_id = 0; node_id < part.n_rank_node; ++node_id) {
                    double d = rank_node_damping[node_id];
                    if (d > 0.0) {
                        int base = node_id * 3;
                        velocity[base + 0] -= d * velocity[base + 0];
                        velocity[base + 1] -= d * velocity[base + 1];
                        velocity[base + 2] -= d * velocity[base + 2];
                    }
                }

                // 5. Source injection into element-local residual
                {
                    int dir = (direction == "x") ? 0 : ((direction == "y") ? 1 : 2);
                    double stf_val = 0.0;
                    if (step < static_cast<int>(cfg.stf_t.size())) {
                        stf_val = cfg.stf_values[step];
                    }
                    if (stf_val != 0.0) {
                        for (int si = 0; si < cfg.n_src_elements; ++si) {
                            int elem_idx = src_elem_to_local[si];
                            if (elem_idx < 0)
                                continue;
                            int weight_off = si * n_node;
                            int dof_base_elem = elem_idx * n_node * 3;
                            for (int k = 0; k < ngll; ++k) {
                                for (int j = 0; j < ngll; ++j) {
                                    for (int i = 0; i < ngll; ++i) {
                                        double w = cfg.src_weights[weight_off +
                                                                   (i * ngll + j) * ngll + k];
                                        if (w == 0.0)
                                            continue;
                                        int node_off = (i * ngll + j) * ngll + k;
                                        local_element_residual[dof_base_elem + node_off * 3 + dir] +=
                                            stf_val * w;
                                    }
                                }
                            }
                        }
                    }
                }

                // 6. Scatter element-local → global (accumulates at shared nodes)
                scatter_to_rank(local_element_residual, part.local_element2rank_node, n_local_element, n_node, residual);

                // 6. MPI halo exchange on global residual
                exchange_halo(exchange_patterns, residual, 3);

                // 7. Newmark corrector (global arrays, global mass)
                newmark_correct(solver_dt, beta, gamma, rank_node_mass, displacement, velocity,
                                acceleration, residual);

            } else {
                // === Legacy element-local path (backward compat) ===
                newmark_predict(solver_dt, beta, displacement, velocity, acceleration,
                                displacement_tilde);

                std::fill(residual.begin(), residual.end(), 0.0);

                compute_element_residual<gf::ActiveBackend>(
                    n_local_element, part.dxi_dx.data(), part.jacobian.data(), part.lambda_.data(),
                    part.mu_.data(), D_mat.data(), gll_wts.data(), ngll,
                    displacement_tilde.data(), residual.data());

                apply_pml_damping(part.pml_damping, displacement_tilde, velocity,
                                  static_cast<int>(velocity.size()));

                {
                    int dir = (direction == "x") ? 0 : ((direction == "y") ? 1 : 2);
                    double stf_val = 0.0;
                    if (step < static_cast<int>(cfg.stf_t.size())) {
                        stf_val = cfg.stf_values[step];
                    }
                    if (stf_val != 0.0) {
                        for (int si = 0; si < cfg.n_src_elements; ++si) {
                            int elem_idx = src_elem_to_local[si];
                            if (elem_idx < 0)
                                continue;
                            int weight_off = si * n_node;
                            int dof_base_elem = elem_idx * n_node * 3;
                            for (int k = 0; k < ngll; ++k) {
                                for (int j = 0; j < ngll; ++j) {
                                    for (int i = 0; i < ngll; ++i) {
                                        double w = cfg.src_weights[weight_off +
                                                                   (i * ngll + j) * ngll + k];
                                        if (w == 0.0)
                                            continue;
                                        int node_off = (i * ngll + j) * ngll + k;
                                        residual[dof_base_elem + node_off * 3 + dir] +=
                                            stf_val * w;
                                    }
                                }
                            }
                        }
                    }
                }
                exchange_halo(exchange_patterns, residual, 3);
                newmark_correct(solver_dt, beta, gamma, part.mass, displacement, velocity,
                                acceleration, residual);
            }

            // --- Write restart (every restart_stride solver steps) ---
            if (do_restart && step > 0 && step % restart_stride == 0) {
                restart_writer.write(step, step * solver_dt, displacement, velocity, acceleration,
                                     part.pml_damping);
            }

            // --- Write snapshot (every snapshot_stride solver steps) ---
            if (cfg.snapshot_stride > 0 && step % cfg.snapshot_stride == 0) {
                std::vector<double> rec_strain;
                std::vector<double> rec_displacement;
                std::vector<double> rec_velocity;
                std::vector<double> rec_acceleration;
                bool recording_mode = cfg.record_depth_max_m > 0.0;
                bool has_recording =
                    part.recording.has_recording && !part.recording.vertex_ids.empty();

                // Helper: extract recorded-vertex values from global or element-local DOF array
                auto extract_recorded = [&](const std::vector<double>& src, int ncomp,
                                            std::vector<double>& dst) {
                    if (!has_recording)
                        return;
                    size_t n_vertices = part.recording.vertex_ids.size();
                    dst.resize(n_vertices * ncomp, 0.0);
                    for (size_t vertex_idx = 0; vertex_idx < n_vertices; ++vertex_idx) {
                        int elem = part.recording.src_elem_local[vertex_idx];
                        int corner = part.recording.src_corner[vertex_idx];
                        int corner_i = (corner & 1) ? (ngll - 1) : 0;
                        int corner_j = (corner & 2) ? (ngll - 1) : 0;
                        int corner_k = (corner & 4) ? (ngll - 1) : 0;
                        int corner_node = (corner_i * ngll + corner_j) * ngll + corner_k;
                        if (use_global_dof) {
                            int node_id = part.local_element2rank_node[elem * n_node + corner_node];
                            for (int d = 0; d < ncomp; ++d) {
                                dst[vertex_idx * ncomp + d] = src[node_id * 3 + d];
                            }
                        } else {
                            int dof_base = (elem * n_node + corner_node) * 3;
                            for (int d = 0; d < ncomp; ++d) {
                                dst[vertex_idx * ncomp + d] = src[dof_base + d];
                            }
                        }
                    }
                };

                if (has_recording) {
                    // For global DOF: gather fresh displacement into element-local
                    // so strain derivative computation can access all GLL nodes.
                    if (use_global_dof) {
                        gather_from_rank(displacement, part.local_element2rank_node, n_local_element, n_node,
                                           local_element_displacement);
                    }
                    // Pointer to the displacement array used for strain computation
                    const double* strain_disp =
                        use_global_dof ? local_element_displacement.data() : displacement.data();

                    size_t n_vertices = part.recording.vertex_ids.size();
                    rec_strain.resize(n_vertices * 6, 0.0);
                    for (size_t vertex_idx = 0; vertex_idx < n_vertices; ++vertex_idx) {
                        int elem = part.recording.src_elem_local[vertex_idx];
                        int corner = part.recording.src_corner[vertex_idx];

                        int corner_i = (corner & 1) ? (ngll - 1) : 0;
                        int corner_j = (corner & 2) ? (ngll - 1) : 0;
                        int corner_k = (corner & 4) ? (ngll - 1) : 0;

                        const int corner_node = (corner_i * ngll + corner_j) * ngll + corner_k;
                        const double* dxi_dx_ptr =
                            &part.dxi_dx[elem * n_node * 9 + corner_node * 9];
                        const double* disp_ptr =
                            &strain_disp[elem * n_node * 3 + corner_node * 3];

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
                        out[0] = du_dx[0][0];                        // exx
                        out[1] = du_dx[1][1];                        // eyy
                        out[2] = du_dx[2][2];                        // ezz
                        out[3] = 0.5 * (du_dx[0][1] + du_dx[1][0]);  // exy
                        out[4] = 0.5 * (du_dx[0][2] + du_dx[2][0]);  // exz
                        out[5] = 0.5 * (du_dx[1][2] + du_dx[2][1]);  // eyz
                    }
                    // Extract displacement, velocity, acceleration at recorded vertices
                    extract_recorded(displacement, 3, rec_displacement);
                    extract_recorded(velocity, 3, rec_velocity);
                    extract_recorded(acceleration, 3, rec_acceleration);
                } else if (!recording_mode) {
                    // For global DOF: gather displacement before full-volume strain
                    if (use_global_dof) {
                        gather_from_rank(displacement, part.local_element2rank_node, n_local_element, n_node,
                                           local_element_displacement);
                    }
                    const double* strain_disp =
                        use_global_dof ? local_element_displacement.data() : displacement.data();

                    rec_strain.resize(n_local_element * n_node * 6, 0.0);
                    for (int elem = 0; elem < n_local_element; ++elem) {
                        for (int i = 0; i < ngll; ++i) {
                            for (int j = 0; j < ngll; ++j) {
                                for (int k = 0; k < ngll; ++k) {
                                    const int node_idx = (i * ngll + j) * ngll + k;
                                    const double* dxi_dx_ptr =
                                        &part.dxi_dx[elem * n_node * 9 + node_idx * 9];
                                    const double* disp_ptr =
                                        &strain_disp[elem * n_node * 3 + node_idx * 3];
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
                record.write_step(step, rec_strain.data(), rec_displacement.data(),
                                  rec_velocity.data(), rec_acceleration.data());
            }
#endif

            // --- Progress report (every log_stride steps, plus last) ---
            if ((step + 1) % cfg.log_stride == 0 || step == cfg.nsteps - 1) {
                auto t_now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                int pct = (step + 1) * 100 / cfg.nsteps;
                double eta =
                    (step + 1 < cfg.nsteps) ? elapsed * (cfg.nsteps - step - 1) / (step + 1) : 0.0;

                // Estimated finish time = now + eta
                auto finish_tp = std::chrono::system_clock::now() +
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

        logger.progress_done();

#ifdef GF_WITH_CUDA
        cuda_free_state(gpu_state);
#endif

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