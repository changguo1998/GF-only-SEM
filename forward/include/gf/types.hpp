#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <Eigen/Dense>

namespace gf {

// --- 3D vector and matrix types ---
using Vec3  = Eigen::Vector3d;
using Mat33 = Eigen::Matrix3d;
using Mat93 = Eigen::Matrix<double, 9, 3>;   // d(xi)/dx (9 partials as 9x3)

// --- Quadrature ---
struct GLLQuad {
    int N;
    std::vector<double> points;        // N+1 GLL nodes in [-1,1]
    std::vector<double> weights;       // N+1 quadrature weights
    std::vector<double> derivatives;   // (N+1)x(N+1) flattened derivative matrix
};

// --- Per-rank data (subset of partition_{r}.h5) ---
// Material, geometric quantities are stored as flat arrays indexed
// by local element + GLL node. No separate MaterialProperties/PMProfile structs --
// everything is precomputed at GLL nodes in partition_{r}.h5.
struct RankData {
    int n_local_elem = 0, n_ghost_elem = 0, n_total_elem = 0;
    int ngll = 0;   // N+1, extracted from partition array shapes

    std::vector<int64_t> local_element_ids;   // 1-based global element IDs
    std::vector<int64_t> ghost_element_ids;
    std::vector<int32_t> ghost_owners;         // which rank owns each ghost

    // Precomputed fields at GLL nodes (element-first, flattened: [n_elem * NGLL^3, ...])
    std::vector<double> coords;       // (x,y,z) per GLL node
    std::vector<double> jacobian;     // det(J) per GLL node
    std::vector<double> dxi_dx;       // d(xi_i)/dx_j per GLL node, 9 values
    std::vector<double> mass;         // lumped mass diagonal per GLL node
    std::vector<double> vp, vs, density;     // material at GLL nodes
    std::vector<double> pml_damping;   // PML damping, 0=interior

    // Precomputed exchange patterns (from /partition/rank_{r}/exchange/)
    // Face-pair send/recv lists per neighbor
    std::vector<int32_t> neighbors;

    // Exchange patterns for MPI halo: per-neighbor send/recv DOF index lists.
    // send_dof_indices[i]: local DOF index to send to neighbor i
    // recv_dof_indices[i]: local DOF index to receive from neighbor i
    struct ExchangePattern {
        int neighbor_rank;
        std::vector<int> send_dof_indices;   // local DOF indices to send
        std::vector<int> recv_dof_indices;   // local DOF indices to receive into (ghost DOFs)
    };
    std::vector<ExchangePattern> exchange_patterns;
};

// --- Time stepping ---
struct NewmarkParams {
    double beta  = 0.0;    // beta=0 for explicit central difference
    double gamma = 0.5;    // gamma=1/2
    double dt    = 0.0;
};

// --- Simulation configuration (from config.h5) ---
struct ConfigData {
    std::string title;
    int polynomial_order = 0;
    double dt = 0.0;
    int nsteps = 0;
    double cfl_safety = 1.0;
    int checkpoint_interval = 0;
    std::string checkpoint_precision = "float64";

    // Domain bounds
    double xmin = 0.0, xmax = 0.0;
    double ymin = 0.0, ymax = 0.0;
    double zmin = 0.0, zmax = 0.0;

    // Source
    std::vector<double> stf_t;
    std::vector<double> stf_values;
    double source_x = 0.0, source_y = 0.0, source_z = 0.0;
};

} // namespace gf