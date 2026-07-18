#pragma once

#include <hdf5.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gf {

/// RestartState: full volume state read back from a restart file.
struct RestartState {
    int step = 0;
    double time_s = 0.0;
    std::vector<double> displacement;
    std::vector<double> velocity;
    std::vector<double> acceleration;
    std::vector<double> pml_damping;
    bool use_global_dof = false;  // if true, arrays are n_rank_node-sized
    int n_rank_node = 0;
};

/// Writes full-volume restart state (latest-only overwrite).
///
/// File layout: restart/{direction}/restart_{r}.h5
///   Attributes: rank, source_direction, step, time_s, ngll
///               use_global_dof (int, 0 or 1)
///               n_rank_node (int, only when use_global_dof=1)
///   Datasets (element-local): displacement, velocity, acceleration (float64, 5D:
///   [n_elem,NGLL,NGLL,NGLL,3])
///                                pml_damping (float64, 4D: [n_elem,NGLL,NGLL,NGLL])
///   Datasets (global DOF):     displacement, velocity, acceleration (float64, 1D: [n_rank_node *
///   3])
///                                pml_damping (float64, 1D: [n_rank_node])
///
/// Overwrites the same file each time — latest only.
class RestartWriter {
public:
    /// Open (or create) the restart file.
    ///
    /// \param output_dir       Top-level output directory
    /// \param source_direction Force direction string ("x", "y", or "z")
    /// \param rank             MPI rank number
    /// \param n_local_cell  Number of local elements on this rank
    /// \param ngll             Number of GLL points per axis (N+1)
    /// \param use_global_dof   If true, state vectors are n_rank_node-sized (flat 1D)
    /// \param n_rank_node      Number of unique nodes on this rank (only used when
    /// use_global_dof=true)
    RestartWriter(const std::string& output_dir, const std::string& source_direction, int rank,
                  int n_local_cell, int ngll, bool use_global_dof = false, int n_rank_node = 0);

    ~RestartWriter();

    /// Write restart state (overwrites previous restart file).
    ///
    /// \param step         Current solver step (0-based)
    /// \param time_s       Current simulation time (seconds)
    /// \param displacement State vector (element-local 5D or global 1D depending on mode)
    /// \param velocity     State vector
    /// \param acceleration State vector
    /// \param pml_damping  PML damping field
    void write(int step, double time_s, const std::vector<double>& displacement,
               const std::vector<double>& velocity, const std::vector<double>& acceleration,
               const std::vector<double>& pml_damping);

    /// Finalize and close the HDF5 file.
    void close();

    // Prevent copying
    RestartWriter(const RestartWriter&) = delete;
    RestartWriter& operator=(const RestartWriter&) = delete;

private:
    hid_t file_id_;
    std::string filepath_;
    int n_elem_local_;
    int ngll_;
    std::string source_direction_;
    bool use_global_dof_;
    int n_rank_node_;
};

/// Read a restart file and return the saved state.
///
/// \param output_dir       Top-level output directory
/// \param source_direction Force direction string
/// \param rank             MPI rank number
/// \return                 RestartState with all saved arrays
RestartState read_restart(const std::string& output_dir, const std::string& source_direction,
                          int rank);

}  // namespace gf