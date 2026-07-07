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
};

/// Writes full-volume restart state (latest-only overwrite).
///
/// File layout: restart/{direction}/restart_{r}.h5
///   Attributes: rank, source_direction, step, time_s, ngll
///   Datasets: displacement, velocity, acceleration (all float64)
///             pml_damping (float64)
///
/// Overwrites the same file each time — latest only.
class RestartWriter {
public:
    /// Open (or create) the restart file.
    ///
    /// \param output_dir       Top-level output directory
    /// \param source_direction Force direction string ("x", "y", or "z")
    /// \param rank             MPI rank number
    /// \param n_local_elem     Number of local elements on this rank
    /// \param ngll             Number of GLL points per axis (N+1)
    RestartWriter(const std::string& output_dir, const std::string& source_direction, int rank,
                  int n_local_elem, int ngll);

    ~RestartWriter();

    /// Write restart state (overwrites previous restart file).
    ///
    /// \param step         Current solver step (0-based)
    /// \param time_s       Current simulation time (seconds)
    /// \param displacement [n_elem_local * NGLL^3 * 3] displacement field
    /// \param velocity     [n_elem_local * NGLL^3 * 3] velocity field
    /// \param acceleration [n_elem_local * NGLL^3 * 3] acceleration field
    /// \param pml_damping  [n_elem_local * NGLL^3] PML damping field
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