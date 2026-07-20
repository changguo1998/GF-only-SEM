#pragma once

#include <hdf5.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gf/types.hpp"

namespace gf {

/// Writes GLL-node field records (strain + displacement/velocity/acceleration)
/// using the preprocess-built recording map.
///
/// Each call to write_step creates a standalone file:
///   wavefields/{direction}/record_{rank}_{step}.h5
///   /strain        [1, n_rec_cell, n_node, 6]  float32 or float64
///   /displacement  [1, n_rec_cell, n_node, 3]
///   /velocity      [1, n_rec_cell, n_node, 3]
///   /acceleration  [1, n_rec_cell, n_node, 3]
///   /gll_node_ids        [n_unique_gll] int64
///   /gll_node_coords     [n_unique_gll, 3] float64
///   /cell_gll_node_index [n_rec_cell, n_node] int32
///   Attributes: rank, source_direction, basis="gll",
///               record_depth_max_m, record_depth_actual_m, excludes_pml, ngll
class RecordWriter {
public:
    /// Store recording parameters (file is created per write_step call).
    ///
    /// \param output_dir       Top-level output directory
    /// \param source_direction Force direction string ("x", "y", or "z")
    /// \param rank             MPI rank number
    /// \param rec_map          Recording map with GLL node IDs and cell indices
    /// \param ngll             Number of GLL points per axis (N+1)
    /// \param use_float32      If true, store fields as 32-bit float
    RecordWriter(const std::string& output_dir, const std::string& source_direction, int rank,
                 const RankData::RecordingMap& rec_map, int ngll, bool use_float32 = false,
                 double record_depth_max_m = 0.0, double record_depth_actual_m = 0.0);
    ~RecordWriter();

    /// Write one snapshot to a standalone file: record_{rank}_{step}.h5.
    /// Creates the file, writes mesh metadata + fields, then closes.
    /// Pointers may be null to skip writing that field.
    ///
    /// \param step           Solver step number (used in filename)
    /// \param strain         Strain array [n_rec_cell * n_node * 6], Voigt order (or null)
    /// \param displacement   Displacement array [n_rec_cell * n_node * 3] (or null)
    /// \param velocity       Velocity array [n_rec_cell * n_node * 3] (or null)
    /// \param acceleration   Acceleration array [n_rec_cell * n_node * 3] (or null)
    void write_step(int step, const double* strain, const double* displacement = nullptr,
                    const double* velocity = nullptr, const double* acceleration = nullptr);

    /// No-op (each write_step creates and closes its own file).
    void close() {}

    /// Return number of recording cells.
    int n_rec_cell() const { return static_cast<int>(n_rec_cell_); }

    // Prevent copying
    RecordWriter(const RecordWriter&) = delete;
    RecordWriter& operator=(const RecordWriter&) = delete;

private:
    hid_t file_id_;
    hsize_t n_rec_cell_;
    int n_node_;  // ngll^3
    int ngll_;
    hsize_t n_unique_gll_;
    bool use_float32_;
    std::string output_dir_;
    std::string source_direction_;
    int rank_;
    std::string basis_;
    bool excludes_pml_;
    double record_depth_max_m_;
    double record_depth_actual_m_;
    std::vector<int64_t> gll_node_ids_;
    std::vector<double> gll_node_coords_;
    std::vector<int32_t> cell_gll_node_index_;
};

}  // namespace gf
