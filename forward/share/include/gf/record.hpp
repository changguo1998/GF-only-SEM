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
/// Each call to write_step creates a field-only snapshot file:
///   wavefields/{direction}/record_{rank}_{step}.h5
///   /strain        [1, n_rec_cell, n_node, 6]  float32 or float64
///   /displacement  [1, n_rec_cell, n_node, 3]
///   /velocity      [1, n_rec_cell, n_node, 3]
///   /acceleration  [1, n_rec_cell, n_node, 3]
///   Attributes: rank, source_direction, source_partition_start, source_partition_count
class RecordWriter {
public:
    /// Store recording parameters for field-only snapshots.
    ///
    /// \param output_dir       Top-level output directory
    /// \param source_direction Force direction string ("x", "y", or "z")
    /// \param rank             MPI rank number
    /// \param rec_map          Recording map with GLL node IDs and cell indices
    /// \param ngll             Number of GLL points per axis (N+1)
    /// \param source_partition_start First source partition represented by this output rank
    /// \param source_partition_count Number of consecutive source partitions represented
    /// \param use_float32      If true, store fields as 32-bit float
    RecordWriter(const std::string& output_dir, const std::string& source_direction, int rank,
                 const RankData::RecordingMap& rec_map, int ngll, int source_partition_start,
                 int source_partition_count, bool use_float32 = false);
    ~RecordWriter();

    /// Write one snapshot to a standalone file: record_{rank}_{step}.h5.
    /// Creates the file, writes fields, then closes.
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
    hsize_t n_rec_cell_;
    int n_node_;  // ngll^3
    bool use_float32_;
    std::string output_dir_;
    std::string source_direction_;
    int rank_;
    int source_partition_start_;
    int source_partition_count_;
};

}  // namespace gf
