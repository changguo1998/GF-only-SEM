#pragma once

#include <hdf5.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gf/types.hpp"

namespace gf {

/// Writes element-local GLL field records.
/// DEBUG builds retain all local cells and dynamic fields; release builds
/// retain compact recording cells and strain only.
///
/// Each call to write_step creates a field-only snapshot file:
///   wavefields/{direction}/record_{rank}_{step}.h5
/// Production: strain [1, n_record_cell, n_node, 6].
/// DEBUG: strain plus three vector fields [1, n_local_cell, n_node, ncomp].
class RecordWriter {
public:
    /// Store snapshot parameters.
    ///
    /// \param output_dir       Top-level output directory
    /// \param source_direction Force direction string ("x", "y", or "z")
    /// \param rank             MPI rank number
#ifdef DEBUG
    /// \param n_local_cell     Number of local cells represented by this output rank
#else
    /// \param rec_map          Compact recording-cell map
#endif
    /// \param ngll             Number of GLL points per axis (N+1)
    /// \param source_partition_start First source partition represented by this output rank
    /// \param source_partition_count Number of consecutive source partitions represented
    /// \param use_float32      If true, store fields as 32-bit float
#ifdef DEBUG
    RecordWriter(const std::string& output_dir, const std::string& source_direction, int rank,
                 int n_local_cell, int ngll, int source_partition_start,
                 int source_partition_count, bool use_float32 = false);
#else
    RecordWriter(const std::string& output_dir, const std::string& source_direction, int rank,
                 const RankData::RecordingMap& rec_map, int ngll, int source_partition_start,
                 int source_partition_count, bool use_float32 = false);
#endif
    ~RecordWriter();

    /// Write one snapshot to a standalone file: record_{rank}_{step}.h5.
    /// Creates the file, writes fields, then closes.
    /// \param step           Solver step number (used in filename)
#ifdef DEBUG
    /// \param strain         Strain array [n_local_cell * n_node * 6], Voigt order
    /// \param displacement   Displacement array [n_local_cell * n_node * 3] (or null)
    /// \param velocity       Velocity array [n_local_cell * n_node * 3] (or null)
    /// \param acceleration   Acceleration array [n_local_cell * n_node * 3] (or null)
    void write_step(int step, const double* strain, const double* displacement = nullptr,
                    const double* velocity = nullptr, const double* acceleration = nullptr);
#else
    /// \param strain         Strain array [n_record_cell * n_node * 6], Voigt order
    void write_step(int step, const double* strain);
#endif

    /// No-op (each write_step creates and closes its own file).
    void close() {}

    /// Return number of cells stored in each snapshot.
    int n_record_cell() const { return static_cast<int>(n_record_cell_); }

    // Prevent copying
    RecordWriter(const RecordWriter&) = delete;
    RecordWriter& operator=(const RecordWriter&) = delete;

private:
    hsize_t n_record_cell_;
    int n_node_;  // ngll^3
    bool use_float32_;
    std::string output_dir_;
    std::string source_direction_;
    int rank_;
    int source_partition_start_;
    int source_partition_count_;
};

}  // namespace gf
