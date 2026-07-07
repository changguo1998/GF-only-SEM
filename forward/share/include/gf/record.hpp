#pragma once

#include <hdf5.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gf/CompressionFilter.h"
#include "gf/types.hpp"

namespace gf {

/// Writes shallow mesh-vertex field records (strain + displacement/velocity/acceleration)
/// using the preprocess-built recording map.
///
/// Each call to write_step creates a standalone file:
///   wavefields/{direction}/record_{rank}_{step}.h5
///   /strain        [1, n_record_vertices, 6]  float32 or float64
///   /displacement  [1, n_record_vertices, 3]
///   /velocity      [1, n_record_vertices, 3]
///   /acceleration  [1, n_record_vertices, 3]
///   /vertex_ids [n_record_vertices] int64
///   Attributes: rank, source_direction, basis="mesh_vertices",
///               record_depth_max_m, record_depth_actual_m, excludes_pml
class RecordWriter {
public:
    /// Store recording parameters (file is created per write_step call).
    ///
    /// \param output_dir       Top-level output directory
    /// \param source_direction Force direction string ("x", "y", or "z")
    /// \param rank             MPI rank number
    /// \param rec_map          Recording map with vertex_ids and source element info
    /// \param ngll             Number of GLL points per axis (N+1)
    /// \param compression      Compression configuration (unused for per-step files)
    /// \param use_float32      If true, store fields as 32-bit float
    RecordWriter(const std::string& output_dir, const std::string& source_direction, int rank,
                 const RankData::RecordingMap& rec_map, int ngll, CompressionConfig compression,
                 bool use_float32 = false, double record_depth_max_m = 0.0,
                 double record_depth_actual_m = 0.0);

    ~RecordWriter();

    /// Write one snapshot to a standalone file: record_{rank}_{step}.h5.
    /// Creates the file, writes vertex_ids + fields, then closes.
    /// Pointers may be null to skip writing that field.
    ///
    /// \param step           Solver step number (used in filename)
    /// \param strain         Strain array [n_record_vertices * 6], Voigt order (or null)
    /// \param displacement   Displacement array [n_record_vertices * 3] (or null)
    /// \param velocity       Velocity array [n_record_vertices * 3] (or null)
    /// \param acceleration   Acceleration array [n_record_vertices * 3] (or null)
    void write_step(int step, const double* strain, const double* displacement = nullptr,
                    const double* velocity = nullptr, const double* acceleration = nullptr);

    /// No-op (each write_step creates and closes its own file).
    void close() {}

    /// Return number of recorded vertices.
    int n_vertices() const { return static_cast<int>(n_vertices_); }

    // Prevent copying
    RecordWriter(const RecordWriter&) = delete;
    RecordWriter& operator=(const RecordWriter&) = delete;

private:
    hid_t file_id_;
    hsize_t n_vertices_;
    int ngll_;
    bool use_float32_;
    std::string output_dir_;
    std::string source_direction_;
    int rank_;
    std::string basis_;
    bool excludes_pml_;
    double record_depth_max_m_;
    double record_depth_actual_m_;
    std::vector<int64_t> vertex_ids_;
};

}  // namespace gf