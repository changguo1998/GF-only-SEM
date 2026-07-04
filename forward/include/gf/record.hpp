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
/// File layout: wavefields/{direction}/record_{rank}.h5
///   /strain        [n_snapshots, n_record_vertices, 6]  extendible (float32 or float64)
///   /displacement  [n_snapshots, n_record_vertices, 3]  extendible (float32 or float64)
///   /velocity      [n_snapshots, n_record_vertices, 3]  extendible (float32 or float64)
///   /acceleration  [n_snapshots, n_record_vertices, 3]  extendible (float32 or float64)
///   /vertex_ids [n_record_vertices] int64
///   Attributes: rank, source_direction, basis="mesh_vertices",
///               record_depth_max_m, record_depth_actual_m, excludes_pml
class RecordWriter {
public:
    /// Open (or create) the record file and set up dataset structure.
    ///
    /// \param output_dir       Top-level output directory
    /// \param source_direction Force direction string ("x", "y", or "z")
    /// \param rank             MPI rank number
    /// \param rec_map          Recording map with vertex_ids and source element info
    /// \param ngll             Number of GLL points per axis (N+1)
    /// \param compression      Compression configuration
    /// \param use_float32      If true, store fields as 32-bit float
    RecordWriter(const std::string& output_dir, const std::string& source_direction, int rank,
                 const RankData::RecordingMap& rec_map, int ngll, CompressionConfig compression,
                 bool use_float32 = false, double record_depth_max_m = 0.0,
                 double record_depth_actual_m = 0.0);

    ~RecordWriter();

    /// Write all fields for the current time step at recorded vertices.
    /// Extends the time dimension (dim 0) of each dataset by 1.
    /// Pointers may be null to skip writing that field.
    ///
    /// \param step           Time step number (0-based)
    /// \param strain         Strain array [n_record_vertices * 6], Voigt order (or null)
    /// \param displacement   Displacement array [n_record_vertices * 3] (or null)
    /// \param velocity       Velocity array [n_record_vertices * 3] (or null)
    /// \param acceleration   Acceleration array [n_record_vertices * 3] (or null)
    void write_step(int step, const double* strain, const double* displacement = nullptr,
                    const double* velocity = nullptr, const double* acceleration = nullptr);

    /// Finalize and close the HDF5 file.
    void close();

    /// Return number of recorded vertices.
    int n_vertices() const { return static_cast<int>(n_vertices_); }

    // Prevent copying
    RecordWriter(const RecordWriter&) = delete;
    RecordWriter& operator=(const RecordWriter&) = delete;

private:
    hid_t file_id_;
    hid_t strain_dset_;
    hid_t displacement_dset_;
    hid_t velocity_dset_;
    hid_t acceleration_dset_;
    hsize_t current_step_;
    hsize_t n_vertices_;
    int ngll_;
    bool use_float32_;
    std::string filepath_;
    std::string source_direction_;

    // Helper: create an extendible dataset with given component count
    hid_t create_field_dset(const std::string& name, int ncomp);
};

}  // namespace gf