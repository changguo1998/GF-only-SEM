#pragma once

#include <hdf5.h>

#include <cstdint>
#include <string>

#include "gf/CompressionFilter.h"

namespace gf {

/// Writes strain records to per-rank HDF5 files with extendible time dimension.
///
/// File layout: wavefields/{direction}/record_{rank}.h5
///   /strain [n_snapshots, n_elem_local, NGLL, NGLL, NGLL, 6]  extendible (float32 or float64)
///   /local_element_ids [n_elem_local] int64
///   Attributes: source_direction, ngll
///
/// Uses compress module for chunking and compression configuration.
class RecordWriter {
public:
    /// Open (or create) the record file and set up dataset structure.
    ///
    /// \param output_dir       Top-level output directory
    /// \param source_direction Force direction string ("x", "y", or "z")
    /// \param rank             MPI rank number
    /// \param n_local_elem     Number of local elements on this rank
    /// \param element_ids      Global 1-based element IDs [n_local_elem]
    /// \param ngll             Number of GLL points per axis (N+1)
    /// \param compression      Compression configuration
    /// \param use_float32      If true, store strain as 32-bit float
    RecordWriter(const std::string& output_dir, const std::string& source_direction, int rank,
                 int n_local_elem, const int64_t* element_ids, int ngll,
                 CompressionConfig compression, bool use_float32 = false);

    ~RecordWriter();

    /// Write strain values for the current time step.
    /// Extends the time dimension (dim 0) of the /strain dataset by 1.
    ///
    /// \param step    Time step number (0-based)
    /// \param strain  Strain array [n_local_elem * NGLL^3 * 6], Voigt order
    void write_step(int step, const double* strain);

    /// Finalize and close the HDF5 file.
    void close();

    // Prevent copying
    RecordWriter(const RecordWriter&) = delete;
    RecordWriter& operator=(const RecordWriter&) = delete;

private:
    hid_t file_id_;
    hid_t strain_dset_;
    hid_t strain_space_;
    hsize_t current_step_;
    hsize_t n_elem_local_;
    int ngll_;
    bool use_float32_;
    std::string filepath_;
    std::string source_direction_;
};

}  // namespace gf