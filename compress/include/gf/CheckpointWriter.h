// compress/include/gf/CheckpointWriter.h
#pragma once
#include "CompressionFilter.h"
#include "ChunkingStrategy.h"
#include "PrecisionPolicy.h"

#include <hdf5.h>
#include <string>
#include <vector>

namespace gf {

/// Configuration for a single checkpoint write.
struct CheckpointConfig {
    CompressionConfig compression{};
    bool use_float32 = false;  ///< Store as float32 (downcast from float64)
    int ngll = 5;              ///< Polynomial order + 1 (N+1)
};

/// Write the initial checkpoint file structure.
///
/// Creates record_r{rank}.h5 with:
///   - Attrs: rank, dt, checkpoint_interval, nsteps
///   - local_element_ids: fixed-size int64 dataset
///   - strain: extendible dataset [n_checkpoints, n_elem_local, NGLL, NGLL, NGLL, 6]
///            float32 default, dim 0 = H5S_UNLIMITED
///
/// Subsequent checkpoints extend dim 0 by 1 and write new time slice.
///
/// \param file_id       Open HDF5 file identifier
/// \param n_elem_local  Number of local elements (this MPI rank)
/// \param element_ids   Global 1-based element IDs for each local element
/// \param config        Compression, precision, and chunking settings
///
/// \return The strain dataset ID (caller should H5Dclose it)
inline hid_t write_checkpoint(
    hid_t file_id,
    hsize_t n_elem_local,
    const int64_t* element_ids,
    const CheckpointConfig& config)
{
    constexpr int ncomps = 6;

    // --- local_element_ids dataset (fixed) ---
    hsize_t elem_dims[1] = {n_elem_local};
    hid_t elem_space = H5Screate_simple(1, elem_dims, nullptr);
    hid_t elem_dset = H5Dcreate2(file_id, "local_element_ids",
                                  H5T_NATIVE_INT64, elem_space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(elem_dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, element_ids);
    H5Dclose(elem_dset);
    H5Sclose(elem_space);

    // --- strain dataset (extendible) ---
    // Shape: [n_checkpoints, n_elem_local, NGLL, NGLL, NGLL, 6]
    // Dim 0 (time) unlimited, others fixed
    hsize_t strain_dims[6] = {1, n_elem_local,
                              static_cast<hsize_t>(config.ngll),
                              static_cast<hsize_t>(config.ngll),
                              static_cast<hsize_t>(config.ngll),
                              ncomps};
    hsize_t max_dims[6] = {H5S_UNLIMITED, n_elem_local,
                           static_cast<hsize_t>(config.ngll),
                           static_cast<hsize_t>(config.ngll),
                           static_cast<hsize_t>(config.ngll),
                           ncomps};
    hid_t strain_space = H5Screate_simple(6, strain_dims, max_dims);

    // Create property list with chunking + compression
    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    if (plist_id < 0) {
        H5Sclose(strain_space);
        throw std::runtime_error("H5Pcreate failed");
    }

    try {
        apply_chunking(plist_id, config.ngll, n_elem_local);
        apply_compression(plist_id, config.compression);
    } catch (...) {
        H5Pclose(plist_id);
        H5Sclose(strain_space);
        throw;
    }

    // Select the write type based on precision config
    hid_t write_type = select_precision_type(config.use_float32);

    // Create the dataset
    hid_t dset_id = H5Dcreate2(
        file_id, "strain",
        write_type,
        strain_space,
        H5P_DEFAULT,
        plist_id,
        H5P_DEFAULT
    );
    if (dset_id < 0) {
        H5Pclose(plist_id);
        H5Sclose(strain_space);
        throw std::runtime_error("H5Dcreate2 failed for strain");
    }

    H5Pclose(plist_id);
    H5Sclose(strain_space);

    return dset_id;
}

}  // namespace gf