// compress/include/gf/ChunkingStrategy.h
#pragma once
#include <hdf5.h>

#include <array>
#include <cstddef>
#include <stdexcept>

namespace gf {

/// Compute chunk dimensions for a 6D strain tensor dataset.
///
/// Shape: [n_checkpoints, n_elem_local, NGLL, NGLL, NGLL, 6]
///
/// Strategy (design §10):
///   - Dim 0 (time): chunk = 1 (one step at a time, matching write pattern)
///   - Dim 1 (element): chunk along element dim, default 64 elements per chunk
///   - Dims 2-5 (spatial + components): full (NGLL^3 x 6) per chunk
///
/// \param ngll            Polynomial degree + 1 (N+1)
/// \param n_elem_local    Number of local elements (per MPI rank)
/// \param chunk_size      Elements per chunk (default 64)
/// \return 6-element chunk size array
inline std::array<hsize_t, 6> compute_chunk_dims(int ngll, hsize_t n_elem_local,
                                                 hsize_t chunk_size = 64) {
    if (chunk_size > n_elem_local)
        chunk_size = n_elem_local;
    constexpr hsize_t ncomps = 6;  // εxx,εyy,εzz,εxy,εxz,εyz

    return {1,                           // time: one step
            chunk_size,                  // elements per chunk
            static_cast<hsize_t>(ngll),  // ξ
            static_cast<hsize_t>(ngll),  // η
            static_cast<hsize_t>(ngll),  // ζ
            ncomps};                     // components
}

/// Apply chunking to an HDF5 dataset creation property list.
///
/// Creates extendible dataset with H5S_UNLIMITED on dim 0 (time axis).
///
/// \param plist         Dataset creation property list
/// \param ngll          Polynomial degree + 1
/// \param n_elem_local  Local elements on this rank
/// \param chunk_size    Elements per chunk (default 64)
/// \return The modified property list (same as input)
inline hid_t apply_chunking(hid_t plist, int ngll, hsize_t n_elem_local, hsize_t chunk_size = 64) {
    auto chunk = compute_chunk_dims(ngll, n_elem_local, chunk_size);
    herr_t status = H5Pset_chunk(plist, 6, chunk.data());
    if (status < 0) {
        throw std::runtime_error("H5Pset_chunk failed");
    }
    return plist;
}

}  // namespace gf