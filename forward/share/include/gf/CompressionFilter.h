// compress/include/gf/CompressionFilter.h
#pragma once
#include <hdf5.h>

#include <stdexcept>
#include <string>

namespace gf {

/// Compression algorithm selection for HDF5 checkpoint datasets.
enum class CompressionMethod {
    None,  ///< No compression
    LZF,   ///< HDF5 LZF filter (fast, modest ratio)
    Zlib,  ///< HDF5 zlib/gzip filter (slower, better ratio)
};

/// Parameters for zlib compression.
struct ZlibConfig {
    int level = 6;  ///< 1 (fast) .. 9 (best ratio)
};

/// Complete compression configuration.
struct CompressionConfig {
    CompressionMethod method = CompressionMethod::None;
    ZlibConfig zlib{};

    /// Human-readable label for benchmarking output.
    std::string label() const;
};

/// Apply compression settings to an HDF5 dataset creation property list.
///
/// Must be called *after* H5Pset_chunk() on the same plist.
/// Returns the modified property list ID (same as input).
///
/// \param plist  Dataset creation property list (H5P_DATASET_CREATE)
/// \param config Compression configuration
/// \throws std::runtime_error if HDF5 call fails
hid_t apply_compression(hid_t plist, const CompressionConfig& config);

// --- inline implementations ---

inline std::string CompressionConfig::label() const {
    switch (method) {
        case CompressionMethod::None:
            return "none";
        case CompressionMethod::LZF:
            return "lzf";
        case CompressionMethod::Zlib:
            return "zlib:" + std::to_string(zlib.level);
    }
    return "unknown";
}

inline hid_t apply_compression(hid_t plist, const CompressionConfig& config) {
    if (config.method == CompressionMethod::None) {
        return plist;
    }

    if (config.method == CompressionMethod::Zlib) {
        // H5Z_FILTER_DEFLATE is built-in
        herr_t status = H5Pset_deflate(plist, config.zlib.level);
        if (status < 0) {
            throw std::runtime_error(
                "H5Pset_deflate failed (level=" + std::to_string(config.zlib.level) + ")");
        }
    }

    if (config.method == CompressionMethod::LZF) {
        // LZF filter must be enabled at HDF5 build time (--with-lzf)
        // Filter ID: H5Z_FILTER_LZF = 32000
        unsigned int filter_id = 32000;  // H5Z_FILTER_LZF
        unsigned int flags = 0;
        size_t cd_nelmts = 0;
        unsigned int cd_values[1] = {0};
        herr_t status = H5Pset_filter(plist, filter_id, flags, cd_nelmts, cd_values);
        if (status < 0) {
            throw std::runtime_error(
                "H5Pset_filter(LZF) failed — is LZF "
                "enabled in your HDF5 build?");
        }
    }

    return plist;
}

}  // namespace gf