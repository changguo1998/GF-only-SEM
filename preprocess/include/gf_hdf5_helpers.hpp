#pragma once
/// gf_hdf5_helpers.hpp — common HDF5 I/O utilities for preprocess C++ code
///
/// Shared by gf_preprocess.cpp, metis_partition.cpp, config_writer.cpp.

#include <hdf5.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gf {
namespace h5 {

/// Create link creation property list with intermediate group creation enabled.
inline hid_t lcpl_with_groups() {
    hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
    H5Pset_create_intermediate_group(lcpl, 1);
    return lcpl;
}

/// Open HDF5 file, exit on failure.
inline hid_t open_or_fail(const char* path, unsigned flags) {
    hid_t fid = H5Fopen(path, flags, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "ERROR: cannot open HDF5 file: %s\n", path);
        std::exit(1);
    }
    return fid;
}

/// Read 1-D dataset into std::vector<double>.
inline std::vector<double> read_double(hid_t fid, const char* name) {
    hid_t ds = H5Dopen2(fid, name, H5P_DEFAULT);
    if (ds < 0) {
        fprintf(stderr, "ERROR: dataset not found: %s\n", name);
        std::exit(1);
    }
    hid_t space = H5Dget_space(ds);
    hsize_t dims[8];
    int ndims = H5Sget_simple_extent_ndims(space);
    H5Sget_simple_extent_dims(space, dims, nullptr);
    hsize_t total = 1;
    for (int i = 0; i < ndims; ++i)
        total *= dims[i];
    std::vector<double> buf(total);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
    return buf;
}

/// Read 1-D dataset into std::vector<int64_t>.
inline std::vector<int64_t> read_int64(hid_t fid, const char* name) {
    hid_t ds = H5Dopen2(fid, name, H5P_DEFAULT);
    if (ds < 0) {
        fprintf(stderr, "ERROR: dataset not found: %s\n", name);
        std::exit(1);
    }
    hid_t space = H5Dget_space(ds);
    hsize_t dims[8];
    int ndims = H5Sget_simple_extent_ndims(space);
    H5Sget_simple_extent_dims(space, dims, nullptr);
    hsize_t total = 1;
    for (int i = 0; i < ndims; ++i)
        total *= dims[i];
    std::vector<int64_t> buf(total);
    H5Dread(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
    return buf;
}

/// Write std::vector<double> as an N-D dataset, overwriting if present.
inline void write_double(hid_t fid, const char* name, const std::vector<double>& data,
                         const std::vector<hsize_t>& dims) {
    H5Ldelete(fid, name, H5P_DEFAULT);
    hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
    hid_t ds = H5Dcreate2(fid, name, H5T_NATIVE_DOUBLE, space, lcpl_with_groups(), H5P_DEFAULT,
                          H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(ds);
    H5Sclose(space);
}

/// Write std::vector<int32_t> as an N-D dataset.
inline void write_int32(hid_t fid, const char* name, const std::vector<int32_t>& data,
                        const std::vector<hsize_t>& dims) {
    H5Ldelete(fid, name, H5P_DEFAULT);
    hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
    hid_t ds = H5Dcreate2(fid, name, H5T_NATIVE_INT32, space, lcpl_with_groups(), H5P_DEFAULT,
                          H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(ds);
    H5Sclose(space);
}

/// Write std::vector<int64_t> as an N-D dataset.
inline void write_int64(hid_t fid, const char* name, const std::vector<int64_t>& data,
                        const std::vector<hsize_t>& dims) {
    H5Ldelete(fid, name, H5P_DEFAULT);
    hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
    hid_t ds = H5Dcreate2(fid, name, H5T_NATIVE_INT64, space, lcpl_with_groups(), H5P_DEFAULT,
                          H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(ds);
    H5Sclose(space);
}

/// Get the shape (dims) of an HDF5 dataset.
inline std::vector<hsize_t> get_dims(hid_t fid, const char* name) {
    hid_t ds = H5Dopen2(fid, name, H5P_DEFAULT);
    if (ds < 0) {
        fprintf(stderr, "ERROR: dataset not found: %s\n", name);
        std::exit(1);
    }
    hid_t space = H5Dget_space(ds);
    int ndims = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims(ndims);
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    H5Dclose(ds);
    H5Sclose(space);
    return dims;
}

}  // namespace h5
}  // namespace gf