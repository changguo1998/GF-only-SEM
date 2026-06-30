#ifndef GF_H5IO_HH
#define GF_H5IO_HH

#include <hdf5.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdlib>

// Lightweight RAII wrappers + helpers for reading HDF5 datasets/attributes
// used by the VTK tools.

namespace gf_h5io {

class H5File {
  public:
    H5File(const std::string &path) {
        file_ = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_ < 0)
            throw std::runtime_error("H5Fopen failed: " + path);
    }
    ~H5File() { if (file_ >= 0) H5Fclose(file_); }
    H5File(const H5File &) = delete;
    H5File &operator=(const H5File &) = delete;
    hid_t id() const { return file_; }
  private:
    hid_t file_ = -1;
};

inline bool dataset_exists(hid_t loc_id, const std::string &name) {
    return H5Lexists(loc_id, name.c_str(), H5P_DEFAULT) > 0;
}

inline bool attr_exists(hid_t loc_id, const std::string &name) {
    htri_t ret = H5Aexists(loc_id, name.c_str());
    return ret > 0;
}

// Read 2-D int64 dataset into vector<vector<int64_t>>
inline std::vector<std::vector<int64_t>> read_int64_2d(hid_t loc_id, const std::string &name) {
    hid_t dset = H5Dopen2(loc_id, name.c_str(), H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("Cannot open dataset: " + name);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    int64_t n0 = (int64_t)dims[0], n1 = (int64_t)dims[1];
    std::vector<int64_t> flat(n0 * n1);
    H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    H5Sclose(space);
    H5Dclose(dset);
    std::vector<std::vector<int64_t>> result(n0);
    for (int64_t i = 0; i < n0; ++i) {
        result[i].resize(n1);
        for (int64_t j = 0; j < n1; ++j)
            result[i][j] = flat[i * n1 + j];
    }
    return result;
}

// Read 1-D int64 dataset
inline std::vector<int64_t> read_int64_1d(hid_t loc_id, const std::string &name) {
    hid_t dset = H5Dopen2(loc_id, name.c_str(), H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("Cannot open dataset: " + name);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    std::vector<int64_t> data(dims[0]);
    H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space);
    H5Dclose(dset);
    return data;
}

// Read 1-D int32 dataset
inline std::vector<int32_t> read_int32_1d(hid_t loc_id, const std::string &name) {
    hid_t dset = H5Dopen2(loc_id, name.c_str(), H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("Cannot open dataset: " + name);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    std::vector<int32_t> data(dims[0]);
    H5Dread(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space);
    H5Dclose(dset);
    return data;
}

// Read 2-D float64 dataset into vector<vector<double>>
inline std::vector<std::vector<double>> read_float64_2d(hid_t loc_id, const std::string &name) {
    hid_t dset = H5Dopen2(loc_id, name.c_str(), H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("Cannot open dataset: " + name);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    int64_t n0 = (int64_t)dims[0], n1 = (int64_t)dims[1];
    std::vector<double> flat(n0 * n1);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
    H5Sclose(space);
    H5Dclose(dset);
    std::vector<std::vector<double>> result(n0);
    for (int64_t i = 0; i < n0; ++i) {
        result[i].resize(n1);
        for (int64_t j = 0; j < n1; ++j)
            result[i][j] = flat[i * n1 + j];
    }
    return result;
}

// Read 1-D float64 dataset
inline std::vector<double> read_float64_1d(hid_t loc_id, const std::string &name) {
    hid_t dset = H5Dopen2(loc_id, name.c_str(), H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("Cannot open dataset: " + name);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    std::vector<double> data(dims[0]);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space);
    H5Dclose(dset);
    return data;
}

// Read N-D float64 dataset (up to 5 dims), returns flattened vector + shape.
// shape[0] = total elements in original array.
inline std::vector<double> read_float64_nd(hid_t loc_id, const std::string &name,
                                           std::vector<hsize_t> &shape) {
    hid_t dset = H5Dopen2(loc_id, name.c_str(), H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("Cannot open dataset: " + name);
    hid_t space = H5Dget_space(dset);
    int ndims = H5Sget_simple_extent_ndims(space);
    shape.resize(ndims);
    H5Sget_simple_extent_dims(space, shape.data(), nullptr);
    hsize_t total = 1;
    for (int i = 0; i < ndims; ++i) total *= shape[i];
    std::vector<double> data(total);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space);
    H5Dclose(dset);
    return data;
}

// Read int32 attribute
inline int32_t read_attr_int32(hid_t loc_id, const std::string &dset_name, const std::string &attr_name) {
    hid_t dset = H5Dopen2(loc_id, dset_name.c_str(), H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("Cannot open dataset for attr: " + dset_name);
    hid_t attr = H5Aopen(dset, attr_name.c_str(), H5P_DEFAULT);
    if (attr < 0) { H5Dclose(dset); throw std::runtime_error("Cannot open attr: " + attr_name); }
    int32_t val;
    H5Aread(attr, H5T_NATIVE_INT32, &val);
    H5Aclose(attr);
    H5Dclose(dset);
    return val;
}

inline int32_t read_attr_int32_group(hid_t loc_id, const std::string &group_name, const std::string &attr_name) {
    hid_t grp = H5Gopen2(loc_id, group_name.c_str(), H5P_DEFAULT);
    if (grp < 0) throw std::runtime_error("Cannot open group: " + group_name);
    hid_t attr = H5Aopen(grp, attr_name.c_str(), H5P_DEFAULT);
    if (attr < 0) { H5Gclose(grp); throw std::runtime_error("Cannot open attr: " + attr_name); }
    int32_t val;
    H5Aread(attr, H5T_NATIVE_INT32, &val);
    H5Aclose(attr);
    H5Gclose(grp);
    return val;
}

inline double read_attr_double_group(hid_t loc_id, const std::string &group_name, const std::string &attr_name) {
    hid_t grp = H5Gopen2(loc_id, group_name.c_str(), H5P_DEFAULT);
    if (grp < 0) throw std::runtime_error("Cannot open group: " + group_name);
    hid_t attr = H5Aopen(grp, attr_name.c_str(), H5P_DEFAULT);
    if (attr < 0) { H5Gclose(grp); throw std::runtime_error("Cannot open attr: " + attr_name); }
    double val;
    H5Aread(attr, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(attr);
    H5Gclose(grp);
    return val;
}

} // namespace gf_h5io

#endif // GF_H5IO_HH