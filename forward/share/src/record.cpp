// forward/share/src/record.cpp
#include "gf/record.hpp"

#include <hdf5.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace gf {

namespace {

/// Select HDF5 native type based on precision flag.
inline hid_t select_precision_type(bool use_float32) noexcept {
    return use_float32 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
}

/// Write scalar attr to HDF5.
void write_scalar_attr(hid_t loc_id, const std::string& name, hid_t type_id, const void* value) {
    hid_t attr_space = H5Screate(H5S_SCALAR);
    if (attr_space < 0)
        throw std::runtime_error("H5Screate failed for attr: " + name);
    hid_t attr_id =
        H5Acreate2(loc_id, name.c_str(), type_id, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr_id < 0) {
        H5Sclose(attr_space);
        throw std::runtime_error("H5Acreate2 failed for attr: " + name);
    }
    herr_t status = H5Awrite(attr_id, type_id, value);
    H5Aclose(attr_id);
    H5Sclose(attr_space);
    if (status < 0)
        throw std::runtime_error("H5Awrite failed for attr: " + name);
}

/// Write string attr to HDF5.
void write_string_attr(hid_t loc_id, const std::string& name, const std::string& value) {
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, value.size());
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr_id =
        H5Acreate2(loc_id, name.c_str(), str_type, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr_id < 0) {
        H5Sclose(attr_space);
        H5Tclose(str_type);
        throw std::runtime_error("H5Acreate2 failed for string attr: " + name);
    }
    herr_t status = H5Awrite(attr_id, str_type, value.c_str());
    H5Aclose(attr_id);
    H5Sclose(attr_space);
    H5Tclose(str_type);
    if (status < 0)
        throw std::runtime_error("H5Awrite failed for string attr: " + name);
}

/// Create a 1-step field dataset, write data, and close.
static void write_field(hid_t file_id, const std::string& name, int ncomp, hsize_t n_vertices,
                        bool use_float32, const double* data) {
    if (data == nullptr)
        return;

    constexpr int ndim = 3;
    hsize_t dims[3] = {1, n_vertices, static_cast<hsize_t>(ncomp)};
    hid_t space = H5Screate_simple(ndim, dims, nullptr);
    if (space < 0)
        throw std::runtime_error("H5Screate_simple failed for " + name);

    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    if (plist < 0) {
        H5Sclose(space);
        throw std::runtime_error("H5Pcreate failed for " + name);
    }

    hsize_t chunk_dims[3] = {1, n_vertices, static_cast<hsize_t>(ncomp)};
    H5Pset_chunk(plist, ndim, chunk_dims);

    hid_t write_type = select_precision_type(use_float32);

    hid_t dset =
        H5Dcreate2(file_id, name.c_str(), write_type, space, H5P_DEFAULT, plist, H5P_DEFAULT);
    if (dset < 0) {
        H5Pclose(plist);
        H5Sclose(space);
        throw std::runtime_error("H5Dcreate2 failed for " + name);
    }
    H5Pclose(plist);
    H5Sclose(space);

    hsize_t total = n_vertices * static_cast<hsize_t>(ncomp);
    if (use_float32) {
        std::vector<float> fbuf(total);
        for (hsize_t i = 0; i < total; ++i)
            fbuf[i] = static_cast<float>(data[i]);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fbuf.data());
    } else {
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    }

    H5Dclose(dset);
}

static void write_field_4d(hid_t file_id, const std::string& name, int ncomp, hsize_t n_rec_cell,
                           int n_node, bool use_float32, const double* data) {
    if (data == nullptr)
        return;

    constexpr int ndim = 4;
    hsize_t dims[4] = {1, n_rec_cell, static_cast<hsize_t>(n_node), static_cast<hsize_t>(ncomp)};
    hid_t space = H5Screate_simple(ndim, dims, nullptr);
    if (space < 0)
        throw std::runtime_error("H5Screate_simple failed for " + name);

    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[4] = {1, n_rec_cell, static_cast<hsize_t>(n_node),
                             static_cast<hsize_t>(ncomp)};
    H5Pset_chunk(plist, ndim, chunk_dims);

    hid_t write_type = select_precision_type(use_float32);

    hid_t dset =
        H5Dcreate2(file_id, name.c_str(), write_type, space, H5P_DEFAULT, plist, H5P_DEFAULT);
    if (dset < 0) {
        H5Pclose(plist);
        H5Sclose(space);
        throw std::runtime_error("H5Dcreate2 failed for " + name);
    }
    H5Pclose(plist);
    H5Sclose(space);

    hsize_t total = n_rec_cell * static_cast<hsize_t>(n_node) * static_cast<hsize_t>(ncomp);
    if (use_float32) {
        std::vector<float> fbuf(total);
        for (hsize_t i = 0; i < total; ++i)
            fbuf[i] = static_cast<float>(data[i]);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fbuf.data());
    } else {
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    }

    H5Dclose(dset);
}

}  // anonymous namespace

RecordWriter::RecordWriter(const std::string& output_dir, const std::string& source_direction,
                           int rank, const RankData::RecordingMap& rec_map, int ngll,
                           int source_partition_start, int source_partition_count,
                           bool use_float32)
    : n_rec_cell_(static_cast<hsize_t>(rec_map.rec_cell_local.size())),
      n_node_(ngll * ngll * ngll),
      use_float32_(use_float32),
      output_dir_(output_dir),
      source_direction_(source_direction),
      rank_(rank),
      source_partition_start_(source_partition_start),
      source_partition_count_(source_partition_count) {}

RecordWriter::~RecordWriter() {
    try {
        close();
    } catch (...) {
    }
}

void RecordWriter::write_step(int step, const double* strain, const double* displacement,
                              const double* velocity, const double* acceleration) {
    if (n_rec_cell_ == 0)
        return;

    std::string dirpath = output_dir_ + "/" + source_direction_;
    std::filesystem::create_directories(dirpath);

    std::string filepath =
        dirpath + "/record_" + std::to_string(rank_) + "_" + std::to_string(step) + ".h5";

    hid_t file_id = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("H5Fcreate failed: " + filepath);
    }

    // Write root group attributes
    write_string_attr(file_id, "source_direction", source_direction_);
    write_scalar_attr(file_id, "rank", H5T_NATIVE_INT, &rank_);
    write_scalar_attr(file_id, "source_partition_start", H5T_NATIVE_INT, &source_partition_start_);
    write_scalar_attr(file_id, "source_partition_count", H5T_NATIVE_INT, &source_partition_count_);
    // Write 4D field datasets [1, n_rec_cell, n_node, ncomp]
    write_field_4d(file_id, "strain", 6, n_rec_cell_, n_node_, use_float32_, strain);
    write_field_4d(file_id, "displacement", 3, n_rec_cell_, n_node_, use_float32_, displacement);
    write_field_4d(file_id, "velocity", 3, n_rec_cell_, n_node_, use_float32_, velocity);
    write_field_4d(file_id, "acceleration", 3, n_rec_cell_, n_node_, use_float32_, acceleration);

    H5Fclose(file_id);
}

}  // namespace gf
