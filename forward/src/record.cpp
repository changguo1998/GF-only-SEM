// forward/src/record.cpp
#include "gf/record.hpp"

#include <hdf5.h>
#include <sys/stat.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "gf/ChunkingStrategy.h"
#include "gf/CompressionFilter.h"
#include "gf/PrecisionPolicy.h"

namespace gf {

namespace {

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

}  // anonymous namespace

RecordWriter::RecordWriter(const std::string& output_dir, const std::string& source_direction,
                           int rank, const RankData::RecordingMap& rec_map, int ngll,
                           CompressionConfig compression, bool use_float32,
                           double record_depth_max_m, double record_depth_actual_m)
    : file_id_(-1),
      strain_dset_(-1),
      displacement_dset_(-1),
      velocity_dset_(-1),
      acceleration_dset_(-1),
      current_step_(0),
      ngll_(ngll),
      use_float32_(use_float32),
      source_direction_(source_direction) {
    n_vertices_ = static_cast<hsize_t>(rec_map.vertex_ids.size());

    // Build file path: {output_dir}/{direction}/record_{rank}.h5
    std::string wavefields_dir = output_dir + "/" + source_direction;
    filepath_ = wavefields_dir + "/record_" + std::to_string(rank) + ".h5";

    // Create output directory if needed
    mkdir(wavefields_dir.c_str(), 0755);

    // Create or open HDF5 file
    file_id_ = H5Fcreate(filepath_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id_ < 0) {
        throw std::runtime_error("H5Fcreate failed: " + filepath_);
    }

    // Write root group attributes
    int rank_int = rank;
    write_string_attr(file_id_, "source_direction", source_direction);
    write_scalar_attr(file_id_, "rank", H5T_NATIVE_INT, &rank_int);
    write_scalar_attr(file_id_, "ngll", H5T_NATIVE_INT, &ngll);
    write_string_attr(file_id_, "basis", "mesh_vertices");
    hbool_t excludes_pml_flag = rec_map.has_recording ? 1 : 0;
    write_scalar_attr(file_id_, "excludes_pml", H5T_NATIVE_HBOOL, &excludes_pml_flag);

    write_scalar_attr(file_id_, "record_depth_max_m", H5T_NATIVE_DOUBLE, &record_depth_max_m);
    write_scalar_attr(file_id_, "record_depth_actual_m", H5T_NATIVE_DOUBLE,
                      &record_depth_actual_m);

    // Write vertex_ids dataset (possibly empty)
    hsize_t elem_dims[1] = {n_vertices_};
    hid_t elem_space = H5Screate_simple(1, elem_dims, nullptr);
    if (elem_space < 0)
        throw std::runtime_error("H5Screate_simple failed for vertex_ids");
    hid_t elem_dset = H5Dcreate2(file_id_, "vertex_ids", H5T_NATIVE_INT64, elem_space, H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);
    if (elem_dset < 0) {
        H5Sclose(elem_space);
        throw std::runtime_error("H5Dcreate2 failed for vertex_ids");
    }
    if (n_vertices_ > 0) {
        H5Dwrite(elem_dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 rec_map.vertex_ids.data());
    }
    H5Dclose(elem_dset);
    H5Sclose(elem_space);

    // Create field datasets: strain (6 comp), displacement/velocity/acceleration (3 comp)
    strain_dset_ = create_field_dset("strain", 6);
    displacement_dset_ = create_field_dset("displacement", 3);
    velocity_dset_ = create_field_dset("velocity", 3);
    acceleration_dset_ = create_field_dset("acceleration", 3);

    current_step_ = n_vertices_ > 0 ? 1 : 0;
}

hid_t RecordWriter::create_field_dset(const std::string& name, int ncomp) {
    constexpr int ndim = 3;
    hsize_t dims[3] = {n_vertices_ > 0 ? hsize_t{1} : hsize_t{0}, n_vertices_,
                       static_cast<hsize_t>(ncomp)};
    hsize_t max_dims[3] = {H5S_UNLIMITED, n_vertices_ > 0 ? n_vertices_ : hsize_t{H5S_UNLIMITED},
                           static_cast<hsize_t>(ncomp)};

    hid_t space = H5Screate_simple(ndim, dims, max_dims);
    if (space < 0)
        throw std::runtime_error("H5Screate_simple failed for " + name);

    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    if (plist < 0) {
        H5Sclose(space);
        throw std::runtime_error("H5Pcreate failed for " + name);
    }

    hsize_t chunk_dims[3] = {1, n_vertices_ > 0 ? n_vertices_ : hsize_t{1},
                             static_cast<hsize_t>(ncomp)};
    H5Pset_chunk(plist, ndim, chunk_dims);

    CompressionConfig comp;
    comp.method = CompressionMethod::None;
    apply_compression(plist, comp);

    hid_t write_type = select_precision_type(use_float32_);

    hid_t dset =
        H5Dcreate2(file_id_, name.c_str(), write_type, space, H5P_DEFAULT, plist, H5P_DEFAULT);
    if (dset < 0) {
        H5Pclose(plist);
        H5Sclose(space);
        throw std::runtime_error("H5Dcreate2 failed for " + name);
    }
    H5Pclose(plist);
    H5Sclose(space);
    return dset;
}

RecordWriter::~RecordWriter() {
    try {
        close();
    } catch (...) {
    }
}

void RecordWriter::write_step(int step, const double* strain, const double* displacement,
                              const double* velocity, const double* acceleration) {
    (void)step;
    if (file_id_ < 0) {
        throw std::runtime_error("RecordWriter: file not open");
    }

    if (n_vertices_ == 0) {
        return;
    }

    auto write_field_dset = [&](hid_t dset_id, const double* data, int ncomp) {
        if (dset_id < 0 || data == nullptr)
            return;

        // Extend dataset along time dimension
        hsize_t new_dims[3] = {current_step_, n_vertices_, static_cast<hsize_t>(ncomp)};
        herr_t status = H5Dset_extent(dset_id, new_dims);
        if (status < 0) {
            throw std::runtime_error("H5Dset_extent failed");
        }

        hid_t filespace = H5Dget_space(dset_id);
        if (filespace < 0) {
            throw std::runtime_error("H5Dget_space failed");
        }

        hsize_t start[3] = {current_step_ - 1, 0, 0};
        hsize_t count[3] = {1, n_vertices_, static_cast<hsize_t>(ncomp)};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);

        hid_t memspace = H5Screate_simple(3, count, nullptr);
        if (memspace < 0) {
            H5Sclose(filespace);
            throw std::runtime_error("H5Screate_simple failed for memspace");
        }

        hsize_t total = n_vertices_ * static_cast<hsize_t>(ncomp);
        if (use_float32_) {
            std::vector<float> fbuf(total);
            for (hsize_t i = 0; i < total; ++i)
                fbuf[i] = static_cast<float>(data[i]);
            H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, fbuf.data());
        } else {
            H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, data);
        }

        H5Sclose(memspace);
        H5Sclose(filespace);
    };

    write_field_dset(strain_dset_, strain, 6);
    write_field_dset(displacement_dset_, displacement, 3);
    write_field_dset(velocity_dset_, velocity, 3);
    write_field_dset(acceleration_dset_, acceleration, 3);

    ++current_step_;
}

void RecordWriter::close() {
    auto close_dset = [](hid_t& id) {
        if (id >= 0) {
            H5Dclose(id);
            id = -1;
        }
    };
    close_dset(strain_dset_);
    close_dset(displacement_dset_);
    close_dset(velocity_dset_);
    close_dset(acceleration_dset_);
    if (file_id_ >= 0) {
        H5Fclose(file_id_);
        file_id_ = -1;
    }
}

}  // namespace gf