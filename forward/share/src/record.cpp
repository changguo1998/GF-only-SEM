// forward/share/src/record.cpp
#include "gf/record.hpp"

#include <hdf5.h>

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

    CompressionConfig comp;
    comp.method = CompressionMethod::None;
    apply_compression(plist, comp);

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

}  // anonymous namespace

RecordWriter::RecordWriter(const std::string& output_dir, const std::string& source_direction,
                           int rank, const RankData::RecordingMap& rec_map, int ngll,
                           CompressionConfig /*compression*/, bool use_float32,
                           double record_depth_max_m, double record_depth_actual_m)
    : file_id_(-1),
      ngll_(ngll),
      use_float32_(use_float32),
      output_dir_(output_dir),
      source_direction_(source_direction),
      rank_(rank),
      basis_("mesh_vertices"),
      excludes_pml_(rec_map.has_recording),
      record_depth_max_m_(record_depth_max_m),
      record_depth_actual_m_(record_depth_actual_m),
      vertex_ids_(rec_map.vertex_ids) {
    n_vertices_ = static_cast<hsize_t>(rec_map.vertex_ids.size());
}

RecordWriter::~RecordWriter() {
    try {
        close();
    } catch (...) {
    }
}

void RecordWriter::write_step(int step, const double* strain, const double* displacement,
                              const double* velocity, const double* acceleration) {
    if (n_vertices_ == 0)
        return;

    // Build file path: {output_dir}/{source_direction}/record_{rank}_{step}.h5
    std::string filepath = output_dir_ + "/" + source_direction_ + "/record_" +
                           std::to_string(rank_) + "_" + std::to_string(step) + ".h5";

    hid_t file_id = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("H5Fcreate failed: " + filepath);
    }

    // Write root group attributes
    write_string_attr(file_id, "source_direction", source_direction_);
    write_scalar_attr(file_id, "rank", H5T_NATIVE_INT, &rank_);
    write_scalar_attr(file_id, "ngll", H5T_NATIVE_INT, &ngll_);
    write_string_attr(file_id, "basis", basis_);
    hbool_t excludes_pml_flag = excludes_pml_ ? 1 : 0;
    write_scalar_attr(file_id, "excludes_pml", H5T_NATIVE_HBOOL, &excludes_pml_flag);
    write_scalar_attr(file_id, "record_depth_max_m", H5T_NATIVE_DOUBLE, &record_depth_max_m_);
    write_scalar_attr(file_id, "record_depth_actual_m", H5T_NATIVE_DOUBLE,
                      &record_depth_actual_m_);

    // Write vertex_ids dataset
    hsize_t elem_dims[1] = {n_vertices_};
    hid_t elem_space = H5Screate_simple(1, elem_dims, nullptr);
    if (elem_space < 0) {
        H5Fclose(file_id);
        throw std::runtime_error("H5Screate_simple failed for vertex_ids");
    }
    hid_t elem_dset = H5Dcreate2(file_id, "vertex_ids", H5T_NATIVE_INT64, elem_space, H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);
    if (elem_dset < 0) {
        H5Sclose(elem_space);
        H5Fclose(file_id);
        throw std::runtime_error("H5Dcreate2 failed for vertex_ids");
    }
    if (n_vertices_ > 0) {
        H5Dwrite(elem_dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, vertex_ids_.data());
    }
    H5Dclose(elem_dset);
    H5Sclose(elem_space);

    // Write field datasets (each 1-step, fixed-size)
    write_field(file_id, "strain", 6, n_vertices_, use_float32_, strain);
    write_field(file_id, "displacement", 3, n_vertices_, use_float32_, displacement);
    write_field(file_id, "velocity", 3, n_vertices_, use_float32_, velocity);
    write_field(file_id, "acceleration", 3, n_vertices_, use_float32_, acceleration);

    H5Fclose(file_id);
}

}  // namespace gf