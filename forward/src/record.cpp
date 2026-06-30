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

    // Create extendible strain dataset: [n_snapshots, n_vertices, 6].
    // Empty recording ranks use shape [0, 0, 6] with unlimited vertex max
    // so HDF5 can use positive chunk dimensions.
    constexpr hsize_t ncomps = 6;
    hsize_t strain_dims[3] = {n_vertices_ > 0 ? hsize_t{1} : hsize_t{0}, n_vertices_, ncomps};
    hsize_t max_dims[3] = {H5S_UNLIMITED, n_vertices_ > 0 ? n_vertices_ : hsize_t{H5S_UNLIMITED},
                           ncomps};

    hid_t strain_space = H5Screate_simple(3, strain_dims, max_dims);
    if (strain_space < 0)
        throw std::runtime_error("H5Screate_simple failed for strain");

    // Apply chunking and compression
    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    if (plist < 0) {
        H5Sclose(strain_space);
        throw std::runtime_error("H5Pcreate failed");
    }

    // Chunk along time dimension: 1 snapshot per chunk. HDF5 requires all
    // chunk dimensions to be positive even when this rank records zero vertices.
    hsize_t chunk_dims[3] = {1, n_vertices_ > 0 ? n_vertices_ : hsize_t{1}, ncomps};
    H5Pset_chunk(plist, 3, chunk_dims);

    // Apply compression filter
    apply_compression(plist, compression);

    hid_t write_type = select_precision_type(use_float32_);

    strain_dset_ =
        H5Dcreate2(file_id_, "strain", write_type, strain_space, H5P_DEFAULT, plist, H5P_DEFAULT);
    if (strain_dset_ < 0) {
        H5Pclose(plist);
        H5Sclose(strain_space);
        throw std::runtime_error("H5Dcreate2 failed for strain");
    }
    H5Pclose(plist);
    H5Sclose(strain_space);

    current_step_ = n_vertices_ > 0 ? 1 : 0;  // non-empty ranks start with 1 allocated step
}

RecordWriter::~RecordWriter() {
    try {
        close();
    } catch (...) {
    }
}

void RecordWriter::write_step(int step, const double* strain) {
    (void)step;
    if (file_id_ < 0 || strain_dset_ < 0) {
        throw std::runtime_error("RecordWriter: file not open");
    }

    if (n_vertices_ == 0) {
        return;
    }

    // Extend dataset along time dimension
    hsize_t new_dims[3] = {current_step_, n_vertices_, 6};
    herr_t status = H5Dset_extent(strain_dset_, new_dims);
    if (status < 0) {
        throw std::runtime_error("H5Dset_extent failed at step " + std::to_string(current_step_));
    }

    // Get the updated dataspace
    hid_t filespace = H5Dget_space(strain_dset_);
    if (filespace < 0) {
        throw std::runtime_error("H5Dget_space failed");
    }

    // Select hyperslab for the new step
    hsize_t start[3] = {current_step_ - 1, 0, 0};
    hsize_t count[3] = {1, n_vertices_, 6};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);

    // Create memory dataspace for this slab
    hid_t memspace = H5Screate_simple(3, count, nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        throw std::runtime_error("H5Screate_simple failed for memspace");
    }

    // If float32, convert doubles to floats on the fly
    if (use_float32_) {
        hsize_t total = n_vertices_ * 6;
        std::vector<float> fbuf(total);
        for (hsize_t i = 0; i < total; ++i)
            fbuf[i] = static_cast<float>(strain[i]);
        H5Dwrite(strain_dset_, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, fbuf.data());
    } else {
        H5Dwrite(strain_dset_, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, strain);
    }

    H5Sclose(memspace);
    H5Sclose(filespace);

    ++current_step_;
}

void RecordWriter::close() {
    if (strain_dset_ >= 0) {
        H5Dclose(strain_dset_);
        strain_dset_ = -1;
    }
    if (file_id_ >= 0) {
        H5Fclose(file_id_);
        file_id_ = -1;
    }
}

}  // namespace gf