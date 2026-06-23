// forward/src/record.cpp
#include "gf/record.hpp"
#include "gf/CompressionFilter.h"
#include "gf/PrecisionPolicy.h"
#include "gf/ChunkingStrategy.h"
#include <hdf5.h>
#include <stdexcept>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace gf {

namespace {
// Create directory if it doesn't exist (POSIX)
void ensure_dir(const std::string& path) {
    std::string dir;
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i] == '/') {
            dir = path.substr(0, i);
            if (!dir.empty()) {
                mkdir(dir.c_str(), 0755);
            }
        }
    }
    mkdir(path.c_str(), 0755);
}

// Helper: write scalar attribute
void write_scalar_attr(hid_t loc_id, const std::string& name, hid_t type_id, const void* value) {
    hid_t attr_space = H5Screate(H5S_SCALAR);
    if (attr_space < 0) throw std::runtime_error("H5Screate failed for attr: " + name);
    hid_t attr_id = H5Acreate2(loc_id, name.c_str(), type_id, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr_id < 0) {
        H5Sclose(attr_space);
        throw std::runtime_error("H5Acreate2 failed for attr: " + name);
    }
    H5Awrite(attr_id, type_id, value);
    H5Aclose(attr_id);
    H5Sclose(attr_space);
}

// Helper: write string attribute
void write_string_attr(hid_t loc_id, const std::string& name, const std::string& value) {
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, value.size());
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr_id = H5Acreate2(loc_id, name.c_str(), str_type, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr_id < 0) {
        H5Sclose(attr_space);
        H5Tclose(str_type);
        throw std::runtime_error("H5Acreate2 failed for string attr: " + name);
    }
    H5Awrite(attr_id, str_type, value.c_str());
    H5Aclose(attr_id);
    H5Sclose(attr_space);
    H5Tclose(str_type);
}
} // anonymous namespace

RecordWriter::RecordWriter(const std::string& output_dir,
                           const std::string& source_direction,
                           int rank,
                           int n_local_elem,
                           const int64_t* element_ids,
                           int ngll,
                           CompressionConfig compression,
                           bool use_float32)
    : file_id_(-1)
    , strain_dset_(-1)
    , strain_space_(-1)
    , current_step_(0)
    , n_elem_local_(static_cast<hsize_t>(n_local_elem))
    , ngll_(ngll)
    , use_float32_(use_float32)
    , source_direction_(source_direction)
{
    // Build file path: wavefields/{direction}/record_{rank}.h5
    std::string wavefields_dir = output_dir + "/wavefields/" + source_direction;
    ensure_dir(wavefields_dir);
    filepath_ = wavefields_dir + "/record_" + std::to_string(rank) + ".h5";

    // Create or open HDF5 file
    file_id_ = H5Fcreate(filepath_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id_ < 0) {
        throw std::runtime_error("H5Fcreate failed: " + filepath_);
    }

    // Write root group attributes
    int ngll_attr = ngll_;
    write_string_attr(file_id_, "source_direction", source_direction);
    write_scalar_attr(file_id_, "ngll", H5T_NATIVE_INT, &ngll_attr);
    write_scalar_attr(file_id_, "rank", H5T_NATIVE_INT, &rank);

    // Write local_element_ids dataset
    hsize_t elem_dims[1] = {n_elem_local_};
    hid_t elem_space = H5Screate_simple(1, elem_dims, nullptr);
    if (elem_space < 0) throw std::runtime_error("H5Screate_simple failed for elem_ids");
    hid_t elem_dset = H5Dcreate2(file_id_, "local_element_ids",
                                  H5T_NATIVE_INT64, elem_space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (elem_dset < 0) {
        H5Sclose(elem_space);
        throw std::runtime_error("H5Dcreate2 failed for local_element_ids");
    }
    H5Dwrite(elem_dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, element_ids);
    H5Dclose(elem_dset);
    H5Sclose(elem_space);

    // Create extendible strain dataset: [1, n_elem_local, NGLL, NGLL, NGLL, 6]
    constexpr hsize_t ncomps = 6;
    hsize_t strain_dims[6] = {1, n_elem_local_,
                              static_cast<hsize_t>(ngll_),
                              static_cast<hsize_t>(ngll_),
                              static_cast<hsize_t>(ngll_),
                              ncomps};
    hsize_t max_dims[6] = {H5S_UNLIMITED, n_elem_local_,
                           static_cast<hsize_t>(ngll_),
                           static_cast<hsize_t>(ngll_),
                           static_cast<hsize_t>(ngll_),
                           ncomps};

    strain_space_ = H5Screate_simple(6, strain_dims, max_dims);
    if (strain_space_ < 0) throw std::runtime_error("H5Screate_simple failed for strain");

    // Apply chunking and compression
    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    if (plist < 0) {
        H5Sclose(strain_space_);
        throw std::runtime_error("H5Pcreate failed");
    }
    apply_chunking(plist, ngll_, n_elem_local_);
    apply_compression(plist, compression);

    hid_t write_type = select_precision_type(use_float32);

    strain_dset_ = H5Dcreate2(file_id_, "strain", write_type, strain_space_,
                               H5P_DEFAULT, plist, H5P_DEFAULT);
    if (strain_dset_ < 0) {
        H5Pclose(plist);
        H5Sclose(strain_space_);
        throw std::runtime_error("H5Dcreate2 failed for strain");
    }
    H5Pclose(plist);

    current_step_ = 1;  // already have 1 step allocated
}

RecordWriter::~RecordWriter() {
    try {
        close();
    } catch (...) {
        // suppress exceptions in destructor
    }
}

void RecordWriter::write_step(int step, const double* strain) {
    (void)step;
    if (file_id_ < 0 || strain_dset_ < 0) {
        throw std::runtime_error("RecordWriter: file not open");
    }

    // Extend the dataset along time dimension (dim 0) to current_step_.
    hsize_t new_dims[6] = {current_step_, n_elem_local_,
                           static_cast<hsize_t>(ngll_),
                           static_cast<hsize_t>(ngll_),
                           static_cast<hsize_t>(ngll_), 6};
    herr_t status = H5Dset_extent(strain_dset_, new_dims);
    if (status < 0) {
        throw std::runtime_error("H5Dset_extent failed at step " + std::to_string(current_step_));
    }

    // Get the updated dataspace
    hid_t filespace = H5Dget_space(strain_dset_);
    if (filespace < 0) {
        throw std::runtime_error("H5Dget_space failed");
    }
    H5Sget_simple_extent_dims(filespace, new_dims, nullptr);

    // Select hyperslab for the new step
    hsize_t start[6] = {current_step_ - 1, 0, 0, 0, 0, 0};
    hsize_t count[6] = {1, n_elem_local_,
                        static_cast<hsize_t>(ngll_),
                        static_cast<hsize_t>(ngll_),
                        static_cast<hsize_t>(ngll_), 6};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);

    // Create memory dataspace for this slab
    hid_t memspace = H5Screate_simple(6, count, nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        throw std::runtime_error("H5Screate_simple failed for memspace");
    }

    // If float32, convert doubles to floats on the fly
    if (use_float32_) {
        hsize_t total = 1;
        for (int i = 0; i < 6; ++i) total *= count[i];
        std::vector<float> fbuf(total);
        for (hsize_t i = 0; i < total; ++i) fbuf[i] = static_cast<float>(strain[i]);
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
    if (strain_space_ >= 0) {
        H5Sclose(strain_space_);
        strain_space_ = -1;
    }
    if (file_id_ >= 0) {
        H5Fclose(file_id_);
        file_id_ = -1;
    }
}

} // namespace gf