// forward/share/src/restart.cpp
#include "gf/restart.hpp"

#include <hdf5.h>
#include <sys/stat.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace gf {

namespace {

// Overwrite scalar attribute: delete if exists, then create+write
void write_scalar_attr(hid_t loc_id, const std::string& name, hid_t type_id, const void* value) {
    if (H5Aexists(loc_id, name.c_str()) > 0) {
        H5Adelete(loc_id, name.c_str());
    }
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

// Overwrite string attribute: delete if exists, then create+write
void write_string_attr(hid_t loc_id, const std::string& name, const std::string& value) {
    if (H5Aexists(loc_id, name.c_str()) > 0) {
        H5Adelete(loc_id, name.c_str());
    }
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

void write_dset(hid_t loc_id, const std::string& name, const std::vector<double>& data, int ndim,
                const hsize_t* dims) {
    // Remove existing dataset if present (overwrite mode)
    if (H5Lexists(loc_id, name.c_str(), H5P_DEFAULT) > 0) {
        H5Ldelete(loc_id, name.c_str(), H5P_DEFAULT);
    }
    hid_t space = H5Screate_simple(ndim, dims, nullptr);
    if (space < 0)
        throw std::runtime_error("H5Screate_simple failed for " + name);
    hid_t dset = H5Dcreate2(loc_id, name.c_str(), H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) {
        H5Sclose(space);
        throw std::runtime_error("H5Dcreate2 failed for " + name);
    }
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(dset);
    H5Sclose(space);
}

}  // anonymous namespace

RestartWriter::RestartWriter(const std::string& output_dir, const std::string& source_direction,
                             int rank, int n_local_element, int ngll)
    : file_id_(-1), n_elem_local_(n_local_element), ngll_(ngll), source_direction_(source_direction) {
    std::string restart_dir = output_dir + "/" + source_direction;
    filepath_ = restart_dir + "/restart_" + std::to_string(rank) + ".h5";

    // Create directory if needed
    mkdir(restart_dir.c_str(), 0755);

    // Create or truncate file
    file_id_ = H5Fcreate(filepath_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id_ < 0) {
        throw std::runtime_error("H5Fcreate failed: " + filepath_);
    }

    write_string_attr(file_id_, "source_direction", source_direction);
    write_scalar_attr(file_id_, "rank", H5T_NATIVE_INT, &rank);
    write_scalar_attr(file_id_, "ngll", H5T_NATIVE_INT, &ngll);
}

RestartWriter::~RestartWriter() {
    try {
        close();
    } catch (...) {
    }
}

void RestartWriter::write(int step, double time_s, const std::vector<double>& displacement,
                          const std::vector<double>& velocity,
                          const std::vector<double>& acceleration,
                          const std::vector<double>& pml_damping) {
    if (file_id_ < 0) {
        throw std::runtime_error("RestartWriter: file not open");
    }

    // Update step and time attributes
    write_scalar_attr(file_id_, "step", H5T_NATIVE_INT, &step);
    write_scalar_attr(file_id_, "time_s", H5T_NATIVE_DOUBLE, &time_s);

    hsize_t dims4[4] = {static_cast<hsize_t>(n_elem_local_), static_cast<hsize_t>(ngll_),
                        static_cast<hsize_t>(ngll_), static_cast<hsize_t>(ngll_)};
    hsize_t dims4_3[5] = {static_cast<hsize_t>(n_elem_local_), static_cast<hsize_t>(ngll_),
                          static_cast<hsize_t>(ngll_), static_cast<hsize_t>(ngll_), 3};

    write_dset(file_id_, "displacement", displacement, 5, dims4_3);
    write_dset(file_id_, "velocity", velocity, 5, dims4_3);
    write_dset(file_id_, "acceleration", acceleration, 5, dims4_3);
    write_dset(file_id_, "pml_damping", pml_damping, 4, dims4);
}

void RestartWriter::close() {
    if (file_id_ >= 0) {
        H5Fclose(file_id_);
        file_id_ = -1;
    }
}

RestartState read_restart(const std::string& output_dir, const std::string& source_direction,
                          int rank) {
    std::string filepath =
        output_dir + "/" + source_direction + "/restart_" + std::to_string(rank) + ".h5";

    hid_t fid = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) {
        throw std::runtime_error("read_restart: cannot open " + filepath);
    }

    RestartState state;

    // Read attributes
    hid_t attr;
    attr = H5Aopen(fid, "step", H5P_DEFAULT);
    if (attr >= 0) {
        H5Aread(attr, H5T_NATIVE_INT, &state.step);
        H5Aclose(attr);
    }
    attr = H5Aopen(fid, "time_s", H5P_DEFAULT);
    if (attr >= 0) {
        H5Aread(attr, H5T_NATIVE_DOUBLE, &state.time_s);
        H5Aclose(attr);
    }

    // Helper lambda to read a dataset
    auto read_dset = [&](const std::string& name, std::vector<double>& out) {
        hid_t dset = H5Dopen2(fid, name.c_str(), H5P_DEFAULT);
        if (dset < 0)
            return;
        hid_t space = H5Dget_space(dset);
        hssize_t nel = H5Sget_select_npoints(space);
        out.resize(static_cast<size_t>(nel));
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
        H5Dclose(dset);
        H5Sclose(space);
    };

    read_dset("displacement", state.displacement);
    read_dset("velocity", state.velocity);
    read_dset("acceleration", state.acceleration);
    read_dset("pml_damping", state.pml_damping);

    H5Fclose(fid);
    return state;
}

}  // namespace gf