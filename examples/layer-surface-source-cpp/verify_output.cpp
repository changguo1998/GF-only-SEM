/// Verify that every Green-function field is finite and nonzero.

#include <hdf5.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool verify_dataset(hid_t file, const char* path) {
    hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
    if (dataset < 0)
        return false;
    hid_t dataspace = H5Dget_space(dataset);
    int rank = H5Sget_simple_extent_ndims(dataspace);
    std::vector<hsize_t> dimensions(rank);
    H5Sget_simple_extent_dims(dataspace, dimensions.data(), nullptr);
    std::size_t value_count = 1;
    for (hsize_t dimension : dimensions)
        value_count *= dimension;
    std::vector<double> values(value_count);
    bool read_ok =
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0;
    H5Sclose(dataspace);
    H5Dclose(dataset);
    if (!read_ok)
        return false;
    bool has_nonzero_value = false;
    for (double value : values) {
        if (!std::isfinite(value))
            return false;
        has_nonzero_value = has_nonzero_value || value != 0.0;
    }
    return has_nonzero_value;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output_directory = argc > 1 ? argv[1] : "greenfun";
    const char* dataset_paths[] = {"field/greens_tensor", "field/displacement_tensor",
                                   "field/velocity_tensor", "field/acceleration_tensor"};
    int tile_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(output_directory)) {
        if (!entry.is_regular_file() || entry.path().filename().string().rfind("tile_", 0) != 0)
            continue;
        hid_t file = H5Fopen(entry.path().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) {
            std::cerr << "Failed to open " << entry.path() << '\n';
            return 1;
        }
        for (const char* dataset_path : dataset_paths) {
            if (!verify_dataset(file, dataset_path)) {
                std::cerr << entry.path() << ':' << dataset_path
                          << " is missing, non-finite, or entirely zero\n";
                H5Fclose(file);
                return 1;
            }
        }
        H5Fclose(file);
        ++tile_count;
    }
    if (tile_count != 16) {
        std::cerr << "Expected 16 tiles, got " << tile_count << '\n';
        return 1;
    }
    std::cout << "Verified 16 tiles: all tensor fields are finite and nonzero\n";
    return 0;
}
