/// Generate the topology-only HDF5 mesh for the pure C++ layered example.

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr int kNx = 22;
constexpr int kNy = 22;
constexpr int kNz = 11;
constexpr double kLengthX = 10000.0;
constexpr double kLengthY = 10000.0;
constexpr double kLengthZ = 5000.0;
constexpr double kInterfaceDepth = 500.0;

using Edge = std::array<int64_t, 2>;
using Face = std::array<int64_t, 4>;
using Cell = std::array<int64_t, 8>;

const std::array<std::array<int, 4>, 6> kHexFaces = {
    {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {3, 7, 6, 2}, {0, 4, 7, 3}, {1, 2, 6, 5}}};

int64_t vertex_id(int x_index, int y_index, int z_index) {
    return static_cast<int64_t>(z_index) * (kNy + 1) * (kNx + 1) +
           static_cast<int64_t>(y_index) * (kNx + 1) + x_index;
}

bool same_orientation(const Face& first, const Face& second) {
    for (int offset = 0; offset < 4; ++offset) {
        if (std::abs(second[offset]) != std::abs(first[0]))
            continue;
        for (int i = 0; i < 4; ++i) {
            if (first[i] != second[(offset + i) % 4])
                return false;
        }
        return true;
    }
    return false;
}

template <typename Value>
void write_dataset(hid_t group, const char* name, hid_t datatype, const std::vector<Value>& values,
                   hsize_t rows, hsize_t columns) {
    hsize_t dimensions[2] = {rows, columns};
    hid_t dataspace = H5Screate_simple(2, dimensions, nullptr);
    hid_t dataset =
        H5Dcreate2(group, name, datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    H5Dclose(dataset);
    H5Sclose(dataspace);
}

void write_count(hid_t group, const char* name, int64_t value) {
    hid_t dataspace = H5Screate(H5S_SCALAR);
    hid_t attribute =
        H5Acreate2(group, name, H5T_NATIVE_INT64, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, H5T_NATIVE_INT64, &value);
    H5Aclose(attribute);
    H5Sclose(dataspace);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string output_path = argc > 1 ? argv[1] : "model.h5";

    std::vector<double> coordinates;
    coordinates.reserve(static_cast<std::size_t>(kNx + 1) * (kNy + 1) * (kNz + 1) * 3);
    for (int z_index = 0; z_index <= kNz; ++z_index) {
        double depth_m =
            z_index == 0
                ? 0.0
                : kInterfaceDepth + (z_index - 1) * (kLengthZ - kInterfaceDepth) / (kNz - 1);
        for (int y_index = 0; y_index <= kNy; ++y_index) {
            for (int x_index = 0; x_index <= kNx; ++x_index) {
                coordinates.push_back(x_index * kLengthX / kNx);
                coordinates.push_back(y_index * kLengthY / kNy);
                coordinates.push_back(depth_m);
            }
        }
    }

    std::vector<Cell> cells;
    cells.reserve(kNx * kNy * kNz);
    for (int z_index = 0; z_index < kNz; ++z_index) {
        for (int y_index = 0; y_index < kNy; ++y_index) {
            for (int x_index = 0; x_index < kNx; ++x_index) {
                cells.push_back({vertex_id(x_index, y_index, z_index),
                                 vertex_id(x_index + 1, y_index, z_index),
                                 vertex_id(x_index + 1, y_index + 1, z_index),
                                 vertex_id(x_index, y_index + 1, z_index),
                                 vertex_id(x_index, y_index, z_index + 1),
                                 vertex_id(x_index + 1, y_index, z_index + 1),
                                 vertex_id(x_index + 1, y_index + 1, z_index + 1),
                                 vertex_id(x_index, y_index + 1, z_index + 1)});
            }
        }
    }

    std::map<Edge, int64_t> edge_ids;
    std::vector<Edge> edges;
    std::map<std::array<int64_t, 4>, int64_t> surface_ids;
    std::vector<Face> surfaces;
    std::vector<int64_t> cell_to_surface;
    cell_to_surface.reserve(cells.size() * 6);

    for (const Cell& cell : cells) {
        for (const auto& local_face : kHexFaces) {
            Face signed_edges{};
            std::array<int64_t, 4> unsigned_edges{};
            for (int edge_index = 0; edge_index < 4; ++edge_index) {
                int64_t first_vertex = cell[local_face[edge_index]];
                int64_t second_vertex = cell[local_face[(edge_index + 1) % 4]];
                Edge edge = {std::min(first_vertex, second_vertex),
                             std::max(first_vertex, second_vertex)};
                auto [iterator, inserted] = edge_ids.emplace(edge, edges.size() + 1);
                if (inserted)
                    edges.push_back(edge);
                int64_t edge_id = iterator->second;
                signed_edges[edge_index] = first_vertex < second_vertex ? edge_id : -edge_id;
                unsigned_edges[edge_index] = edge_id;
            }
            std::sort(unsigned_edges.begin(), unsigned_edges.end());
            auto [iterator, inserted] = surface_ids.emplace(unsigned_edges, surfaces.size() + 1);
            if (inserted)
                surfaces.push_back(signed_edges);
            int64_t surface_id = iterator->second;
            cell_to_surface.push_back(same_orientation(signed_edges, surfaces[surface_id - 1])
                                          ? surface_id
                                          : -surface_id);
        }
    }

    std::vector<int64_t> edge_to_vertex;
    edge_to_vertex.reserve(edges.size() * 2);
    for (const Edge& edge : edges) {
        edge_to_vertex.push_back(edge[0] + 1);
        edge_to_vertex.push_back(edge[1] + 1);
    }
    std::vector<int64_t> surface_to_edge;
    surface_to_edge.reserve(surfaces.size() * 4);
    for (const Face& surface : surfaces)
        surface_to_edge.insert(surface_to_edge.end(), surface.begin(), surface.end());

    hid_t file = H5Fcreate(output_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "Failed to create " << output_path << '\n';
        return 1;
    }
    hid_t topology = H5Gcreate2(file, "topology", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_dataset(topology, "vertex_to_coord", H5T_NATIVE_DOUBLE, coordinates,
                  coordinates.size() / 3, 3);
    write_dataset(topology, "edge_to_vertex", H5T_NATIVE_INT64, edge_to_vertex, edges.size(), 2);
    write_dataset(topology, "surface_to_edge", H5T_NATIVE_INT64, surface_to_edge, surfaces.size(),
                  4);
    write_dataset(topology, "cell_to_surface", H5T_NATIVE_INT64, cell_to_surface, cells.size(), 6);
    write_count(topology, "n_vertex", coordinates.size() / 3);
    write_count(topology, "n_edge", edges.size());
    write_count(topology, "n_surface", surfaces.size());
    write_count(topology, "n_cell", cells.size());
    H5Gclose(topology);
    H5Fclose(file);

    std::cout << "Wrote " << output_path << ": " << cells.size() << " elements, "
              << coordinates.size() / 3 << " vertices\n";
    return 0;
}
