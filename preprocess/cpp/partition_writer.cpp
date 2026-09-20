/// partition_writer.cpp — build solver-ready per-rank partition files

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gf_config.h"
#include "gf_hdf5_helpers.hpp"

namespace gf {
namespace {

struct RecordingData {
    std::vector<int64_t> global_cell_ids;
    std::vector<int32_t> local_cell_indices;
    std::vector<int64_t> gll_node_ids;
    std::vector<double> gll_node_coords;
    std::vector<int32_t> cell_gll_node_index;
};

struct RankPartition {
    std::vector<int64_t> local_cell_ids;
    std::vector<int64_t> ghost_cell_ids;
    std::vector<int32_t> ghost_owners;
    std::vector<int32_t> local_cell2rank_node;
    std::vector<int32_t> local_cell2global_node;
    int n_rank_node = 0;
    std::map<int, std::vector<int32_t>> exchange_dofs;
    RecordingData recording;
};

void fail(const std::string& message) {
    fprintf(stderr, "ERROR: %s\n", message.c_str());
    std::exit(1);
}

void delete_link_if_present(hid_t location, const char* path) {
    if (H5Lexists(location, path, H5P_DEFAULT) > 0)
        H5Ldelete(location, path, H5P_DEFAULT);
}

hid_t create_group(hid_t location, const char* path) {
    hid_t link_properties = h5::lcpl_with_groups();
    hid_t group = H5Gcreate2(location, path, link_properties, H5P_DEFAULT, H5P_DEFAULT);
    H5Pclose(link_properties);
    if (group < 0)
        fail(std::string("cannot create HDF5 group: ") + path);
    return group;
}

void write_int_attribute(hid_t location, const char* name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attribute = H5Acreate2(location, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, H5T_NATIVE_INT, &value);
    H5Aclose(attribute);
    H5Sclose(space);
}

void write_double_attribute(hid_t location, const char* name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attribute =
        H5Acreate2(location, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attribute);
    H5Sclose(space);
}

void write_string_attribute(hid_t location, const char* name, const char* value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, std::strlen(value) + 1);
    H5Tset_strpad(type, H5T_STR_NULLTERM);
    hid_t attribute = H5Acreate2(location, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, type, value);
    H5Aclose(attribute);
    H5Tclose(type);
    H5Sclose(space);
}

template <typename T>
void write_vector(hid_t location, const char* name, const std::vector<T>& values, hid_t type,
                  const std::vector<hsize_t>& dimensions) {
    if (values.empty())
        return;
    hid_t space =
        H5Screate_simple(static_cast<int>(dimensions.size()), dimensions.data(), nullptr);
    hid_t dataset = H5Dcreate2(location, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset < 0)
        fail(std::string("cannot create HDF5 dataset: ") + name);
    H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    H5Dclose(dataset);
    H5Sclose(space);
}

/// Copy selected rows of an element-first dataset without changing its datatype or shape tail.
void copy_selected_dataset(hid_t source_file, const char* source_path, hid_t destination_group,
                           const char* destination_name, const std::vector<int64_t>& element_ids,
                           bool required) {
    if (H5Lexists(source_file, source_path, H5P_DEFAULT) <= 0) {
        if (required)
            fail(std::string("required dataset not found: ") + source_path);
        return;
    }

    hid_t source = H5Dopen2(source_file, source_path, H5P_DEFAULT);
    hid_t source_space = H5Dget_space(source);
    int rank = H5Sget_simple_extent_ndims(source_space);
    std::vector<hsize_t> source_dimensions(rank);
    H5Sget_simple_extent_dims(source_space, source_dimensions.data(), nullptr);
    if (rank < 1)
        fail(std::string("element dataset has no leading dimension: ") + source_path);

    hid_t file_type = H5Dget_type(source);
    hid_t native_type = H5Tget_native_type(file_type, H5T_DIR_ASCEND);
    hsize_t row_values = 1;
    for (int dimension = 1; dimension < rank; ++dimension)
        row_values *= source_dimensions[dimension];
    std::vector<unsigned char> selected_values(element_ids.size() * row_values *
                                               H5Tget_size(native_type));

    // Select all owned element rows in one HDF5 read; local IDs are globally sorted.
    H5Sselect_none(source_space);
    std::vector<hsize_t> start(rank, 0), count = source_dimensions;
    count[0] = 1;
    for (size_t local_index = 0; local_index < element_ids.size(); ++local_index) {
        int64_t global_index = element_ids[local_index];
        if (global_index < 0 || static_cast<hsize_t>(global_index) >= source_dimensions[0])
            fail(std::string("element index outside dataset: ") + source_path);
        start[0] = static_cast<hsize_t>(global_index);
        H5Sselect_hyperslab(source_space, local_index == 0 ? H5S_SELECT_SET : H5S_SELECT_OR,
                            start.data(), nullptr, count.data(), nullptr);
    }

    std::vector<hsize_t> destination_dimensions = source_dimensions;
    destination_dimensions[0] = static_cast<hsize_t>(element_ids.size());
    hid_t destination_space = H5Screate_simple(rank, destination_dimensions.data(), nullptr);
    hid_t destination = H5Dcreate2(destination_group, destination_name, file_type,
                                   destination_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (destination < 0)
        fail(std::string("cannot create partition dataset: ") + destination_name);
    H5Dread(source, native_type, destination_space, source_space, H5P_DEFAULT,
            selected_values.data());
    H5Dwrite(destination, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, selected_values.data());

    H5Dclose(destination);
    H5Sclose(destination_space);
    H5Tclose(native_type);
    H5Tclose(file_type);
    H5Sclose(source_space);
    H5Dclose(source);
}

std::vector<int32_t> read_global_node_ids(hid_t model_file) {
    std::vector<int64_t> values = h5::read_int64(model_file, "partition/global_cell2global_node");
    std::vector<int32_t> result(values.size());
    for (size_t index = 0; index < values.size(); ++index)
        result[index] = static_cast<int32_t>(values[index]);
    return result;
}

/// Build local/ghost element lists from shared topology surfaces.
std::vector<RankPartition> build_rank_elements(const std::vector<int64_t>& cell_to_surface,
                                               int n_cell, int n_faces,
                                               const std::vector<int32_t>& element_to_rank,
                                               int n_ranks) {
    std::vector<RankPartition> ranks(n_ranks);
    for (int cell = 0; cell < n_cell; ++cell)
        ranks[element_to_rank[cell]].local_cell_ids.push_back(cell);

    std::map<int64_t, std::vector<int>> surface_cells;
    for (int cell = 0; cell < n_cell; ++cell) {
        for (int face = 0; face < n_faces; ++face) {
            int64_t surface = std::abs(cell_to_surface[cell * n_faces + face]);
            if (surface > 0)
                surface_cells[surface].push_back(cell);
        }
    }

    for (const auto& [surface, cells] : surface_cells) {
        (void)surface;
        for (size_t left = 0; left < cells.size(); ++left) {
            for (size_t right = left + 1; right < cells.size(); ++right) {
                int first_cell = cells[left];
                int second_cell = cells[right];
                int first_rank = element_to_rank[first_cell];
                int second_rank = element_to_rank[second_cell];
                if (first_rank == second_rank)
                    continue;
                const std::pair<int, int> additions[] = {{first_rank, second_cell},
                                                         {second_rank, first_cell}};
                for (const auto& [owner_rank, ghost_cell] : additions) {
                    auto& rank = ranks[owner_rank];
                    if (std::find(rank.ghost_cell_ids.begin(), rank.ghost_cell_ids.end(),
                                  ghost_cell) == rank.ghost_cell_ids.end()) {
                        rank.ghost_cell_ids.push_back(ghost_cell);
                        rank.ghost_owners.push_back(element_to_rank[ghost_cell]);
                    }
                }
            }
        }
    }
    return ranks;
}

/// Compact global node IDs per rank and build all co-owner exchange patterns.
void build_rank_nodes(std::vector<RankPartition>& ranks, const std::vector<int32_t>& global_nodes,
                      int n_cell, int nodes_per_cell,
                      const std::vector<int32_t>& element_to_rank) {
    int maximum_global_node =
        global_nodes.empty() ? -1 : *std::max_element(global_nodes.begin(), global_nodes.end());
    std::vector<std::set<int>> node_owner_ranks(maximum_global_node + 1);
    for (int cell = 0; cell < n_cell; ++cell) {
        int owner = element_to_rank[cell];
        const int32_t* cell_nodes = global_nodes.data() + cell * nodes_per_cell;
        for (int node = 0; node < nodes_per_cell; ++node)
            node_owner_ranks[cell_nodes[node]].insert(owner);
    }

    for (int rank_index = 0; rank_index < static_cast<int>(ranks.size()); ++rank_index) {
        auto& rank = ranks[rank_index];
        std::vector<int32_t> compact_global_nodes;
        for (int64_t cell : rank.local_cell_ids) {
            const int32_t* cell_nodes = global_nodes.data() + cell * nodes_per_cell;
            compact_global_nodes.insert(compact_global_nodes.end(), cell_nodes,
                                        cell_nodes + nodes_per_cell);
        }
        for (int64_t cell : rank.ghost_cell_ids) {
            const int32_t* cell_nodes = global_nodes.data() + cell * nodes_per_cell;
            compact_global_nodes.insert(compact_global_nodes.end(), cell_nodes,
                                        cell_nodes + nodes_per_cell);
        }
        std::sort(compact_global_nodes.begin(), compact_global_nodes.end());
        compact_global_nodes.erase(
            std::unique(compact_global_nodes.begin(), compact_global_nodes.end()),
            compact_global_nodes.end());
        rank.n_rank_node = static_cast<int>(compact_global_nodes.size());

        std::map<int, std::vector<std::pair<int32_t, int32_t>>> shared_nodes;
        for (int64_t cell : rank.local_cell_ids) {
            const int32_t* cell_nodes = global_nodes.data() + cell * nodes_per_cell;
            for (int node = 0; node < nodes_per_cell; ++node) {
                int32_t global_node = cell_nodes[node];
                int32_t compact_node = static_cast<int32_t>(
                    std::lower_bound(compact_global_nodes.begin(), compact_global_nodes.end(),
                                     global_node) -
                    compact_global_nodes.begin());
                rank.local_cell2global_node.push_back(global_node);
                rank.local_cell2rank_node.push_back(compact_node);
                for (int other_rank : node_owner_ranks[global_node]) {
                    if (other_rank != rank_index)
                        shared_nodes[other_rank].emplace_back(global_node, compact_node);
                }
            }
        }

        for (auto& [other_rank, nodes] : shared_nodes) {
            std::sort(nodes.begin(), nodes.end());
            nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
            auto& dofs = rank.exchange_dofs[other_rank];
            dofs.reserve(nodes.size() * 3);
            for (const auto& [global_node, compact_node] : nodes) {
                (void)global_node;
                dofs.push_back(compact_node * 3);
                dofs.push_back(compact_node * 3 + 1);
                dofs.push_back(compact_node * 3 + 2);
            }
        }
    }
}

double snap_record_depth(const std::vector<double>& coordinates, int n_cell, int nodes_per_cell,
                         double zmin, double zmax, double requested_depth) {
    if (requested_depth <= 0.0)
        return 0.0;
    std::vector<double> face_levels;
    face_levels.reserve(n_cell * 2);
    for (int cell = 0; cell < n_cell; ++cell) {
        double cell_zmin = std::numeric_limits<double>::max();
        double cell_zmax = std::numeric_limits<double>::lowest();
        for (int node = 0; node < nodes_per_cell; ++node) {
            double z = coordinates[(cell * nodes_per_cell + node) * 3 + 2];
            cell_zmin = std::min(cell_zmin, z);
            cell_zmax = std::max(cell_zmax, z);
        }
        if (cell_zmin >= zmin && cell_zmin <= zmax)
            face_levels.push_back(cell_zmin);
        if (cell_zmax >= zmin && cell_zmax <= zmax)
            face_levels.push_back(cell_zmax);
    }
    std::sort(face_levels.begin(), face_levels.end());
    face_levels.erase(std::unique(face_levels.begin(), face_levels.end()), face_levels.end());
    double target_z = zmin + requested_depth;
    auto level = std::lower_bound(face_levels.begin(), face_levels.end(), target_z);
    return level == face_levels.end() ? zmax - zmin : *level - zmin;
}

/// Build per-rank recording maps from non-PML cells above the snapped depth.
void build_recording_maps(std::vector<RankPartition>& ranks,
                          const std::vector<int32_t>& global_nodes,
                          const std::vector<double>& coordinates,
                          const std::vector<int64_t>& is_pml, int nodes_per_cell, double zmin,
                          double actual_depth) {
    if (actual_depth <= 0.0)
        return;
    double target_z = zmin + actual_depth;
    for (auto& rank : ranks) {
        std::set<int64_t> unique_node_ids;
        for (size_t local_cell = 0; local_cell < rank.local_cell_ids.size(); ++local_cell) {
            int64_t global_cell = rank.local_cell_ids[local_cell];
            if (is_pml[global_cell] != 0)
                continue;
            double centroid_z = 0.0;
            for (int node = 0; node < nodes_per_cell; ++node)
                centroid_z += coordinates[(global_cell * nodes_per_cell + node) * 3 + 2];
            centroid_z /= nodes_per_cell;
            if (centroid_z <= target_z + 1.0e-12) {
                rank.recording.global_cell_ids.push_back(global_cell);
                rank.recording.local_cell_indices.push_back(static_cast<int32_t>(local_cell));
                const int32_t* cell_nodes = global_nodes.data() + global_cell * nodes_per_cell;
                unique_node_ids.insert(cell_nodes, cell_nodes + nodes_per_cell);
            }
        }

        rank.recording.gll_node_ids.assign(unique_node_ids.begin(), unique_node_ids.end());
        std::unordered_map<int64_t, int32_t> node_to_recording_index;
        for (size_t index = 0; index < rank.recording.gll_node_ids.size(); ++index)
            node_to_recording_index[rank.recording.gll_node_ids[index]] =
                static_cast<int32_t>(index);
        rank.recording.gll_node_coords.assign(rank.recording.gll_node_ids.size() * 3,
                                              std::numeric_limits<double>::quiet_NaN());

        for (int64_t global_cell : rank.recording.global_cell_ids) {
            const int32_t* cell_nodes = global_nodes.data() + global_cell * nodes_per_cell;
            for (int node = 0; node < nodes_per_cell; ++node) {
                int32_t recording_index = node_to_recording_index[cell_nodes[node]];
                rank.recording.cell_gll_node_index.push_back(recording_index);
                double* destination = rank.recording.gll_node_coords.data() + recording_index * 3;
                if (std::isnan(destination[0])) {
                    const double* source =
                        coordinates.data() + (global_cell * nodes_per_cell + node) * 3;
                    std::copy(source, source + 3, destination);
                }
            }
        }
    }
}

std::vector<int64_t> compute_tile_index(const Config& cfg, int n_cell, double zmin, double zmax,
                                        double actual_depth) {
    std::vector<int64_t> tile_index(n_cell, -1);
    if (cfg.tilex_elements.empty() || cfg.tiley_elements.empty() || cfg.nx_elements <= 0 ||
        cfg.ny_elements <= 0)
        return tile_index;
    int xy_elements = cfg.nx_elements * cfg.ny_elements;
    if (n_cell % xy_elements != 0)
        fail("n_cell is not divisible by nx_elements * ny_elements");
    int nz_elements = n_cell / xy_elements;
    double element_depth = (zmax - zmin) / nz_elements;

    std::vector<int> tilex_boundaries(1, 0), tiley_boundaries(1, 0);
    for (int width : cfg.tilex_elements)
        tilex_boundaries.push_back(tilex_boundaries.back() + width);
    for (int width : cfg.tiley_elements)
        tiley_boundaries.push_back(tiley_boundaries.back() + width);

    for (int cell = 0; cell < n_cell; ++cell) {
        int x = cell % cfg.nx_elements;
        int y = (cell / cfg.nx_elements) % cfg.ny_elements;
        int z = cell / xy_elements;
        int interior_x = x - cfg.pml_xmin;
        int interior_y = y - cfg.pml_ymin;
        if (interior_x < 0 || interior_y < 0 ||
            interior_x >= cfg.nx_elements - cfg.pml_xmin - cfg.pml_xmax ||
            interior_y >= cfg.ny_elements - cfg.pml_ymin - cfg.pml_ymax)
            continue;
        if (actual_depth > 0.0 && zmin + (z + 0.5) * element_depth > zmin + actual_depth)
            continue;
        auto tile_x =
            std::upper_bound(tilex_boundaries.begin(), tilex_boundaries.end(), interior_x);
        auto tile_y =
            std::upper_bound(tiley_boundaries.begin(), tiley_boundaries.end(), interior_y);
        int x_index = static_cast<int>(tile_x - tilex_boundaries.begin()) - 1;
        int y_index = static_cast<int>(tile_y - tiley_boundaries.begin()) - 1;
        if (x_index >= 0 && x_index < static_cast<int>(cfg.tilex_elements.size()) &&
            y_index >= 0 && y_index < static_cast<int>(cfg.tiley_elements.size()))
            tile_index[cell] = y_index * cfg.tilex_elements.size() + x_index;
    }
    return tile_index;
}

void expose_field_cell_group(hid_t model_file) {
    delete_link_if_present(model_file, "field/cell");
    if (H5Lcreate_hard(model_file, "/field/element", model_file, "/field/cell", H5P_DEFAULT,
                       H5P_DEFAULT) < 0)
        fail("cannot create /field/cell hard link");
}

void write_recording_group(hid_t partition_file, const RecordingData& recording,
                           double requested_depth, double actual_depth) {
    if (recording.gll_node_ids.empty())
        return;
    hid_t group = create_group(partition_file, "recording");
    write_string_attribute(group, "basis", "gll");
    write_double_attribute(group, "record_depth_max_m", requested_depth);
    write_double_attribute(group, "record_depth_actual_m", actual_depth);
    write_int_attribute(group, "excludes_pml", 1);
    write_int_attribute(group, "n_unique_gll", static_cast<int>(recording.gll_node_ids.size()));
    write_int_attribute(group, "n_rec_cell", static_cast<int>(recording.global_cell_ids.size()));
    write_vector(group, "gll_node_ids", recording.gll_node_ids, H5T_NATIVE_INT64,
                 {recording.gll_node_ids.size()});
    write_vector(group, "gll_node_coords", recording.gll_node_coords, H5T_NATIVE_DOUBLE,
                 {recording.gll_node_ids.size(), 3});
    write_vector(group, "rec_cell_local", recording.local_cell_indices, H5T_NATIVE_INT32,
                 {recording.local_cell_indices.size()});
    write_vector(group, "rec_cell_global_ids", recording.global_cell_ids, H5T_NATIVE_INT64,
                 {recording.global_cell_ids.size()});
    write_vector(group, "cell_gll_node_index", recording.cell_gll_node_index, H5T_NATIVE_INT32,
                 {recording.cell_gll_node_index.size()});
    H5Gclose(group);
}

void write_one_partition(hid_t model_file, const std::filesystem::path& path,
                         const RankPartition& rank, const std::vector<int32_t>& element_to_rank,
                         const std::vector<int64_t>& tile_index, int n_ranks, int ngll,
                         double requested_depth, double actual_depth) {
    hid_t partition_file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (partition_file < 0)
        fail("cannot create partition file: " + path.string());
    hid_t field_cell = create_group(partition_file, "field/cell");

    const char* required_fields[] = {"coords", "dxi_dx",  "jacobian", "mass", "vp",
                                     "vs",     "density", "lambda",   "mu",   "damping"};
    for (const char* field : required_fields) {
        std::string source_path = std::string("field/element/") + field;
        copy_selected_dataset(model_file, source_path.c_str(), field_cell, field,
                              rank.local_cell_ids, true);
    }
    const std::pair<const char*, const char*> optional_fields[] = {
        {"is_pml", "is_pml"},
        {"pml_region", "pml_region"},
        {"cpml_K", "pml_K"},
        {"cpml_d", "pml_d"},
        {"cpml_alpha", "pml_alpha"},
        {"pml_coef_alpha", "pml_coef_alpha"},
        {"pml_coef_beta", "pml_coef_beta"},
        {"pml_coef_abar", "pml_coef_abar"},
        {"pml_coef_strain", "pml_coef_strain"},
        {"tau_sigma", "tau_sigma"},
        {"tau_epsilon_mu", "tau_epsilon_mu"},
        {"tau_epsilon_kappa", "tau_epsilon_kappa"},
        {"q_mu", "q_mu"},
        {"q_kappa", "q_kappa"},
    };
    for (const auto& [source_name, destination_name] : optional_fields) {
        std::string source_path = std::string("field/element/") + source_name;
        copy_selected_dataset(model_file, source_path.c_str(), field_cell, destination_name,
                              rank.local_cell_ids, false);
    }

    std::vector<int64_t> local_tiles;
    local_tiles.reserve(rank.local_cell_ids.size());
    for (int64_t cell : rank.local_cell_ids)
        local_tiles.push_back(tile_index[cell]);
    write_vector(field_cell, "tile_index", local_tiles, H5T_NATIVE_INT64, {local_tiles.size()});
    write_vector(field_cell, "local_cell2rank_node", rank.local_cell2rank_node, H5T_NATIVE_INT32,
                 {rank.local_cell_ids.size(), static_cast<hsize_t>(ngll),
                  static_cast<hsize_t>(ngll), static_cast<hsize_t>(ngll)});
    write_vector(field_cell, "local_cell2global_node", rank.local_cell2global_node,
                 H5T_NATIVE_INT32,
                 {rank.local_cell_ids.size(), static_cast<hsize_t>(ngll),
                  static_cast<hsize_t>(ngll), static_cast<hsize_t>(ngll)});
    write_int_attribute(field_cell, "n_rank_node", rank.n_rank_node);
    H5Gclose(field_cell);

    hid_t field_surface = create_group(partition_file, "field/surface");
    std::vector<hsize_t> boundary_dimensions =
        h5::get_dims(model_file, "field/surface/boundary_tag");
    std::vector<int64_t> boundary_indices(boundary_dimensions[0]);
    std::iota(boundary_indices.begin(), boundary_indices.end(), 0);
    copy_selected_dataset(model_file, "field/surface/boundary_tag", field_surface, "boundary_tag",
                          boundary_indices, true);
    H5Gclose(field_surface);

    hid_t partition_group = create_group(partition_file, "partition");
    write_int_attribute(partition_group, "n_ranks", n_ranks);
    write_vector(partition_group, "element_to_rank", element_to_rank, H5T_NATIVE_INT32,
                 {element_to_rank.size()});
    write_vector(partition_group, "local_cell_ids", rank.local_cell_ids, H5T_NATIVE_INT64,
                 {rank.local_cell_ids.size()});
    write_vector(partition_group, "ghost_cell_ids", rank.ghost_cell_ids, H5T_NATIVE_INT64,
                 {rank.ghost_cell_ids.size()});
    write_vector(partition_group, "ghost_owners", rank.ghost_owners, H5T_NATIVE_INT32,
                 {rank.ghost_owners.size()});
    if (!rank.exchange_dofs.empty()) {
        hid_t exchange_group = create_group(partition_group, "exchange");
        for (const auto& [neighbor, dofs] : rank.exchange_dofs) {
            std::string neighbor_name = "neighbor_" + std::to_string(neighbor);
            hid_t neighbor_group = create_group(exchange_group, neighbor_name.c_str());
            write_vector(neighbor_group, "send_dof", dofs, H5T_NATIVE_INT32, {dofs.size()});
            write_vector(neighbor_group, "recv_dof", dofs, H5T_NATIVE_INT32, {dofs.size()});
            H5Gclose(neighbor_group);
        }
        H5Gclose(exchange_group);
    }
    H5Gclose(partition_group);

    write_recording_group(partition_file, rank.recording, requested_depth, actual_depth);
    H5Fclose(partition_file);
}

}  // namespace

double write_partition_files(const char* model_path, const Config& cfg,
                             const std::vector<int32_t>& element_to_rank) {
    fprintf(stderr, "=== Writing solver partitions ===\n");
    hid_t model_file = h5::open_or_fail(model_path, H5F_ACC_RDWR);
    std::vector<hsize_t> coordinate_dimensions = h5::get_dims(model_file, "field/element/coords");
    int n_cell = static_cast<int>(coordinate_dimensions[0]);
    int ngll = static_cast<int>(coordinate_dimensions[1]);
    int nodes_per_cell = ngll * ngll * ngll;
    if (static_cast<int>(element_to_rank.size()) != n_cell)
        fail("element_to_rank size does not match n_cell");
    for (int32_t rank : element_to_rank) {
        if (rank < 0 || rank >= cfg.n_ranks)
            fail("element_to_rank contains an invalid rank");
    }

    std::vector<int64_t> cell_to_surface = h5::read_int64(model_file, "topology/cell_to_surface");
    std::vector<hsize_t> surface_dimensions = h5::get_dims(model_file, "topology/cell_to_surface");
    std::vector<int32_t> global_nodes = read_global_node_ids(model_file);
    std::vector<double> coordinates = h5::read_double(model_file, "field/element/coords");
    std::vector<int64_t> is_pml = h5::read_int64(model_file, "field/element/is_pml");

    double zmin = 0.0, zmax = 0.0;
    hid_t domain_group = H5Gopen2(model_file, "domain", H5P_DEFAULT);
    if (domain_group < 0)
        fail("model.h5 is missing /domain");
    for (const auto& [name, destination] :
         std::array<std::pair<const char*, double*>, 2>{{{"zmin", &zmin}, {"zmax", &zmax}}}) {
        hid_t attribute = H5Aopen(domain_group, name, H5P_DEFAULT);
        if (attribute < 0)
            fail(std::string("missing /domain attribute: ") + name);
        H5Aread(attribute, H5T_NATIVE_DOUBLE, destination);
        H5Aclose(attribute);
    }
    H5Gclose(domain_group);

    std::vector<RankPartition> ranks =
        build_rank_elements(cell_to_surface, n_cell, static_cast<int>(surface_dimensions[1]),
                            element_to_rank, cfg.n_ranks);
    build_rank_nodes(ranks, global_nodes, n_cell, nodes_per_cell, element_to_rank);
    double actual_depth =
        snap_record_depth(coordinates, n_cell, nodes_per_cell, zmin, zmax, cfg.record_depth_max_m);
    build_recording_maps(ranks, global_nodes, coordinates, is_pml, nodes_per_cell, zmin,
                         actual_depth);
    std::vector<int64_t> tile_index = compute_tile_index(cfg, n_cell, zmin, zmax, actual_depth);

    h5::write_int64(model_file, "field/element/tile_index", tile_index,
                    {static_cast<hsize_t>(n_cell)});
    h5::write_int32(model_file, "field/element/global_cell2global_node", global_nodes,
                    {static_cast<hsize_t>(n_cell), static_cast<hsize_t>(ngll),
                     static_cast<hsize_t>(ngll), static_cast<hsize_t>(ngll)});
    expose_field_cell_group(model_file);

    std::filesystem::path partition_directory =
        std::filesystem::path(model_path).parent_path() / "partitions";
    std::filesystem::create_directories(partition_directory);
    for (const auto& entry : std::filesystem::directory_iterator(partition_directory)) {
        std::string filename = entry.path().filename().string();
        constexpr const char* prefix = "partition_";
        constexpr const char* suffix = ".h5";
        if (!entry.is_regular_file() || filename.rfind(prefix, 0) != 0 ||
            filename.size() <= std::strlen(prefix) + std::strlen(suffix) ||
            filename.compare(filename.size() - std::strlen(suffix), std::strlen(suffix), suffix) !=
                0)
            continue;
        std::string rank_text = filename.substr(
            std::strlen(prefix), filename.size() - std::strlen(prefix) - std::strlen(suffix));
        if (std::all_of(rank_text.begin(), rank_text.end(),
                        [](unsigned char character) { return std::isdigit(character); }))
            std::filesystem::remove(entry.path());
    }
    for (int rank = 0; rank < cfg.n_ranks; ++rank) {
        if (ranks[rank].local_cell_ids.empty())
            fail("METIS produced an empty rank " + std::to_string(rank));
        std::filesystem::path partition_path =
            partition_directory / ("partition_" + std::to_string(rank) + ".h5");
        write_one_partition(model_file, partition_path, ranks[rank], element_to_rank, tile_index,
                            cfg.n_ranks, ngll, cfg.record_depth_max_m, actual_depth);
        fprintf(stderr, "  rank %d: %zu cells, %d nodes, %zu recording cells\n", rank,
                ranks[rank].local_cell_ids.size(), ranks[rank].n_rank_node,
                ranks[rank].recording.global_cell_ids.size());
    }

    H5Fclose(model_file);
    return actual_depth;
}

}  // namespace gf
