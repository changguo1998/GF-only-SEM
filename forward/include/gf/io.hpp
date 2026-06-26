#pragma once

#include <hdf5.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gf/types.hpp"

namespace gf {

/// Read partition_{r}.h5 for this MPI rank.
///
/// Reads local subset data from HDF5 partition file:
///   - Geometry and material fields (coords, jacobian, dxi_dx, mass, vp, vs, density, pml_damping)
///   - Topology (local_element_ids, ghost_element_ids, ghost_owners)
///   - Exchange patterns (neighbors)
///   - Polynomial order (ngll) from array shapes
///
/// \param path  Path to partition file, e.g. "partitions/partition_0.h5"
/// \return      Populated RankData for this rank
RankData read_partition(const std::string& path, int rank);

/// Read config.h5 (rank-invariant simulation configuration).
///
/// Reads all simulation parameters from the config file:
///   - Title, polynomial_order, solver_dt, output_dt_s, snapshot_stride, nsteps, cfl_safety
///   - Snapshot settings (stride, precision)
///   - Domain bounds, source STF
ConfigData read_config(const std::string& path);

/// Helper: read a double-precision dataset from an open HDF5 file.
std::vector<double> read_dataset_double(hid_t file_id, const std::string& name);

/// Helper: read an int64_t dataset from an open HDF5 file.
std::vector<int64_t> read_dataset_int64(hid_t file_id, const std::string& name);

/// Helper: read an int32_t dataset from an open HDF5 file.
std::vector<int32_t> read_dataset_int32(hid_t file_id, const std::string& name);

}  // namespace gf