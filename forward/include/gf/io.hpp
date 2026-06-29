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

/// Read ALL partitions and merge into a single RankData (single-rank mode).
///
/// Scans partition_dir for partition_{r}.h5 files, reads each, concatenates
/// element-based arrays, merges recording maps, and clears ghost/exchange
/// data. Used when nprocs==1 (non-MPI or mpirun -n 1).
///
/// \param partition_dir  Directory containing partition_{r}.h5 files, e.g. "partitions"
/// \return               Merged RankData for the full domain
RankData read_partition_all(const std::string& partition_dir);

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

// Read int64 dataset, return as std::vector<int> (narrowing conversion).
std::vector<int> read_dataset_int(hid_t file_id, const std::string& name);

}  // namespace gf