#pragma once

#include <string>

namespace gf {

/// Main forward simulation driver.
///
/// Reads partition files and config, runs the Newmark time loop,
/// and writes strain checkpoint records.
///
/// \param partition_dir  Directory containing partition_{r}.h5 files
/// \param config_path    Path to config.h5 (rank-invariant)
/// \param output_dir     Directory for output wavefields
/// \param direction      Force direction string ("x", "y", or "z")
/// \return 0 on success, non-zero on failure
int run_forward(const std::string& partition_dir,
                const std::string& config_path,
                const std::string& output_dir,
                const std::string& direction);

} // namespace gf