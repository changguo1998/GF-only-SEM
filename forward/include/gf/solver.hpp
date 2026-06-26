#pragma once

#include <string>

namespace gf {

/// Main forward simulation driver.
///
/// Reads partition files and config from frozen paths relative to CWD:
///   - config.h5
///   - partitions/partition_{r}.h5
/// Writes strain records to:
///   - wavefields/{direction}/record_{r}.h5
///
/// \param direction      Force direction string ("x", "y", or "z")
/// \return 0 on success, non-zero on failure
int run_forward(const std::string& direction);

} // namespace gf