#pragma once

#include <string>

namespace gf {

/// Main forward simulation driver.
///
/// Reads partition files and config from frozen paths relative to CWD:
///   - config.h5
///   - partitions/partition_{r}.h5
/// Writes strain records to:
///   - wavefields/{direction}/record_{r}_{step}.h5
///
/// \param direction        Force direction string ("x", "y", or "z")
/// \param resume_mode      Resume from restart file
/// \param effective_nprocs Override MPI size (0 = auto from MPI). Used when
///                          MPI ranks exceed GPUs — solver reduces effective
///                          rank count and redistributes partitions.
/// \return 0 on success, non-zero on failure
int run_forward(const std::string& direction, bool resume_mode = false, int effective_nprocs = 0);

}  // namespace gf