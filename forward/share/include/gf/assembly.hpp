#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gf/types.hpp"

namespace gf {

/// Scatter element-local residual to per-rank global residual.
///
/// For each element e and GLL node n,
///   rank_node_residual[3 * local_cell2rank_node[e*n_node+n] + d] += local_cell_residual[3 *
///   (e*n_node+n) + d]
///
/// At shared physical nodes (same node_id), contributions from all
/// sharing elements accumulate.  The global array is zeroed before
/// accumulation.
void scatter_to_rank(const std::vector<double>& local_cell_residual,
                     const std::vector<int32_t>& local_cell2rank_node, int n_local_cell,
                     int n_node, std::vector<double>& rank_node_residual);

/// Gather per-rank global field to element-local array.
///
/// For each element e and GLL node n,
///   local_cell_field[3 * (e*n_node+n) + d] = rank_node_field[3 *
///   local_cell2rank_node[e*n_node+n] + d]
void gather_from_rank(const std::vector<double>& rank_node_field,
                      const std::vector<int32_t>& local_cell2rank_node, int n_local_cell,
                      int n_node, std::vector<double>& local_cell_field);

// ---- Legacy API (kept for existing test linkage, DEPRECATED) ----

void assemble_residual(const std::vector<double>& local_cell_residual, const RankData& rank_data,
                       std::vector<double>& rank_node_residual);

void add_source_to_rhs(int elem_idx, int gll_i, int gll_j, int gll_k, double fx, double fy,
                       double fz, const RankData& rank_data, std::vector<double>& rhs);

}  // namespace gf