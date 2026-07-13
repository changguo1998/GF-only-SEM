#pragma once

#include <cstddef>
#include <vector>

#include "gf/types.hpp"

namespace gf {

/// Point force source located at an arbitrary sub-element position.
///
/// Uses precomputed GLL nodal coordinates and inverse Jacobian dxi_dx
/// to locate the containing element and map to natural coordinates.
/// Once located, the force is distributed to GLL nodes via interpolation
/// weights (Lagrange polynomial evaluations at the source's natural coordinates).
///
/// Usage:
///   PointForceSource src;
///   src.locate(px, py, pz, coords, dxi_dx, n_local_element, ngll);
///   src.apply(fx, fy, fz, rank_data, rhs);
class PointForceSource {
public:
    int element_id = 0;                   // 1-based global element id (set by locate)
    int gll_i = 0, gll_j = 0, gll_k = 0;  // GLL indices for closest node
    double wx = 0.0, wy = 0.0, wz = 0.0;  // Lagrange interpolation weights

    /// Locate the source in the mesh and compute interpolation weights.
    ///
    /// Searches over local elements for the one containing the source point
    /// in natural coordinates (xi, eta, zeta) ∈ [-1, 1]³.
    ///
    /// @param[in] src_x,src_y,src_z  physical coordinates of source
    /// @param[in] coords             [n_local_element * ngll^3 * 3] GLL nodal coordinates
    /// @param[in] dxi_dx             [n_local_element * ngll^3 * 9] inverse Jacobian per GLL node
    /// @param[in] n_local_element       number of local elements
    /// @param[in] ngll               GLL order per axis
    /// @return true if source location was found in a local element
    bool locate(double src_x, double src_y, double src_z, const std::vector<double>& coords,
                const std::vector<double>& dxi_dx, int n_local_element, int ngll);

    /// Apply the source force to the RHS at the current time step.
    ///
    /// The force is distributed to GLL nodes via the precomputed
    /// interpolation weights (Lagrange interpolation). For a point source
    /// at the exact location of a GLL node, all weight goes to that node.
    ///
    /// @param[in] force_x,force_y,force_z  source force components
    /// @param[in] rank_data                per-rank metadata
    /// @param[in,out] rhs                  right-hand side, accumulated into
    void apply(double force_x, double force_y, double force_z, const RankData& rank_data,
               std::vector<double>& rhs) const;
};

}  // namespace gf