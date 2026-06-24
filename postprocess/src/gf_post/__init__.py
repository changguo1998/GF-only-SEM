"""gf_post: Strain Green's function extraction from SEM checkpoint files.

Reads HDF5 strain checkpoints from the C++ SEM solver and mesh.h5 for
GLL-node geometry, assembles the full 3×6 strain Green's tensor at every
GLL node, and writes spatially tiled HDF5 output.

No receiver positions — output is the full GLL-node field.
"""

__version__ = "0.2.0"

from gf_post.geometry import (
    gll_nodes_1d,
    gll_weights_1d,
    gll_nodes_3d,
    lagrange_basis_1d,
    lagrange_basis_3d,
)
__all__ = [
    "__version__",
    "gll_nodes_1d",
    "gll_weights_1d",
    "gll_nodes_3d",
    "lagrange_basis_1d",
    "lagrange_basis_3d",
]