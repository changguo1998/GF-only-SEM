"""gf_post: Strain Green's function extraction from SEM checkpoint files.

Reads HDF5 strain checkpoints from the C++ SEM solver and mesh.h5 for
GLL-node geometry, locates receiver positions within mesh elements,
performs GLL basis interpolation at receiver positions, and outputs
strain Green's functions as spatially tiled HDF5 files.
"""

__version__ = "0.1.0"

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