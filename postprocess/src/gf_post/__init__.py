"""gf_post: Strain Green's function extraction from SEM checkpoint files.

Reads HDF5 strain checkpoints from the C++ SEM solver and mesh.h5 for
GLL-node geometry, assembles the full 3×6 strain Green's tensor at every
GLL node, and writes spatially tiled HDF5 output.

No receiver positions — output is the full GLL-node field.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]