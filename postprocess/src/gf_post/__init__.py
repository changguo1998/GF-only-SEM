"""gf_post: Strain Green's function extraction from SEM record files.

Reads shallow mesh-vertex strain records from the C++ SEM solver and mesh.h5
vertex geometry, assembles the full 3×6 strain Green's tensor at recorded
mesh vertices, and writes horizontally tiled HDF5 output.

No receiver positions — output is the configured shallow full-volume field.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
