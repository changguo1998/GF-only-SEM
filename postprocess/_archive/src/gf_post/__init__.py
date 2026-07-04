"""gf_post: Strain Green's function extraction from SEM record files.

Reads shallow mesh-vertex strain records from the C++ SEM solver.
Assembles 3×6 strain Green tensors at recorded vertices. Writes horizontal
HDF5 tiles.

No receivers. Output is the configured shallow field.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
