"""Green's tensor assembly from three force-direction waveforms."""

import numpy as np
import numpy.typing as npt


def assemble_greens_tensor(
    waveforms: dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Assemble strain Green's tensor from 3 force-direction strain fields.
    
    Args:
        waveforms: dict with keys "fx", "fy", "fz".
            Each value has shape [nt, n_cell, NGLL, NGLL, NGLL, 6]
            (strain at all GLL nodes).
    
    Returns:
        [nt, n_cell, NGLL, NGLL, NGLL, 6, 3] Green's tensor.
        Shape[-2] = strain component (Voigt: xx,yy,zz,xy,xz,yz)
        Shape[-1] = force direction (0=x, 1=y, 2=z)
    """
    nt, n_cell, ngll1, ngll2, ngll3, _ = waveforms["fx"].shape
    
    tensor = np.zeros((nt, n_cell, ngll1, ngll2, ngll3, 6, 3), dtype=np.float64)
    tensor[:, :, :, :, :, :, 0] = waveforms["fx"]
    tensor[:, :, :, :, :, :, 1] = waveforms["fy"]
    tensor[:, :, :, :, :, :, 2] = waveforms["fz"]
    
    return tensor