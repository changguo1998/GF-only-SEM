"""Source Time Function (STF) evaluator.

Evaluates a user-provided STF callable at each time step
t = 0, dt, 2*dt, ..., (nsteps-1)*dt.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import numpy as np
import numpy.typing as npt

# Accept either a Python scalar or a NumPy scalar
STFOutput = Union[float, np.floating[np.float64]]


def evaluate_stf(
    stf_func: Callable[[float], STFOutput], dt: float, nsteps: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate source time function at each time step.

    Args:
        stf_func: Callable taking a time (float) and returning the
            STF amplitude (float or np.floating).
        dt:   Time step in seconds.
        nsteps: Number of time steps.

    Returns:
        stf_t:    [nsteps] time values: ``0, dt, 2*dt, ...``
        stf_vals: [nsteps] corresponding STF amplitudes.
    """
    if nsteps <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    stf_t = np.empty(nsteps, dtype=np.float64)
    stf_vals = np.empty(nsteps, dtype=np.float64)

    for i in range(nsteps):
        t = i * dt
        stf_t[i] = t
        stf_vals[i] = stf_func(t)

    return stf_t, stf_vals
