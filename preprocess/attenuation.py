"""
SLS attenuation preprocessor — τ-method relaxation time computation.

Computes τ_σ^l and τ_ε^l for each GLL node from Q_mu using the
τ-method (logarithmically spaced τ_σ, linear least-squares fit
for the anelastic weights w_l = τ_ε^l/τ_σ^l − 1).

References:
  - Blanch, Robertson & Symes (1995), "Modeling of a constant Q..."
  - Komatitsch & Tromp (1999), "Introduction to the spectral element method..."
  - SPECFEM3D attenuation implementation
"""

from __future__ import annotations

import numpy as np
import h5py
from typing import Optional, Tuple


def compute_tau_from_q(
    q_mu: np.ndarray,  # [n_cell, NGLL, NGLL, NGLL]
    ngll: int,
    n_sls: int = 3,
    f0: float = 2.0,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute τ_σ^l and τ_ε^l for each GLL node.

    τ-method algorithm:
      1. Choose τ_σ^l = 1 / (2π f_l) where f_l are log-spaced in
         [f_min, f_max] centered around f0.
      2. For each node with Q < ∞, solve the linear system:
           1/Q(ω_l) ≈ Σ_m w_m · (ω_l τ_σ^m) / (1 + ω_l² (τ_σ^m)²)
         for the anelastic weights w_m.
      3. Compute τ_ε^l = τ_σ^l · (1 + w_l).

      For nodes with Q = 0 or Q → ∞ (no attenuation):
         τ_ε^l = τ_σ^l  →  no relaxation.

    Parameters
    ----------
    q_mu : ndarray [n_cell, NGLL, NGLL, NGLL]
        Shear quality factor per GLL node.  Values <= 0 or inf
        mean no attenuation at that node.
    ngll : int
        Number of GLL points per dimension.
    n_sls : int
        Number of SLS relaxation mechanisms (default 3).
    f0 : float
        Reference frequency (Hz) at which Q is defined.
    f_min, f_max : float or None
        Frequency band edges.  Default: f0/200, f0*5 (covers ~2.5 decades).

    Returns
    -------
    tau_sigma : ndarray [n_cell, NGLL, NGLL, NGLL, n_sls]
        Stress relaxation times τ_σ^l (seconds).
    tau_epsilon : ndarray [n_cell, NGLL, NGLL, NGLL, n_sls]
        Strain relaxation times τ_ε^l (seconds).
        τ_ε > τ_σ for attenuation; τ_ε = τ_σ for no attenuation.
    """
    shape = q_mu.shape
    if len(shape) != 4:
        raise ValueError(f"q_mu must be 4D [n_cell, NGLL, NGLL, NGLL], got shape {shape}")
    n_cell = shape[0]

    # --- Step 1: Choose log-spaced τ_σ ---
    if f_min is None:
        f_min = f0 / 200.0
    if f_max is None:
        f_max = f0 * 5.0

    f_l = np.logspace(np.log10(f_min), np.log10(f_max), n_sls)
    tau_sigma_l = 1.0 / (2.0 * np.pi * f_l)  # [n_sls]

    # --- Step 2: Build the frequency-domain linear system ---
    omega_l = 2.0 * np.pi * f_l  # [n_sls]

    # Contribution matrix A[l, m]: response of mechanism m at frequency f_l
    A = np.zeros((n_sls, n_sls))
    for l in range(n_sls):
        for m in range(n_sls):
            w_ts = omega_l[l] * tau_sigma_l[m]
            A[l, m] = w_ts / (1.0 + w_ts * w_ts)

    # Pre-invert A (same for all nodes since tau_sigma_l is global)
    A_inv = np.linalg.inv(A)

    # --- Step 3: Compute per-node τ_ε ---
    tau_sigma = np.zeros((n_cell, ngll, ngll, ngll, n_sls))
    tau_epsilon = np.zeros((n_cell, ngll, ngll, ngll, n_sls))

    for cell in range(n_cell):
        for i in range(ngll):
            for j in range(ngll):
                for k in range(ngll):
                    q_val = q_mu[cell, i, j, k]

                    # Default τ_σ values (always valid)
                    tau_sigma[cell, i, j, k, :] = tau_sigma_l

                    if q_val <= 0.0 or not np.isfinite(q_val):
                        # No attenuation: τ_ε = τ_σ
                        tau_epsilon[cell, i, j, k, :] = tau_sigma_l
                        continue

                    # Solve A · w = b, where b[l] = 1/Q at each frequency
                    inv_q_target = 1.0 / q_val
                    b = np.full(n_sls, inv_q_target)
                    w = A_inv @ b

                    # τ_ε^l = τ_σ^l * (1 + w_l)
                    # Guard: w_l must be >= 0 (τ_ε ≥ τ_σ for stability)
                    tau_epsilon[cell, i, j, k, :] = tau_sigma_l * (1.0 + np.maximum(w, 0.0))

    return tau_sigma, tau_epsilon


def write_attenuation_to_model(
    model_path: str,
    q_kappa: np.ndarray,
    q_mu: np.ndarray,
    ngll: int,
    n_sls: int = 3,
    f0: float = 2.0,
) -> None:
    """
    Write tau_sigma, tau_epsilon, and Q fields to model.h5.

    Creates/overwrites datasets under /field/cell/.

    Parameters
    ----------
    model_path : str
        Path to existing model.h5 file.
    q_kappa : ndarray [n_cell, NGLL, NGLL, NGLL]
        Bulk quality factor.
    q_mu : ndarray [n_cell, NGLL, NGLL, NGLL]
        Shear quality factor.
    ngll : int
        GLL points per dimension.
    n_sls : int
        Number of SLS mechanisms.
    f0 : float
        Reference frequency (Hz).
    """
    tau_sigma, tau_epsilon = compute_tau_from_q(q_mu, ngll, n_sls, f0)

    with h5py.File(model_path, "a") as f:
        field_cell = f.require_group("field/cell")

        for name, data in [
            ("tau_sigma", tau_sigma),
            ("tau_epsilon", tau_epsilon),
            ("q_kappa", q_kappa),
            ("q_mu", q_mu),
        ]:
            if name in field_cell:
                del field_cell[name]
            field_cell.create_dataset(name, data=data, dtype="float64", compression=None)

        # Store metadata as attributes
        field_cell["tau_sigma"].attrs["n_sls"] = n_sls
        field_cell["tau_sigma"].attrs["f0_Hz"] = f0
        field_cell["tau_sigma"].attrs["description"] = (
            "Stress relaxation times τ_σ^l per GLL node, shape [n_cell, NGLL, NGLL, NGLL, n_sls]"
        )
        field_cell["tau_epsilon"].attrs["description"] = (
            "Strain relaxation times τ_ε^l per GLL node, shape [n_cell, NGLL, NGLL, NGLL, n_sls]"
        )
