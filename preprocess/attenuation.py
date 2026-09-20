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
from scipy.optimize import nnls

NO_ATTENUATION_Q = 1.0e8
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
      2. Fit non-negative normalized memory weights over 100 logarithmic
         frequency samples so Re(M*) / Im(M*) approximates Q.
      3. Convert normalized weights to τ_ε/τ_σ ratios used by the solver.

      For nodes with Q <= 0, Q >= 1e8, or non-finite Q (no attenuation):
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
        Frequency band edges. Default: f0/20, f0*3, matching the range used
        by the SPECFEM three-mechanism reference setup.

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
    # --- Step 1: Choose log-spaced τ_σ ---
    if f_min is None:
        f_min = f0 / 20.0
    if f_max is None:
        f_max = f0 * 3.0

    f_l = np.logspace(np.log10(f_min), np.log10(f_max), n_sls)
    tau_sigma_l = 1.0 / (2.0 * np.pi * f_l)  # [n_sls]

    # --- Step 2: Fit the normalized memory weights over the frequency band ---
    sample_frequency_hz = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    angular_frequency = 2.0 * np.pi * sample_frequency_hz[:, np.newaxis]
    frequency_tau = angular_frequency * tau_sigma_l[np.newaxis, :]

    # --- Step 3: Compute per-node τ_ε, caching each distinct Q value ---
    output_shape = (*shape, n_sls)
    tau_sigma = np.broadcast_to(tau_sigma_l, output_shape).copy()
    tau_epsilon = tau_sigma.copy()
    valid_q = np.isfinite(q_mu) & (q_mu > 0.0) & (q_mu < NO_ATTENUATION_Q)
    for q_value in np.unique(q_mu[valid_q]):
        inverse_q = 1.0 / float(q_value)
        # Im(M*) - Re(M*) / Q = 0 is linear in normalized memory weights.
        fit_matrix = (frequency_tau + inverse_q) / (1.0 + frequency_tau**2)
        normalized_weights, _ = nnls(fit_matrix, np.full(sample_frequency_hz.size, inverse_q))
        weight_sum = float(np.sum(normalized_weights))
        if weight_sum >= 1.0:
            raise ValueError(f"Q={q_value:g} produces an unstable SLS fit")
        ratio_minus_one = n_sls * normalized_weights / (1.0 - weight_sum)
        tau_epsilon[q_mu == q_value] = tau_sigma_l * (1.0 + ratio_minus_one)

    return tau_sigma, tau_epsilon


def compute_unrelaxed_modulus_scale(
    tau_sigma: np.ndarray, tau_epsilon: np.ndarray, reference_frequency_hz: float
) -> np.ndarray:
    """Return the factor converting reference-frequency modulus to unrelaxed modulus."""
    ratio = tau_epsilon / tau_sigma
    normalized_weights = (ratio - 1.0) / np.sum(ratio, axis=-1, keepdims=True)
    frequency_tau = 2.0 * np.pi * reference_frequency_hz * tau_sigma
    real_modulus_ratio = 1.0 - np.sum(normalized_weights / (1.0 + frequency_tau**2), axis=-1)
    return 1.0 / real_modulus_ratio


def write_attenuation_to_model(
    model_path: str,
    q_kappa: np.ndarray,
    q_mu: np.ndarray,
    ngll: int,
    n_sls: int = 3,
    f0: float = 2.0,
) -> None:
    """
    Write shear/bulk relaxation times and Q fields to model.h5.

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
    tau_sigma, tau_epsilon_mu = compute_tau_from_q(q_mu, ngll, n_sls, f0)
    tau_sigma_kappa, tau_epsilon_kappa = compute_tau_from_q(q_kappa, ngll, n_sls, f0)
    if not np.array_equal(tau_sigma, tau_sigma_kappa):
        raise ValueError("Q_mu and Q_kappa produced inconsistent tau_sigma values")

    with h5py.File(model_path, "a") as f:
        field_root = f.require_group("field")
        if "cell" in field_root and "coords" in field_root["cell"]:
            field_cell = field_root["cell"]
        elif "element" in field_root and "coords" in field_root["element"]:
            field_cell = field_root["element"]
        else:
            field_cell = field_root.require_group("cell")

        for name, data in [
            ("tau_sigma", tau_sigma),
            ("tau_epsilon_mu", tau_epsilon_mu),
            ("tau_epsilon_kappa", tau_epsilon_kappa),
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
        field_cell["tau_epsilon_mu"].attrs["description"] = (
            "Shear strain relaxation times per GLL node, shape [n_cell, NGLL, NGLL, NGLL, n_sls]"
        )
        field_cell["tau_epsilon_kappa"].attrs["description"] = (
            "Bulk strain relaxation times per GLL node, shape [n_cell, NGLL, NGLL, NGLL, n_sls]"
        )

        # Vp/Vs describe the material at the reference frequency. Convert the
        # stored elastic coefficients to the unrelaxed moduli used by the SLS kernel.
        if all(name in field_cell for name in ("vp", "vs", "density", "lambda", "mu")):
            density = np.asarray(field_cell["density"])
            shear_modulus_reference = density * np.asarray(field_cell["vs"]) ** 2
            bulk_modulus_reference = density * (
                np.asarray(field_cell["vp"]) ** 2 - 4.0 * np.asarray(field_cell["vs"]) ** 2 / 3.0
            )
            shear_scale = compute_unrelaxed_modulus_scale(tau_sigma, tau_epsilon_mu, f0)
            bulk_scale = compute_unrelaxed_modulus_scale(tau_sigma, tau_epsilon_kappa, f0)
            shear_modulus = shear_modulus_reference * shear_scale
            bulk_modulus = bulk_modulus_reference * bulk_scale
            field_cell["mu"][...] = shear_modulus
            field_cell["lambda"][...] = bulk_modulus - 2.0 * shear_modulus / 3.0
