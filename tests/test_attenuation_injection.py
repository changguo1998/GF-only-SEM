"""
Unit tests for the config-driven SLS attenuation auto-injection
(preprocess Step 10b, `preprocess/cli.py:write_attenuation_if_configured`).

Covers: writes when q_mu/q_kappa present, skips when absent, dataset shapes and
attrs, independent Q_mu/Q_kappa handling, n_sls clamping, and the Q -> infinity
elastic limit.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from preprocess.attenuation import compute_tau_from_q, compute_unrelaxed_modulus_scale
from preprocess.cli import write_attenuation_if_configured

NGLL = 4
N_CELL = 6
TOTAL = N_CELL * NGLL * NGLL * NGLL


@pytest.fixture
def model_path(tmp_path):
    """An empty-but-valid model.h5 (field/cell group gets created on demand)."""
    path = tmp_path / "model.h5"
    with h5py.File(path, "w") as f:
        f.require_group("field/cell")
    return str(path)


def make_config(**kwargs):
    defaults = dict(q_mu=1.0e9, q_kappa=1.0e9, n_sls=3, f0_for_pml_hz=2.0)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def read_field(path, name):
    with h5py.File(path, "r") as f:
        ds = f[f"field/cell/{name}"]
        assert isinstance(ds, h5py.Dataset)
        return np.asarray(ds[...], dtype=np.float64)


def has_key(path, name):
    with h5py.File(path, "r") as f:
        return name in f["field/cell"]


# --------------------------------------------------------------------------
# write_attenuation_if_configured
# --------------------------------------------------------------------------


def test_writes_when_configured(model_path):
    cfg = make_config()
    assert write_attenuation_if_configured(model_path, cfg, N_CELL, NGLL) is True

    for name in ("tau_sigma", "tau_epsilon_mu", "tau_epsilon_kappa", "q_mu", "q_kappa"):
        assert has_key(model_path, name), f"missing {name}"
    # tau_* shape [n_cell, NGLL, NGLL, NGLL, n_sls]; Q fields [n_cell, NGLL^3]
    assert read_field(model_path, "tau_sigma").shape == (N_CELL, NGLL, NGLL, NGLL, 3)
    assert read_field(model_path, "q_mu").shape == (N_CELL, NGLL, NGLL, NGLL)

    with h5py.File(model_path, "r") as f:
        ds = f["field/cell/tau_sigma"]
        assert ds.attrs["n_sls"] == 3
        assert np.allclose(ds.attrs["f0_Hz"], 2.0)


def test_skips_without_q(model_path):
    cfg = make_config()
    del cfg.q_mu  # q_kappa only -> treated as unset
    assert write_attenuation_if_configured(model_path, cfg, N_CELL, NGLL) is False
    assert not has_key(model_path, "q_mu")
    assert not has_key(model_path, "tau_sigma")

    cfg = SimpleNamespace(q_mu=1.0e9)  # no q_kappa, no n_sls, no f0
    assert write_attenuation_if_configured(model_path, cfg, N_CELL, NGLL) is False
    assert not has_key(model_path, "tau_sigma")


def test_q_infinity_elastic_limit(model_path):
    """Q -> inf: both relaxation channels approach the elastic limit."""
    cfg = make_config(q_mu=1.0e9, q_kappa=1.0e9)
    write_attenuation_if_configured(model_path, cfg, N_CELL, NGLL)
    ts = read_field(model_path, "tau_sigma")
    te_mu = read_field(model_path, "tau_epsilon_mu")
    te_kappa = read_field(model_path, "tau_epsilon_kappa")
    np.testing.assert_array_equal(te_mu, ts)
    np.testing.assert_array_equal(te_kappa, ts)
    assert np.all(ts > 0.0)


def test_n_sls_clamped_to_three(model_path):
    """n_sls != 3 must be clamped to the solver-fixed N_SLS=3."""
    cfg = make_config(n_sls=5)
    write_attenuation_if_configured(model_path, cfg, N_CELL, NGLL)
    ts = read_field(model_path, "tau_sigma")
    assert ts.shape[-1] == 3
    with h5py.File(model_path, "r") as f:
        assert f["field/cell/tau_sigma"].attrs["n_sls"] == 3


def test_f0_default_from_config(model_path):
    cfg = make_config(f0_for_pml_hz=2.5)
    write_attenuation_if_configured(model_path, cfg, N_CELL, NGLL)
    with h5py.File(model_path, "r") as f:
        assert np.allclose(f["field/cell/tau_sigma"].attrs["f0_Hz"], 2.5)


def test_scalar_q_supported(model_path):
    """cli passes constant arrays, but scalar q values must be float()-coercible."""
    cfg = make_config(q_mu=1e9, q_kappa=1e9)
    write_attenuation_if_configured(model_path, cfg, N_CELL, NGLL)
    assert np.all(read_field(model_path, "q_mu") == 1.0e9)


def test_q_mu_and_q_kappa_are_independent(model_path):
    cfg = make_config(q_mu=20.0, q_kappa=1.0e9)
    write_attenuation_if_configured(model_path, cfg, N_CELL, NGLL)
    tau_sigma = read_field(model_path, "tau_sigma")
    assert np.all(read_field(model_path, "tau_epsilon_mu") > tau_sigma)
    assert np.allclose(read_field(model_path, "tau_epsilon_kappa"), tau_sigma, rtol=1e-6)


def test_pipeline_fields_include_attenuation_and_unrelaxed_moduli(model_path):
    shape = (N_CELL, NGLL, NGLL, NGLL)
    fields = {"mu": np.full(shape, 2.0), "lambda": np.full(shape, 3.0)}
    cfg = make_config(q_mu=20.0, q_kappa=10.0, f0_for_pml_hz=18.0)

    assert write_attenuation_if_configured(model_path, cfg, N_CELL, NGLL, fields=fields)

    assert "tau_epsilon_mu" in fields
    assert "tau_epsilon_kappa" in fields
    assert np.all(fields["mu"] > 2.0)
    reference_bulk_modulus = 3.0 + 2.0 * 2.0 / 3.0
    unrelaxed_bulk_modulus = fields["lambda"] + 2.0 * fields["mu"] / 3.0
    assert np.all(unrelaxed_bulk_modulus > reference_bulk_modulus)


# --------------------------------------------------------------------------
# compute_tau_from_q (pure core)
# --------------------------------------------------------------------------


def q_array(q_value):
    return np.full((N_CELL, NGLL, NGLL, NGLL), q_value, dtype=np.float64)


def test_finite_q_gives_attenuation():
    """Finite Q: tau_epsilon > tau_sigma (relaxation present)."""
    ts, te = compute_tau_from_q(q_array(30.0), NGLL, 3, 2.0)
    assert np.all(te > ts + 1e-12)
    # Q >= 1e8 is the exact elastic sentinel used by the example configs.
    ts_inf, te_inf = compute_tau_from_q(q_array(1.0e9), NGLL, 3, 2.0)
    np.testing.assert_array_equal(te_inf, ts_inf)


def test_fitted_parameters_reproduce_target_q_at_reference_frequency():
    target_q = 20.0
    reference_frequency_hz = 18.0
    tau_sigma, tau_epsilon = compute_tau_from_q(q_array(target_q), NGLL, 3, reference_frequency_hz)
    tau_sigma_node = tau_sigma[0, 0, 0, 0]
    ratio = tau_epsilon[0, 0, 0, 0] / tau_sigma_node
    weights = (ratio - 1.0) / np.sum(ratio)
    frequency_tau = 2.0 * np.pi * reference_frequency_hz * tau_sigma_node
    modulus_ratio = 1.0 - np.sum(weights / (1.0 + 1j * frequency_tau))
    measured_q = modulus_ratio.real / modulus_ratio.imag
    assert measured_q == pytest.approx(target_q, rel=0.03)


def test_fitted_parameters_hold_q_over_configured_band():
    target_q = 20.0
    reference_frequency_hz = 18.0
    tau_sigma, tau_epsilon = compute_tau_from_q(q_array(target_q), NGLL, 3, reference_frequency_hz)
    tau_sigma_node = tau_sigma[0, 0, 0, 0]
    ratio = tau_epsilon[0, 0, 0, 0] / tau_sigma_node
    weights = (ratio - 1.0) / np.sum(ratio)

    for frequency_hz in np.logspace(
        np.log10(reference_frequency_hz / 20.0), np.log10(reference_frequency_hz * 3.0), 100
    ):
        frequency_tau = 2.0 * np.pi * frequency_hz * tau_sigma_node
        modulus_ratio = 1.0 - np.sum(weights / (1.0 + 1j * frequency_tau))
        measured_q = modulus_ratio.real / modulus_ratio.imag
        assert measured_q == pytest.approx(target_q, rel=0.03)


def test_unrelaxed_scale_preserves_reference_frequency_modulus():
    reference_frequency_hz = 18.0
    tau_sigma, tau_epsilon = compute_tau_from_q(q_array(20.0), NGLL, 3, reference_frequency_hz)
    scale = compute_unrelaxed_modulus_scale(tau_sigma, tau_epsilon, reference_frequency_hz)
    ratio = tau_epsilon[0, 0, 0, 0] / tau_sigma[0, 0, 0, 0]
    weights = (ratio - 1.0) / np.sum(ratio)
    frequency_tau = 2.0 * np.pi * reference_frequency_hz * tau_sigma[0, 0, 0, 0]
    real_ratio = 1.0 - np.sum(weights / (1.0 + frequency_tau**2))
    assert scale[0, 0, 0, 0] * real_ratio == pytest.approx(1.0)


def test_nonpositive_q_node_no_attenuation():
    """q <= 0 or nan : that GLL node gets tau_epsilon == tau_sigma."""
    q = np.ones((N_CELL, NGLL, NGLL, NGLL), dtype=np.float64) * 30.0
    q[0, 0, 0, 0] = 0.0
    q[1, 1, 1, 1] = np.nan
    ts, te = compute_tau_from_q(q, NGLL, 3, 2.0)
    # q <= 0 or NaN -> no relaxation for that GLL node (tau_epsilon == tau_sigma)
    assert np.allclose(te[0, 0, 0, 0], ts[0, 0, 0, 0])
    assert np.allclose(te[1, 1, 1, 1], ts[1, 1, 1, 1])
    assert np.all(np.isfinite(te[1, 1, 1, 1]))
    # neighbors with finite Q still attenuate (all n_sls mechanisms)
    assert np.all(te[0, 0, 0, 1] > ts[0, 0, 0, 1] + 1e-12)


def test_tau_sigma_shape_and_values():
    ts, te = compute_tau_from_q(q_array(30.0), NGLL, 3, 2.0)
    assert ts.shape == (N_CELL, NGLL, NGLL, NGLL, 3)
    assert te.shape == ts.shape
    # tau_sigma = 1/(2 pi f_l) with f in [f0/20, 3 f0]
    f_l = np.logspace(np.log10(2.0 / 20.0), np.log10(2.0 * 3.0), 3)
    expected = 1.0 / (2.0 * np.pi * f_l)
    assert np.allclose(ts[0, 0, 0, 0, :], expected, rtol=1e-12)


def test_bad_q_shape_raises():
    with pytest.raises(ValueError, match="4D"):
        compute_tau_from_q(np.zeros(4), NGLL, 3, 2.0)
