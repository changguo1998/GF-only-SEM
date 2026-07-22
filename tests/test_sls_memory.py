"""
Unit tests for SLS memory variable update logic.

Covers: zero initial state, single-step accumulation, multi-step decay,
tau_epsilon > tau_sigma constraint, no-attenuation limit, Voigt indexing.
"""

import numpy as np
import pytest

# Constants (mirror C++ SLS namespace)
N_SLS = 3
VOIGT_COMPONENTS = 6
MEMORY_PER_NODE = N_SLS * VOIGT_COMPONENTS  # 18


def sls_memory_offset(node, mechanism, voigt):
    return node * MEMORY_PER_NODE + mechanism * VOIGT_COMPONENTS + voigt


def sigma_old_offset(node, voigt):
    return node * VOIGT_COMPONENTS + voigt


def coef_offset(node, mechanism):
    return node * N_SLS + mechanism


def update_sls_memory_numpy(rmemory_sls, sigma_old, sigma_current,
                            sls_coef_a, sls_coef_b, n_node):
    """Reference implementation matching the C++ element kernel SLS block."""
    vmap = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for node in range(n_node):
        for sls in range(N_SLS):
            a = sls_coef_a[coef_offset(node, sls)]
            b = sls_coef_b[coef_offset(node, sls)]
            for v, (l, m) in enumerate(vmap):
                off = sls_memory_offset(node, sls, v)
                sigma_curr = sigma_current[sigma_old_offset(node, v)]
                sigma_prev = sigma_old[sigma_old_offset(node, v)]
                delta = sigma_curr - sigma_prev
                rmemory_sls[off] = a * rmemory_sls[off] + b * delta
        for v, (l, m) in enumerate(vmap):
            sigma_old[sigma_old_offset(node, v)] = \
                sigma_current[sigma_old_offset(node, v)]


class TestSLSZeroInit:
    """R stays zero when sigma never changes from zero."""

    def test_zero_initial(self):
        n_node = 5
        rmemory_sls = np.zeros(n_node * MEMORY_PER_NODE)
        sigma_old = np.zeros(n_node * VOIGT_COMPONENTS)
        sigma_current = np.zeros(n_node * VOIGT_COMPONENTS)
        sls_coef_a = np.full(n_node * N_SLS, 0.9)
        sls_coef_b = np.full(n_node * N_SLS, 0.05)

        update_sls_memory_numpy(rmemory_sls, sigma_old, sigma_current,
                                sls_coef_a, sls_coef_b, n_node)

        np.testing.assert_array_equal(rmemory_sls, 0.0)
        np.testing.assert_array_equal(sigma_old, 0.0)


class TestSLSSingleStep:
    """First non-zero stress step: R accumulates from zero."""

    def test_accumulation(self):
        n_node = 1
        rmemory_sls = np.zeros(n_node * MEMORY_PER_NODE)
        sigma_old = np.zeros(n_node * VOIGT_COMPONENTS)
        sigma_current = np.ones(n_node * VOIGT_COMPONENTS)

        a = 0.8
        b = 0.1
        sls_coef_a = np.full(n_node * N_SLS, a)
        sls_coef_b = np.full(n_node * N_SLS, b)

        update_sls_memory_numpy(rmemory_sls, sigma_old, sigma_current,
                                sls_coef_a, sls_coef_b, n_node)

        # R = a * 0 + b * (1 - 0) = b
        for sls in range(N_SLS):
            for v in range(VOIGT_COMPONENTS):
                off = sls_memory_offset(0, sls, v)
                assert rmemory_sls[off] == pytest.approx(b, abs=1e-12)

        # sigma_old should now be 1.0
        for v in range(VOIGT_COMPONENTS):
            assert sigma_old[sigma_old_offset(0, v)] == 1.0


class TestSLSDecay:
    """Constant strain → R decays exponentially to zero."""

    def test_multi_step_decay(self):
        n_node = 1
        rmemory_sls = np.zeros(n_node * MEMORY_PER_NODE)
        sigma_old = np.zeros(n_node * VOIGT_COMPONENTS)
        sigma_current = np.full(n_node * VOIGT_COMPONENTS, 1.0)

        a = 0.8
        b = 0.1
        sls_coef_a = np.full(n_node * N_SLS, a)
        sls_coef_b = np.full(n_node * N_SLS, b)

        # Step 1: sigma goes 0→1
        update_sls_memory_numpy(rmemory_sls, sigma_old, sigma_current,
                                sls_coef_a, sls_coef_b, n_node)
        r1 = rmemory_sls[sls_memory_offset(0, 0, 0)]
        assert r1 == pytest.approx(b, abs=1e-12)

        # Step 2: sigma stays 1→1, delta=0, R = a*R
        update_sls_memory_numpy(rmemory_sls, sigma_old, sigma_current,
                                sls_coef_a, sls_coef_b, n_node)
        r2 = rmemory_sls[sls_memory_offset(0, 0, 0)]
        assert r2 == pytest.approx(a * b, abs=1e-12)

        # Step 3
        update_sls_memory_numpy(rmemory_sls, sigma_old, sigma_current,
                                sls_coef_a, sls_coef_b, n_node)
        r3 = rmemory_sls[sls_memory_offset(0, 0, 0)]
        assert r3 == pytest.approx(a * a * b, abs=1e-12)


class TestSLSCoefficients:
    """τ_ε > τ_σ is required for physical attenuation."""

    def test_b_positive(self):
        solver_dt = 0.001
        tau_s = 0.01
        tau_e = 0.015  # > tau_s -> attenuation

        a = np.exp(-solver_dt / tau_s)
        b = (tau_e / tau_s - 1.0) * (1.0 - a)

        assert b > 0, f"b should be positive for attenuation, got {b}"
        assert b < 1.0

    def test_no_attenuation_limit(self):
        solver_dt = 0.001
        tau_s = 0.01
        tau_e = 0.01  # equal -> no attenuation

        a = np.exp(-solver_dt / tau_s)
        b = (tau_e / tau_s - 1.0) * (1.0 - a)

        assert b == 0.0, f"b should be 0 when tau_e == tau_s, got {b}"


class TestSLSSteadyState:
    """After many steps with constant strain, R → 0 (stress relaxation)."""

    def test_decay_to_zero(self):
        n_node = 1
        rmemory_sls = np.zeros(n_node * MEMORY_PER_NODE)
        sigma_old = np.zeros(n_node * VOIGT_COMPONENTS)

        tau_s = 0.01
        tau_e = 0.015
        solver_dt = 0.0005
        a = np.exp(-solver_dt / tau_s)
        b = (tau_e / tau_s - 1.0) * (1.0 - a)

        sls_coef_a = np.full(n_node * N_SLS, a)
        sls_coef_b = np.full(n_node * N_SLS, b)
        sigma_current = np.full(n_node * VOIGT_COMPONENTS, 1.0)

        # Step 1: introduce strain
        update_sls_memory_numpy(rmemory_sls, sigma_old, sigma_current,
                                sls_coef_a, sls_coef_b, n_node)

        # Steps 2..N: hold strain constant → R decays
        for _ in range(200):
            update_sls_memory_numpy(rmemory_sls, sigma_old, sigma_current,
                                    sls_coef_a, sls_coef_b, n_node)

        r_final = rmemory_sls[sls_memory_offset(0, 0, 0)]
        assert r_final < 0.01, f"R should decay to near zero, got {r_final}"


class TestSLSMultipleMechanisms:
    """Each SLS mechanism updates independently."""

    def test_independent_mechanisms(self):
        n_node = 1
        rmemory_sls = np.zeros(n_node * MEMORY_PER_NODE)
        sigma_old = np.zeros(n_node * VOIGT_COMPONENTS)
        sigma_current = np.full(n_node * VOIGT_COMPONENTS, 1.0)

        # Different coefficients per mechanism
        sls_coef_a = np.array([0.9, 0.8, 0.7])
        sls_coef_b = np.array([0.05, 0.04, 0.03])

        update_sls_memory_numpy(rmemory_sls, sigma_old, sigma_current,
                                sls_coef_a, sls_coef_b, n_node)

        for sls in range(N_SLS):
            r = rmemory_sls[sls_memory_offset(0, sls, 0)]
            expected = sls_coef_b[sls]
            assert r == pytest.approx(expected, abs=1e-12)