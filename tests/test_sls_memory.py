"""Unit tests for the strain-driven SLS memory recurrence."""

import numpy as np
import pytest

N_SLS = 3
N_COMPONENT = 6
MEMORY_PER_NODE = N_SLS * N_COMPONENT
TRACE = 2


def memory_offset(node, mechanism, component):
    return node * MEMORY_PER_NODE + mechanism * N_COMPONENT + component


def strain_offset(node, component):
    return node * N_COMPONENT + component


def forcing_offset(node, mechanism, time_level):
    return (node * N_SLS + mechanism) * 2 + time_level


def update_memory(
    memory,
    strain_old,
    strain_current,
    decay,
    forcing_mu,
    forcing_kappa,
    shear_modulus,
    bulk_modulus,
    n_node,
):
    """Reference implementation matching the CPU and CUDA element kernels."""
    for node in range(n_node):
        for mechanism in range(N_SLS):
            coefficient_decay = decay[node * N_SLS + mechanism]
            for component in range(N_COMPONENT):
                forcing = forcing_kappa if component == TRACE else forcing_mu
                modulus = bulk_modulus if component == TRACE else 2.0 * shear_modulus
                previous_weight = forcing[forcing_offset(node, mechanism, 0)]
                current_weight = forcing[forcing_offset(node, mechanism, 1)]
                offset = memory_offset(node, mechanism, component)
                memory[offset] = coefficient_decay * memory[offset] + modulus * (
                    previous_weight * strain_old[strain_offset(node, component)]
                    + current_weight * strain_current[strain_offset(node, component)]
                )
        for component in range(N_COMPONENT):
            strain_old[strain_offset(node, component)] = strain_current[
                strain_offset(node, component)
            ]


def coefficient_arrays(n_node, decay_value=0.8, previous=0.03, current=0.02):
    decay = np.full(n_node * N_SLS, decay_value)
    forcing_mu = np.empty(n_node * N_SLS * 2)
    forcing_kappa = np.empty_like(forcing_mu)
    forcing_mu[0::2] = previous
    forcing_mu[1::2] = current
    forcing_kappa[:] = forcing_mu
    return decay, forcing_mu, forcing_kappa


def test_zero_strain_keeps_zero_memory():
    n_node = 5
    memory = np.zeros(n_node * MEMORY_PER_NODE)
    strain_old = np.zeros(n_node * N_COMPONENT)
    strain_current = np.zeros_like(strain_old)
    coefficients = coefficient_arrays(n_node)

    update_memory(memory, strain_old, strain_current, *coefficients, 2.0, 3.0, n_node)

    np.testing.assert_array_equal(memory, 0.0)
    np.testing.assert_array_equal(strain_old, 0.0)


def test_first_step_uses_current_strain_weight():
    memory = np.zeros(MEMORY_PER_NODE)
    strain_old = np.zeros(N_COMPONENT)
    strain_current = np.ones(N_COMPONENT)
    coefficients = coefficient_arrays(1, current=0.02)

    update_memory(memory, strain_old, strain_current, *coefficients, 2.0, 3.0, 1)

    for mechanism in range(N_SLS):
        assert memory[memory_offset(0, mechanism, 0)] == pytest.approx(0.08)
        assert memory[memory_offset(0, mechanism, TRACE)] == pytest.approx(0.06)
    np.testing.assert_array_equal(strain_old, 1.0)


def test_constant_strain_relaxes_to_nonzero_memory():
    memory = np.zeros(MEMORY_PER_NODE)
    strain_old = np.zeros(N_COMPONENT)
    strain_current = np.ones(N_COMPONENT)
    coefficients = coefficient_arrays(1, decay_value=0.8, previous=0.03, current=0.02)

    for _ in range(200):
        update_memory(memory, strain_old, strain_current, *coefficients, 2.0, 3.0, 1)

    # Steady memory = modulus * (previous + current) / (1 - decay).
    assert memory[memory_offset(0, 0, 0)] == pytest.approx(1.0)
    assert memory[memory_offset(0, 0, TRACE)] == pytest.approx(0.75)


def test_shear_and_bulk_forcing_are_independent():
    memory = np.zeros(MEMORY_PER_NODE)
    strain_old = np.zeros(N_COMPONENT)
    strain_current = np.ones(N_COMPONENT)
    decay, forcing_mu, forcing_kappa = coefficient_arrays(1, current=0.02)
    forcing_kappa[1::2] = 0.04

    update_memory(
        memory, strain_old, strain_current, decay, forcing_mu, forcing_kappa, 2.0, 3.0, 1
    )

    assert memory[memory_offset(0, 0, 0)] == pytest.approx(0.08)
    assert memory[memory_offset(0, 0, TRACE)] == pytest.approx(0.12)


def test_zero_forcing_is_elastic_limit():
    memory = np.zeros(MEMORY_PER_NODE)
    strain_old = np.zeros(N_COMPONENT)
    strain_current = np.ones(N_COMPONENT)
    decay, forcing_mu, forcing_kappa = coefficient_arrays(1, previous=0.0, current=0.0)

    update_memory(
        memory, strain_old, strain_current, decay, forcing_mu, forcing_kappa, 2.0, 3.0, 1
    )

    np.testing.assert_array_equal(memory, 0.0)
