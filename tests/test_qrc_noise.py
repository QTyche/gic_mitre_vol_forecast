from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from qtyche_qrc.models.qrc.encoding import initial_density_matrix
from qtyche_qrc.models.qrc.hamiltonian import ring_edges
from qtyche_qrc.models.qrc.noise import (
    QRCMeasurementConfig,
    apply_local_depolarizing_channel,
    measurement_rng,
    observable_estimates_from_bits,
    sample_commuting_observables,
)
from qtyche_qrc.models.qrc.observables import ObservableSet
from qtyche_qrc.models.qrc.reservoir import QRCConfig, QuantumReservoir
from qtyche_qrc.models.qrc.robust_features import RobustQuantumReservoir


def _mixed_diagonal_state() -> NDArray[np.complex128]:
    return np.asarray(
        np.diag(np.asarray([0.42, 0.18, 0.27, 0.13], dtype=complex)),
        dtype=complex,
    )


def test_finite_shot_sampling_is_deterministic_for_fixed_measurement_seed() -> None:
    state = _mixed_diagonal_state()

    first, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=512,
        rng=measurement_rng(1),
    )
    repeated, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=512,
        rng=measurement_rng(1),
    )

    assert np.array_equal(first, repeated)


def test_different_measurement_seeds_produce_different_finite_shot_features() -> None:
    state = _mixed_diagonal_state()

    first, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=128,
        rng=measurement_rng(0),
    )
    second, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=128,
        rng=measurement_rng(2),
    )

    assert not np.array_equal(first, second)


def test_high_shot_estimates_converge_toward_analytic_expectations() -> None:
    state = _mixed_diagonal_state()
    observables = ObservableSet.build(2, ring_edges(2), 1)
    analytic, _ = observables.expectations(state)
    low, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=128,
        rng=measurement_rng(0),
    )
    high, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=32768,
        rng=measurement_rng(0),
    )

    assert np.linalg.norm(high - analytic) < np.linalg.norm(low - analytic)
    assert np.max(np.abs(high - analytic)) < 0.02


def test_all_commuting_observables_reuse_the_same_joint_bitstrings() -> None:
    bits = np.asarray([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=np.int8)

    values, connected = observable_estimates_from_bits(bits, edges=ring_edges(2))

    assert np.array_equal(values, np.asarray([0.0, 0.0, 1.0]))
    assert np.array_equal(connected, np.asarray([1.0]))


def test_local_depolarizing_channel_preserves_trace_hermiticity_and_positivity() -> None:
    bell = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)
    state = np.outer(bell, bell.conj())

    noisy = apply_local_depolarizing_channel(
        state,
        n_qubits=2,
        probability=0.2,
    )

    np.testing.assert_allclose(np.trace(noisy), 1.0, atol=1e-12)
    assert np.allclose(noisy, noisy.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(noisy).min() >= -1e-12


def test_measurement_bit_flip_probability_one_flips_each_output_bit() -> None:
    state = initial_density_matrix(2)

    clean, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=16,
        rng=measurement_rng(0),
        measurement_bit_flip_probability=0.0,
    )
    flipped, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=16,
        rng=measurement_rng(0),
        measurement_bit_flip_probability=1.0,
    )

    assert np.array_equal(clean, np.asarray([1.0, 1.0, 1.0]))
    assert np.array_equal(flipped, np.asarray([-1.0, -1.0, 1.0]))


def test_bit_flip_rng_does_not_perturb_later_basis_sampling_batches() -> None:
    state = _mixed_diagonal_state()
    clean_sampling_rng = measurement_rng(1, stream=0)
    noisy_sampling_rng = measurement_rng(1, stream=0)
    flip_rng = measurement_rng(1, stream=1)

    clean_first, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=256,
        rng=clean_sampling_rng,
    )
    clean_second, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=256,
        rng=clean_sampling_rng,
    )
    noisy_first, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=256,
        rng=noisy_sampling_rng,
        measurement_bit_flip_probability=1.0,
        bit_flip_rng=flip_rng,
    )
    noisy_second, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=256,
        rng=noisy_sampling_rng,
        measurement_bit_flip_probability=1.0,
        bit_flip_rng=flip_rng,
    )

    assert np.array_equal(noisy_first, clean_first * np.asarray([-1.0, -1.0, 1.0]))
    assert np.array_equal(noisy_second, clean_second * np.asarray([-1.0, -1.0, 1.0]))


def test_zero_probability_channels_recover_clean_results_exactly() -> None:
    state = _mixed_diagonal_state()
    depolarized = apply_local_depolarizing_channel(
        state,
        n_qubits=2,
        probability=0.0,
    )
    base_config = QRCMeasurementConfig(shots=512, measurement_seed=2)
    zero_noise_config = replace(
        base_config,
        depolarizing_probability=0.0,
        measurement_bit_flip_probability=0.0,
    )

    base, _ = sample_commuting_observables(
        state,
        n_qubits=2,
        edges=ring_edges(2),
        shots=512,
        rng=measurement_rng(2),
    )
    zero_noise, _ = sample_commuting_observables(
        depolarized,
        n_qubits=2,
        edges=ring_edges(2),
        shots=512,
        rng=measurement_rng(2),
        measurement_bit_flip_probability=zero_noise_config.measurement_bit_flip_probability,
    )

    assert np.array_equal(depolarized, state)
    assert np.array_equal(base, zero_noise)


def test_analytic_robustness_path_recovers_existing_exact_features() -> None:
    qrc_config = QRCConfig(n_qubits=2, virtual_nodes=2, reservoir_seed=2026)
    inputs = np.random.default_rng(9).normal(size=(6, 3))
    exact = QuantumReservoir(3, qrc_config).transform(inputs, reset=True)
    robust = RobustQuantumReservoir(
        3,
        qrc_config,
        QRCMeasurementConfig(),
    ).transform(inputs, reset=True)

    assert np.array_equal(exact, robust)
