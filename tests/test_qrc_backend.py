import numpy as np
import pytest

from qtyche_qrc.models.qrc.backends import ExactDensityMatrixBackend, trace_distance
from qtyche_qrc.models.qrc.encoding import (
    initial_density_matrix,
    input_projection,
    partial_trace_qubit,
    reset_and_encode_input,
    ry_state,
)
from qtyche_qrc.models.qrc.hamiltonian import (
    generate_hamiltonian,
    pauli_operators,
    ring_edges,
)
from qtyche_qrc.models.qrc.observables import ObservableSet


def test_pauli_operators_have_correct_dimensions_and_are_hermitian() -> None:
    for operator in pauli_operators().values():
        assert operator.shape == (2, 2)
        assert np.allclose(operator, operator.conj().T)


def test_ring_graph_contains_expected_edges() -> None:
    assert ring_edges(4) == ((0, 1), (1, 2), (2, 3), (3, 0))


def test_hamiltonian_is_hermitian() -> None:
    definition = generate_hamiltonian(3, seed=7)
    assert definition.matrix.shape == (8, 8)
    assert np.allclose(definition.matrix, definition.matrix.conj().T)


def test_hamiltonian_generation_is_deterministic_for_fixed_seed() -> None:
    first = generate_hamiltonian(4, seed=11)
    second = generate_hamiltonian(4, seed=11)
    assert np.array_equal(first.matrix, second.matrix)
    assert first.checksum == second.checksum


def test_different_seeds_produce_different_hamiltonians() -> None:
    assert generate_hamiltonian(3, seed=1).checksum != generate_hamiltonian(3, seed=2).checksum


def test_exact_backend_rejects_more_than_six_qubits() -> None:
    with pytest.raises(ValueError, match="n_qubits <= 6"):
        generate_hamiltonian(7, seed=1)


def test_cached_delta_unitary_is_unitary_within_tolerance() -> None:
    definition = generate_hamiltonian(3, seed=7)
    backend = ExactDensityMatrixBackend(definition.matrix, 3, 0.5)
    identity = np.eye(8)
    product = np.einsum("ji,jk->ik", backend.unitary.conj(), backend.unitary)
    assert np.allclose(product, identity, atol=1e-10)


def test_initial_density_matrix_has_trace_one_and_is_positive_semidefinite() -> None:
    state = initial_density_matrix(3)
    assert np.isclose(np.trace(state), 1.0)
    assert np.min(np.linalg.eigvalsh(state)) >= 0.0


def test_partial_trace_returns_expected_bell_subsystem_state() -> None:
    bell = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)
    density = np.outer(bell, bell.conj())
    reduced = partial_trace_qubit(density, 0, 2)
    assert np.allclose(reduced, np.eye(2) / 2.0)


def test_input_reset_produces_expected_encoded_marginal_and_tensor_order() -> None:
    previous = initial_density_matrix(3)
    theta = 0.7
    reset = reset_and_encode_input(previous, theta, 3)
    marginal = partial_trace_qubit(reset, 1, 3)
    expected = np.kron(
        np.outer(ry_state(theta), ry_state(theta)),
        np.asarray([[1.0, 0.0], [0.0, 0.0]]),
    )
    assert np.allclose(marginal, expected)


def test_reset_channel_preserves_trace_and_positivity() -> None:
    definition = generate_hamiltonian(3, seed=2)
    backend = ExactDensityMatrixBackend(definition.matrix, 3, 1.0)
    evolved = backend.evolve(initial_density_matrix(3))
    reset = reset_and_encode_input(evolved, -2.4, 3)
    assert np.isclose(np.trace(reset), 1.0)
    assert np.min(np.linalg.eigvalsh(reset)) >= -1e-12


def test_exact_evolution_preserves_trace_and_hermiticity() -> None:
    definition = generate_hamiltonian(3, seed=5)
    backend = ExactDensityMatrixBackend(definition.matrix, 3, 0.25)
    evolved = backend.evolve(initial_density_matrix(3))
    assert np.isclose(np.trace(evolved), 1.0)
    assert np.allclose(evolved, evolved.conj().T)


def test_measurement_of_known_product_state_has_known_expectations() -> None:
    observables = ObservableSet.build(3, ring_edges(3), 1)
    values, connected = observables.expectations(initial_density_matrix(3))
    assert np.allclose(values, 1.0)
    assert np.allclose(connected, 0.0)


def test_observable_feature_ordering_is_deterministic() -> None:
    first = ObservableSet.build(3, ring_edges(3), 2)
    second = ObservableSet.build(3, ring_edges(3), 2)
    assert first.feature_names == second.feature_names
    assert first.checksum == second.checksum


def test_qrc_feature_dimension_matches_virtual_nodes_times_nodes_and_edges() -> None:
    observables = ObservableSet.build(4, ring_edges(4), 2)
    assert observables.raw_feature_dimension == 2 * (4 + 4) == 16


def test_input_projection_is_deterministic_for_same_seed() -> None:
    assert np.array_equal(input_projection(2, 5, 2026), input_projection(2, 5, 2026))


def test_input_projection_rows_have_unit_norm() -> None:
    projection = input_projection(3, 7, 8)
    assert np.allclose(np.linalg.norm(projection, axis=1), 1.0)


def test_trace_distance_is_zero_for_identical_states() -> None:
    state = initial_density_matrix(3)
    assert trace_distance(state, state) == 0.0


def test_trace_distance_is_bounded_between_zero_and_one() -> None:
    zero = initial_density_matrix(3)
    one = np.zeros_like(zero)
    one[-1, -1] = 1.0
    assert 0.0 <= trace_distance(zero, one) <= 1.0


def test_sampled_couplings_and_fields_respect_configured_ranges() -> None:
    definition = generate_hamiltonian(4, seed=22, j_strength=2.0, h_strength=3.0)
    assert np.all(np.abs(definition.couplings) <= 2.0)
    assert np.all((definition.fields >= 1.5) & (definition.fields <= 4.5))
