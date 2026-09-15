import numpy as np
from scipy import sparse

from src.malecns_circuit import MaleCNSCircuit, hidden_features
from src.simulation import CircuitSimulator


def circuit() -> MaleCNSCircuit:
    # W[post, pre]: input neuron 0 drives downstream neuron 1.
    W = sparse.csr_matrix(([1.0], ([1], [0])), shape=(3, 3), dtype=np.float32)
    return MaleCNSCircuit(W=W, metadata={}, input_indices=np.array([0]))


def test_simulation_is_deterministic_and_has_expected_shape() -> None:
    simulator = CircuitSimulator(circuit().W, np.array([0]), steps=4)
    first = simulator.simulate(np.array([[0.8], [0.2]], dtype=np.float32))
    second = simulator.simulate(np.array([[0.8], [0.2]], dtype=np.float32))
    assert first.final_state.shape == (2, 3)
    np.testing.assert_allclose(first.final_state, second.final_state)
    assert np.isfinite(simulator.simulate(np.zeros((1, 1), dtype=np.float32)).mean_state).all()


def test_input_changes_downstream_and_features_exclude_inputs() -> None:
    demo_circuit = circuit()
    simulator = CircuitSimulator(demo_circuit.W, demo_circuit.input_indices, steps=4)
    quiet = simulator.simulate(np.array([0.0], dtype=np.float32))
    active = simulator.simulate(np.array([1.0], dtype=np.float32))
    assert active.final_state[1] > quiet.final_state[1]
    features = hidden_features(active.final_state, active.mean_state, demo_circuit)
    assert features.shape == (4,)
    np.testing.assert_allclose(features[:2], active.final_state[1:])


def test_matrix_orientation_is_post_pre() -> None:
    simulator = CircuitSimulator(circuit().W, np.array([0]), steps=2, decay=0)
    result = simulator.simulate(np.array([1.0], dtype=np.float32))
    assert result.final_state[1] > 0
    assert result.final_state[2] == 0
