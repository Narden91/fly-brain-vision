import numpy as np
import torch
from scipy import sparse

from src.differentiable_circuit import (
    DifferentiableCircuitClassifier,
    DifferentiableCircuitSimulator,
    place_model_with_sparse_fallback,
    resolve_torch_device,
)
from src.malecns_circuit import MaleCNSCircuit
from src.simulation import CircuitSimulator


def test_torch_simulator_matches_numpy_at_initial_dynamics() -> None:
    W = sparse.csr_matrix(([0.4, -0.2, 0.7], ([1, 2, 2], [0, 0, 1])), shape=(3, 3), dtype=np.float32)
    inputs = np.array([[0.7], [0.1]], dtype=np.float32)
    expected = CircuitSimulator(W, np.array([0]), steps=4).simulate(inputs)
    simulator = DifferentiableCircuitSimulator(W, np.array([0]), steps=4)
    actual = simulator(torch.from_numpy(inputs))
    np.testing.assert_allclose(actual.final_state.detach().numpy(), expected.final_state, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual.mean_state.detach().numpy(), expected.mean_state, rtol=1e-5, atol=1e-6)


def test_connectome_weights_are_fixed_and_temporal_inputs_have_expected_shape() -> None:
    W = sparse.csr_matrix(([1.0], ([1], [0])), shape=(3, 3), dtype=np.float32)
    simulator = DifferentiableCircuitSimulator(W, np.array([0]), steps=3)
    result = simulator(torch.ones((2, 3, 1)))
    assert result.final_state.shape == (2, 3)
    assert simulator.W.requires_grad is False
    assert "W" not in dict(simulator.named_parameters())
    np.testing.assert_allclose(simulator.W.values().detach().numpy(), W.data)


def test_device_selection_has_a_cpu_override_and_sparse_smoke_path() -> None:
    W = sparse.csr_matrix(([1.0], ([1], [0])), shape=(3, 3), dtype=np.float32)
    circuit = MaleCNSCircuit(
        W=W,
        metadata={"input_xy": [[0.0, 0.0]]},
        input_indices=np.array([0]),
    )
    model = DifferentiableCircuitClassifier(circuit, steps=3)
    assert resolve_torch_device("cpu").type == "cpu"
    assert place_model_with_sparse_fallback(model, "cpu").type == "cpu"
    assert next(model.parameters()).device.type == "cpu"
