import numpy as np
from scipy import sparse

from src.malecns_circuit import MaleCNSCircuit
from src.rewiring import degree_weight_preserving_rewire


def test_degree_weight_preserving_rewire_keeps_degrees_and_weights() -> None:
    rows = np.array([1, 2, 3, 4, 5, 0, 2, 4, 0, 3])
    cols = np.array([0, 1, 2, 3, 4, 5, 0, 2, 5, 1])
    values = np.arange(1, len(rows) + 1, dtype=np.float32)
    circuit = MaleCNSCircuit(sparse.csr_matrix((values, (rows, cols)), shape=(6, 6)), {}, np.array([0]))
    rewired = degree_weight_preserving_rewire(circuit, seed=7, swaps_per_edge=1)
    original = circuit.W.tocsr()
    changed = rewired.W.tocsr()
    np.testing.assert_array_equal(np.diff(original.indptr), np.diff(changed.indptr))
    np.testing.assert_array_equal(np.diff(original.tocsc().indptr), np.diff(changed.tocsc().indptr))
    np.testing.assert_allclose(np.sort(original.data), np.sort(changed.data))
