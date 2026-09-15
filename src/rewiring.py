"""Matched directed rewiring controls for connectome experiments."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from src.malecns_circuit import MaleCNSCircuit


def degree_weight_preserving_rewire(circuit: MaleCNSCircuit, *, seed: int, swaps_per_edge: int = 5) -> MaleCNSCircuit:
    """Swap directed targets while preserving every node degree and weight multiset."""
    if swaps_per_edge < 1:
        raise ValueError("swaps_per_edge must be positive.")
    coo = circuit.W.tocoo()
    source = coo.col.astype(np.int64, copy=True)
    target = coo.row.astype(np.int64, copy=True)
    pairs = set(zip(source.tolist(), target.tolist(), strict=True))
    rng = np.random.default_rng(seed)
    swaps, attempts, wanted = 0, 0, len(source) * swaps_per_edge
    max_attempts = wanted * 25
    while swaps < wanted and attempts < max_attempts:
        attempts += 1
        left, right = rng.integers(0, len(source), size=2)
        if left == right or source[left] == source[right] or target[left] == target[right]:
            continue
        replacement_left = (int(source[left]), int(target[right]))
        replacement_right = (int(source[right]), int(target[left]))
        if replacement_left[0] == replacement_left[1] or replacement_right[0] == replacement_right[1]:
            continue
        old_left = (int(source[left]), int(target[left]))
        old_right = (int(source[right]), int(target[right]))
        if replacement_left in pairs or replacement_right in pairs:
            continue
        pairs.remove(old_left)
        pairs.remove(old_right)
        pairs.add(replacement_left)
        pairs.add(replacement_right)
        target[left], target[right] = target[right], target[left]
        swaps += 1
    if swaps < wanted:
        raise RuntimeError(f"Only completed {swaps} of {wanted} requested degree-preserving swaps.")
    W = sparse.csr_matrix((coo.data, (target, source)), shape=circuit.W.shape, dtype=np.float32)
    return MaleCNSCircuit(W=W, metadata=circuit.metadata, input_indices=circuit.input_indices)


def target_permutation_rewire(circuit: MaleCNSCircuit, *, seed: int) -> MaleCNSCircuit:
    """Legacy global target permutation control retained for comparison."""
    coo = circuit.W.tocoo()
    target_map = np.random.default_rng(seed).permutation(circuit.n_neurons)
    W = sparse.csr_matrix((coo.data, (target_map[coo.row], coo.col)), shape=circuit.W.shape, dtype=np.float32)
    return MaleCNSCircuit(W=W, metadata=circuit.metadata, input_indices=circuit.input_indices)
