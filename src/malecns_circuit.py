"""Offline MaleCNS circuit artifact loading and feature selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class MaleCNSCircuit:
    """Normalized matrix with ``W[post, pre]`` connection orientation."""

    W: sparse.csr_matrix
    metadata: dict
    input_indices: np.ndarray

    @property
    def n_neurons(self) -> int:
        return self.W.shape[0]

    @cached_property
    def hidden_indices(self) -> np.ndarray:
        mask = np.ones(self.n_neurons, dtype=bool)
        mask[self.input_indices] = False
        return np.flatnonzero(mask)


def load_circuit(matrix_path: Path, metadata_path: Path) -> MaleCNSCircuit:
    """Load an offline artifact and check the metadata needed by the demo."""
    if not matrix_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "MaleCNS circuit artifact is missing. Run `python scripts/build_circuit.py` "
            "with NEUPRINT_APPLICATION_CREDENTIALS, then train the probe."
        )
    try:
        W = sparse.load_npz(matrix_path).tocsr().astype(np.float32)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("MaleCNS circuit metadata or matrix is corrupt.") from exc

    body_ids = metadata.get("body_ids")
    input_ids = metadata.get("input_body_ids")
    input_xy = metadata.get("input_xy")
    if not isinstance(body_ids, list) or not isinstance(input_ids, list) or not isinstance(input_xy, list):
        raise ValueError("MaleCNS circuit metadata is missing body IDs or visual-column coordinates.")
    if W.shape[0] != W.shape[1] or W.shape[0] != len(body_ids):
        raise ValueError("MaleCNS circuit matrix shape does not match its metadata.")
    index = {int(body_id): i for i, body_id in enumerate(body_ids)}
    try:
        input_indices = np.asarray([index[int(body_id)] for body_id in input_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("A visual input body ID is not present in the circuit matrix.") from exc
    if len(input_indices) == 0 or len(input_xy) != len(input_indices):
        raise ValueError("MaleCNS circuit has no valid visual input neurons or coordinates.")
    return MaleCNSCircuit(W=W, metadata=metadata, input_indices=input_indices)


def hidden_features(final_state: np.ndarray, mean_state: np.ndarray, circuit: MaleCNSCircuit) -> np.ndarray:
    """Return only downstream state; visual input neurons never enter this readout."""
    final_state = np.asarray(final_state)
    mean_state = np.asarray(mean_state)
    if final_state.shape != mean_state.shape or final_state.shape[-1] != circuit.n_neurons:
        raise ValueError("Simulation state shape does not match the circuit.")
    hidden = circuit.hidden_indices
    return np.concatenate((final_state[..., hidden], mean_state[..., hidden]), axis=-1)
