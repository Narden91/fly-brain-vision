"""Small deterministic toy dynamics for a connectome-derived circuit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SimulationResult:
    final_state: np.ndarray
    mean_state: np.ndarray


class CircuitSimulator:
    def __init__(
        self,
        W: sparse.spmatrix,
        input_indices: np.ndarray,
        *,
        steps: int = 12,
        decay: float = 0.65,
        gain: float = 1.0,
        input_gain: float = 1.0,
    ) -> None:
        if W.shape[0] != W.shape[1]:
            raise ValueError("Circuit matrix must be square with W[post, pre] orientation.")
        if steps < 1:
            raise ValueError("steps must be positive.")
        self.W = W.tocsr().astype(np.float32)
        self.input_indices = np.asarray(input_indices, dtype=np.int64)
        self.steps, self.decay, self.gain, self.input_gain = steps, decay, gain, input_gain

    def simulate(self, inputs: np.ndarray) -> SimulationResult:
        inputs = np.asarray(inputs, dtype=np.float32)
        single = inputs.ndim == 1
        if single:
            inputs = inputs[None, :]
        if inputs.ndim != 2 or inputs.shape[1] != len(self.input_indices):
            raise ValueError("inputs must have shape (batch, number of visual input neurons).")

        state = np.zeros((inputs.shape[0], self.W.shape[0]), dtype=np.float32)
        input_current = np.zeros_like(state)
        input_current[:, self.input_indices] = self.input_gain * inputs
        mean_state = np.zeros_like(state)
        for _ in range(self.steps):
            recurrent = self.W.dot(state.T).T
            state = np.tanh(self.decay * state + self.gain * recurrent + input_current)
            mean_state += state
        mean_state /= self.steps
        if single:
            return SimulationResult(final_state=state[0], mean_state=mean_state[0])
        return SimulationResult(final_state=state, mean_state=mean_state)
