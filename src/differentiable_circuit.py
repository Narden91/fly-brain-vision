"""Differentiable, fixed-connectome dynamics and sparse classifier readout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse
from torch import nn

from src.malecns_circuit import MaleCNSCircuit
from src.simulation import DEFAULT_DECAY, DEFAULT_GAIN, DEFAULT_INPUT_GAIN


@dataclass(frozen=True)
class TorchSimulationResult:
    final_state: torch.Tensor
    mean_state: torch.Tensor


class DeviceSelectionError(RuntimeError):
    """Raised when a caller explicitly requires an unavailable accelerator."""


def resolve_torch_device(requested: str = "auto") -> torch.device:
    """Select CUDA when available, with an explicit and reproducible CPU override.

    ``auto`` is intentionally conservative: a later sparse-matrix smoke test can
    still demote it to CPU. ``cuda`` means CUDA is required and therefore fails
    loudly instead of silently changing an experiment's device.
    """
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise DeviceSelectionError(
                "CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu."
            )
        return torch.device("cpu")

    # RTX tensor cores can accelerate eligible float32 dense operations. The
    # sparse recurrence itself remains float32 because that path is more broadly
    # supported and numerically dependable than reduced-precision sparse CSR.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def device_label(device: torch.device) -> str:
    """Return a stable, user-visible description without exposing configuration."""
    if device.type != "cuda":
        return "cpu"
    return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(device)})"


def _softplus_inverse(value: float) -> float:
    return float(np.log(np.expm1(value)))


def scipy_to_torch_csr(matrix: sparse.spmatrix) -> torch.Tensor:
    matrix = matrix.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64, copy=False)),
        torch.from_numpy(matrix.indices.astype(np.int64, copy=False)),
        torch.from_numpy(matrix.data),
        size=matrix.shape,
        dtype=torch.float32,
    )


class DifferentiableCircuitSimulator(nn.Module):
    """Unroll fixed MaleCNS wiring while learning only global stable dynamics."""

    def __init__(
        self,
        W: sparse.spmatrix,
        input_indices: np.ndarray,
        *,
        steps: int = 8,
        decay: float = DEFAULT_DECAY,
        gain: float = DEFAULT_GAIN,
        input_gain: float = DEFAULT_INPUT_GAIN,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive.")
        if not 0 < decay < 1 or gain <= 0 or input_gain <= 0:
            raise ValueError("Initial dynamics must be stable and positive.")
        self.register_buffer("W", scipy_to_torch_csr(W), persistent=True)
        self.register_buffer("input_indices", torch.as_tensor(input_indices, dtype=torch.long), persistent=True)
        self.steps = steps
        self.raw_decay = nn.Parameter(torch.tensor(np.log(decay / (1 - decay)), dtype=torch.float32))
        self.raw_gain = nn.Parameter(torch.tensor(_softplus_inverse(gain), dtype=torch.float32))
        self.raw_input_gain = nn.Parameter(torch.tensor(_softplus_inverse(input_gain), dtype=torch.float32))

    @property
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_decay)

    @property
    def gain(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_gain)

    @property
    def input_gain(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_input_gain)

    def dynamics(self) -> dict[str, float]:
        return {
            "decay": float(self.decay.detach()),
            "gain": float(self.gain.detach()),
            "input_gain": float(self.input_gain.detach()),
        }

    def forward(self, inputs: torch.Tensor) -> TorchSimulationResult:
        """Run static ``(batch, inputs)`` or temporal ``(batch, steps, inputs)`` data."""
        if inputs.ndim == 2:
            inputs = inputs[:, None, :].expand(-1, self.steps, -1)
        if inputs.ndim != 3 or inputs.shape[1] != self.steps or inputs.shape[2] != len(self.input_indices):
            raise ValueError("inputs must have shape (batch, inputs) or (batch, simulation steps, inputs).")

        state = torch.zeros((inputs.shape[0], self.W.shape[0]), dtype=inputs.dtype, device=inputs.device)
        mean_state = torch.zeros_like(state)
        decay, gain, input_gain = self.decay, self.gain, self.input_gain
        for frame in inputs.unbind(dim=1):
            recurrent = torch.sparse.mm(self.W, state.T).T
            state = decay * state + gain * recurrent
            state = state.index_add(1, self.input_indices, input_gain * frame)
            state = torch.tanh(state)
            mean_state = mean_state + state
        return TorchSimulationResult(final_state=state, mean_state=mean_state / self.steps)


class DifferentiableCircuitClassifier(nn.Module):
    """Fixed connectome plus a sparsity-regularized linear downstream readout."""

    def __init__(self, circuit: MaleCNSCircuit, *, steps: int = 8, classes: int = 10) -> None:
        super().__init__()
        self.simulator = DifferentiableCircuitSimulator(circuit.W, circuit.input_indices, steps=steps)
        self.register_buffer("hidden_indices", torch.as_tensor(circuit.hidden_indices, dtype=torch.long), persistent=True)
        self.readout = nn.Linear(2 * len(circuit.hidden_indices), classes)

    def features(self, result: TorchSimulationResult) -> torch.Tensor:
        return torch.cat(
            (result.final_state.index_select(1, self.hidden_indices), result.mean_state.index_select(1, self.hidden_indices)),
            dim=1,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, TorchSimulationResult]:
        result = self.simulator(inputs)
        return self.readout(self.features(result)), result

    def readout_l1(self) -> torch.Tensor:
        return self.readout.weight.abs().mean()


def place_model_with_sparse_fallback(
    model: DifferentiableCircuitClassifier, requested: str = "auto"
) -> torch.device:
    """Place a model and verify its fixed sparse recurrence on the selected device.

    Some PyTorch/CUDA combinations expose CUDA but do not implement the sparse
    CSR operation used by this experiment. Auto mode catches precisely that case
    and retains a functional CPU service; explicit CUDA mode reports the error.
    """
    device = resolve_torch_device(requested)
    try:
        model.to(device)
        probe = torch.zeros((1, len(model.simulator.input_indices)), dtype=torch.float32, device=device)
        with torch.inference_mode():
            model(probe)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return device
    except RuntimeError as exc:
        if requested.lower() != "auto" or device.type != "cuda":
            raise DeviceSelectionError(f"Could not run the fixed sparse circuit on {device}: {exc}") from exc
        model.to("cpu")
        torch.cuda.empty_cache()
        return torch.device("cpu")


def load_differentiable_probe(path: Path, circuit: MaleCNSCircuit) -> tuple[DifferentiableCircuitClassifier, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["n_neurons"] != circuit.n_neurons or config["n_inputs"] != len(circuit.input_indices):
        raise ValueError("Differentiable probe does not match the loaded circuit.")
    model = DifferentiableCircuitClassifier(circuit, steps=int(config["steps"]), classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict_differentiable(
    model: DifferentiableCircuitClassifier, input_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    inputs = torch.as_tensor(np.asarray(input_values, dtype=np.float32), device=device)[None, :]
    logits, result = model(inputs)
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probabilities, result.final_state[0].cpu().numpy(), result.mean_state[0].cpu().numpy()
