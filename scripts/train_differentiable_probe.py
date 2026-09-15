"""Train a sparse readout over differentiable, fixed MaleCNS dynamics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_probe import split_indices  # noqa: E402
from src.differentiable_circuit import (  # noqa: E402
    DifferentiableCircuitClassifier,
    device_label,
    place_model_with_sparse_fallback,
)
from src.malecns_circuit import MaleCNSCircuit, load_circuit  # noqa: E402
from src.temporal_digits import VARIANT_NAMES, TemporalDigitSet, build_temporal_digit_set, sample_sequence_batch  # noqa: E402


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 8
    epochs: int = 20
    # Zero picks a conservative CPU batch or a larger RTX-friendly CUDA batch.
    batch_size: int = 0
    learning_rate: float = 0.01
    l1_weight: float = 0.0005
    device: str = "auto"


def _loader(
    data: TemporalDigitSet, batch_size: int, *, shuffle: bool, seed: int, pin_memory: bool = False
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(data.frames), torch.from_numpy(data.labels), torch.from_numpy(data.variants))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=pin_memory,
    )


@torch.inference_mode()
def evaluate_model(
    model: DifferentiableCircuitClassifier, data: TemporalDigitSet, xy: np.ndarray, batch_size: int
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    non_blocking = device.type == "cuda"
    correct = np.zeros(len(VARIANT_NAMES), dtype=np.int64)
    totals = np.zeros(len(VARIANT_NAMES), dtype=np.int64)
    for frames, labels, variants in _loader(data, batch_size, shuffle=False, seed=0, pin_memory=non_blocking):
        frames = frames.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        variants = variants.to(device, non_blocking=non_blocking)
        inputs = sample_sequence_batch(frames, xy)
        logits, _ = model(inputs)
        predicted = logits.argmax(dim=1)
        for variant in variants.unique():
            mask = variants == variant
            index = int(variant)
            correct[index] += int((predicted[mask] == labels[mask]).sum())
            totals[index] += int(mask.sum())
    by_variant = {name: float(correct[index] / totals[index]) for index, name in enumerate(VARIANT_NAMES)}
    temporal = correct[5:].sum() / totals[5:].sum()
    return {
        "accuracy": float(correct.sum() / totals.sum()),
        "static_accuracy": by_variant["static"],
        "temporal_accuracy": float(temporal),
        "accuracy_by_variant": by_variant,
    }


def fit_differentiable_model(
    circuit: MaleCNSCircuit,
    images: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    seed: int,
    config: TrainingConfig,
) -> tuple[DifferentiableCircuitClassifier, dict[str, Any]]:
    """Fit without allowing augmentation to cross the original image split."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_data = build_temporal_digit_set(images[train_indices], labels[train_indices], frames=config.steps, source_indices=train_indices)
    test_data = build_temporal_digit_set(images[test_indices], labels[test_indices], frames=config.steps, source_indices=test_indices)
    model = DifferentiableCircuitClassifier(circuit, steps=config.steps)
    device = place_model_with_sparse_fallback(model, config.device)
    batch_size = config.batch_size or (512 if device.type == "cuda" else 64)
    non_blocking = device.type == "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    objective = nn.CrossEntropyLoss()
    xy = np.asarray(circuit.metadata["input_xy"], dtype=np.float32)

    model.train()
    for _ in range(config.epochs):
        for frames, targets, _ in _loader(
            train_data, batch_size, shuffle=True, seed=seed, pin_memory=non_blocking
        ):
            optimizer.zero_grad(set_to_none=True)
            frames = frames.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            logits, _ = model(sample_sequence_batch(frames, xy))
            loss = objective(logits, targets) + config.l1_weight * model.readout_l1()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    metrics = evaluate_model(model, test_data, xy, batch_size)
    metrics["train_examples"] = int(len(train_data.labels))
    metrics["test_examples"] = int(len(test_data.labels))
    metrics["train_source_examples"] = int(len(train_indices))
    metrics["test_source_examples"] = int(len(test_indices))
    metrics["dynamics"] = model.simulator.dynamics()
    metrics["device"] = device_label(device)
    metrics["effective_batch_size"] = batch_size
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=0, help="0 selects 512 on CUDA or 64 on CPU.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l1-weight", type=float, default=0.0005)
    parser.add_argument("--output", type=Path, default=ROOT / "models/differentiable_probe.pt")
    args = parser.parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l1_weight=args.l1_weight,
        device=args.device,
    )
    circuit = load_circuit(ROOT / "data/malecns_circuit.npz", ROOT / "data/malecns_circuit_meta.json")
    digits = load_digits()
    train_indices, test_indices = split_indices(digits.target, args.seed)
    model, metrics = fit_differentiable_model(
        circuit, digits.images, digits.target, train_indices, test_indices, seed=args.seed, config=config
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "classes": list(range(10)),
        "config": {
            "model_kind": "differentiable-fixed-connectome",
            "model_version": "1.0.0",
            "steps": config.steps,
            "n_neurons": circuit.n_neurons,
            "n_inputs": len(circuit.input_indices),
        },
    }
    torch.save(checkpoint, args.output)
    metadata = {
        "model_kind": "differentiable-fixed-connectome",
        "model_version": "1.0.0",
        "dataset": circuit.metadata.get("dataset"),
        "random_seed": args.seed,
        "training": asdict(config),
        "feature_construction": "fixed sparse MaleCNS recurrence; final and mean downstream states; sparse linear readout",
        "edges_trainable": False,
        **metrics,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
