"""Train linear digit readouts from prepared MaleCNS circuit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.image_encoder import sample_digit_batch  # noqa: E402
from src.malecns_circuit import MaleCNSCircuit, hidden_features, load_circuit  # noqa: E402
from src.simulation import CircuitSimulator, DEFAULT_DECAY, DEFAULT_GAIN, DEFAULT_STEPS  # noqa: E402


def classifier() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def digit_inputs(images: np.ndarray, circuit: MaleCNSCircuit) -> np.ndarray:
    return sample_digit_batch(images, np.asarray(circuit.metadata["input_xy"], dtype=np.float32))


def circuit_features(inputs: np.ndarray, circuit: MaleCNSCircuit, batch_size: int) -> np.ndarray:
    simulator = CircuitSimulator(circuit.W, circuit.input_indices)
    parts: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        result = simulator.simulate(inputs[start : start + batch_size])
        parts.append(hidden_features(result.final_state, result.mean_state, circuit))
    return np.concatenate(parts).astype(np.float32)


def split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(labels))
    return train_test_split(all_indices, test_size=0.2, random_state=seed, stratify=labels)


def fit_accuracy(features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[object, float]:
    probe = classifier()
    probe.fit(features[train_idx], labels[train_idx])
    return probe, float(accuracy_score(labels[test_idx], probe.predict(features[test_idx])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear readout on MaleCNS hidden-neuron activity.")
    parser.add_argument("--samples", type=int, default=0, help="Use this many fixed-seed digit examples; 0 uses all.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "models/digit_probe.joblib")
    args = parser.parse_args()
    try:
        circuit = load_circuit(ROOT / "data/malecns_circuit.npz", ROOT / "data/malecns_circuit_meta.json")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    digits = load_digits()
    images, labels = digits.images, digits.target
    if args.samples:
        if not 100 <= args.samples <= len(labels):
            raise SystemExit(f"--samples must be between 100 and {len(labels)} (or 0).")
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(labels), args.samples, replace=False)
        images, labels = images[chosen], labels[chosen]

    inputs = digit_inputs(images, circuit)
    features = circuit_features(inputs, circuit, args.batch_size)
    train_idx, test_idx = split_indices(labels, args.seed)
    probe, accuracy = fit_accuracy(features, labels, train_idx, test_idx)
    _, baseline_accuracy = fit_accuracy(inputs, labels, train_idx, test_idx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(probe, args.output)
    metadata = {
        "accuracy": accuracy,
        "input_only_accuracy": baseline_accuracy,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "circuit_neurons": circuit.n_neurons,
        "circuit_edges": int(circuit.W.nnz),
        "input_neurons": int(len(circuit.input_indices)),
        "feature_dim": int(features.shape[1]),
        "feature_construction": "concatenate(final_state[hidden], mean_state[hidden]); visual input neurons excluded",
        "simulation_steps": DEFAULT_STEPS,
        "decay": DEFAULT_DECAY,
        "gain": DEFAULT_GAIN,
        "dataset": circuit.metadata.get("dataset"),
        "random_seed": args.seed,
        "digits_dataset": "scikit-learn handwritten digits dataset",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
