"""Compare input-only, real MaleCNS, and randomized-edge controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_probe import circuit_features, digit_inputs, fit_accuracy, split_indices  # noqa: E402
from src.malecns_circuit import MaleCNSCircuit, load_circuit  # noqa: E402


def randomized_edge_circuit(circuit: MaleCNSCircuit, seed: int) -> MaleCNSCircuit:
    """Permute postsynaptic targets; edges, weights, inputs, and shape stay unchanged."""
    coo = circuit.W.tocoo()
    rng = np.random.default_rng(seed)
    target_map = rng.permutation(circuit.n_neurons)
    W = sparse.csr_matrix((coo.data, (target_map[coo.row], coo.col)), shape=circuit.W.shape)
    return MaleCNSCircuit(W=W, metadata=circuit.metadata, input_indices=circuit.input_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a deterministic randomized-edge control.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=ROOT / "models/benchmark.json")
    args = parser.parse_args()
    try:
        circuit = load_circuit(ROOT / "data/malecns_circuit.npz", ROOT / "data/malecns_circuit_meta.json")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    digits = load_digits()
    inputs = digit_inputs(digits.images, circuit)
    train_idx, test_idx = split_indices(digits.target, args.seed)
    _, input_accuracy = fit_accuracy(inputs, digits.target, train_idx, test_idx)
    _, malecns_accuracy = fit_accuracy(circuit_features(inputs, circuit, args.batch_size), digits.target, train_idx, test_idx)
    randomized = randomized_edge_circuit(circuit, args.seed)
    _, randomized_accuracy = fit_accuracy(circuit_features(inputs, randomized, args.batch_size), digits.target, train_idx, test_idx)
    results = {
        "input_only_accuracy": input_accuracy,
        "malecns_accuracy": malecns_accuracy,
        "randomized_accuracy": randomized_accuracy,
        "randomized_control": "postsynaptic-target permutation; same edge count and normalized edge-weight multiset",
        "random_seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
