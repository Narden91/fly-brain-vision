"""Evaluate fixed MaleCNS dynamics against matched rewired controls."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_differentiable_probe import TrainingConfig, fit_differentiable_model  # noqa: E402
from scripts.train_probe import split_indices  # noqa: E402
from src.malecns_circuit import MaleCNSCircuit, load_circuit  # noqa: E402
from src.rewiring import degree_weight_preserving_rewire, target_permutation_rewire  # noqa: E402

CONTROL_NAMES = ("target_permutation", "degree_weight_preserving")


def paired_sign_flip_pvalue(differences: list[float]) -> float:
    """Exact one-sided paired sign-flip test for a positive mean difference."""
    observed = float(np.mean(differences))
    signed_means = [
        np.mean(np.asarray(signs) * differences) for signs in itertools.product((-1, 1), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(signed_means) >= observed - 1e-12))


def build_controls(circuit: MaleCNSCircuit, seed: int, swaps_per_edge: int) -> dict[str, MaleCNSCircuit]:
    return {
        "target_permutation": target_permutation_rewire(circuit, seed=seed),
        "degree_weight_preserving": degree_weight_preserving_rewire(
            circuit, seed=seed, swaps_per_edge=swaps_per_edge
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=0, help="0 selects 512 on CUDA or 64 on CPU.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l1-weight", type=float, default=0.0005)
    parser.add_argument("--swaps-per-edge", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "models/differentiable_benchmark.json")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if len(seeds) < 2:
        raise SystemExit("At least two seeds are required for paired controls.")
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l1_weight=args.l1_weight,
        device=args.device,
    )
    circuit = load_circuit(ROOT / "data/malecns_circuit.npz", ROOT / "data/malecns_circuit_meta.json")
    digits = load_digits()
    runs: list[dict] = []
    for seed in seeds:
        train_indices, test_indices = split_indices(digits.target, seed)
        _, real_metrics = fit_differentiable_model(
            circuit, digits.images, digits.target, train_indices, test_indices, seed=seed, config=config
        )
        controls = {}
        for name, control_circuit in build_controls(circuit, seed, args.swaps_per_edge).items():
            _, controls[name] = fit_differentiable_model(
                control_circuit, digits.images, digits.target, train_indices, test_indices, seed=seed, config=config
            )
        runs.append({"seed": seed, "real": real_metrics, "controls": controls})

    real_scores = np.asarray([run["real"]["temporal_accuracy"] for run in runs])
    control_summary = {}
    for name in CONTROL_NAMES:
        scores = np.asarray([run["controls"][name]["temporal_accuracy"] for run in runs])
        differences = (real_scores - scores).tolist()
        control_summary[name] = {
            "mean_temporal_accuracy": float(scores.mean()),
            "mean_difference": float(np.mean(differences)),
            "paired_sign_flip_pvalue": paired_sign_flip_pvalue(differences),
            "passes": bool(np.mean(differences) >= 0.01 and paired_sign_flip_pvalue(differences) < 0.05),
        }
    results = {
        "model_kind": "differentiable-fixed-connectome",
        "dataset": circuit.metadata.get("dataset"),
        "seeds": seeds,
        "training": asdict(config),
        "real_mean_temporal_accuracy": float(real_scores.mean()),
        "controls": control_summary,
        "release_criterion": "real temporal accuracy exceeds every control by >= 0.01 with paired sign-flip p < 0.05",
        "release_passed": all(summary["passes"] for summary in control_summary.values()),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
