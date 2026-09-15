from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fly_encoder import FlyEncoder  # noqa: E402


def balanced_subset(labels: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    if n_samples < 100:
        raise ValueError("Use at least 100 samples so every digit is represented reasonably.")

    rng = np.random.default_rng(seed)
    per_class = n_samples // 10
    chosen: list[np.ndarray] = []
    for digit in range(10):
        candidates = np.flatnonzero(labels == digit)
        rng.shuffle(candidates)
        chosen.append(candidates[:per_class])
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return idx


def resize_mnist(images_28: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(images_28, dtype=torch.float32)[:, None] / 255.0
    tensor = torch.nn.functional.interpolate(
        tensor, size=(64, 64), mode="bilinear", align_corners=False
    )
    return tensor[:, 0].numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tiny linear MNIST readout on frozen FlyVis features."
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "models" / "mnist_medulla_probe.joblib"
    )
    args = parser.parse_args()

    dataset = MNIST(root=ROOT / "data" / "mnist", train=True, download=True)
    labels_all = dataset.targets.numpy()
    idx = balanced_subset(labels_all, args.samples, args.seed)
    images = resize_mnist(dataset.data.numpy()[idx])
    labels = labels_all[idx]

    encoder = FlyEncoder()
    features: list[np.ndarray] = []
    for start in range(0, len(images), args.batch_size):
        end = min(start + args.batch_size, len(images))
        batch_features, _, _ = encoder.encode_batch(images[start:end])
        features.append(batch_features)
        print(f"encoded {end}/{len(images)}", flush=True)
    X = np.concatenate(features, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=1000, solver="lbfgs"),
    )
    probe.fit(X_train, y_train)
    prediction = probe.predict(X_test)
    accuracy = float(accuracy_score(y_test, prediction))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(probe, args.output)

    metadata = {
        "accuracy": accuracy,
        "samples": int(len(labels)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "feature_types": encoder.feature_types,
        "feature_dim": int(X.shape[1]),
        "flyvis_model": "flow/0000/000",
        "flyvis_version": "1.2.0",
        "dt_seconds": 0.01,
        "frames": 20,
        "note": "Frozen FlyVis features + multinomial logistic-regression readout.",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(metadata, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
