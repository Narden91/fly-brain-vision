"""FastAPI backend for the MaleCNS Fly Brain Classifier React app.

Thin HTTP wrapper around the existing offline pipeline (src/image_encoder.py,
src/malecns_circuit.py, src/simulation.py) — none of that code changes here.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.image_encoder import plot_samples, sample_image  # noqa: E402
from src.malecns_circuit import hidden_features, load_circuit  # noqa: E402
from src.simulation import CircuitSimulator  # noqa: E402

MATRIX_PATH = ROOT / "data/malecns_circuit.npz"
CIRCUIT_META_PATH = ROOT / "data/malecns_circuit_meta.json"
MODEL_PATH = ROOT / "models/digit_probe.joblib"
MODEL_META_PATH = ROOT / "models/digit_probe.json"
BENCHMARK_PATH = ROOT / "models/benchmark.json"
FRONTEND_DIST = ROOT / "frontend/dist"

TOP_CELL_TYPES = 15


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


try:
    circuit = load_circuit(MATRIX_PATH, CIRCUIT_META_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Classifier is missing. Run `python scripts/train_probe.py` after building the circuit.")
    probe = joblib.load(MODEL_PATH)
except (FileNotFoundError, ValueError, OSError) as exc:
    raise RuntimeError(str(exc)) from exc

simulator = CircuitSimulator(circuit.W, circuit.input_indices)
model_meta = read_json(MODEL_META_PATH)
benchmark = read_json(BENCHMARK_PATH)
INPUT_XY = np.asarray(circuit.metadata["input_xy"], dtype=np.float32)

app = FastAPI(title="MaleCNS Fly Brain Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image: str  # "data:image/png;base64,...."


def decode_data_url(data_url: str) -> Image.Image:
    try:
        _, encoded = data_url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    except (ValueError, OSError, UnidentifiedImageError) as exc:
        raise HTTPException(400, f"Could not decode image data URL: {exc}") from exc


def encode_png(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def figure_to_png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return encode_png(buf.getvalue())


def grid_to_png(grid: np.ndarray, scale: int = 20) -> str:
    """Upscale the small (e.g. 8x8) prepared-image grid with nearest-neighbor for a crisp preview."""
    pixels = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return encode_png(buf.getvalue())


def top_cell_type_activity(mean_state: np.ndarray) -> list[dict]:
    hidden = circuit.hidden_indices
    types = circuit.metadata.get("cell_types", [])
    body_ids = circuit.metadata["body_ids"]
    labels = [types[i] if i < len(types) and types[i] else f"bodyId {body_ids[i]}" for i in hidden]

    activity = pd.DataFrame({"cell_type": labels, "activity": mean_state[hidden]})
    activity = activity.assign(magnitude=activity["activity"].abs()).groupby("cell_type", as_index=False).sum()
    top = activity.nlargest(TOP_CELL_TYPES, "magnitude")
    return [{"cellType": row.cell_type, "activity": float(row.activity)} for row in top.itertuples()]


@app.get("/api/meta")
def get_meta():
    """Static facts about the circuit and classifier — fetched once, not per prediction."""
    return {
        "dataset": circuit.metadata.get("dataset"),
        "nNeurons": circuit.n_neurons,
        "nEdges": int(circuit.W.nnz),
        "nInputNeurons": int(len(circuit.input_indices)),
        "simulationSteps": simulator.steps,
        "accuracy": {
            "inputOnly": model_meta.get("input_only_accuracy", benchmark.get("input_only_accuracy")),
            "maleCns": model_meta.get("accuracy", benchmark.get("malecns_accuracy")),
            "randomizedControl": benchmark.get("randomized_accuracy"),
        },
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    image = decode_data_url(req.image)
    prepared, input_values = sample_image(image, INPUT_XY)

    if float(prepared.max()) <= 0.0:
        return {"prediction": None, "confidence": 0.0}

    result = simulator.simulate(input_values)
    features = hidden_features(result.final_state, result.mean_state, circuit)
    probabilities = probe.predict_proba(features[None])[0]
    classes = probe.classes_.astype(int)
    best = int(np.argmax(probabilities))

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    plot_samples(ax, INPUT_XY, input_values)

    return {
        "prediction": int(classes[best]),
        "confidence": float(probabilities[best]),
        "probabilities": {str(int(c)): float(p) for c, p in zip(classes, probabilities)},
        "whatFlySeesPng": grid_to_png(prepared),
        "columnSamplingPng": figure_to_png(fig),
        "topCellTypes": top_cell_type_activity(result.mean_state),
    }


# Serve the built React app (npm run build in frontend/) so `uvicorn backend.main:app`
# alone can run the whole product. In dev, Vite's own server + proxy is used instead
# and frontend/dist won't exist yet.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
