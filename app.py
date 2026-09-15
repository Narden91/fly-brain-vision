from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.image_encoder import plot_samples, sample_image
from src.malecns_circuit import hidden_features, load_circuit
from src.simulation import CircuitSimulator

ROOT = Path(__file__).resolve().parent
MATRIX_PATH = ROOT / "data/malecns_circuit.npz"
CIRCUIT_META_PATH = ROOT / "data/malecns_circuit_meta.json"
MODEL_PATH = ROOT / "models/digit_probe.joblib"
MODEL_META_PATH = ROOT / "models/digit_probe.json"
BENCHMARK_PATH = ROOT / "models/benchmark.json"

st.set_page_config(page_title="MaleCNS Fly Brain Classifier", page_icon="🪰", layout="wide")
st.title("MaleCNS Fly Brain Classifier")
st.caption(
    "A handwritten digit is projected onto visual columns from the real MaleCNS connectome, "
    "propagated through measured neuron-to-neuron wiring, and classified from the resulting neural activity."
)
st.info(
    "This is a toy computational model built on real anatomical connectivity. "
    "The connectome supplies the wiring; the dynamics and classifier are simplified software assumptions."
)


def read_json(path: Path, name: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} metadata is missing or corrupt.") from exc


@st.cache_resource
def resources():
    circuit = load_circuit(MATRIX_PATH, CIRCUIT_META_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Classifier is missing. Run `python scripts/train_probe.py` after building the circuit.")
    return circuit, CircuitSimulator(circuit.W, circuit.input_indices), joblib.load(MODEL_PATH)


try:
    circuit, simulator, probe = resources()
    model_meta = read_json(MODEL_META_PATH, "Classifier") if MODEL_META_PATH.exists() else {}
    benchmark = read_json(BENCHMARK_PATH, "Benchmark") if BENCHMARK_PATH.exists() else {}
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Where does the brain data come from?"):
    st.markdown(
        "MaleCNS v1.0 is the public complete adult male *Drosophila* CNS connectome. "
        "It is a collaboration involving HHMI Janelia/FlyEM, the University of Cambridge, "
        "MRC Laboratory of Molecular Biology, Google Research, and collaborators. The release "
        "contains more than 166,000 neurons and about 125 million synaptic connections. This app "
        "uses a small visual subgraph. MaleCNS data are CC-BY. Connection values start as anatomical "
        "synapse counts, then use explicit mathematical normalization for numerical stability."
    )

uploaded = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg", "webp"])
if uploaded is None:
    st.info("Upload one centered, high-contrast digit to run the connectome-derived circuit.")
    st.stop()
try:
    image = Image.open(uploaded)
    image.load()
    xy = np.asarray(circuit.metadata["input_xy"], dtype=np.float32)
    _, input_values = sample_image(image, xy)
except (UnidentifiedImageError, OSError, ValueError) as exc:
    st.error(f"Could not read that image: {exc}")
    st.stop()

result = simulator.simulate(input_values)
features = hidden_features(result.final_state, result.mean_state, circuit)
probabilities = probe.predict_proba(features[None])[0]
classes = probe.classes_.astype(int)
order = np.argsort(probabilities)[::-1]
prediction = int(classes[order[0]])

first, second, third, fourth = st.columns(4)
with first:
    st.subheader("1. Input image")
    st.image(image, use_container_width=True)
with second:
    st.subheader("2. MaleCNS visual-column sampling")
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    plot_samples(ax, xy, input_values)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with third:
    st.subheader("3. Connectome activity")
    st.write("MaleCNS v1.0")
    st.write(f"{circuit.n_neurons:,} biological neurons")
    st.write(f"{circuit.W.nnz:,} biological connections")
    st.write(f"{len(circuit.input_indices):,} image-input neurons")
    st.write(f"{simulator.steps} simulation steps")
with fourth:
    st.subheader("4. Prediction")
    st.metric("Predicted digit", prediction)
    st.metric("Classifier confidence", f"{probabilities[order[0]]:.1%}")
    st.bar_chart(pd.DataFrame({"probability": probabilities}, index=classes))

hidden = circuit.hidden_indices
types = circuit.metadata.get("cell_types", [])
body_ids = circuit.metadata["body_ids"]
labels = [types[i] if i < len(types) and types[i] else f"bodyId {body_ids[i]}" for i in hidden]
activity = pd.DataFrame({"cell type": labels, "activity": result.mean_state[hidden]})
activity = activity.assign(magnitude=activity["activity"].abs()).groupby("cell type", as_index=False).sum()
st.subheader("Top activated downstream cell types")
st.bar_chart(activity.nlargest(15, "magnitude").set_index("cell type")["activity"])

metrics = {
    "Input-only baseline": model_meta.get("input_only_accuracy", benchmark.get("input_only_accuracy")),
    "MaleCNS hidden-state readout": model_meta.get("accuracy", benchmark.get("malecns_accuracy")),
    "Randomized-edge control": benchmark.get("randomized_accuracy"),
}
available = {name: f"{value:.1%}" for name, value in metrics.items() if isinstance(value, (float, int))}
if available:
    st.caption("Experimental held-out accuracy: " + " · ".join(f"{name}: {value}" for name, value in available.items()))
