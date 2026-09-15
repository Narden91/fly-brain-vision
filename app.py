from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.fly_encoder import FlyEncoder


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "mnist_medulla_probe.joblib"
META_PATH = MODEL_PATH.with_suffix(".json")

st.set_page_config(page_title="Fly Brain Vision Demo", page_icon="🪰", layout="wide")
st.title("Fly Brain Vision Demo")
st.caption(
    "A frozen connectome-constrained Drosophila visual system turns an image into neural activity; "
    "a tiny linear readout predicts the digit."
)

with st.expander("What this demo actually uses"):
    st.markdown(
        """
This app uses **FlyVis**, a PyTorch model whose architecture is constrained by measured
Drosophila visual-system connectivity. It does **not** execute the entire 2026 MaleCNS
connectome as a neural network. MaleCNS is a structural wiring dataset; FlyVis supplies
neural dynamics and pretrained parameters that make this kind of demo executable.
        """
    )

if not MODEL_PATH.exists():
    st.error(
        "The linear readout has not been trained yet. Run `flyvis download-pretrained`, "
        "then `python scripts/train_probe.py --samples 2000`, and restart the app."
    )
    st.stop()


@st.cache_resource
def load_encoder() -> FlyEncoder:
    return FlyEncoder()


@st.cache_resource
def load_probe():
    return joblib.load(MODEL_PATH)


encoder = load_encoder()
probe = load_probe()
metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

uploaded = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg", "webp"],
    help="Best results: one digit, centered, high contrast.",
)

if uploaded is None:
    st.info("Upload a digit to run it through the fly visual-system model.")
    if metadata:
        st.write(
            f"Current probe: {metadata.get('samples', '?')} MNIST examples, "
            f"held-out accuracy {metadata.get('accuracy', 0):.1%}."
        )
    st.stop()

image = Image.open(uploaded)
result = encoder.encode_image(image)
probs = probe.predict_proba(result.features[None])[0]
classes = probe.classes_.astype(int)
order = np.argsort(probs)[::-1]
prediction = int(classes[order[0]])
confidence = float(probs[order[0]])

left, middle, right = st.columns([1, 1, 1])
with left:
    st.subheader("1. Input")
    st.image(image, use_container_width=True)

with middle:
    st.subheader("2. Fly-eye sampling")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(
        encoder.hex_x,
        encoder.hex_y,
        c=result.retina,
        s=42,
        marker="h",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.subheader("3. Linear readout")
    st.metric("Predicted digit", prediction)
    st.metric("Readout confidence", f"{confidence:.1%}")
    top = pd.DataFrame(
        {
            "digit": classes[order[:5]],
            "probability": probs[order[:5]],
        }
    ).set_index("digit")
    st.bar_chart(top)

st.subheader("What the fly visual circuit did")
activity = pd.DataFrame(
    {
        "cell_type": result.activity_names,
        "mean_activity": result.activity_values,
        "magnitude": np.abs(result.activity_values),
    }
).sort_values("magnitude", ascending=False)

st.bar_chart(activity.head(15).set_index("cell_type")["mean_activity"])
st.caption(
    "The chart shows the strongest mean responses among selected medulla/Tm cell types. "
    "The classifier sees the spatial response pattern across these biological cell types; "
    "it does not see the raw uploaded image directly."
)
