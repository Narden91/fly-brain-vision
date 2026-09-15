from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.image_encoder import plot_samples, sample_image
from src.malecns_circuit import hidden_features, load_circuit
from src.simulation import CircuitSimulator

ROOT = Path(__file__).resolve().parent
MATRIX_PATH = ROOT / "data/malecns_circuit.npz"
CIRCUIT_META_PATH = ROOT / "data/malecns_circuit_meta.json"
MODEL_PATH = ROOT / "models/digit_probe.joblib"
MODEL_META_PATH = ROOT / "models/digit_probe.json"
BENCHMARK_PATH = ROOT / "models/benchmark.json"

CANVAS_SIZE = 280
# ponytail: calibration knob, not a tunable feature. Training digits are thick, blocky
# 8x8 strokes; a thin pen stroke downsamples to a faint smear. Raise/lower if handwriting
# on your display draws thinner/thicker than expected.
STROKE_WIDTH = 18

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


def fly_mascot(digit: int | None, confidence: float) -> str:
    """Render a cartoon fly that reports the predicted digit in a speech bubble.

    Built entirely from <div>s with inline `style=` attributes — no <svg>, no <style>
    block. Measured in a real browser: st.html runs content through DOMPurify, and this
    Streamlit version's DOMPurify config silently drops the whole <svg> subtree (0
    console warning, just gone) and any <style> tag. Inline styles on plain divs are the
    one thing confirmed to survive, so the whole mascot is built that way. All positioning
    is absolute children of one `position:relative` div, so there's no risk of a wing
    escaping to some unrelated positioned ancestor and rendering off in space.
    """
    if digit is None:
        mouth = '<div style="position:absolute;top:66px;left:40px;width:16px;height:3px;border-radius:2px;background:#2a1a10"></div>'
        badge, message, sub = "", "Draw a digit for me!", ""
    elif confidence >= 0.5:
        mouth = (
            '<div style="position:absolute;top:58px;left:34px;width:28px;height:16px;'
            'border-radius:0 0 16px 16px;background:#2a1a10"></div>'
        )
        badge, message, sub = "", f"It's a <strong>{digit}</strong>!", f"{confidence:.0%} sure"
    else:
        mouth = '<div style="position:absolute;top:66px;left:40px;width:16px;height:3px;border-radius:2px;background:#2a1a10"></div>'
        badge = (
            '<div style="position:absolute;top:-6px;right:-4px;width:20px;height:20px;border-radius:50%;'
            'background:#ffffff;border:2px solid #3a2a1a;color:#3a2a1a;font-size:0.75rem;font-weight:700;'
            'line-height:16px;text-align:center">?</div>'
        )
        message, sub = f"Hmm&hellip; maybe a <strong>{digit}</strong>?", f"only {confidence:.0%} sure"

    wing = (
        "position:absolute;top:22px;{side}:-20px;width:48px;height:26px;border-radius:50%;"
        "background:linear-gradient(135deg,#ffffff,#bcdff5);opacity:0.8;"
        "border:1px solid #9fd3ee;transform:rotate({angle}deg)"
    )
    eye = (
        "position:absolute;top:26px;{side}:14px;width:32px;height:32px;border-radius:50%;"
        "background:radial-gradient(circle at 35% 30%,#e0566c,#7a1020 75%)"
    )
    antenna = "position:absolute;top:-14px;{side}:26px;width:2px;height:18px;background:#3a2a1a;transform:rotate({angle}deg)"
    antenna_tip = "position:absolute;top:-19px;{side}:20px;width:7px;height:7px;border-radius:50%;background:#3a2a1a"
    leg = "position:absolute;top:{top}px;{side}:6px;width:2px;height:20px;background:#3a2a1a;transform:rotate({angle}deg)"

    fly = f"""
    <div style="position:relative;width:96px;height:100px;flex-shrink:0">
      <div style="{wing.format(side="left", angle=-18)}"></div>
      <div style="{wing.format(side="right", angle=18)}"></div>
      <div style="{antenna.format(side="left", angle=-25)}"></div>
      <div style="{antenna.format(side="right", angle=25)}"></div>
      <div style="{antenna_tip.format(side="left")}"></div>
      <div style="{antenna_tip.format(side="right")}"></div>
      <div style="{leg.format(top=68, side="left", angle=35)}"></div>
      <div style="{leg.format(top=68, side="right", angle=-35)}"></div>
      <div style="{leg.format(top=56, side="left", angle=20)}"></div>
      <div style="{leg.format(top=56, side="right", angle=-20)}"></div>
      <div style="position:absolute;top:8px;left:4px;width:88px;height:80px;border-radius:50%;
                  background:radial-gradient(circle at 32% 24%,#8a6a4a,#3a2a1a 75%);
                  box-shadow:0 8px 10px rgba(0,0,0,0.35)">
        {badge}
      </div>
      <div style="{eye.format(side="left")}">
        <div style="position:absolute;top:4px;left:5px;width:9px;height:9px;border-radius:50%;background:#ffffff;opacity:0.85"></div>
      </div>
      <div style="{eye.format(side="right")}">
        <div style="position:absolute;top:4px;left:5px;width:9px;height:9px;border-radius:50%;background:#ffffff;opacity:0.85"></div>
      </div>
      {mouth}
    </div>
    """

    sub_html = f'<div style="font-size:0.85rem;color:#666;margin-top:0.25rem">{sub}</div>' if sub else ""
    return (
        '<div style="display:flex;align-items:center;gap:1.25rem;padding:0.5rem 0">'
        f"{fly}"
        '<div style="background:#ffffff;border:2px solid #3a2a1a;border-radius:14px;'
        'padding:0.75rem 1.1rem;color:#1a1a1a;font-size:1.1rem">'
        f"<div>{message}</div>{sub_html}"
        "</div>"
        "</div>"
    )


# A white HTML5 canvas the visitor draws a digit on. Hand-rolled with st.components.v2
# (CCv2) instead of the third-party streamlit-drawable-canvas package, which is
# incompatible with this Streamlit version (it still targets the old v1 component API).
_DIGIT_CANVAS = st.components.v2.component(
    "digit_canvas",
    html="""
    <div class="digit-canvas">
      <canvas id="pad"></canvas>
      <button id="clear" type="button">Clear</button>
    </div>
    """,
    css="""
    .digit-canvas { display: flex; flex-direction: column; align-items: flex-start; gap: 0.6rem; }
    #pad { background: #ffffff; border: 2px solid #3a2a1a; border-radius: 8px; touch-action: none; cursor: crosshair; }
    #clear {
      padding: 0.35rem 0.9rem;
      border-radius: 6px;
      border: 1px solid #3a2a1a;
      background: #ffffff;
      cursor: pointer;
      font-size: 0.9rem;
    }
    #clear:hover { background: #f0f0f0; }
    """,
    js="""
    export default function (component) {
      const { data, parentElement, setStateValue } = component
      const canvas = parentElement.querySelector("#pad")
      const clearBtn = parentElement.querySelector("#clear")
      if (!canvas || !clearBtn) return

      const size = (data && data.size) || 280
      const strokeWidth = (data && data.stroke_width) || 18
      const ctx = canvas.getContext("2d")

      // Only paint the initial (blank, or restored-on-rerun) frame once per canvas
      // element. The user's live strokes are the source of truth after that; we never
      // fight them by re-hydrating from `data` on every script rerun.
      if (!canvas.dataset.initialized) {
        canvas.width = size
        canvas.height = size
        ctx.fillStyle = "#ffffff"
        ctx.fillRect(0, 0, size, size)
        if (data && data.image) {
          const img = new Image()
          img.onload = () => ctx.drawImage(img, 0, 0, size, size)
          img.src = data.image
        }
        canvas.dataset.initialized = "1"
      }

      ctx.lineJoin = "round"
      ctx.lineCap = "round"
      ctx.strokeStyle = "#000000"
      ctx.lineWidth = strokeWidth

      const posFromEvent = (e) => {
        const rect = canvas.getBoundingClientRect()
        const scaleX = canvas.width / rect.width
        const scaleY = canvas.height / rect.height
        return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY]
      }
      const emit = () => setStateValue("image", canvas.toDataURL("image/png"))

      let drawing = false
      let lastX = 0
      let lastY = 0

      canvas.onpointerdown = (e) => {
        if (e.button !== 0) return
        drawing = true
        ;[lastX, lastY] = posFromEvent(e)
        ctx.beginPath()
        ctx.arc(lastX, lastY, strokeWidth / 2, 0, Math.PI * 2)
        ctx.fillStyle = "#000000"
        ctx.fill()
        canvas.setPointerCapture(e.pointerId)
      }
      canvas.onpointermove = (e) => {
        // Require the primary button to still be down, not just our own `drawing`
        // flag — guards against a stray pointermove (hover, scroll, any event that
        // isn't a real held-button drag) silently extending the last stroke.
        if (!drawing || !(e.buttons & 1)) return
        const [x, y] = posFromEvent(e)
        ctx.beginPath()
        ctx.moveTo(lastX, lastY)
        ctx.lineTo(x, y)
        ctx.stroke()
        ;[lastX, lastY] = [x, y]
      }
      const endStroke = (e) => {
        if (!drawing) return
        drawing = false
        if (e && e.pointerId != null) canvas.releasePointerCapture(e.pointerId)
        emit()
      }
      canvas.onpointerup = endStroke
      canvas.onpointerleave = endStroke
      canvas.onpointercancel = endStroke

      clearBtn.onclick = () => {
        ctx.fillStyle = "#ffffff"
        ctx.fillRect(0, 0, size, size)
        emit()
      }
    }
    """,
)


def digit_canvas(*, size: int, stroke_width: int, key: str) -> str | None:
    """Mount the drawing canvas and return its content as a PNG data URL, or None."""
    component_state = st.session_state.get(key, {})
    result = _DIGIT_CANVAS(
        key=key,
        data={"size": size, "stroke_width": stroke_width, "image": component_state.get("image")},
        default={"image": None},
        on_image_change=lambda: None,
    )
    return result.image


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

with st.container(border=True):
    draw_col, mascot_col = st.columns([1, 1], vertical_alignment="center")
    with draw_col:
        st.subheader("Draw a digit")
        canvas_value = digit_canvas(size=CANVAS_SIZE, stroke_width=STROKE_WIDTH, key="digit_canvas")

    xy = np.asarray(circuit.metadata["input_xy"], dtype=np.float32)
    prepared = None
    if canvas_value:
        _, encoded = canvas_value.split(",", 1)
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        prepared, input_values = sample_image(image, xy)

    prediction, confidence = None, 0.0
    if prepared is not None and float(prepared.max()) > 0.0:
        result = simulator.simulate(input_values)
        features = hidden_features(result.final_state, result.mean_state, circuit)
        probabilities = probe.predict_proba(features[None])[0]
        classes = probe.classes_.astype(int)
        order = np.argsort(probabilities)[::-1]
        prediction = int(classes[order[0]])
        confidence = float(probabilities[order[0]])

    with mascot_col:
        st.html(fly_mascot(prediction, confidence))

if prediction is None:
    st.stop()

st.divider()
first, second, third, fourth = st.columns(4)
with first:
    st.subheader("1. What the fly sees")
    st.image(prepared, clamp=True, width="stretch")
with second:
    st.subheader("2. MaleCNS visual-column sampling")
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    plot_samples(ax, xy, input_values)
    st.pyplot(fig, width="stretch")
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
    st.metric("Classifier confidence", f"{confidence:.1%}")
    # st.bar_chart pulls in altair, which crashes at import on this Python/altair
    # combo (TypedDict(closed=True) mismatch) — plain matplotlib avoids it.
    fig_p, ax_p = plt.subplots(figsize=(3.4, 2.0))
    ax_p.bar(classes.astype(str), probabilities, color="#4a3222")
    ax_p.set_ylim(0, 1)
    ax_p.set_ylabel("probability")
    st.pyplot(fig_p, width="stretch")
    plt.close(fig_p)

hidden = circuit.hidden_indices
types = circuit.metadata.get("cell_types", [])
body_ids = circuit.metadata["body_ids"]
labels = [types[i] if i < len(types) and types[i] else f"bodyId {body_ids[i]}" for i in hidden]
activity = pd.DataFrame({"cell type": labels, "activity": result.mean_state[hidden]})
activity = activity.assign(magnitude=activity["activity"].abs()).groupby("cell type", as_index=False).sum()
st.divider()
st.subheader("Top activated downstream cell types")
top_activity = activity.nlargest(15, "magnitude").set_index("cell type")["activity"]
fig_a, ax_a = plt.subplots(figsize=(9, 3.2))
ax_a.bar(top_activity.index, top_activity.values, color=["#8b1e2b" if v >= 0 else "#4a3222" for v in top_activity.values])
ax_a.set_ylabel("activity")
ax_a.tick_params(axis="x", rotation=60)
fig_a.tight_layout()
st.pyplot(fig_a, width="stretch")
plt.close(fig_a)

metrics = {
    "Input-only baseline": model_meta.get("input_only_accuracy", benchmark.get("input_only_accuracy")),
    "MaleCNS hidden-state readout": model_meta.get("accuracy", benchmark.get("malecns_accuracy")),
    "Randomized-edge control": benchmark.get("randomized_accuracy"),
}
available = {name: f"{value:.1%}" for name, value in metrics.items() if isinstance(value, (float, int))}
if available:
    st.caption("Experimental held-out accuracy: " + " · ".join(f"{name}: {value}" for name, value in available.items()))
