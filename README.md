# Fly Brain Vision Demo

A deliberately small demo showing how a **connectome-constrained fruit-fly visual system** can be used as a frozen feature extractor for image classification.

Upload a handwritten digit → sample it through a simulated fly eye → run the stimulus through **FlyVis** → classify the resulting neural activity with a linear readout.

> **Scientific accuracy:** this repository does **not** claim that the complete 2026 MaleCNS connectome is itself a pretrained image classifier. MaleCNS is a structural wiring map. The executable network here is **FlyVis**, whose architecture is constrained by measured Drosophila visual-system connectivity and whose dynamics were task-optimized in the published FlyVis work.

## Why this demo

The point is not MNIST accuracy. The point is to make the idea tangible:

```text
image
  ↓
simulated fly eye (721 photoreceptors)
  ↓
connectome-constrained FlyVis network
  ↓
medulla / Tm neural activity
  ↓
small linear classifier
  ↓
digit
```

The biological network is frozen. Only the tiny readout is trained for digit labels.

## Quick start

FlyVis 1.2.0 supports Python 3.9–3.12. Python 3.11 is a safe choice.

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
flyvis download-pretrained
python scripts/train_probe.py --samples 2000
streamlit run app.py
```

Or:

```bash
make prepare
make run
```

The one-time `train_probe.py` step can be compute-heavy on CPU because it actually simulates the connectome-constrained visual network. For a polished public demo, run it once on a GPU, then commit or attach the generated `models/mnist_medulla_probe.joblib` and metadata file to a release so visitors only need to run the app.

## What the application shows

1. The uploaded image.
2. The image after sampling by FlyVis's hexagonal `BoxEye` receptor layout.
3. The digit prediction from a multinomial logistic-regression readout.
4. A chart of the strongest selected medulla/Tm cell-type responses.

This makes the biological intermediate representation visible rather than presenting a black-box classifier.

## Repository structure

```text
app.py                       Streamlit UI
src/fly_encoder.py           FlyVis image → neural feature encoder
scripts/train_probe.py       Train the small MNIST linear readout
models/                      Generated readout + metadata
requirements.txt             Runtime dependencies
Makefile                     One-command setup helpers
```

## Relationship to the 2026 complete fly CNS map

In September 2026, Google Research described its collaboration with HHMI Janelia and others on the complete male Drosophila CNS connectome. Janelia's MaleCNS v1.0 release exposes neuron annotations, neurotransmitter predictions, skeletons and the full segment-to-segment connection graph. The full connection-weight table is about 1.1 GB.

That dataset is extremely useful, but a connectome table alone does not specify all neuronal dynamics or effective synaptic parameters required to run it as an artificial neural network. That is why this demo starts with FlyVis: it is already an executable PyTorch model built around measured fly visual-system connectivity.

A natural v2 of this repository is to replace or augment FlyVis with a visual subgraph extracted directly from MaleCNS v1.0 and explicitly define the missing dynamics.

## Expected benchmark

A separate public 2026 FlyVis/MNIST experiment reported about **92.7%** static-digit accuracy from medulla-level features with a linear probe, versus about **88.4%** from the retinal hexal input. Treat that result as a useful sanity check, not as a guarantee for this smaller training script; accuracy varies with sample count, preprocessing and feature selection.

## Sources

- Google Research, *A connectomics milestone: Mapping the complete male fruit fly brain* (2026-09-03)
- HHMI Janelia, *Male CNS Connectome* and dataset downloads
- TuragaLab, `flyvis` and its documentation
- Lappalainen et al., *Connectome-constrained networks predict neural activity across the fly visual system*, Nature (2024)
- Hikotty, `flyvis-sagemaker-mnist` (independent proof-of-concept)

## License

This demo repository is provided under the MIT License. FlyVis, its pretrained models, MaleCNS data, MNIST and other upstream resources retain their own licenses and attribution requirements.
