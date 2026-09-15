# MaleCNS Fly Brain Classifier

Can a real biological wiring diagram be used like a computational network?

This project extracts a small visual circuit directly from the public MaleCNS v1.0 fruit-fly connectome, stimulates its visual-column neurons with an image, propagates activity through measured synaptic connections, and trains a tiny linear classifier on the resulting activity.

```text
image
  ↓
real MaleCNS optic-column neurons
  ↓
real MaleCNS connectivity
  ↓
toy dynamical simulation
  ↓
downstream neural activity
  ↓
linear classifier
  ↓
digit
```

> This is not a simulated fly mind and not a biologically complete neural simulation. MaleCNS provides measured neuron-to-neuron wiring and synapse counts. Activity dynamics, input encoding, normalization, and classifier are simplified computational assumptions.

The goal is to show the path from public anatomical data to a small computational graph. It is not a claim about biological digit recognition or classifier accuracy.

## Run the prepared demo

If this checkout includes `data/malecns_circuit.npz`, its metadata, and the model files:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs offline after those files have been prepared. It does not need a neuPrint token.

## Rebuild the data and models

Create a free neuPrint token at <https://neuprint.janelia.org>. Keep it in your shell environment, never in a file committed to this repository.

```bash
export NEUPRINT_APPLICATION_CREDENTIALS="..."
python scripts/build_circuit.py
python scripts/train_probe.py
python scripts/benchmark.py
streamlit run app.py
```

Windows PowerShell:

```powershell
$env:NEUPRINT_APPLICATION_CREDENTIALS = "..."
python scripts/build_circuit.py
```

`build_circuit.py` downloads and caches the official `optic-column-type-assignments-v1.0.xlsx` table from `flyconnectome/2025malecns`. It selects valid right-eye L1 neurons as visual input neurons, parses their published optic-column labels into positions, collects two downstream hops from MaleCNS through neuPrint, then stores a normalized sparse matrix. L1 neurons are not called photoreceptors here.

Raw edge values are anatomical synapse counts. The builder applies `log1p(count)` and normalizes each postsynaptic matrix row, so the toy recurrent update stays numerically stable. The matrix uses `W[post, pre]` for a connection from `pre` to `post`.

## Experiment

Training uses the scikit-learn handwritten digits dataset, with no external image download:

```text
digit image → optic-column samples → MaleCNS toy dynamics → hidden-neuron state → logistic regression
```

The classifier uses concatenated final and mean activity from downstream neurons only. It does not receive raw pixels or input-neuron activity as its main feature vector.

`train_probe.py` also measures an input-only linear baseline. `benchmark.py` compares that baseline, the MaleCNS graph, and a deterministic randomized-edge control. Results are saved only when the scripts run; this repository does not invent accuracy values.

## Commands

```bash
make install
make circuit
make train
make benchmark
make run
make test
```

## Scientific limits

MaleCNS gives biological connectivity and anatomical synapse counts. It does not supply measured activity for this task, an image encoder, effective synaptic signs, or a digit classifier. The simulated states are software outputs, not measured fly neural activity. Neurotransmitter annotations, where retained in metadata, are predictions and are not required for the default unsigned simulator.

## Data source and attribution

MaleCNS v1.0 is the public complete adult male *Drosophila* CNS connectome, produced by a collaboration involving HHMI Janelia/FlyEM, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research, and collaborators. The release contains more than 166,000 neurons and about 125 million synaptic connections. This demo uses only a small visual subgraph.

MaleCNS-derived data are CC-BY and retain their upstream attribution requirements. The code in this repository is MIT licensed; that does not relicense the MaleCNS data.

Sources: [MaleCNS supplemental data](https://github.com/flyconnectome/2025malecns), [neuPrint](https://neuprint.janelia.org).
