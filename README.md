# MaleCNS Fly Brain Classifier

Can a real biological wiring diagram be used like a computational network?

This project extracts a small visual circuit from the public MaleCNS v1.0 fruit-fly connectome, stimulates its visual-column neurons with an image, propagates activity through measured synaptic connections, and classifies the resulting neural activity with a linear probe.

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
digit prediction
```

> This is not a simulated fly mind or biologically complete neural simulation. MaleCNS provides measured neuron-to-neuron wiring and synapse counts; dynamics, input encoding, and classifier are simplified software assumptions. This demonstrates the path from public connectome data to a computational graph, not biological digit recognition capability.

## Quick Start

### Requirements

- Python 3.10+
- Node.js 18+
- npm 9+

### Run the demo (with pre-built data and models)

```bash
# 1. Clone and navigate
git clone <repo-url>
cd fly-brain-vision

# 2. Install Python backend dependencies
pip install -r requirements.txt

# 3. Install and build frontend
cd frontend
npm install
npm run build
cd ..

# 4. Start the application
python -m uvicorn backend.main:app --port 8000
```

Open <http://localhost:8000> in your browser. Draw a digit on the white canvas; the fly mascot predicts the digit and displays:
- **What the fly sees**: 8×8 grayscale image after preprocessing
- **Visual-column sampling**: MaleCNS optic-column neuron positions and activation
- **Connectome activity**: Circuit statistics (neurons, synapses, simulation steps)
- **Prediction**: Confidence scores for each digit 0-9
- **Top cell types**: Downstream neurons with highest activity

**Shortcut:** `make run` does steps 2-4.

## Architecture

**Backend** (`backend/main.py`): FastAPI wraps the offline pipeline (`src/`) with two HTTP endpoints:
- `GET /api/meta` — Returns circuit metadata and benchmark accuracy
- `POST /api/predict` — Accepts canvas image, runs simulation, returns prediction and visualizations

**Frontend** (`frontend/`): React 19 + TypeScript with:
- HTML5 canvas for freehand digit drawing
- Animated SVG fly mascot with expression feedback
- Real-time prediction with loading/error states
- CSS-based bar charts (no external chart library)

**Full-stack deployment:** Backend serves the built React app, so one process on one port in production.

## Development

### Frontend hot reload (Vite dev server + backend)

**Terminal 1** — Backend with auto-reload:
```bash
python -m uvicorn backend.main:app --port 8000 --reload
```

**Terminal 2** — Frontend dev server:
```bash
cd frontend
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` calls to the backend at `:8000`. Changes to React components hot-reload instantly.

### Build and test

```bash
make install          # Install Python + Node dependencies
make lint             # Run oxlint (frontend) and Python linters
make test             # Run pytest suite
make frontend-build   # Build optimized React bundle
```

## Rebuild data and models from scratch

Requires a free neuPrint token from <https://neuprint.janelia.org>. Store in your shell environment; never commit it.

**Bash/Zsh:**

```bash
export NEUPRINT_APPLICATION_CREDENTIALS="your-token-here"
python scripts/build_circuit.py
python scripts/train_probe.py
python scripts/benchmark.py
```

**PowerShell:**

```powershell
$env:NEUPRINT_APPLICATION_CREDENTIALS = "your-token-here"
python scripts/build_circuit.py
python scripts/train_probe.py
python scripts/benchmark.py
```

Or use `make`:

```bash
make circuit
make train
make benchmark
```

### What each script does

**`build_circuit.py`:**

- Downloads `optic-column-type-assignments-v1.0.xlsx` from [flyconnectome/2025malecns](https://github.com/flyconnectome/2025malecns)
- Selects valid right-eye L1 neurons as visual input
- Queries MaleCNS via neuPrint API for 2-hop downstream connectivity
- Normalizes sparse adjacency matrix (log1p counts, row normalization)
- Outputs: `data/malecns_circuit.npz`, `data/malecns_circuit_meta.json`

**`train_probe.py`:**

- Uses scikit-learn handwritten digits (1,797 train, 600 test, no external download)
- Preprocesses: crop to ink, center, downsample to 8×8
- Simulates activity through MaleCNS circuit
- Trains logistic regression on hidden-layer activity
- Outputs: `models/digit_probe.joblib`, `models/digit_probe.json`

**`benchmark.py`:**

- Compares three models on held-out test set:
  1. Input-only linear baseline (no circuit)
  2. MaleCNS circuit + hidden-state readout
  3. Randomized-edge control (circuit with shuffled weights)
- Outputs: `models/benchmark.json`

**Expected accuracy:** ~95.6% (MaleCNS), 94.4% (input-only), 96.7% (randomized control)

## Testing

Run the full test suite:

```bash
make test
```

Tests verify:

- Image preprocessing (crop, center, downsample to 8×8)
- End-to-end pipeline on simulated digits
- API endpoints return correct JSON shapes

## Experimental design

The classifier pipeline:

```text
digit image → optic-column samples → MaleCNS dynamics → hidden activity → logistic regression
```

Key design choices:

- **Feature vector:** Concatenated final and mean activity from downstream neurons (not raw pixels)
- **Classifier:** Logistic regression (scikit-learn), no hyperparameter tuning
- **Baseline comparison:** Input-only linear classifier to isolate circuit contribution
- **Control:** Randomized-weight circuit to validate that specific wiring matters

Results are computed fresh each run; no pre-computed accuracy values in the repo.

## Project structure

```text
fly-brain-vision/
├── backend/                    # FastAPI server
│   └── main.py                # API endpoints and frontend serving
├── frontend/                   # React + TypeScript application
│   ├── src/
│   │   ├── components/        # UI components (canvas, mascot, panels)
│   │   ├── hooks/             # Custom hooks (useMeta, usePrediction)
│   │   ├── api.ts             # HTTP client
│   │   ├── types.ts           # TypeScript interfaces
│   │   ├── App.tsx            # Root component
│   │   └── App.css            # Styles
│   ├── dist/                  # Built production bundle
│   └── package.json
├── src/                        # Python pipeline (unchanged)
│   ├── image_encoder.py       # Image preprocessing
│   ├── malecns_circuit.py     # Circuit loading
│   └── simulation.py          # Dynamics simulator
├── scripts/                    # Data prep and training
│   ├── build_circuit.py       # Download and build MaleCNS subgraph
│   ├── train_probe.py         # Train digit classifier
│   └── benchmark.py           # Compare baseline, MaleCNS, and control
├── data/                       # Pre-computed circuit and metadata
│   ├── malecns_circuit.npz
│   └── malecns_circuit_meta.json
├── models/                     # Trained classifier
│   ├── digit_probe.joblib
│   └── benchmark.json
├── tests/                      # Python unit tests
├── requirements.txt            # Python dependencies
├── Makefile                    # Task automation
└── README.md                   # This file
```

## Troubleshooting

**Port already in use:** Change the port in the startup command:

```bash
python -m uvicorn backend.main:app --port 8001
```

**Missing data/models files:** Run the full rebuild sequence (see "Rebuild data and models from scratch").

**Frontend build fails:** Clear npm cache and reinstall:

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Canvas not responding:** Check browser console for errors. Clear browser cache and reload.

**neuPrint API errors during rebuild:** Verify your token has permission scope. Regenerate at <https://neuprint.janelia.org/account>.

## Scientific limits

MaleCNS gives biological connectivity and anatomical synapse counts. It does not supply measured activity for this task, an image encoder, effective synaptic signs, or a digit classifier. The simulated states are software outputs, not measured fly neural activity. Neurotransmitter annotations, where retained in metadata, are predictions and are not required for the default unsigned simulator.

## Data source and attribution

MaleCNS v1.0 is the public complete adult male *Drosophila* CNS connectome, produced by a collaboration involving HHMI Janelia/FlyEM, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research, and collaborators. The release contains more than 166,000 neurons and about 125 million synaptic connections. This demo uses only a small visual subgraph.

MaleCNS-derived data are CC-BY and retain their upstream attribution requirements. The code in this repository is MIT licensed; that does not relicense the MaleCNS data.

Sources: [MaleCNS supplemental data](https://github.com/flyconnectome/2025malecns), [neuPrint](https://neuprint.janelia.org).

## License

**Code:** MIT License (see LICENSE file)

**Data:** MaleCNS data is CC-BY 4.0. When using the connectome data, cite:
> [MaleCNS citation — see flyconnectome/2025malecns repository]

## Citation

If you use this demo in research, cite MaleCNS v1.0:

```bibtex
@article{maelecns2025,
  title={MaleCNS: Complete connectome of the adult Drosophila male brain},
  year={2025},
  organization={FlyEM, HHMI Janelia and collaborators}
}
```

## Getting help

- **API questions:** Check the FastAPI docs at <http://localhost:8000/docs>
- **neuPrint API issues:** See [neuPrint documentation](https://neuprint.janelia.org)
- **MaleCNS data questions:** Visit [flyconnectome/2025malecns](https://github.com/flyconnectome/2025malecns)
- **Bug reports:** Open an issue on GitHub
