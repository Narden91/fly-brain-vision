PYTHON ?= python

.PHONY: install circuit train benchmark frontend-install frontend-build run dev test

install:
	$(PYTHON) -m pip install -r requirements.txt

circuit:
	$(PYTHON) scripts/build_circuit.py

train:
	$(PYTHON) scripts/train_probe.py

benchmark:
	$(PYTHON) scripts/benchmark.py

frontend-install:
	cd frontend && npm install

frontend-build:
	cd frontend && npm run build

# Builds the React app, then runs the FastAPI backend which serves it (single port).
run: frontend-build
	$(PYTHON) -m uvicorn backend.main:app --port 8000

# Two dev servers instead: Vite (hot reload, :5173) proxies /api to FastAPI (:8000).
dev:
	$(PYTHON) -m uvicorn backend.main:app --port 8000 --reload

test:
	$(PYTHON) -m compileall .
	$(PYTHON) -m pytest -q
