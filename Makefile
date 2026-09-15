PYTHON ?= python

.PHONY: install circuit train benchmark run test

install:
	$(PYTHON) -m pip install -r requirements.txt

circuit:
	$(PYTHON) scripts/build_circuit.py

train:
	$(PYTHON) scripts/train_probe.py

benchmark:
	$(PYTHON) scripts/benchmark.py

run:
	streamlit run app.py

test:
	$(PYTHON) -m compileall .
	$(PYTHON) -m pytest -q
