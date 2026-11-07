PY ?= python
.PHONY: install data eval eval-offline full test lint label dashboard docker

install:
	$(PY) -m pip install -e ".[dash,dev]"
data:
	$(PY) -m indic_judge.cli build-data
eval:            ## 20-item sample, end to end (uses cache; calls APIs only on misses)
	$(PY) -m indic_judge.cli run configs/sample20.yaml
eval-offline:    ## same, cache only (CI regression)
	$(PY) -m indic_judge.cli run configs/sample20.yaml --offline
full:
	$(PY) -m indic_judge.cli run configs/full.yaml
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests dashboard
label:
	$(PY) -m streamlit run dashboard/label_app.py
dashboard:
	$(PY) -m streamlit run dashboard/app.py
docker:
	docker build -t indic-judge-eval . && docker run --rm -p 8501:8501 indic-judge-eval
