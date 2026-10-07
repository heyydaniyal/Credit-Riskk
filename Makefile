# Credit Risk Scoring System — phase targets
# Phases must run in order; do not start a phase before its predecessor's checkpoint passes.

.PHONY: install phase0 eda test lint check features features-resume scorecard tune \
        model-final calibrate decision final-pipeline explain fairness sensitivity-dev \
        serve demo artifacts docker-build docker-run ceremony-dry-run holdout-ceremony notebooks

install:
	pip install -r requirements-dev.txt

phase0:
	python3 scripts/run_phase0.py

eda:
	python3 scripts/run_eda.py

test:
	python3 -m pytest -v -rs

lint:
	ruff check src/ tests/ scripts/ config/ app/

check: lint test

# ── Phase 2: feature engineering (staged, restartable) ────────────
# null-importance runs are checkpointed: re-run `make features` until
# the null-chunks step prints DONE (single-CPU boxes need ~4 passes).
features:
	python3 scripts/run_phase2.py build select
	python3 scripts/run_phase2_null_chunks.py --chunk 12
	python3 scripts/run_phase2.py woe report check

features-resume:
	python3 scripts/run_phase2_null_chunks.py --chunk 12

# ── Phase 3: modeling ─────────────────────────────────────────────
scorecard:
	python3 scripts/run_phase3_scorecard.py

tune:
	python3 scripts/run_phase3_tune.py --minutes 50

model-final:
	python3 scripts/run_phase3_final.py

# ── Phase 4: calibration + decision layer ────────────────────────
calibrate:
	python3 scripts/run_phase4_calibration.py

decision:
	python3 scripts/run_phase4_decision.py

# ── Phase 5 sequence (spec: must not be half-run) ─────────────────
# retrain constrained folds → new OOF → refit isotonic → re-verify → recompute cost curve
# Constrained retrain -> monotonicity GATE -> recalibration -> decision recompute.
# The sequence that cannot be half-run (spec Phase 5).
final-pipeline:
	python3 scripts/run_phase5_final_pipeline.py

explain:
	python3 scripts/run_phase5_explain.py

fairness:
	python3 scripts/run_phase5_fairness.py

# Post-holdout sensitivity analysis on DEVELOPMENT data only (all six frozen
# grids + the cure-rate axis + a fitted segment-policy benchmark). Never
# touches the holdout; results are labelled as development evidence.
sensitivity-dev:
	python3 scripts/run_sensitivity_dev.py

# ── Phase 6: deployment + monitoring ─────────────────────────────
serve:
	python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8008

demo:
	python3 -m streamlit run app/streamlit_app.py

# rebuild the hash-gated serving bundle from training outputs; then `make test`
artifacts:
	python3 scripts/build_artifacts.py

docker-build:
	docker build -t credit-risk:latest .

docker-run:
	docker run -p 8501:7860 credit-risk:latest   # open http://localhost:8501

# ── Phase 7: the one-shot holdout evaluation ──────────────────────
ceremony-dry-run:
	python3 scripts/run_phase7_holdout_ceremony.py --dry-run

holdout-ceremony:
	python3 scripts/run_phase7_holdout_ceremony.py

notebooks:
	python3 scripts/build_notebooks.py
	python3 scripts/build_notebook_02.py
	python3 scripts/build_notebook_03.py
	python3 scripts/build_notebooks_04_06.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb --ExecutePreprocessor.timeout=900
