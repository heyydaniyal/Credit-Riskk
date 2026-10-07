# Credit Risk Scoring — single-container deployment (HF Spaces pattern)
# Build gate: artifact hashes are re-verified and the artifact-free unit
# contracts (test_decision.py) run INSIDE the image; a drifted artifact
# fails the build. (The full golden regression runs in CI, not in-image,
# to keep the image lean.)
FROM python:3.11-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/
COPY tests/test_decision.py tests/
# artifacts land where the code expects them
RUN mkdir -p models data/processed && \
    cp artifacts/lgbm_constrained_fold*.txt artifacts/isotonic_final.pkl \
       artifacts/scorecard_folds.pkl models/ && \
    cp artifacts/lgbm_features.json artifacts/categorical_levels.json \
       artifacts/golden_scoring.json artifacts/feature_manifest.csv \
       artifacts/monitoring_reference.json artifacts/demo_pool.parquet data/processed/ 2>/dev/null; \
    cp artifacts/term_fallback_rule.json config/
# build-time verification: hashes + unit contracts (artifact-free tests)
RUN python - <<'PY'
import hashlib, json
m = json.load(open("artifacts/MANIFEST.json"))["sha256"]
for name, h in m.items():
    if name == "MANIFEST.json": continue
    actual = hashlib.sha256(open(f"artifacts/{name}", "rb").read()).hexdigest()
    assert actual == h, f"ARTIFACT DRIFT: {name}"
print("artifact hashes verified")
PY
RUN python -m pytest tests/test_decision.py -q

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
