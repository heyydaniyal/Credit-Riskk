# Credit Risk Scoring — single-container deployment (HF Spaces pattern:
# Streamlit on the public port, FastAPI on an internal port, launched by
# the Streamlit process).
#
# Three stages:
#   deps    runtime wheels into a venv (no build tools, no dev deps)
#   verify  dev deps + the hash gate + the bundle-backed test suite
#           (golden regression in-process AND through HTTP). A drifted
#           artifact or a broken contract fails the BUILD.
#   runtime slim image: venv + code + artifacts only.
# runtime copies a marker file from verify, so BuildKit cannot skip the
# verify stage (an unreferenced stage is silently never built).

FROM python:3.12-slim AS deps
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM deps AS verify
WORKDIR /app
COPY requirements.txt requirements-dev.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY docs/ docs/
COPY app/ app/
COPY config/ config/
COPY src/ src/
COPY artifacts/ artifacts/
COPY tests/ tests/
COPY scripts/run_phase7_holdout_ceremony.py scripts/
RUN python -m pytest -q tests/test_artifacts.py::test_manifest_hashes_match_bundle \
    && python -m pytest -q -p no:cacheprovider \
    && touch /verified

FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
COPY --from=deps /opt/venv /opt/venv
COPY --from=verify /verified /verified
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /app
COPY config/ config/
COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/
RUN mkdir -p data && chown -R app:app /app
USER app
# 7860 = Hugging Face Spaces default; override with -e PORT=8501 elsewhere
ENV PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health')"
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]
