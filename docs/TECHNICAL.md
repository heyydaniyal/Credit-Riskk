# Technical Reference

Architecture, API contract, and how to run everything. Companion docs: [`METHODOLOGY.md`](METHODOLOGY.md) (derivations, evaluation discipline) and the executive [README](../README.md).

---

## Architecture

```
 raw Kaggle tables                    ┌── development (Phases 0–5) ──────────────┐
 application_train ─┐                 │ 5-fold CV on the 80% dev split           │
 bureau ────────────┼─► aggregations ─┤ Optuna-tuned LightGBM + WOE scorecard    │
 bureau_balance ────┤   + domain      │ isotonic calibration (cross-fit metrics)  │
 previous_appl. ────┘   ratios        │ monotonic constraints + gate              │
                                      └───────────────┬───────────────────────────┘
                                                      ▼
                                                      ▼  make artifacts
                     artifacts/  (hash-gated bundle, MANIFEST.json — what CI tests and serving loads)
                                                      ▼
                    ┌─────────────── DeployedPipeline (src/models/pipeline.py) ───────────────┐
   feature vector ─►│ categorical contract cast → one float matrix → 5 constrained boosters   │
   (input gate:     │ (mean) → isotonic_final → t*(x) decision → reason codes → 300–850 score │
    src/api)        └───────────────────────────┬─────────────────────────────────────────────┘
                                                ▼
                       FastAPI /score ──► SQLite request log ──► Streamlit Monitoring tab
                       (src/api/app.py)                          (rolling approval rate, mean t*, PSI)
```

**One canonical scoring path.** `DeployedPipeline` is the *only* place the chain is assembled; the API, the demo and the holdout ceremony all call it. Serving uses `DeployedPipeline.from_artifacts()`, which loads **only** the hash-gated bundle in `artifacts/` — the same bytes CI tests and the Docker build verifies. A **golden regression test** freezes 25 applicants' PDs, decisions, thresholds, scores and top reason codes to 10 decimal places; their inputs ship in `artifacts/golden_inputs.parquet`, so the regression runs in CI and inside the image (in-process **and** through HTTP), not only on a machine holding the training matrix.

**One cost model.** `src/models/decision.attach_cost_params` is the single implementation of LGD/EAD/margin/`t*(x)`, including bounded sensitivity overrides and the cure-rate axis. (A parallel `src/decisioning/cost_model.py`, previously kept in step by a test, was deleted.) Term is derived by one function, `src/features/stateless.implied_term_years`, used both to build the training matrix and by the API.

---

## API contract

`POST /score` — accepts a **precomputed feature vector** (59 fields: the 58 model features plus `NAME_CONTRACT_TYPE`, `term_years`, `AMT_CREDIT` for the decision layer). It does **not** re-run bureau joins; in production that belongs to an upstream feature pipeline, and `scripts/featurize.py` bridges the gap for the demo.

**Request notes.** Numeric fields are `Optional[float]` — JSON `null` becomes NaN, which LightGBM routes natively. Categorical fields are deliberately **unconstrained strings**: unseen values are NaN-routed via the serving contract and *counted in the response*, because an unseen-category spike is itself a drift signal.

**Input gate — what the service refuses (422), and why.** Each rule reproduces a defect found in review:

| Rule | Why |
|---|---|
| Impossible values: `EXT_SOURCE_*` outside [0,1], negative amounts, missing `AMT_CREDIT` / `NAME_CONTRACT_TYPE`, unknown contract type | cost parameters would be undefined or nonsensical |
| **Derived features must match their sources.** `term_years`, `credit_goods_ratio`, `ext_source_mean/min/n_missing`, `EXT_SOURCE_1/3_is_missing` are recomputed from the fields they come from; a supplied value that disagrees is rejected, an omitted one is filled | `term_years` drives both the PD and `t*(x)`: a 43.8%-PD reject became an approve by sending `term_years=7` (t* 0.067 → 0.227); `term_years=1000` gave t* = 0.98 |
| **Training-support gate.** At most 12 of 36 application-side features missing, at most 16 of 22 credit-history aggregates missing — the maxima of the 307,511-row training matrix (`config/constants.py`, verified by a test) | a payload of only `AMT_CREDIT` + contract type returned **APPROVE at PD 2.2%**: LightGBM follows its NaN branches and returns a confident low PD. "History unknown" (all 22 aggregates null) never occurs in training; a genuine no-history applicant has six zero-valued count/sum aggregates |
| **Unknown fields forbidden** (`extra="forbid"`), in the body and in `cost_overrides` | the protected attribute `CODE_GENDER` cannot even reach the scoring path |
| `cost_overrides` outside the frozen sensitivity grids | the demo explores sensitivity; it can never present unregistered economics as system output |

**Optional `cost_overrides`** — `lgd_cash` 0.60–0.80, `lgd_revolving` 0.75–0.95, `ead_cash` 0.75–0.95, `ead_revolving` 0.85–1.00, `r_net_cash` 0.03–0.08, `m_revolving` 0.10–0.30, `cure_rate` 0–0.50 (the spans of the frozen grids; `GET /cost-bounds` returns them). Overrides move `t*(x)` and the decision, never the PD.

**Response** — every input to the decision, so any decision is auditable:

```json
{
  "pd_calibrated": 0.3480,          "decision": "REJECT",
  "threshold_applied": 0.0545,      "score_scaled": 433,
  "lgd_used": 0.70, "ead_factor_used": 0.85, "margin_used": 0.0431,
  "cure_rate_used": 0.0,            "term_years_used": 1.72,
  "term_fallback_used": false,
  "reason_codes": [{"feature": "ext_source_mean",
                    "plain_language": "low average external credit score",
                    "contribution": 1.42}, ...],
  "unseen_category_counts": {},
  "cost_overrides_used": {},
  "model_version": "69ef259f612a",
  "note": "cost parameters are illustrative economic assumptions; ..."
}
```

`GET /health` returns service status, fold count, feature count and model version (hash of the artifact manifest).

**Latency.** Each request builds **one float matrix** shared by the five fold models and the reason-code pass; LightGBM's per-call pandas conversion had been over half of single-row time. Outputs are bit-identical (asserted by `test_matrix_fast_path_is_bit_identical_to_pandas_path`, the golden tests, and a 61,503-row holdout re-score with max |Δ| = 0.0). On the same 2-vCPU sandbox, HTTP single-row p95 with reason codes fell from **170 ms to 119 ms**. Absolute latency is hardware-dependent (an earlier benchmark on a different idle machine measured p95 68 ms before this change); measure on the deployment host before quoting a number — `reports/tables/latency_benchmark.json`.

---

## How to run

Python **3.12**. `requirements.txt` is the runtime set (what the image serves with); `requirements-dev.txt` adds tests, lint (ruff pinned), training and notebook tooling. Both are verified to resolve in a clean venv with pip and uv. (The previous single file pinned `pyarrow==25.0.0` against `streamlit 1.59.2`'s `pyarrow<25`, so it could not be installed at all.)

```bash
pip install -r requirements-dev.txt

# development pipeline (in dependency order; each phase gates the next)
make phase0          # frozen splits + folds
make features        # feature matrix + selection  (re-run until null-chunks prints DONE)
make scorecard       # WOE logistic baseline
make tune            # resumable Optuna study
make model-final     # fold ensemble + OOF predictions
make calibrate       # isotonic + cross-fitted metrics + alignment overlay
make decision        # t*(x), gap checkpoint, curves, swap-set
make final-pipeline  # constrained retrain → monotonicity GATE → recalibration
make explain fairness
make sensitivity-dev # all six cost grids + cure rate, cross-fitted comparators (dev only)

# serving
make artifacts       # rebuild the hash-gated bundle from training outputs (reproducible:
                     # a rebuild reproduces every existing hash byte-for-byte)
make serve           # uvicorn on :8008
make demo            # Streamlit (launches the API internally)
make docker-build && make docker-run   # http://localhost:8501

# evaluation (one shot)
make ceremony-dry-run   # rehearse the mechanism on dev — never touches holdout
make holdout-ceremony   # THE run; guarded, logged, consistency-checked (already spent)

make notebooks       # rebuild + execute 00–06
make check           # ruff + pytest
```

---

## Tests and CI

`ruff` (pinned, explicit rule set in `pyproject.toml`) + `pytest` on every push, then a Docker build. **83 tests.** A fresh clone has no training outputs (`data/processed/`, `models/` are gitignored) but does have the committed `artifacts/` bundle, so in CI: **73 pass, 10 skip** (only tests that need the 60 MB feature matrix or training outputs), **0 fail** — and the golden regression, through HTTP, is among the 73. Tests never write into the service's request log (`tests/conftest.py`).

| Area | What it protects |
|---|---|
| Aggregations, WOE, selection | join correctness, hand-computed WOE/IV, NaN and unseen-category handling, protected-attribute quarantine |
| Decision layer | `t*` anchor 0.1119, amortization on cash not revolving, term capping, cure-rate axis, overrides bounded by frozen grids, hand-computed ledger, best-flat selection, cross-fitted comparators are not oracles |
| Calibration | monotone + bounded output, clip on out-of-range scores, cross-fit improves Brier, PSI identity/shift |
| Models | per-fold WOE refit (leakage), OOF rows scored by unseen models, ensemble = mean of folds, no class weights on the GBM |
| Monotonicity | gate catches a non-monotone model and passes a constrained one |
| Artifacts | manifest hashes, bundle == training outputs, **golden regression from the bundle**, fast path bit-identical to the pandas path, gate constants = measured training maxima, stored `term_years` = the single definition, feature lists and serving contract consistent, every third-party import declared |
| API | golden parity **through HTTP**, 422 rejections, client cannot choose `term_years`, derived features must match sources, near-empty and history-unknown payloads refused, `CODE_GENDER` and unknown fields refused, cure-rate override, exactly 3 reason codes on rejection |
| Monitoring | PSI delegation (single implementation), categorical PSI, band labels, drift scenario direction, request-log round trip and rolling series |
| Ceremony | pre-flight aborts on unknown contract types, run-once guard implemented |

---

## Deployment

**Artifacts.** `make artifacts` builds a 15-file bundle (5 constrained boosters, `isotonic_final.pkl`, scorecard folds, feature list, categorical contract, golden outputs **and inputs**, term-fallback rule, feature manifest, monitoring reference, demo pool) with a **sha256 MANIFEST**. The bundle is committed in full — an earlier `.gitignore` excluded `monitoring_reference.json`, which made the Docker hash gate crash on any git checkout. A rebuild reproduces every pre-existing hash byte-for-byte.

**Container — three stages** (`Dockerfile`):

1. `deps` — runtime requirements into a venv on `python:3.12-slim`.
2. `verify` — dev requirements, then the hash gate and the bundle-backed test suite (golden regression in-process and through HTTP). A drifted artifact or a broken contract **fails the build**. The runtime stage copies a marker file from this stage, so BuildKit cannot silently skip it.
3. `runtime` — slim image with the venv, code and artifacts only; non-root user; `HEALTHCHECK` on Streamlit; port from `$PORT` (default 7860, the Hugging Face Spaces convention).

The Streamlit process launches uvicorn on an internal port (`$API_PORT`, default 8008), so the demo exercises the real API rather than shortcutting into the pipeline.

> **Verification note.** The image build itself has not been run from this environment (no registry access to pull `python:3.12-slim`). Every step was reproduced file-for-file instead: both requirement sets installed into clean 3.12 venvs, the verify stage's exact file layout ran the hash gate and the suite (73 passed, 10 skipped), and the runtime layout served the Streamlit app end-to-end with runtime-only dependencies. CI's `docker` job runs the real build on every push.

### Deploying to Hugging Face Spaces

1. Create a Space with the **Docker** SDK.
2. Push this repository to it (the Space builds the `Dockerfile`; the runtime listens on `$PORT` = 7860). Add this header to the Space's README:
   ```yaml
   ---
   title: Credit Risk Scoring
   sdk: docker
   app_port: 7860
   ---
   ```
3. The build fails if any artifact hash or contract test fails — a broken model cannot go live.
4. Put the Space URL in the first line of the README.

**Monitoring.** Every `/score` request logs prediction, threshold, decision and timestamp to SQLite — deliberately **no applicant features** (plaintext PII with no retention control). `read_log()` turns the log into rolling approval-rate and rolling mean-`t*(x)` series; the Monitoring tab plots them and can push demo or simulated-recession traffic through the API to move them. Mean `t*(x)` is tracked because instance-dependent thresholds let a product-mix shift move approvals with no score drift at all. The tab also shows PSI on the top-10 SHAP features (numeric via quantile bins, categorical via category shares) and **score PSI**. Bands: <0.10 stable, 0.10–0.25 investigate, >0.25 retrain. The log path is `$CREDIT_RISK_REQUEST_LOG` if set, else `data/request_log.sqlite`.

**Simulated drift.** The "simulate drift" button applies a severe recession scenario (external scores −0.10, payment burden +25%, income −20%) to 1,000 applicants. Verified to fire: worst feature PSI 0.08 (stable) → **0.65 (retrain band)**, score PSI 0.011 → 0.32, approval rate 62.5% → 44.4%; mean `t*(x)` unchanged (thresholds depend on product and term, not scores). It is a *framework demonstrated with simulated drift*, labeled as such in the UI and the docs — not production telemetry.

**Retraining policy.** Score PSI > 0.25, or ≥3 watched features > 0.25 → retrain on a refreshed window, then re-run the full Phase 0 validation, recalibration, and fairness audit before promotion.

**What cannot be monitored here:** realized default rates. Labels arrive 12+ months late, which is exactly why credit monitoring leans on PSI rather than live AUC.

---

## Repository layout

```
config/          frozen specs: cost convention, term fallback, holdout plan, phase designs
src/features/    stateless transforms, aggregations, selection, WOE, serving contract
src/models/      scorecard, lgbm, calibration, decision, monotonicity, explain, pipeline
src/monitoring/  PSI, drift scenario, request log (write + rolling read), report
src/api/         FastAPI service
app/             Streamlit demo (Score + Monitoring tabs)
scripts/         phase runners, featurize CLI, artifact builder, holdout ceremony
notebooks/       00–06 executed teaching notebooks (06 = serving + monitoring)
reports/         phase summaries, tables, figures
docs/            TECHNICAL.md, METHODOLOGY.md
artifacts/       hash-gated deployment bundle (built by `make artifacts`, committed)
tests/           83 tests; conftest isolates the request log
```

**Known gap, stated plainly:** the API scores feature vectors; real-time bureau aggregation is upstream-pipeline work this project does not claim to have built. `featurize.py lookup` serves the demo path from the precomputed matrix. `featurize.py raw` builds a payload from a bare application row with stateless transforms only; its credit-history aggregates are *unknown*, which the API refuses (no training applicant lacks all history). `--assume-no-history` states explicitly that the applicant has no bureau or previous record (count/sum aggregates = 0, as for the 2,470 no-history training applicants), and that payload scores.
