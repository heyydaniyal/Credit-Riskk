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
                    ┌─────────────── DeployedPipeline (src/models/pipeline.py) ───────────────┐
   feature vector ─►│ categorical contract cast → 5 constrained boosters (mean) →             │
                    │ isotonic_final → t*(x) decision → reason codes → 300–850 score          │
                    └───────────────────────────┬─────────────────────────────────────────────┘
                                                ▼
                       FastAPI /score ──► SQLite request log ──► Streamlit Monitoring tab
                       (src/api/app.py)                          (PSI, approval rate, mean t*)
```

**One canonical scoring path.** `DeployedPipeline` is the *only* place the chain is assembled; the API and the holdout ceremony both call it rather than re-implementing it. A **golden regression test** freezes 25 applicants' PDs, decisions, thresholds, scores, and top reason codes to 10 decimal places, so any drift in artifacts or code fails a test before it can reach production.

---

## API contract

`POST /score` — accepts a **precomputed feature vector** (59 fields: the 58 model features plus `NAME_CONTRACT_TYPE`, `term_years`, `AMT_CREDIT` for the decision layer). It does **not** re-run bureau joins; in production that belongs to an upstream feature pipeline, and `scripts/featurize.py` bridges the gap for the demo.

**Request notes.** Numeric fields are `Optional[float]` — JSON `null` becomes NaN, which LightGBM routes natively. Categorical fields are deliberately **unconstrained strings**: unseen values are NaN-routed via the serving contract and *counted in the response*, because an unseen-category spike is itself a drift signal. Impossible values are rejected with 422 (EXT_SOURCE outside [0,1], negative amounts, missing `AMT_CREDIT`).

**Optional `cost_overrides`** are bounded by the frozen sensitivity grids (`lgd_cash` 0.60–0.80, `r_net_cash` 0.03–0.08, `ead_cash` 0.75–0.95, `m_revolving` 0.10–0.30); anything outside returns 422, so the demo can explore sensitivity but can never present un-registered economics as system output.

**Response** — every input to the decision, so any decision is auditable:

```json
{
  "pd_calibrated": 0.3480,          "decision": "REJECT",
  "threshold_applied": 0.0545,      "score_scaled": 433,
  "lgd_used": 0.70, "ead_factor_used": 0.85, "margin_used": 0.0431,
  "term_fallback_used": false,
  "reason_codes": [{"feature": "ext_source_mean",
                    "plain_language": "low average external credit score",
                    "contribution": 1.42}, ...],
  "unseen_category_counts": {},
  "note": "cost parameters are illustrative economic assumptions"
}
```

`GET /health` returns service status, fold count, and feature count.

**Measured latency:** p50 65 ms, p95 **68 ms** single-row including exact TreeSHAP reason codes (46 ms without), on a single idle CPU — confirmed by a second clean benchmark. Note: a reading taken while a training job saturated the CPU returned 143 ms; a contended benchmark is a load artifact, not a regression.

---

## How to run

```bash
pip install -r requirements.txt

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

# serving
make artifacts       # build the hashed deployment set
make serve           # uvicorn on :8008
make demo            # Streamlit (launches the API internally)
make docker-build && make docker-run

# evaluation (one shot)
make ceremony-dry-run   # rehearse the mechanism on dev — never touches holdout
make holdout-ceremony   # THE run; guarded, logged, consistency-checked

make test lint
```

---

## Tests and CI

`ruff` + `pytest` on every push (GitHub Actions). **67 tests**, all passing. Tests requiring run artifacts skip cleanly in a fresh checkout, so CI stays green without the trained models.

| Area | What it protects |
|---|---|
| Aggregations, WOE, selection | join correctness, hand-computed WOE/IV, NaN and unseen-category handling, protected-attribute quarantine |
| Decision layer | `t*` anchor 0.1119, amortization applied to cash but not revolving, hand-computed ledger, best-flat selection |
| Calibration | monotone + bounded output, clip on out-of-range scores, cross-fit improves Brier, PSI identity/shift |
| Models | per-fold WOE refit (leakage), OOF rows scored by unseen models, ensemble = mean of folds, no class weights on the GBM |
| Monotonicity | gate catches a non-monotone model and passes a constrained one |
| Artifacts | feature lists ⊆ matrix, no TARGET/ID/gender contamination, monotonic features survived selection, serving contract matches training levels, **golden scoring regression** |
| API | golden parity **through HTTP**, 422 schema rejections, exactly 3 reason codes on rejection, overrides bounded by frozen grids |
| Monitoring | PSI delegation (single implementation), categorical PSI, band labels, drift scenario direction |
| Ceremony | pre-flight aborts on unknown contract types, run-once guard implemented |

---

## Deployment

**Artifacts.** `make artifacts` builds a 13-file whitelist (5 constrained boosters, `isotonic_final.pkl`, scorecard folds, feature list, categorical contract, golden file, term-fallback rule, feature manifest, monitoring reference) with a **sha256 MANIFEST**. The Docker build re-verifies every hash and fails on drift — proven by tampering: appending one byte to a feature list fails verification on exactly that file. Model files are gitignored, so the artifacts directory is how a deployment gets them.

**Container.** `python:3.11-slim`, pinned requirements, artifact hash check plus unit-contract tests as build gates. The Streamlit process launches uvicorn on an internal port (single-container host pattern, e.g. HF Spaces), so the demo exercises the real API rather than shortcutting into the pipeline.

**Monitoring.** Every `/score` request logs features, predictions, applied threshold, and decision to SQLite. The Monitoring tab shows PSI on the top-10 SHAP features (numeric via quantile bins, categorical via category shares), **score PSI**, rolling approval rate, and **mean `t*(x)`** — the last is necessary precisely because instance-dependent thresholds let a product-mix shift move approvals with no score drift at all. Bands: <0.10 stable, 0.10–0.25 investigate, >0.25 retrain.

**Simulated drift.** The "simulate drift" button applies a severe recession scenario (external scores −0.10, payment burden +25%, income −20%) to 1,000 applicants. Verified to fire: baseline worst PSI 0.085 (stable) → **0.66 (retrain band)** across four features, score PSI 0.29, approval rate 60% → 44%. It is a *framework demonstrated with simulated drift*, labeled as such in the UI and the docs — not production telemetry.

**Retraining policy.** Score PSI > 0.25, or ≥3 watched features > 0.25 → retrain on a refreshed window, then re-run the full Phase 0 validation, recalibration, and fairness audit before promotion.

**What cannot be monitored here:** realized default rates. Labels arrive 12+ months late, which is exactly why credit monitoring leans on PSI rather than live AUC.

---

## Repository layout

```
config/          frozen specs: cost convention, term fallback, holdout plan, phase designs
src/features/    stateless transforms, aggregations, selection, WOE, serving contract
src/models/      scorecard, lgbm, calibration, decision, monotonicity, explain, pipeline
src/monitoring/  PSI, drift scenario, request logging, report
src/api/         FastAPI service
app/             Streamlit demo (Score + Monitoring tabs)
scripts/         phase runners, featurize CLI, artifact builder, holdout ceremony
notebooks/       00–05 executed teaching notebooks
reports/         phase summaries, tables, figures
docs/            TECHNICAL.md, METHODOLOGY.md
artifacts/       hashed deployment set (built by `make artifacts`)
```

**Known gap, stated plainly:** the API scores feature vectors; real-time bureau aggregation is upstream-pipeline work this project does not claim to have built. `featurize.py lookup` serves the demo path from the precomputed matrix; `featurize.py raw` handles a bare application row with stateless transforms only, with the degradation noted in the payload itself.
