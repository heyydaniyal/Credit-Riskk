# Phase 6 — Deployment + Monitoring

*The serving layer is a validated transport around the golden-guarded `DeployedPipeline` — it adds validation and observability, never logic. Every design decision was frozen (with experimental evidence) in `config/phase6_design_frozen.json` before this code existed.*

## The API (`src/api/app.py`)

`POST /score` accepts the 59-field feature vector (pydantic model generated programmatically from the manifest; JSON `null` → NaN for numerics; categoricals are **deliberately unconstrained strings** — unseen values NaN-route through the serving contract with counts returned, because drift observability beats strict rejection). The response returns *every input to the decision*: calibrated PD (display-clipped), decision, t\*(x), LGD/EAD/margin used, fallback flag, exactly 3 adverse reason codes, the 300–850 score, unseen-category counts, and the illustrative-assumptions note. Optional `cost_overrides` are **bounded by the frozen sensitivity grids** — out-of-grid economics is a 422, so the demo can explore sensitivity but never present un-registered assumptions as system output. Impossible values (EXT_SOURCE outside [0,1], negative amounts, missing AMT_CREDIT) reject with 422; everything else is the model's job, not the transport's.

**Transport proven, twice:** the pre-build experiment showed JSON round-trip parity on the golden set, and `test_golden_through_http` now holds the HTTP layer to the same 10-decimal standard permanently. **Latency, measured honestly:** p95 = 65ms single-row with reason codes (46ms without) on a 1-CPU sandbox — the spec's <100ms met with margin, and the measured number is the quoted number. One found-and-fixed bug: FastAPI's two-body-parameter embedding broke the flat payload contract; `cost_overrides` now nests inside the single request model. *(October 2026: absolute latency is hardware-dependent — the same-machine before/after comparison and the serving fast path are in [AUDIT_RESPONSE](AUDIT_RESPONSE.md#second-review-october-2026--the-system-layer).)*

## featurize CLI (`scripts/featurize.py`)

`lookup --id` pulls an applicant's vector from the precomputed matrix (the demo path — bureau joins are upstream-pipeline work, said plainly); `raw --csv` handles a bare application row with stateless transforms only, aggregates NaN'd + flagged, degradation documented in the payload itself. End-to-end proven: featurize → HTTP → decision (applicant 100002, a known defaulter, correctly REJECTED at PD 0.348).

## Monitoring (`src/monitoring/`) — demonstrated with **simulated** drift

Watchlist per the frozen design: PSI on the top-10 SHAP features (numeric via the tested quantile-bin PSI; **categorical via category-share PSI** — a gap the first gate run exposed when `ORGANIZATION_TYPE` turned out to be watched), score PSI, rolling approval rate, and mean t\*(x) — the metric instance-dependent thresholds make necessary. Bands 0.10/0.25 (Siddiqi). Requests log to SQLite. The label-latency paragraph is in the UI itself: realized default rates are unobservable for 12+ months, which is *why* credit monitoring leans on PSI.

**The drift-alarm gate — failed twice before passing, and both failures were the system working:**

1. *First run:* the recession scenario drifted income ratios — features **not on the watchlist** — and the alarm correctly stayed silent (score PSI barely moved because the model barely uses them). Lesson recorded in the gate artifact: scenario design must target watched, model-relevant features, and an alarm that fires on irrelevant drift would be the actual bug.
2. *Second run:* the realistic scenario (bureau scores deteriorate) landed at PSI = 0.25 — exactly on the band. Deepened to a **severe stress** (−0.10 on EXT_SOURCE features, labeled as such).
3. *Final:* baseline clean (worst 0.085) → severe recession fires **four features into RETRAIN band (up to 0.66), score PSI 0.29, approvals 60%→44%** — a coherent economic story, end to end. The scenario is defined **once** (`simulate_recession`) so the UI button and the validation gate can never diverge.

## Two latent bugs the tests caught during the build

**1. Package-attribute shadowing (would have broken the live dashboard).** `src/monitoring/__init__.py` did `from src.models.calibration import psi`, but the package also contains a submodule named `psi` — so importing `src.monitoring.psi` anywhere rebinds the name on the package and the numeric PSI path raises `'module' object is not callable`. It passed every run until a test imported the submodule first, which is exactly how it would have failed in production: fine in dev, broken once import order changed. Fixed with an explicit alias (`_psi`) and a comment naming the trap.

**2. Two PSI implementations (a divergence waiting to happen).** The scaffold's `compute_psi` coexisted with the canonical one. They were verified to agree numerically (0.238286 vs 0.238286 on a shifted normal) — but the dashboard could have drifted from the tested path silently. The legacy function now delegates to the canonical implementation; the scaffold-era test still passes, guarding the wrapper.

**Artifact gate proven by tampering:** appending one byte to `lgbm_features.json` makes verification fail on exactly that file — the Docker build cannot ship a drifted artifact.

## Streamlit (`app/streamlit_app.py`) + Docker

Two tabs: *Score* (applicant picker, metrics, reason codes, grid-bounded sliders labeled "illustrative assumptions — explore sensitivity") and *Monitoring* (PSI dashboard + the simulated-drift button). The Streamlit process launches uvicorn internally — the demo exercises the real API. The Dockerfile (python:3.11-slim) copies the 13-artifact whitelist and **fails the build** on hash drift or a failing unit-test contract — a stale model cannot ship. Container start command targets the single-container host pattern.

## Honest scopes

The live demo URL requires the owner's hosting credentials — this phase delivers the deployable repo, container, and instructions. The Docker build itself is validated to the extent this sandbox allows (all gate steps run identically outside Docker; docker-engine execution happens on the host). The API scores *feature vectors*; the production feature pipeline (real-time bureau joins) is named as upstream work, not silently claimed.
