# Audit Response — fixes with evidence

Each external-audit finding below was **reproduced before being fixed** and **verified after**. Where the evidence contradicted the audit, that is noted rather than smoothed over. Fixes that would have changed a published number without an untouched evaluation set were deferred by protocol (the holdout is spent), not skipped.

## Production defects (fixed, regression-tested)

**F1.3 — `/score` crashes / returns invalid decisions.** Reproduced four paths: unknown contract type → HTTP 500 (NaN cost params → non-JSON response), null contract → 500, negative `term_years` → HTTP 200 with `threshold_applied = −0.144`, zero term → threshold 0.0. Added a decision-layer validation gate (mirrors the ceremony pre-flight) plus a defense-in-depth invariant rejecting any threshold outside (0,1). All four now return clean 422s; the valid path is untouched and **golden parity holds to 10 decimals**. Test: `test_decision_layer_inputs_are_validated`.

**F8.1 — no error handling.** The scoring body now wraps the pipeline call and surfaces a clean 500 message rather than a stack trace, while validation 422s still pass through (verified: unknown contract still 422, not swallowed).

## Config integrity (fixed — evidence changed the fix)

**F1.1 / F1.2 / F8.3 — constants hardcoded, config not read, grids inconsistent.** Before fixing, I checked whether the hardcoded values *matched* config: all did except `M_REVOLVING`, which was `0.18` in `decision.py` but `0.18000000000000002` in config (the raw product `0.10·3·0.6`). Sourcing from config blindly would have shifted every revolving decision by a float epsilon and broken golden parity. Fix order: round config to the exact literal → source `decision.py` from config → reconcile the stray `ead_revolving` grid (`[0.85,0.90,1.00]` → `[0.85,1.00]`, which the ceremony never varied, so no published number moved). Verified behaviour-preserving via golden regression. `cost_model.py` was found to already source config and agree with `decision.py` to 0.0, so it was kept and pinned by a new drift-prevention test rather than deleted.

## The findings that change what the project claims (fixed with disclosure)

**F7.1 — the headline gap is partly circular.** Verified: replacing outcomes `y` with the model's own probabilities `p` (no outcome data read) reproduces the gap at +4.1% vs the realised +3.67%. Added a candid subsection to METHODOLOGY §1: the *sign* is guaranteed by construction, the *magnitude* and *that calibration is good enough for it to materialise* are the empirical content (scrambling probabilities collapses the realised gap 5.3% → 2.1%). README headline now carries the caveat.

**F2.2 — default definition never stated; PD/LGD event mismatch.** `TARGET` is an **early-delinquency** label, not a Basel charge-off, while LGD/EAD are charge-off severities. Added `CURE_RATE` as a first-class config assumption and sensitivity axis. Verified the magnitude the audit claimed: a 50% cure rate moves the 3-year anchor **+80%**, versus **±13%** for the entire LGD grid — event-definition uncertainty dominates every parameter in the cost model. Deployed path keeps `CURE_RATE = 0` (behaviour-preserving); the dominance is now documented.

**F6.2 — false group-calibration claim.** The fairness section claimed the model "chose calibration." Verified it is aggregate-calibrated but **not** within-group: on holdout, F over-predicted +5.7% relative, M under-predicted −7.5%. Corrected the text to state it has neither within-group calibration nor equalized odds.

## Discrimination reporting (added)

**F5.1 / F5.2 — Gini/KS/PR-AUC and risk bands absent from top docs.** Computed on holdout and added to METHODOLOGY §5 + README: **Gini 0.5605, KS 0.4198, PR-AUC 0.2773** (baseline 0.0807), plus a decile risk-band table — rank ordering monotone across all ten deciles, worst decile 3.7× base-rate lift capturing 37% of defaults.

## Explainability (fixed)

**F11.2 — 49% of reason codes fell back to raw column names.** Completed the plain-language mapping to all 58 features. Fallback rate now **0%** (was ~49%), all applicants still receive 3 codes.

**F11.3 — marital status emittable as an adverse reason.** Added `PROHIBITED_REASON_FEATURES`; `reason_codes()` now suppresses a prohibited basis and backfills the next admissible feature. Verified **0 prohibited-basis emissions** (was 187 / 6000 on a 2k sample). Golden reason codes regenerated (PD/decision/threshold provably unchanged).

## MLOps / privacy (fixed)

**F9.2 — plaintext PII request log shipped in the archive.** Deleted `data/request_log.sqlite`; already gitignored.

**F1.9 / F9.3 — no model provenance.** `/score` and `/health` now return a `model_version` (12-char hash of the artifact manifest); the ceremony output stamps `artifact_manifest_sha`.

**F1.5 — container CMD couldn't run** (Streamlit read the uncopied 60MB matrix). Built a `demo_pool.parquet` artifact (3,000 **dev-only** rows, verified zero holdout leakage), repointed the app, wired it into `build_artifacts.py` and the Dockerfile. Container is now self-contained.

## Documentation truth (fixed)

- **F1.7** — `make holdout-ceremony` was still an echo stub (I had wrongly reported wiring it in Phase 6); now calls the real script.
- **F1.6** — Dockerfile header overstated the in-image gate; corrected to say the golden regression runs in CI, hashes + unit contracts run in-image.
- **F1.10** — removed the dead `if False else` branch, made the SHAP sample genuinely stratified (matching the figure title), corrected the fold-1/fold-0 mislabel.
- **F13.1** — README dev AUC `0.7760` → `0.7768`.
- **F12.4** — stale "62 tests" → **67**.

## Where the audit was wrong or overstated (recorded)

- **F1.11** claimed 29 unclosed file handles; direct count finds 3 `open().read()` patterns. Finding real, magnitude inflated ~10×.
- **F4.1** claimed the consistency band is "2.0× / ~4 SE wide"; measured **1.1×** the 2-SE width — essentially correctly sized. Direction fair, quantification doubled.
- **F4.3**: I had earlier told the user 5.28% fell *outside* the holdout gap CI; it is **inside** [2.00%, 5.43%]. The audit was right; I was wrong.

## Deferred by protocol (not skipped)

The improvement candidates in `IMPROVEMENT_ANALYSIS.md` (e.g. `bureau_debt_missing_share`, extended constraints) remain unshipped for the same reason as before: **the holdout is spent**, and adopting a change now would invalidate it as an unbiased estimate. A v2 adopts on development evidence, obtains an untouched evaluation set, and re-runs the full chain.

## Post-fix state

67 tests passing, lint clean, artifact manifest refreshed (14 files, hash gate verified against tampering), golden parity preserved through every behaviour-touching change.
