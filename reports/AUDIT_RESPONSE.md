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

67 tests passing (at the time), lint clean, artifact manifest refreshed (14 files, hash gate verified against tampering), golden parity preserved through every behaviour-touching change.

---

# Second review (October 2026) — the system layer

The first audit hardened the science. This review asked a narrower question — *does the system actually install, build, test and serve as claimed?* — and found that it did not, outside one machine. Same protocol: every finding reproduced before the fix, verified after; golden parity (25 applicants, 10 decimals) re-checked after every behaviour-touching change, and the full 61,503-row holdout **re-scored with the shipped artifacts to confirm the evaluated model is the served model** (max |Δ p_raw| = 0.0; AUC, Brier, ECE, gap, savings and swap-set all reproduce exactly). No model, calibrator, constant or published number was changed.

## Install / CI / build (all reproduced in clean environments)

**R1 — `requirements.txt` could not be installed.** `streamlit==1.59.2` requires `pyarrow<25`; the file pinned `pyarrow==25.0.0`. `pip install -r requirements.txt` failed with `ResolutionImpossible`, so the quickstart, CI and the Docker build all failed at step one. Fixed: `pyarrow==24.0.0`; runtime and dev requirements split; both verified to resolve in clean 3.12 venvs with pip and uv.

**R2 — the Docker image could not build even with R1 fixed.** Base `python:3.11-slim`, but `shap==0.52.0` requires Python ≥ 3.12. Fixed: 3.12 throughout (CI already used 3.12); SHAP moved to dev requirements (serving uses LightGBM's exact `pred_contrib`, not the SHAP package).

**R3 — the in-image test step could not pass.** `test_decision.py` needed `data/processed/features_full.parquet`, which the image never contains (13 tests: 12 passed, 1 failed in a file-for-file mock-up of the image). Fixed: the test runs from the bundle's demo pool.

**R4 — the deployment bundle was partly gitignored.** `.gitignore` excluded `artifacts/monitoring_reference.json`, which is in the manifest; from any git checkout the Docker hash gate raised `FileNotFoundError`. Fixed; the bundle is committed in full, and `test_manifest_hashes_match_bundle` runs the gate in CI.

**R5 — CI was not green, and the golden HTTP test never ran there.** In a simulated fresh clone: 52 passed, 15 skipped, **1 failed** — contradicting TECHNICAL.md ("skip cleanly, CI stays green"). All five API tests, including the golden-through-HTTP regression the README advertised, were among the skips. Fixed: serving loads only from `artifacts/` (`DeployedPipeline.from_artifacts()`), golden **inputs** ship in the bundle, and tests read from it. Fresh clone now: **73 passed, 10 skipped (training-data-only), 0 failed**, golden regression included.

**R6 — "lint-clean" depended on the ruff version.** CI installed unpinned ruff; ruff 0.16 reported 64 errors on code that was clean under 0.15.22. Fixed: ruff pinned, rule set made explicit in `pyproject.toml`, `app/` added to the lint scope, and the 27 findings under the stricter set resolved (including six `zip()` calls made explicit about length).

**R7 — CI relied on one test file's `sys.path` hack.** Plain `pytest` (as CI invokes it) only worked because `test_core.py` happened to insert the repo root. Fixed: `pythonpath = ["."]` in `pyproject.toml`.

## Serving correctness (each probe is now a regression test)

**R8 — the client could choose the loan term.** `term_years` is both a model feature and the main driver of `t*(x)`, and was accepted unbounded and unreconciled with `AMT_CREDIT`/`AMT_ANNUITY`. Probe: an applicant at PD 43.8% (REJECT, t* 0.067) became APPROVE with `term_years=7` (t* 0.227); `term_years=1000` gave t* 0.977. Fixed: derived features are recomputed server-side by the same functions that built the training matrix (`implied_term_years`, now the single definition — `test_stored_term_years_equals_the_single_definition` asserts byte equality over 307,511 rows); a contradicting value is a 422.

**R9 — an empty request was approved.** `AMT_CREDIT` + contract type alone (57/58 features null) → 200 APPROVE at PD 2.2%. Fixed: a training-support gate per block — application-side ≤ 12 of 36 missing, credit history ≤ 16 of 22 missing — the measured maxima of the training matrix (asserted by `test_input_gate_limits_match_training_data`). This also showed that `featurize.py raw` produced payloads ("history unknown": all 22 aggregates null) that **no training row resembles**; they are now refused, and `--assume-no-history` makes the no-record assumption explicit (count aggregates = 0, exactly like the 2,470 no-history training applicants).

**R10 — unknown fields were silently accepted, including `CODE_GENDER`.** Fixed: `extra="forbid"` on the request and on `cost_overrides`.

**R11 — the drift demo raised a SyntaxError.** A multi-line `st.error(...) if ... else None` expression tripped Streamlit's auto-display; clicking "Simulate drift" rendered an exception box under the alarm. Fixed; the app was smoke-tested end-to-end with runtime-only dependencies (sliders, live traffic, drift alarm: zero exceptions).

## Methodology consistency

**R12 — the holdout sensitivity table re-optimised the flat threshold on the holdout** at every grid point (0.085 at the frozen point vs the headline's dev-frozen 0.08), which is why its central row (5.92M CU/10k) did not equal the headline (6.04M). It also covered 4 of the 6 frozen grids and counted the central point four times ("12 points" = 9 distinct). **Disclosed, not re-run** (the holdout is spent); the oracle flat is the stronger competitor, so every row was conservative. The complete sweep — six grids + cure rate, cross-fitted flat competitors — runs on development data (`make sensitivity-dev`): positive at all **14** distinct points, 1.60%–10.94%.

**R13 — `CURE_RATE` was documentation, not code.** The constant existed; nothing imported it; METHODOLOGY's table was hand-computed. Fixed: the cure rate is an axis of the single cost model (also a bounded API override and a demo slider). `test_cure_rate_axis` reproduces the documented table exactly (0.1119 → 0.1526 → 0.2013). On dev data it compresses the gain the most of any axis (5.64% → 1.60% at c = 0.5).

**R14 — two implementations of the cost model.** `src/decisioning/cost_model.py` duplicated `src/models/decision.py`, kept in step by a test. Deleted; its tests now exercise the production implementation. Override logic previously duplicated in the API moved into the same function; PSI bands defined in three places (one with names shifted by a band) now come from `config/constants.py`.

**R15 — how much of the gain is segmentation?** New development evidence: a cross-fitted per-segment flat policy (revolving + cash in 5 term quintiles) gains +5.15% over a single flat threshold vs +5.64% for `t*(x)` — ~91% of the gain is term/contract segmentation, which the closed-form rule captures with nothing fitted.

## Smaller fixes

- **Latency:** one float matrix per request shared by the five boosters and the reason-code pass; bit-identical (tested); HTTP p95 170 → 119 ms on the same 2-vCPU box. The earlier 68 ms figure was from a different machine and is not comparable.
- **Monitoring:** `read_log()` gives rolling approval-rate and mean-`t*` series from live traffic; the tab plots them (previously the log was write-only). Tests and notebooks no longer write into the service's request log.
- **Streamlit:** sliders now move the whole `t*(x)` distribution over the demo pool (spec DoD), including revolving parameters and the cure rate.
- **Notebooks:** 04 and 05 rewritten to the teaching standard (≈200 → ≈1,400 and ≈900 words of narration each, with live calls into `src/`); new 06 (serving + monitoring); 00 executed; `make notebooks` builds all of them. Notebook 05 no longer repeats the retracted "system chose calibration" claim and shows the within-group miscalibration on dev data.
- **Docs:** test counts, METHODOLOGY's duplicated §6, a dead README anchor, Python version, deployment steps for Hugging Face Spaces.

## Post-fix state

83 tests (fresh clone: 73 passed, 10 skipped, 0 failed), lint clean under the pinned ruff and explicit rule set, artifact bundle 15 files (14 pre-existing hashes reproduced byte-for-byte by a rebuild, `golden_inputs.parquet` added), golden parity preserved, holdout re-score exact. Not done from this environment: a real `docker build` (no registry access — every stage reproduced file-for-file instead; CI's `docker` job runs it) and the public deployment.
