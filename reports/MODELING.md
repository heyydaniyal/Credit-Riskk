# Phase 3 — Modeling Summary

*Both models trained on the same frozen 5-fold splits (paired comparison). All numbers below are OOF — development estimates. Nothing here is a final result; final numbers come from the one-shot holdout in Phase 7.*

## Headline

| Model | OOF AUC | Spec target | Train−OOF gap |
|---|---|---|---|
| WOE logistic scorecard (30 features) | **0.7447** | 0.74–0.76 ✓ | 0.0011 |
| LightGBM tuned (58 features) | **0.7774** | 0.775–0.790 ✓ | 0.1169 (audited — see below) |
| **GBM − scorecard** | **+0.0326** | 2–4 points ✓ | |

Both models land inside their pre-registered corridors, and the gap between them (3.3 AUC points) is exactly what the spec predicted the non-linear model would buy. Leakage alarm (>0.81): not triggered.

## 3a. Scorecard (the industry baseline done properly)

`sklearn.LogisticRegression` on 30 WOE features, `class_weight='balanced'`, C tuned over {0.01…100} by OOF AUC — **with the WOE binner refit inside every fold** (fit on 4, transform the 5th; a unit test verifies the five folds produce five different WOE tables). Findings:

- **The C grid is flat** (all five values within 0.00003 AUC). With n = 246k ≫ p = 30 well-conditioned WOE features, the likelihood swamps any reasonable L2 prior — WOE binning already did the variance control regularization would otherwise provide. A property of the technique, worth saying in interviews.
- **Gap 0.0011** — a linear model on 30 features cannot overfit 246k rows; underfitting is its nature (spec 3c), and the remedy is the GBM, not a bigger linear model.
- **Probability caveat carried forward:** class weighting deliberately distorts probabilities (~11× on the implied base rate). Rank metrics are unaffected; euro decisions must wait for Phase 4's recalibration.

## 3b. LightGBM (tuning under stated compute constraints)

Optuna TPE over the spec's frozen search space; objective = mean fold AUC on the frozen folds; 10,000-round ceiling with 200-round early stopping (best iterations landed at ~2,300–3,300 trees — the data chose the size); **no class weights** (AUC-invariant, and probabilities calibrate better — confirmed: ensemble mean prediction 0.0808 vs base rate 0.0807 before any calibration).

**Compute honesty (1-CPU box):** the study ran as a resumable journal-storage daemon with median pruning plus a 600s per-trial time cap. **The cap's bias was caught and corrected by process:** TPE converged into the slow low-learning-rate region, where two trials were time-capped *while beating the incumbent on partial means*. Both were re-evaluated fully with pruning and cap disabled — and both landed at **0.77735**, a dead heat, +0.0014 over the best capped trial. Two configs from the same low-learning-rate neighborhood converging to identical OOF is *neighborhood-plateau* evidence — the claim is scoped: it demonstrates local convergence, not a certified global optimum, and the recorded search carries a stated time-cap bias against the slowest corner (lr ≲ 0.013). The study remains resumable if another point of AUC ever matters; for this project it does not — the centerpiece is the decision layer, and ±0.002 AUC does not move its story. 13 trials adjudicated (8 complete, 5 pruned); the study remains resumable (`make tune`).

Deployed configuration (trial 20): `learning_rate=0.0138, num_leaves=49, min_data_in_leaf=215, feature_fraction=0.682, bagging_fraction=0.875, lambda_l1≈6e-7, lambda_l2≈4e-4`.

## 3c. The gap alarm — fired, audited, adjudicated

The spec's overfitting alarm (train−OOF gap > 0.025) **tripped**: gap = 0.117 (train AUC ≈ 0.894). The audit, using the study's own trials as evidence (`gap_adjudication.json`, figure 33):

1. **Every** completed trial — including the most heavily regularized (λ₂=5) — sits at gap 0.10–0.14. The 0.025 level is unreachable anywhere inside the frozen search space: the gap is a property of boosted trees on this data, not a tunable pathology.
2. Across the 8 completed trials, gap shows **no relationship with OOF AUC** (r = −0.065 — and with n=8 that correlation is itself statistically uninformative; the range argument in (1) is the load-bearing evidence). The deployed config in fact has among the *smallest* gaps and the best OOF.
3. Playbook step 5 is the live check that remains: if the holdout later disagrees badly with OOF, the CV was leaking — the one-shot Phase 7 run adjudicates finally. (CV integrity was independently verified at AUC 0.502 in Phase 2.)

A first-draft narrative for this audit claimed lower-gap trials generalized worse — **the data said otherwise and the text was corrected**. Alarms exist to force audits; audits exist to follow the evidence.

## Post-phase audit additions (senior review)

- **Serving categorical contract** (`data/processed/categorical_levels.json`): an audit experiment on the saved boosters showed reordered category levels are safe and plain strings fail loudly — but an *incomplete* level set at serving time **silently corrupted 90/500 predictions with no error**. The training levels for all 8 categorical features are now persisted as the explicit contract Phase 6's featurize must cast against (unseen values → NaN deliberately, logged), guarded by `test_serving_categorical_contract_matches_matrix`.
- **Term fallback rule frozen pre-Phase-4** (`config/term_fallback_rule.json`): the 10 dev rows with missing `AMT_ANNUITY` get the dev-median term of their contract type (cash 1.72y, revolving 1.67y) + a `term_fallback_used` flag — frozen *before* any cost-gap number exists, per the pre-registration discipline.

## Deep-audit findings (recorded for future phases)

**External validity — the split is random, not temporal.** Production credit models are validated *out-of-time* (train on the past, validate on a later window) because applicant populations drift; a random split overstates deployable performance by construction. This dataset contains no application date, so OOT validation is *impossible here* — which is worth stating plainly: it is the single largest gap between these numbers and what a bank would accept as evidence, it is why the spec's monitoring phase (6b) leans on PSI rather than assuming stationarity, and it belongs in Phase 7's "what a real bank does differently" alongside reject inference.

**Fold heterogeneity, quantified for Phase 4:** best iterations span 1,140–1,753 (1.5×) and fold AUCs span 0.0026 across the five models. The pooled isotonic calibrator assumes the folds are roughly exchangeable — this spread says the assumption is reasonable but not free, and it is exactly what the 4b cross-fitted metrics and OOF-vs-ensemble overlay will price.

**Serving contract hardened (pandas 4):** the audit found the previously documented cast (`astype(CategoricalDtype)`) is deprecated and *raises* under pandas 4 on precisely the unseen-value case it exists to handle. Replaced by `src/features/serving.cast_with_contract()` — unseen → NaN explicitly (LightGBM missing branch) with counts returned for drift logging; the full serving path is verified to reproduce training predictions exactly (Δp = 0 on 500 rows), under warnings-promoted-to-errors.

**Holdout analysis plan frozen** (`config/holdout_analysis_plan.json`): the exact ceremony outputs, in order, pre-registered *before any Phase 4 result exists* — closing the forking-paths hole the spec's prose left open.

**Advisories for Phases 5–6:** (a) `monotone_constraints` is a positional vector — it must be built from the feature list order with 0 for categoricals; a mis-aligned vector constrains the wrong features *silently*. (b) TreeSHAP on 5 boosters × 246k rows is expensive on this box — the spec's "explain fold-1 and say so" clause is the intended path. (c) `models/*.txt` is gitignored, so the Phase 6 Docker build must either retrain via `make` or copy artifacts explicitly — decide before writing the Dockerfile, not during.

## What flows to Phase 4

| Artifact | Contents |
|---|---|
| `data/processed/oof_lgbm.parquet` | (ID, fold, y, p_raw) — validated: complete, unique IDs, AUC reproduces |
| `data/processed/oof_scorecard.parquet` | same, for the baseline |
| `models/lgbm_fold{0..4}.txt` | the deployment fold ensemble — each saved model verified to reproduce its OOF slice exactly |
| `models/scorecard_folds.pkl` | 5 × (binner, logistic) fold artifacts |
| `reports/tables/lgbm_final.json`, `scorecard_cv.json`, `model_comparison.csv`, `gap_adjudication.json`, `optuna_study_state.json` | full audit trail |

Phase 4 fits isotonic on the LightGBM OOF, cross-fits the calibration *metrics*, verifies the OOF-vs-ensemble distribution overlay, and builds t*(x) on the frozen cost constants. SHAP is deferred to Phase 5 so explanations describe the *constrained* shipped model.
