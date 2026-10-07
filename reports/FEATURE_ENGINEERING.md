# Phase 2 — Feature Engineering Summary

*All stateful steps (selection thresholds, correlations, importance distributions, WOE bins) were fit on the DEV set (246,008 rows) only. The holdout was transformed, never inspected — confirmed by the pipeline-integrity check below.*

## What was built

**182-column feature matrix** (307,511 rows) = 119 raw application columns + 41 per-applicant aggregates from `bureau`/`bureau_balance`/`previous_application` + 22 engineered features (domain ratios, missingness flags, sentinel fixes). `CODE_GENDER` is quarantined into a separate audit frame **at build time** — no downstream code can accidentally train on it.

**Selection funnel (all fit on dev):** 179 candidates → near-constant drop (−19) → |Spearman| > 0.98 drop, keep the higher-AUC twin (−28) → null-importance filter, 50 shuffled-target runs, gain must beat the feature's own 95th-percentile noise floor (−75) → **58 final LightGBM features**.

**Scorecard set:** WOE-binned (5 quantile bins, NaN = own bin, rare categories → Other), screened by Information Value ∈ [0.02, 0.5] → **30 features** for the logistic baseline.

## Headline findings

1. **58 features match 133.** Fold-1 AUC with the selected 58: **0.7720**; with all 133 post-correlation candidates: 0.7712 (Δ = −0.0009, noise). The spec's "fewer that match is a better result" branch, verified empirically rather than asserted (`selection_performance_check.json`).

2. **The engineered composite is the single strongest feature.** `ext_source_mean` carries 31× its null-importance noise floor — more gain than the next four features combined. `ext_source_min` (worst-case bureau view) and `term_years` (the cost-model input) are #3 and #4. The credit-officer features earn their keep.

3. **Selection agrees with EDA.** `credit_income_ratio` was dropped by null-importance (gain 1205 vs noise floor 1238) — consistent with EDA finding #8, where leverage deciles were surprisingly flat. The payment-burden ratio (`annuity_income_ratio`) survived instead. Two independent methods, one conclusion.

4. **IV audit fired, was audited, rule held.** `ext_source_mean` scored IV = 0.572 > 0.50 and was flagged per the frozen rule. Audit conclusion: **not leakage** — a composite of the three strongest legitimate predictors (EXT_SOURCE_3 alone: IV 0.31) naturally exceeds the ceiling. It stays out of the *scorecard* because the rule is the rule; no signal is lost since its components are individually in-range and a linear model weights them itself. (It remains in the LightGBM set, where the IV rule does not apply.)

5. **The correlation dropper caught our own duplicates.** `age_years` vs `DAYS_BIRTH` (Spearman = −1.000) and `employment_years` vs `DAYS_EMPLOYED` — engineered readability duplicates, correctly pruned. Also the notorious housing `_AVG/_MODE/_MEDI` triplets (18 of the 28 drops) and `AMT_CREDIT` vs `AMT_GOODS_PRICE` (ρ = 0.984; the higher-AUC goods price kept, with the credit information preserved via `credit_goods_ratio`, `payment_rate`, and `term_years`). Note: `AMT_CREDIT` leaving the *model's* feature set does not affect the Phase 4 cost model, which reads it from the applicant record, not from model features.

6. **A found interaction: `is_pensioner` ≈ ¬`FLAG_EMP_PHONE`** (ρ = −0.9998). Pensioners are exactly the applicants without a work phone — one kept, one dropped.

## Preprocessing policy (justified omissions)

**No feature scaling.** Trees split on rank order — any monotone rescaling produces the identical model. For the scorecard, WOE *is* the scaling: every feature lands on the same log-odds scale.

**No imputation for the GBM set.** LightGBM's native NaN routing learns the best branch for missing values — imputing would destroy the missingness signal documented in EDA finding #3. Explicit `_is_missing` flags surface that signal to both models.

**No imputation for the scorecard either.** NaN is its own WOE bin with its own empirically-estimated risk (the EXT_SOURCE_1-missing bin carries the 8.5%-vs-7.5% differential directly).

**NaN policy after joins:** counts/sums for applicants with no history → 0 (factually zero prior loans); means/maxes of empty history → NaN (undefined, and the flag carries the signal). The two "no history" flags are kept separate because they point in **opposite directions** (no bureau file = riskiest at 10.1%; no prior HC application = safest at 6.0% — EDA findings #11/#12).

## Validation gates (all passed)

| Check | Result | Meaning |
|---|---|---|
| Pipeline integrity (dev vs holdout, final features) | AUC **0.5024** | No processing statistic crossed the split boundary |
| Sanity fold-1 AUC (default params) | **0.7720** | Learnable, inside the expected 0.76–0.80 corridor |
| Leakage alarm (> 0.81) | not triggered | No too-good-to-be-true signal |
| Aggregates vs hand computation (3 random applicants + no-history case) | exact match | Join logic correct |
| TARGET alignment after joins | exact match | Rows never scrambled |
| 23 unit tests (aggregation math, WOE hand-computed, NaN/unseen-category handling, protected-attribute quarantine) | all pass | |

## Leakage discipline notes for Phase 3

- The dev-fit `WOEBinner` artifact is for reporting/holdout-transform only. **Phase 3 CV must refit the binner inside each fold** (fit on 4, transform the 5th) — the class is a fit/transform object precisely so this is possible.
- Fitting *selection* on full dev (rather than per fold) makes OOF estimates very slightly optimistic — standard trade-off for one stable feature list, stated here rather than hidden; the untouched holdout keeps final numbers honest.
- `MONOTONIC_FEATURES` (config) now lists 9 constraint candidates, each verified to exist in the final 58 — a verification that caught one dead reference (`bureau_days_overdue_max`, which did not survive selection) before it could silently no-op in Phase 5.
- Use `load_modeling_frame("dev")` (src/data/load.py) as the ONLY way to assemble features + folds — one loader, one behavior, asserted every time.

## Known limitations (found in review, quantified, deliberately deferred)

**1. Selection is seed-dependent at the boundary.** 14 of 133 features sit within ±10% of their null-importance threshold (8 kept, 6 dropped) — for these, a different seed could flip the call. Why this is acceptable rather than fixed: the damage is *bounded by measurement* (the 58-vs-133 equivalence test: Δ = −0.0009), and the borderline-kept `annuity_income_ratio` — which also carries a monotonic constraint — is guarded by `test_monotonic_features_all_survived_selection`, so any future re-run that drops it fails loudly instead of silently no-opping in Phase 5.

**2. All-NaN debt sums read as 0.** 8,372 applicants (2.7% of those with bureau history) have *every* `AMT_CREDIT_SUM_DEBT` value missing; the pandas sum convention makes their `bureau_debt_sum` = 0, conflating "debt-free" with "debt unknown". This is the universal convention in Home Credit solutions and the features were validated as-is, so it is **documented now and deferred to Phase 3's underfitting playbook**: if more signal is needed, a `bureau_debt_missing_share` feature is the clean fix — followed by a re-run of selection and this phase's validation gates, because changing a frozen feature set has a defined process, not a shortcut.

**3. `bureau_bb_share_dpd_max` is 70% missing** (only applicants with `bureau_balance` coverage) and carries a monotonic constraint. LightGBM handles this correctly — the constraint binds observed values while NaN takes a learned default direction — noted here so Phase 5 doesn't rediscover it.

## ⚠ Known contingency for Phase 3 (flagged now, not discovered later)

**`installments_payments.csv` is not in `data/raw`.** It is the spec's #1 remedy if the tuned model underfits (Phase 3c: *"add installments_payments — late-payment ratios are gold — before touching hyperparameters"*). Current trajectory suggests it won't be needed (fold-1 = 0.772 untuned; tuning + fold-averaging typically adds ~0.005–0.01, landing inside the 0.775–0.790 target), but if Phase 3 OOF comes in below 0.775, **download it from the Kaggle competition data page first** — the aggregation pattern in `src/features/aggregations.py` extends directly (late-payment count, max days late, share of late installments per applicant), followed by a re-run of selection and this phase's validation gates.

## Artifacts

| Path | Contents |
|---|---|
| `data/processed/features_full.parquet` | 307,511 × 182 matrix (all rows; stateless + per-applicant only) |
| `data/processed/audit_frame.parquet` | ID, TARGET, CODE_GENDER, contract type — Phase 5b only |
| `data/processed/feature_manifest.csv` | name, dtype, source, missing share → feeds Phase 6 pydantic |
| `data/processed/lgbm_features.json` | the 58, with the selection pipeline recorded |
| `data/processed/woe_features.json` | the 30, with the IV audit trail |
| `models/woe_binner_dev.pkl` | dev-fit binner (see discipline note above) |
| `reports/tables/` | near-constant, correlation, null-importance, IV, aggregation-summary tables |
| `reports/figures/21–23_*.png` | null-importance top-20, IV ranking, WOE bin examples |
