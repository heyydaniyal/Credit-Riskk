# Phase 1 — EDA Summary

*Analysis on the DEV set (246,008 rows) only. The 20% holdout was never inspected.*

## Headline findings

1. **Severe imbalance (8.07% default).** A reject-nothing model scores 92% accuracy — accuracy is useless. PR-AUC baseline = 0.08. We optimize AUC and calibrated probability, then decide with a cost model.

2. **External scores dominate.** `EXT_SOURCE_2` shows a clean monotonic signal: bottom decile defaults at 18.3%, top decile at 3.0% — a 6× spread. Expect `EXT_SOURCE_1/2/3` to lead every importance ranking. Flagged upfront so a high AUC isn't mistaken for leakage.

3. **Missingness is signal.** `EXT_SOURCE_1` is missing for 56.3% of applicants. Being unscored by a bureau is itself predictive (missing → 8.5% default vs 7.5% present). We create explicit `_is_missing` flags rather than silently imputing.

4. **Pensioner sentinel.** `DAYS_EMPLOYED == 365243` affects 17.9% of rows (pensioners with no employment record). Converted to NaN + flag in feature engineering.

5. **Segment structure for the cost model.** Cash loans default at 8.3% (90.5% of the book); revolving at 5.5% (9.5%). Because revolving is a small share, most instance-dependent threshold variation will be **term-driven**, not contract-driven — known before interpreting the Phase 4 gap.

6. **Gender base rates differ (fairness pre-registration).** Female applicants default at 7.0%, male at 10.1%. Because base rates differ, a disparate-impact flag at the deployed decision rule is a *live possibility* — pre-registered in Phase 5b as something to report, not remediate.


7. **Term structure (cost-model input).** Implied terms cluster in 1–3 years; almost no loans beyond 5 years. Default rate is **non-monotonic** in term: peak 10.3% at 1–1.5y, falling to 4.6% at 3–5y (longer loans go to better-vetted borrowers). Term-bucket table in `reports/tables/segment_scan_term.csv`.

8. **Leverage deciles are surprisingly flat** (6.8%–9.2% default across credit/income deciles) — raw leverage is weaker than expected; interaction with income level likely needed. The ratio still earns its place as a credit-officer feature.

9. **Data-quality audit (scripted).** Income outlier 248× p99 (max 117M!), 10 near-constant columns to drop, 2 XNA genders, 10 missing AMT_ANNUITY rows (cost-model input — needs explicit rule). Full table: `reports/tables/data_quality_audit.csv`.

10. **Categorical risk is strong and sensible.** Lower-secondary education ≈ 2× default of higher education; 'Working' riskier than pensioners/state servants. These same variables are why the Phase 5b gender-proxy check will fire.

11. **Bureau table: no history = riskiest.** Applicants with zero bureau records default at 10.1% — thin-file risk. `has_bureau_history` flag mandatory; quality aggregates > count aggregates.

12. **Previous applications: refusal share is a 2.2× monotone signal** (7.1% → 15.9%). Strongest non-EXT_SOURCE signal found. Note: *no prior HC history* is the SAFEST group (6.0%) — the two aux tables' "no history" groups point in opposite directions; keep separate flags.

## Implications carried forward
- Imbalance → no accuracy metrics; calibration + cost decisioning is the point.
- EXT_SOURCE dominance → audit any AUC > 0.81 for leakage.
- Missingness flags → explicit features in Phase 2.
- Segment scan → revolving moves fewer applicants; expect a modest cost gap.
- Gender base rates → fairness audit measures and discloses, does not remediate.

## Figures
See `reports/figures/` — 15 plots (spec cap), each captioned with its takeaway.
