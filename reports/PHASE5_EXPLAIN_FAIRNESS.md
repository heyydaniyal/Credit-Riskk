# Phase 5 — Explainability, Monotonic Constraints, Fairness Audit

*All numbers are dev/OOF estimates. The deployed artifact from this phase forward is the **constrained** 5-fold ensemble + **its** calibrator (`isotonic_final.pkl`); the Phase 4 calibrator is retired.*

## 5a — The defensibility trade, priced and verified

**The guarantee is a tested property, not a parameter.** The design-review experiment showed LightGBM's `advanced` constraint method violating monotonicity by up to 0.166 raw log-odds — so the pipeline uses `basic`, and a **monotonicity gate** (15-point sweeps per constrained feature, raw-score space) must show zero violations on every fold model before recalibration may proceed. Result: **gate passed, zero violations across all 5 folds and all 9 constrained features.** Scope, then strengthened on challenge: the original gate swept the 2–98% range on 300 rows; the strengthened gate covers the **full observed range (0–100th pct), 1000 rows, 25-point grids, all 5 folds — still zero violations**, closing the could-hide-in-the-tails objection — the strict guarantee itself comes from `basic`'s tree-building constraint by construction; the gate exists because "the parameter was set" is not evidence, as the `advanced` experiment proved.

**The price was near-zero:** constrained OOF AUC 0.77677 vs 0.77735 unconstrained — **cost 0.00059**, five times under the spec's "typically ≤0.003." The feared over-constraining of `basic` did not materialize; the domain directions we imposed are ones the data mostly agrees with.

**Recalibration re-run in full** (the sequence is `make final-pipeline`, un-half-runnable): new OOF → refit isotonic → cross-fitted Brier **0.06671** / ECE **0.00078** → alignment re-verified (PSI 0.00019, ensemble std 1.9% larger than OOF for the same in-sample reason as Phase 4). **The centerpiece survives:** constrained-pipeline gap = **5.28%**, inside the unconstrained bootstrap CI [4.2%, 5.8%]; best-flat re-selected once at 0.080 per the pre-amended convention.

**Explainability:** exact TreeSHAP via `pred_contrib` (additivity 3.5e-14); global beeswarm from a stated 25k fold-0 sample; reason codes under the pinned adverse-to-applicant sign convention with plain-language mapping; score scale with exact anchors (PD 5% → 600; doubling good-odds → +50 points). SHAP explains the raw score; isotonic is monotone, so the ranking carries to the calibrated decision.

## 5b — Fairness audit: the pre-registration did its job, including where reality exceeded it

**Proxy check: OOF AUC 0.907 — ABOVE the pre-registered 0.75–0.85 band.** Gender is even more strongly encoded in legitimate underwriting variables than anticipated. Top carriers: `OCCUPATION_TYPE` (dominant by 3×), `OWN_CAR_AGE`, and — notably — `EXT_SOURCE_1`: the external bureau score itself carries gender signal. **Ablation evidence (added on challenge):** removing `OCCUPATION_TYPE` drops the proxy only to 0.865, and removing all top-3 carriers still leaves 0.840 — the encoding is *diffuse*, so feature deletion cannot sanitize the model as a measured fact, not a talking point. The pre-written framing stands verbatim: excluding the protected attribute does not sanitize the model; we therefore audit at the decision level rather than pretending the feature drop settled the question. No remediation is implemented, and the reason is legal, not lazy: per-group thresholds in credit are disparate treatment.

**Decisions of the deployed rule (dev, pinned definitions):**

| | F | M |
|---|---|---|
| n | 161,887 | 84,119 |
| Base default rate | 7.0% | 10.1% |
| Approval rate | 66.5% | 54.4% |
| Goods rejected (FPR) | 30.8% | 42.0% |
| Defaulters approved (FNR) | 30.8% | 21.9% |
| Mean t\*(x) | 0.0815 | 0.0777 |

- **Disparate impact ratio 0.817 — passes the four-fifths flag by 0.017.** Reported as narrow, not as clean; in a real bank a margin this thin sits inside governance review regardless of the flag.
- **The policy-vs-model distinction the instance rule creates — answered:** mean t\*(x) differs between groups by only **0.004** (revolving share 9.8% vs 9.1%), so the thresholds themselves are near-neutral. The approval gap is **score-driven**, tracking the genuine base-rate difference — model-level, not policy-level disparity. This is the question only an instance-dependent system has to ask, and it now has a measured answer.
- **The equalized-odds gaps (|ΔFPR| 0.112, |ΔFNR| 0.089) are the canonical impossibility pattern:** with different base rates, a calibrated score cannot equalize both error rates across groups (Kleinberg et al. 2016; Chouldechova 2017). Our system chose calibration — the choice every pricing-based lender makes — and the audit reports what that choice costs on the other axes rather than hiding it.
- **Limits, disclosed:** dev-only, gender-only (no ethnicity in the data — the audit is partial); framing ceiling "inspired by" ECOA/Reg B and EU practice, never "compliant with."

## Artifacts

Constrained fold models + `isotonic_final.pkl` (deployed) · `oof_constrained_calibrated.parquet` · `final_pipeline.json` (gate report, AUC cost, recalibration, gap) · `fairness_audit.json` · `reason_codes_demo.json` · `shap_mean_abs_top30.csv` · figures 51–53 · monotonicity verifier + gate in `src/models/monotonicity.py` with its own violation-catching test.

## Carried to Phases 6–7

The `/score` path uses: constrained ensemble → `isotonic_final` → t\*(x) → reason codes (top-3 positive contributions) → score scale. The holdout ceremony (plan frozen at Phase 3) evaluates **this** pipeline; its fairness metrics recompute on holdout decisions per the same pinned definitions.
