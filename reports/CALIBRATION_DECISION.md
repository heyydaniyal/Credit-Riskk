# Phase 4 — Calibration + Cost-Based Decisioning (the centerpiece)

*All numbers here are OOF development estimates under the frozen illustrative cost assumptions. The quoted results come from the one-shot holdout ceremony (Phase 7), whose analysis plan was frozen before this phase ran.*

## 4a/4b — Calibration, measured honestly

| Metric | Raw | Calibrated (cross-fit) | Benchmark |
|---|---|---|---|
| Brier | 0.06666 | **0.06670** | climatology 0.0736 (beaten by 0.0069); target 0.065–0.070 ✓ |
| ECE (10 quantile bins) | 0.00326 | **0.00069** | 5× improvement |

The honest reading: the unweighted GBM was **already nearly calibrated in level** (a designed consequence of Phase 3's no-class-weights decision — ensemble mean 0.0808 vs base rate 0.0807), so Brier barely moves; isotonic's real contribution is **shape** (ECE ÷5). Metrics are cross-fitted (fit on 4 folds' OOF, scored on the 5th, rotated) — the deployed calibrator, fit on all OOF with `out_of_bounds='clip'`, never grades itself.

**Alignment — measured, and the measurement corrected our expectation.** The calibrator trains on OOF scores; deployment averages 5 fold models. Overlay verdict: **PSI = 0.0002, KS = 0.004** — ~500× below the 0.10 monitoring band. And the naive prediction (averaging compresses dispersion) was *wrong*: the dev ensemble is 2.7% **more** dispersed, because 4 of 5 models saw each dev row in training. On truly new data all 5 are out-of-sample, so this overlay slightly overstates the mismatch. Second-order either way, now with numbers attached.

> **Currency note (added at final audit):** the dataset specifies no currency; amounts below are in its own units (CU), not euros. Ratios and percentage gaps are unaffected — the loan amount divides out of `t*(x)`.

## 4c — Instance-dependent thresholds and THE gap checkpoint

`t*(x) = m(x) / (m(x) + LGD(x)·ead(x))` with the spec-frozen constants; term fallback applied to 10 rows per the pre-frozen rule; the 3-year-cash anchor reproduces (0.1119). The empirical check passes: the best flat threshold (0.095) lands near the €-weighted mean t\* (0.086).

**The checkpoint (narrative locked here, before any holdout contact):**

| Policy | Profit €/10k apps | Approval rate |
|---|---|---|
| Naive 0.5 | 67.9M | 99.5% |
| Best flat (0.095, selected on dev, frozen) | 163.5M | 72.5% |
| **Instance t\*(x)** | **171.6M** | 62.2% |

**Instance vs best-flat: +€8.13M per 10k applications = +4.97%** — *above* the pre-registered 1–3% plausibility band. Because a better-than-expected number deserves the same suspicion as a too-good AUC, it was audited before being locked:

- **Independent recompute** (raw numpy, no shared code with the pipeline): matches to the euro.
- **Decomposition:** cash contributes €6.76M/10k (13.4% of cash applicants decided differently — the t\* band 0.03–0.11 straddles the densest region of the calibrated PD distribution) and revolving €1.37M/10k (its 0.175 threshold differs sharply from flat 0.095). Term-driven ≈ 5× revolving — exactly the ordering the spec anticipated.
- **No parameter was touched.** The gap came out where it came out; the pre-registered framing absorbed the size, not the other way around.

The vs-0.5 number (+€103.7M/10k) is reported as what it is: a naive baseline, never a substitute for the best-flat comparison.

**Post-checkpoint uncertainty audit (the gap is interval-backed, not a point):** 1000-resample bootstrap of the per-row ledger difference gives a 95% CI of **[4.21%, 5.83%]** — entirely above the pre-registered 1–3% band, P(gap>0)=1.000. Calibration is excellent precisely where the decisions happen (band ECE 0.00072 in p∈[0.02,0.20], worst bin error 0.0013), the gap is broad-based rather than whale-driven (top-1% loans: 1.0% of the gap vs 3.6% of exposure), and isotonic's 590-step lumpiness is absorbed by the bootstrap. One asymmetry is pre-registered for Phase 7: best-flat carries one dev-fitted parameter while the instance policy carries zero, so a holdout gap slightly *above* the OOF point is the expected direction, not a surprise.

## 4d — Business framing

**Profit vs approval rate** (global scaling of the per-applicant thresholds) shows the committee view: the deployed rule sits at 62% approval. **Swap-set vs 0.5:** the 91,660 applicants the instance policy rejects that naive approves default at **15.1%** — nearly 2× the book's base rate — accounting for €2.55B of avoided net losses under the baseline ledger; the 1,212 both-rejected applicants default at 55%. The decision layer is finding real risk, not reshuffling noise.

## Artifacts

`models/isotonic_dev.pkl` (deployed calibrator) · `oof_calibrated.parquet` (measurement-grade p_cal) · `dev_ensemble_scores.parquet` · tables: calibration metrics, alignment report, gap checkpoint (LOCKED), flat sweep, policy comparison, swap-set, profit-vs-approval · figures 41–45 · 9 new unit tests (anchor math, amortization-only-for-cash, ledger by hand, clip behavior, cross-fit improvement, PSI identity/shift).

## Standing discipline for Phase 5

The constrained retrain **invalidates this calibrator**: new OOF → refit isotonic → re-verify alignment → recompute the cost curve against t\*(x). That sequence becomes `make final-pipeline` so it cannot be half-run. The deployed artifact is the *constrained* ensemble + *its* calibrator — never this one with that model.
