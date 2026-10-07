# Improvement Analysis — measured, not proposed

Every candidate below was **tested on development data**, and the verdict is the measurement. Nothing here was adopted into the shipped system; the reason why is in §8, and it is the most important part of this document.

Artifacts: `reports/tables/improvement_experiments.json`, `feature_improvement_replication.json`, `latency_benchmark.json`.

---

## 1. Blend the WOE scorecard into the GBM (Phase 3) — **REJECTED**

The spec forbids stacking; that call had never been *measured*. Rank-averaged the deployed model's OOF with the scorecard's across weights 0.05–0.50:

| Blend weight (scorecard) | OOF AUC | vs deployed |
|---|---|---|
| 0.00 (deployed) | 0.77677 | — |
| 0.20 | 0.77522 | −0.00155 |
| 0.30 | 0.77348 | −0.00329 |

**Optimal weight is 0.00.** Every non-zero blend hurts — the scorecard carries no signal the GBM lacks, only noise. The spec's no-stacking decision is now empirically validated rather than merely obeyed.

## 2. Alternative calibrators (Phase 4) — **REJECTED**, judged where decisions happen

Cross-fitted, evaluated on ECE **inside the decision band** p ∈ [0.02, 0.20] rather than globally, because that band is where every `t*(x)` sits:

| Calibrator | Brier | Decision-band ECE |
|---|---|---|
| Isotonic (deployed) | 0.06671 | 0.00099 |
| Platt | 0.06671 | 0.00310 |
| Per-segment isotonic (cash / revolving) | 0.06671 | **0.00083** |

Platt is 3× worse where it matters. Per-segment isotonic *looks* better — so it got the only test that counts: **does it change decisions?** It flips 1,329 of 246,008 decisions (0.54%) and moves realized profit by **−0.014%**. The ECE gain is cosmetic; the added complexity (two calibrators, two artifacts, two drift surfaces) buys nothing. Rejected on evidence, not taste.

## 3. `bureau_debt_missing_share` — the deferred Phase 2 item — **INCONCLUSIVE, and that is the finding**

Phase 2 documented that applicants whose bureau debt values are *all* missing get `bureau_debt_sum = 0`, conflating "debt-free" with "debt unknown" (8,372 applicants, 2.7%). The fix is a missing-share feature. Controlled A/B on fold 0 (same seed, params, constraints):

**fold 0: +0.00237 AUC** — above the ~0.001 fold-noise level. A convincing single-fold win.

Then it was replicated:

| Fold | Baseline | +feature | Δ |
|---|---|---|---|
| 0 | 0.77304 | 0.77541 | **+0.00237** |
| 1 | 0.77353 | 0.77325 | **−0.00028** |
| 2 | 0.77591 | 0.77783 | **+0.00192** |
| **mean** | | | **+0.00134**, *not all positive* |

**Verdict: a candidate, not a proven win.** One fold is negative and the mean sits at roughly one fold-standard-deviation — indistinguishable from noise at this sample size. Reporting the single-fold +0.0024 as an improvement would have been over-claiming by exactly the margin this project spends its time avoiding. A full 5-fold OOF evaluation is required before it earns adoption.

## 4. Extend monotonic constraints 9 → 14 (Phase 5) — **ADOPT in v2 (free)**

Added five constraints with defensible domain direction (`bureau_max_overdue_amt` ↑risk, `prev_approval_rate` ↓risk, `ext_source_n_missing` ↑risk, `employment_years` ↓risk, `age_years` ↓risk):

**AUC cost: −0.00000** (0.77304 both arms, identical to five decimals).

More regulatory defensibility for literally zero measured accuracy cost — the strongest result in this analysis. The reason it is not already shipped is §8, not doubt about the finding.

## 5. Serving latency (Phase 6) — **REJECTED as unnecessary, plus a measurement lesson**

Decomposition on an idle machine: full path **p50 65.2 ms / p95 68.0 ms**; without reason codes p95 46.5 ms; five boosters alone 54 ms; TreeSHAP ~36% of cost. Truncating TreeSHAP to 50% of trees saves 20 ms but **changes the top-3 reason codes** — trading explanation fidelity for latency the 100 ms budget does not require is a bad trade.

**Measurement lesson worth more than the optimization:** an earlier reading of this same code returned **142.9 ms p95** — a 2.1× apparent regression. Cause: a LightGBM training job was saturating the single CPU during the benchmark. A contended reading is a load artifact, not a regression, and benchmarks must run on an idle machine. Had that number been published, it would have looked like a real performance bug.

## 6. Untestable here, honestly flagged

`installments_payments.csv`, `POS_CASH_balance.csv`, and `credit_card_balance.csv` are the remaining Home Credit tables and the spec's first underfitting remedy (late-payment ratios). **They are not in this environment and cannot be downloaded**, so no claim is made about their value — only that the aggregation pattern in `src/features/aggregations.py` extends to them directly, and that any adoption requires re-running selection and the full validation chain.

More Optuna trials: the study completed 8 of a 150 budget, and the recorded history shows the best value flat across the final candidates (two independent configurations converged to identical OOF AUC). More trials remain plausible but were not run; the honest statement is that the search is *locally* converged, not that further search is worthless.

## 7. Summary

| Candidate | Measured effect | Verdict |
|---|---|---|
| Scorecard blend | −0.0016 AUC at best non-zero weight | reject |
| Platt calibration | 3× worse decision-band ECE | reject |
| Per-segment calibration | −0.014% profit, 0.5% decisions flipped | reject |
| `bureau_debt_missing_share` | +0.0013 mean, one fold negative | inconclusive — needs 5-fold OOF |
| Extended constraints (14) | −0.00000 AUC | **adopt in v2** |
| Latency optimization | 20 ms saved, reason codes change | reject |

Four of six candidates were rejected **by their own measurements**. That ratio is the point: without the experiments, at least two (per-segment calibration, the single-fold feature win) would have looked like clear improvements.

---

## 8. Why none of this was shipped: the holdout is spent

The 61,503-row holdout has been consumed — once, by the ceremony, exactly as designed. **Any model change adopted now invalidates it as an unbiased estimate**, because the decision to adopt would be informed by knowing how the current model performed on it. Re-running the ceremony on a modified model would produce a number that looks like a result and is not one; the run-once guard exists precisely to make that inconvenient.

The correct protocol for a v2 is therefore:

1. Adopt changes on the basis of **development evidence only** (as above).
2. Obtain an **untouched evaluation set** — a fresh split from re-partitioned data, or nested CV with the outer loop untouched by selection.
3. Re-run the full validation chain — Phase 0 splits, recalibration, monotonicity gate, fairness audit — before quoting any new figure.

A project that quietly re-uses a spent holdout to book a +0.002 AUC improvement has traded a credible result for a slightly larger uncredible one. The measurements above are the honest deliverable: they say what a v2 should try, and they say it without spending something that has already been spent.
