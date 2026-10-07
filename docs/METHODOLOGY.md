# Methodology

Derivations, diagnostics, and the evaluation discipline behind every number in this project. Companion docs: [`TECHNICAL.md`](TECHNICAL.md) (architecture, API, how to run) and the executive [README](../README.md).

---

## 1. The decision rule

`t*(x) = m(x) / (m(x) + LGD(x)·EAD(x))`

### Derivation (Elkan, IJCAI 2001, instance-dependent form)

For applicant *x* with calibrated default probability *p*, loan amount *A*:

| | applicant repays (1−p) | applicant defaults (p) |
|---|---|---|
| **Approve** | `+ m(x)·A` (net margin earned) | `− LGD(x)·EAD(x)·A` (loss given default) |
| **Reject** | `0` (margin foregone) | `0` (loss avoided) |

Approve when expected value of approving exceeds rejecting:

```
(1−p)·m(x)·A  −  p·LGD(x)·ead(x)·A   >   0
```

`A > 0` divides out of the **threshold**:

```
p  <  m(x) / ( m(x) + LGD(x)·ead(x) )  ≡  t*(x)
```

**Sanity anchor:** 3-year cash loan → `m = 0.05 × 3/2 = 0.075`, `L = 0.70 × 0.85 = 0.595`, `t* = 0.075/0.670 = 0.1119`. Reproduced by unit test.


### The default event, and a definitional caveat that dominates the cost model

**What `TARGET = 1` means.** The Home Credit label is an **early-delinquency** event: the client had *payment difficulties — late beyond a threshold on the first installments of the loan*. This is a front-end vintage / early-delinquency label, **not** a Basel Article 178 charge-off (90 days past due or unlikeliness-to-pay). Everything downstream — the PD, the reason codes, the fairness metrics, the euro figures — is predicting *early delinquency*, and the documentation now states this rather than treating `TARGET` as a primitive.

**The consequence for the cost model (the single largest uncertainty in the decision engine).** The decision rule multiplies this early-delinquency PD by `LGD = 0.70` and `EAD = 0.85` — which are **charge-off** severities. The two refer to different events, and Elkan's framework requires them to refer to the same one. Early delinquencies cure at high rates in consumer lending; every cured borrower is charged the full `LGD·EAD·A` in the as-built ledger and denied the margin they in fact paid, so the false-negative cost is systematically overstated and thresholds systematically too strict.

`CURE_RATE` is now an implemented axis of the cost model (`attach_cost_params(..., overrides={"cure_rate": c})`, also a bounded `/score` override and a demo slider): effective loss = `(1−c)·LGD·EAD`, cured borrowers booked at zero (conservative — they earn no margin), so `t* = m / (m + (1−c)·LGD·EAD)`. Its effect on the threshold **dominates every other cost parameter**:

| Assumption | effective L = LGD·EAD·(1−cure) | t*(3y cash) | shift |
|---|---|---|---|
| As-built (implicit 0% cure) | 0.595 | 0.1119 | — |
| 30% cure | 0.416 | 0.1526 | +36% |
| 50% cure | 0.297 | 0.2013 | **+80%** |

For comparison, moving LGD across its entire frozen grid (±0.10) shifts the same anchor by only **±13%**. The dominant uncertainty in the centerpiece is therefore not a parameter *value* — it is the *event definition*, and it was outside the original sensitivity analysis. The deployed path keeps `CURE_RATE = 0` (preserving every published number), but the honest reading is that a bank would estimate this cure rate from internal data and it would move thresholds more than any LGD/EAD/margin choice in the grid.

**What it does to the gap (development data, added after the holdout):** sweeping the cure rate on dev OOF with cross-fitted flat competitors, the instance-vs-best-flat gain falls from **5.64%** (c = 0) to **2.90%** (c = 0.3) and **1.60%** (c = 0.5) — the largest compression of any axis (§7, *Robustness*). The gain stays positive, but its size depends more on what `TARGET` measures than on any frozen constant. Because this axis was added after the ceremony, it is development evidence only; no holdout number is revised.

### The two traps, both closed

**Trap 1 — the degenerate flat threshold.** If LGD, EAD and margin are *constants*, `t*` is the same number for every applicant: instance-dependent accounting, flat decision. The variation here is real and comes from two sources: **term** (`m(x) = r_net · term_years(x) / 2` for cash) and **contract type** (revolving carries its own LGD/EAD/margin). Measured on holdout: `t*` spans **0.027 → 0.175**.

**Trap 2 — the amortization error.** Pricing a rejected good customer as `rate × term × full principal` assumes the entire principal stays outstanding for the entire term. Amortizing loans repay monthly, so average outstanding balance ≈ `A/2`; the naive formula roughly **doubles** false-positive costs and biases every threshold toward over-approval. Hence the `/2` in `m(x)`. Symmetrically, exposure *at default* is not the original amount: `ead_factor` = 0.85 cash (amortization has repaid part, but consumer defaults cluster early, so well above 0.5), 1.00 revolving (drawdown before default; Basel-style retail CCF ≈ 0.75–1.0, conservative end).

### Currency: unspecified, and therefore reported as units

The Home Credit dataset documents **no currency** for `AMT_*` fields, and the magnitudes rule out euros (median borrower income 147,150; median loan 513,531 — implausible for a mass-market consumer lender in EUR). All monetary results are therefore reported in the dataset's own **currency units (CU)**. This does not affect any conclusion: the decision rule is scale-invariant (the amount divides out of `t*`), and the headline comparison — **+3.67% over the best flat threshold** — is a ratio, hence currency-free. Only the absolute figures carry the unit.

### The constants are illustrative assumptions, and that is stated everywhere

LGD, EAD, net margin, and revolving life/utilization **cannot be estimated from this dataset** — there is no recovery data, no balance-at-default, no internal pricing, no multi-year account behaviour. In a bank each is a separate model on internal data (the Basel IRB triplet PD × LGD × EAD, plus a pricing engine). These constants are directionally plausible placeholders occupying exactly the slots where those models plug in — and precisely because they are *chosen* rather than estimated, they were **frozen in the specification before any result existed** and never revised. `t*(x) = m/(m+L)` is the interface where richer LGD/EAD models attach.


### What the headline gap does and does not measure (a candid note)

The evaluation ledger uses the *same* `m(x)`, `LGD(x)`, `EAD(x)` that define `t*(x)`. Under that ledger, "approve iff p < t*(x)" is the Bayes-optimal rule, so it must weakly dominate any flat threshold **in expectation** — the *sign* of the gap is guaranteed by construction, not discovered from the data. We verified this directly: replacing realised outcomes `y` with the model's own calibrated probabilities `p` (reading no outcome data at all) reproduces the gap at **+4.1%**, versus the realised **+3.67%** — a difference within sampling noise.

So the gap is not a pure empirical finding. What *is* empirical, and what the number legitimately establishes, is twofold: (1) that calibration is good enough for the theoretical advantage to materialise against real outcomes rather than collapse — with deliberately scrambled probabilities the realised gap falls from 5.3% to 2.1%, so calibration quality is doing real work; and (2) the *magnitude* of the threshold heterogeneity the assumed economics imply, ~4%. The pre-registration, bootstrap CI, sensitivity grid and one-shot ceremony remain worth doing — they establish that the calibration holds out of sample and that the magnitude is stable — but they are rigour applied to a quantity that is partly analytic. Reporting the gap without this note would overstate how much of it is a finding versus an implication of the cost model. The honest one-line summary: *the system correctly applies Bayes-optimal instance-dependent thresholds, and calibration is good enough (verified) for that to be worth ~4% against real outcomes under the illustrative economics.*

---

## 2. Validation protocol and evaluation discipline

**Split.** 20% stratified holdout (`random_state=42`), sealed at Phase 0, touched exactly once at Phase 7. Development used the remaining 80% with stratified 5-fold CV; the same folds were reused for every model so comparisons are paired, not noisy.

**OOF everywhere.** Calibration, threshold selection, and model comparison all use out-of-fold predictions. The deployed artifact is the **mean of the 5 fold models**, not a model retrained on all data — so the calibrator's training distribution closely matches the deployed one (verified, §3).

**Leakage controls.** Auxiliary tables aggregated per `SK_ID_CURR` only; WOE bins refit *inside* each fold (fit on 4, transform the 5th — a test asserts the five folds produce five different WOE tables); imputation/binning statistics from training folds only. **Pipeline-integrity check:** a classifier trained to separate dev from holdout rows on the final feature set scored **AUC 0.502** — as expected by construction for a random split, which is exactly the point: it detects processing bugs (statistics fit across the split boundary), not covariate shift.

**Pre-registration.** Cost parameters, the cost-accounting ledger, the best-flat selection rule, the fairness metric definitions, the Phase 5 constraint design, the eight ceremony outputs, and the expected holdout consistency bands were each frozen in `config/*.json` **before** the results they govern existed. The ceremony script enforces this in code: a label-free pre-flight gate, a run-once guard with an append-only run log, and an automatic check of results against the pre-registered bands.

**Feature selection caveat, stated not hidden.** Selection (correlation drop, null-importance) was fit on the full dev set rather than inside each fold, making OOF estimates very slightly optimistic. The untouched holdout is what keeps the final numbers honest — and it agreed (§6).

---

## 3. Calibration

**Why first.** The cost analysis multiplies probabilities by euro amounts. If a predicted 0.20 really means 0.12, every threshold decision is wrong.

**Method.** Isotonic regression fit on all OOF predictions (~19.9k positives — far above the ~1k where isotonic's flexibility stops overfitting; Niculescu-Mizil & Caruana, ICML 2005), with `out_of_bounds='clip'` so scores beyond the training range clip rather than becoming NaN.

**Cross-fitted measurement.** Isotonic fit on OOF and scored on the same OOF is in-sample. Every development calibration metric therefore comes from a rotation: fit on four folds' OOF, score the fifth, concatenate. The deployed calibrator is unchanged by this; it simply is not what those numbers grade.

**Alignment, quantified rather than asserted.** The deployed scorer averages 5 fold models while each OOF score came from one. Measured: **PSI 0.0002, KS 0.004** between the two distributions. The naive expectation (averaging compresses dispersion) was *wrong* on dev — the ensemble is ~2% more dispersed, because 4 of 5 models saw each dev row in training; on genuinely new data all five are out-of-sample, so the dev overlay slightly overstates the mismatch. Second-order either way.

**Results.** Development (cross-fitted): Brier 0.0667, ECE 0.0008. Holdout: **Brier 0.0662, ECE 0.0021** vs a climatology baseline of 0.0736. Isotonic's contribution here is *shape*, not level — the unweighted GBM was already nearly calibrated in level (mean prediction 0.0808 vs base rate 0.0807), which was the reason for refusing class weights on the GBM in the first place.

---

## 4. Modeling and the diagnostics playbook

**Scorecard baseline.** `sklearn.LogisticRegression` on 30 WOE-binned features, `class_weight='balanced'`, `C` tuned over {0.01…100}. The C grid came out flat (all within 0.00003 AUC): with n ≫ p and WOE features already on a common log-odds scale, the likelihood swamps any reasonable L2 prior — WOE binning already did the variance control. Class weighting deliberately distorts probabilities, which is one more reason calibration is a separate stage.

**LightGBM.** Optuna TPE over a frozen search space, objective = mean fold AUC on the frozen folds, 10,000-round ceiling with 200-round early stopping (chosen sizes: 1,140–1,753 trees). No class weights, no SMOTE: AUC is invariant to class weights and unweighted models produce less-distorted probabilities.

| Parameter | Deployed value |
|---|---|
| `learning_rate` | 0.0138 |
| `num_leaves` | 49 |
| `min_data_in_leaf` | 215 |
| `feature_fraction` | 0.682 |
| `bagging_fraction` | 0.875 |
| `lambda_l1` / `lambda_l2` | 6.3e-7 / 4.2e-4 |
| `monotone_constraints_method` | `basic` (9 constrained features) |

**Overfitting/underfitting playbook.** Diagnostic = mean train AUC − mean OOF AUC; alarm at 0.025. **The alarm fired** (gap 0.117) and was audited rather than obeyed: every completed trial in the frozen search space — including the most heavily regularized — sits at gap 0.10–0.14, so the alarm level is unreachable there; and gap carries no relationship to OOF AUC across trials. The gap is benign boosted-tree memorization. The live check was playbook step 5: *if the holdout disagrees badly with OOF, the CV leaked.* It did not (§6).

**Monotonic constraints.** LightGBM's `advanced` method was tested first and **violated monotonicity by up to 0.166 raw log-odds** on this feature structure — so the pipeline uses `basic`, and a gate verifies the property directly (full observed range, 1,000 rows, 25-point sweeps, all 5 folds): **zero violations**. Cost of the guarantee: **0.00059 AUC** — five times under the spec's typical ceiling.

---

## 5. Discrimination — the standard credit-model reporting set

Reported here because AUC alone is not the reporting unit a credit-model validation pack uses. Computed on the holdout (n = 61,503):

| Metric | Deployed GBM | Note |
|---|---|---|
| AUC | 0.7802 | |
| **Gini** | **0.5605** | 2·AUC−1; the standard EU reporting unit |
| **KS** | **0.4198** | max separation of good/bad score CDFs; the most-requested consumer-credit statistic |
| **PR-AUC** | 0.2773 | baseline 0.0807 at the 8.1% base rate |

**Risk-band table (deciles of the deployed score, worst = highest risk):**

| Decile | n | mean PD | observed | lift | cum. bad capture |
|---|---|---|---|---|---|
| 0 | 6,151 | 0.0094 | 0.0104 | 0.13× | 100.0% |
| 1 | 6,150 | 0.0167 | 0.0154 | 0.19× | 98.7% |
| 2 | 6,150 | 0.0235 | 0.0234 | 0.29× | 96.8% |
| 3 | 6,150 | 0.0314 | 0.0312 | 0.39× | 93.9% |
| 4 | 6,151 | 0.0410 | 0.0426 | 0.53× | 90.0% |
| 5 | 6,150 | 0.0536 | 0.0579 | 0.72× | 84.8% |
| 6 | 6,150 | 0.0708 | 0.0756 | 0.94× | 77.6% |
| 7 | 6,150 | 0.0974 | 0.0982 | 1.22× | 68.2% |
| 8 | 6,150 | 0.1461 | 0.1535 | 1.90× | 56.1% |
| 9 | 6,151 | 0.3071 | 0.2990 | 3.70× | 37.0% |

Rank ordering is **monotone across all ten deciles** (true), predicted tracks observed in every band, and the worst decile carries **3.7× the base rate** and captures 37% of all defaults. These are solid values for this dataset (Lessmann et al. band).

## 6. Fairness audit

Framed as **inspired by** adverse-action and governance practice (ECOA/Reg B, GDPR Art. 22, SR 11-7 style review) — never as compliance. A public-dataset project with no legal review supports nothing stronger.

**Protected attribute excluded at build time.** `CODE_GENDER` is removed from the modeling matrix in Phase 2 and kept only in a separate audit frame, so no downstream code can train on it.

**Proxy check (pre-registered to fire at 0.75–0.85).** Measured: **0.907** — above the pre-registered band, disclosed as such. Top carriers: `OCCUPATION_TYPE`, `OWN_CAR_AGE`, and notably `EXT_SOURCE_1` (the external bureau score itself carries gender signal). **Ablations:** removing `OCCUPATION_TYPE` drops the proxy only to 0.865; removing the top-3 carriers still leaves 0.840. The encoding is *diffuse* — "excluding the protected attribute does not sanitize the model" is a measured fact here, not a talking point.

**Metrics at deployed decisions (holdout, definitions pinned in advance):**

| | F | M |
|---|---|---|
| n | 40,561 | 20,940 |
| Approval rate | 66.5% | 54.3% |
| Goods rejected (FPR) | 30.9% | 42.0% |
| Defaulters approved (FNR) | 31.8% | 21.4% |
| Mean `t*(x)` | 0.0811 | 0.0778 |

- **Disparate impact ratio 0.817** — passes the four-fifths flag by 0.017. Reported as *narrow*, not clean; a margin this thin sits inside governance review in a real institution regardless of the flag. Dev and holdout agreed to three decimals (0.817 both), so this is a stable property, not sampling noise.
- **Policy-level vs model-level disparity — the question an instance-dependent rule creates.** Mean `t*(x)` differs between groups by only 0.003, and revolving share is nearly identical, so the *thresholds* are near-neutral: the approval gap is **score-driven**, tracking genuine base-rate differences (7.0% vs 10.1% on dev), not threshold-driven. Naming that distinction is the point of measuring it.
- **Equalized-odds gaps (|ΔFPR| 0.111, |ΔFNR| 0.104) reflect the calibration/equalized-odds tension — with an important correction.** With different base rates, a score calibrated *within each group* cannot also equalize both error rates (Kleinberg, Mullainathan & Raghavan 2016; Chouldechova 2017). **Corrected under audit:** this system is calibrated in *aggregate* but **not within groups** — on holdout, female applicants are over-predicted by +5.7% relative (observed 6.99% vs predicted 7.39%) and male applicants under-predicted by −7.5% (observed 10.17% vs predicted 9.40%). An earlier version of this section claimed the system "chose calibration"; that was wrong — it has neither within-group calibration nor equalized odds, only aggregate calibration. The group miscalibration is material and runs in opposite directions; group-wise recalibration would close it but trade against the equalized-odds gaps. Measured and disclosed here; a production model would remediate it.
- **Remediation named, not implemented, with the legal reason.** Reweighing, adversarial debiasing, and per-group thresholds all exist in the literature; per-group thresholds in credit are themselves **disparate treatment** (using the protected attribute in the decision), which is why real lenders cannot reach for the textbook fix. That tension is the finding.
- **Limits.** Gender is the only protected attribute observable in this data; there is no ethnicity field. The audit is therefore partial, and saying so is part of the result.

---

## 7. The holdout ceremony — results and consistency

Run once, by script, on the 61,503-row holdout, with a label-free pre-flight gate, a run-once guard, and automatic checking against bands pre-registered from dev-only bootstraps.

| Output | Holdout | Development | Pre-registered band |
|---|---|---|---|
| AUC (deployed constrained ensemble) | **0.7802** | 0.7760 OOF | [0.7689, 0.7822] ✓ |
| Scorecard AUC | 0.7488 | 0.7447 OOF | — |
| Brier / ECE | **0.0662 / 0.0021** | 0.0667 / 0.0008 | beats climatology 0.0736 |
| Gap: instance vs best-flat | **+3.67%** (6.04M CU/10k) | 5.28% | [3.42%, 7.15%] ✓ |
| Savings vs naive 0.5 | **104.5M CU/10k** | — | *illustrative assumptions* |
| Disparate impact ratio | 0.817 | 0.817 | flagged if < 0.8 |

**Verdict: consistent with development.** AUC landed slightly *above* OOF — playbook step 5 finds no evidence of CV leakage.

**One pre-registered expectation was not confirmed, and that is reported rather than quietly dropped.** The asymmetry argument (best-flat carries one dev-fitted parameter, the instance rule carries zero) predicted the holdout gap would land slightly *above* the dev point. It landed *below* — 3.67% vs 5.28%, inside the magnitude band but contrary to the predicted direction. The honest reading: sampling variation in the gap (band width 3.7 points) swamps the small selection-bias effect that argument identified, so the directional prediction was under-powered. Magnitude confirmed, direction not.

**Robustness — as run, and as corrected.** The ceremony's sensitivity table reports a positive gap at every row it computed. Three defects in that table were found in the October 2026 review and are disclosed rather than re-run (the holdout is spent): (1) it sweeps **4 of the 6** frozen grids (LGD_revolving and ead_revolving were omitted); (2) its central point appears four times, so "12 grid points" are **9 distinct** ones; (3) at every row it re-selects the best flat threshold **on the holdout** (0.085 at the frozen point), whereas the headline uses the dev-frozen 0.08 — which is why its central row (5.92M CU/10k) does not reconcile with the headline (6.04M). Point (3) makes every row *conservative* for the instance rule (an oracle flat threshold is the stronger competitor), so the sign of the conclusion is unaffected.

The complete sweep — all six grids **plus the cure-rate axis**, with every flat competitor **cross-fitted** (threshold chosen on four folds, scored on the fifth) — is run on development data by `make sensitivity-dev` (`reports/tables/sensitivity_dev_full.*`, figure 46). The gain is positive at **all 14 distinct points, 1.60%–10.94%**; it is most sensitive to the cash net margin (2.1% at r = 8%, 10.9% at r = 3%) and to the cure rate (1.6% at c = 0.5). These are development numbers, labelled as such.

**How much of the gain is segmentation?** (Same script, development data.) A fitted flat-threshold policy with one cross-fitted threshold per segment — revolving, plus cash loans in five term quintiles — gains **+5.15%** over a single flat threshold; per contract type alone, **+2.17%**; the closed-form `t*(x)`, **+5.64%** with nothing fitted. About **91%** of the instance-dependent gain is therefore term-and-contract segmentation, and the cost formula captures it without fitting a parameter — and still edges out the fitted 6-threshold competitor.

**Swap-set vs naive 0.5.** The 22,916 applicants the instance policy rejects that a 0.5 threshold would approve default at **15.0%** — roughly double the book's 8.1% base rate. The 261 rejected by both default at 62.8%. The decision layer is separating real risk, not reshuffling noise.

---

## 8. Reject inference (a stated limitation, not a solved problem)

Every model here is trained on **accepted applicants only**. Home Credit's data contains applications that were approved and subsequently observed; the population the model would actually score in production is the full **through-the-door** population, including applicants a prior policy rejected — whose outcomes are unobserved by construction.

**Consequence.** Every metric in this document — AUC, Brier, the euro figures — is an estimate on the *accepted* population. Deployed against through-the-door traffic, the score distribution shifts toward the rejected region where the model has the least evidence, and performance would be expected to degrade to an unknown degree. Nothing in this dataset can bound that degradation.

**Classic remedies, and why none is implemented:** *parcelling* (assign inferred outcomes to rejects by score band), *fuzzy augmentation* (duplicate each reject with fractional good/bad weights), *rejects-as-unlabeled* (semi-supervised / Heckman-style selection correction), and the honest-but-expensive one — deliberately approving a random slice of marginal applicants to buy unbiased data. **All require reject data this dataset does not contain**, which is precisely the point: the limitation is structural, not an omission. Crook & Banasik (2004) find the gains from inference methods are modest and depend heavily on the rejection rate, which is itself an argument against claiming a fix.

---

## 9. What a real bank does differently

- **PD is one model of three.** Basel IRB decomposes expected loss as PD × LGD × EAD; LGD and EAD are separately modeled on internal recovery and exposure data. Here they are constants, sitting exactly where those models would attach.
- **Margins come from a pricing engine,** not a single net-rate assumption — risk-based pricing means the margin is itself a function of the score, making the threshold problem simultaneous rather than sequential.
- **Model risk management.** New models face independent validation and governance sign-off (SR 11-7 in the US; ECB/EBA guidance in the EU) covering conceptual soundness, data quality, and ongoing monitoring — a process this project imitates in structure (pre-registration, frozen artifacts, audit trails) but cannot substitute for.
- **Champion–challenger deployment.** A new model ships alongside the incumbent on a traffic slice and must prove itself on realized outcomes before promotion, rather than switching on a single validation result.
- **Out-of-time validation.** Production validation is temporal — train on the past, validate on a later window — because applicant populations drift. **This dataset has no application dates, so out-of-time validation is impossible here.** That is the single largest gap between these numbers and what a bank would accept as evidence, and it is why the monitoring framework leans on PSI rather than assuming stationarity.
- **Reject inference and policy loops** are standing concerns, revisited every model cycle, with data purchased or generated deliberately.

---

## References

- Elkan (2001), *The Foundations of Cost-Sensitive Learning*, IJCAI — threshold formula, instance-dependent generalization.
- Niculescu-Mizil & Caruana (2005), *Predicting Good Probabilities with Supervised Learning*, ICML — isotonic vs Platt.
- Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*, NeurIPS — SHAP.
- Siddiqi (2006), *Credit Risk Scorecards* — WOE/IV, score scaling, PSI bands.
- Lessmann et al. (2015), *Benchmarking state-of-the-art classification algorithms for credit scoring*, EJOR.
- Hardt, Price & Srebro (2016), *Equality of Opportunity in Supervised Learning*, NeurIPS.
- Kleinberg, Mullainathan & Raghavan (2016); Chouldechova (2017) — calibration vs equalized-odds impossibility.
- Crook & Banasik (2004), *Does reject inference really improve application scoring?*, J. Banking & Finance.
- Basel Committee on Banking Supervision, IRB approach (PD/LGD/EAD; CCF for revolving).
