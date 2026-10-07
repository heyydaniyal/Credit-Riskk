# Credit Risk Scoring System

**A calibrated, explainable credit-risk service with cost-based decisioning — a working system, not a notebook.**

▶ **Live demo:** _add your Hugging Face Space URL here_ ([deploy steps](docs/TECHNICAL.md#deploying-to-hugging-face-spaces)) · `make demo` runs it locally
📄 [Technical reference](docs/TECHNICAL.md) · [Methodology](docs/METHODOLOGY.md) · [Audit trail](reports/AUDIT_RESPONSE.md)

---

## The problem

A lender approving loans with a single 0.5 probability cutoff treats every mistake as equal. It is not: rejecting a good short loan and approving a long loan that defaults are different amounts of money. This system predicts default probability, **calibrates it so the number means what it says**, then gives every applicant their own approval threshold derived from that applicant's loan economics.

## The headline

> Against a naive 0.5 threshold, cost-optimal decisioning saves **104.5M currency units per 10,000 applications** — and against the *best possible single threshold*, still **+3.67% (6.0M CU per 10,000)**.
> *Measured once on a 61,503-applicant holdout, under stated illustrative economic assumptions. The Home Credit dataset specifies **no currency** (median borrower income 147,150, median loan 513,531 — not plausibly euros), so monetary figures are in the dataset's own units; the percentage gap is the currency-free result.*

![profit vs approval rate](reports/figures/71_holdout_profit_vs_approval.png)

The honest comparison is the second number. The gap is partly an implication of the assumed economics rather than a pure empirical finding — see [what it does and does not measure](docs/METHODOLOGY.md#1-the-decision-rule). On development data, a *fitted* policy with one threshold per contract type × term bucket recovers ~91% of the gain: **most of the value is term-and-contract segmentation, which the closed-form rule gets without fitting anything** — and it still beats that fitted competitor.

**Why the decisions are better, in one line:** the 22,916 applicants this system rejects that a 0.5 threshold would approve default at **15.0%** — roughly double the book's 8.1% base rate.

## Results (holdout, one shot)

| | Result | Development | Pre-registered band |
|---|---|---|---|
| AUC | **0.7802** | 0.7768 | [0.7689, 0.7822] ✓ |
| Gini / KS | **0.5605 / 0.4198** | — | monotone deciles, 3.7× top-decile lift |
| Brier / ECE | **0.0662 / 0.0021** | 0.0667 / 0.0008 | beats climatology 0.0736 |
| Gap vs best flat threshold | **+3.67%** | 5.28% | [3.42%, 7.15%] ✓ |
| Disparate impact ratio | 0.817 | 0.817 | flagged if < 0.80 |

**Model comparison, honestly:** tuned LightGBM 0.7802 vs a WOE logistic scorecard 0.7488 — the gradient booster buys ~3 AUC points over a fully auditable linear scorecard.

Everything above was **pre-registered before it was measured**: cost parameters frozen in the spec, the ledger and comparison thresholds frozen before any money figure existed, consistency bands computed from development data alone. One prediction was *wrong* and is reported as such ([methodology §7](docs/METHODOLOGY.md#7-the-holdout-ceremony--results-and-consistency)), as are three defects in the holdout's sensitivity table found in a later review (disclosed, not re-run; the complete sweep over all six cost grids plus a cure-rate axis is on development data: the gain stays positive at all 14 distinct points, 1.6%–10.9%).

## What's in the box

- **Calibrated probabilities** — isotonic on out-of-fold predictions; metrics cross-fitted so the calibrator never grades itself.
- **Per-applicant thresholds** — `t*(x) = m(x)/(m(x)+LGD·EAD)` with amortization-corrected loan economics and an explicit cure-rate axis ([derivation](docs/METHODOLOGY.md#1-the-decision-rule)).
- **Monotonic constraints** — risk moves the domain-sensible direction on 9 features; guarantee *verified* (zero violations), cost 0.0006 AUC.
- **Reason codes** — top-3 adverse factors per decision in plain language, modeled on adverse-action practice.
- **Fairness audit** — gender excluded, proxy measured at 0.907 and disclosed, decision-level metrics, within-group miscalibration disclosed, and the disparate-treatment tension discussed rather than papered over.
- **A scoring API that refuses what it cannot score** — derived inputs (loan term above all, which drives both PD and threshold) recomputed server-side; inputs outside the training support, unknown fields (incl. `CODE_GENDER`) and economics outside the frozen grids rejected with 422.
- **Deployment** — FastAPI + Streamlit, three-stage Docker build whose middle stage re-verifies artifact hashes and runs the test suite; 83 tests, and the 25-applicant golden regression runs **through HTTP in CI** from the committed, hash-gated bundle.
- **Monitoring** — PSI on features and score, rolling approval rate and mean threshold from the live request log, and a simulated-drift demo *verified to fire* (worst feature PSI 0.08 → 0.65).

## Honest limitations

Monetary amounts are in **unspecified currency units**, not euros. `TARGET` is an **early-delinquency** label while LGD/EAD are charge-off severities; the cure-rate axis (now implemented) shows this definitional gap moves thresholds more than any other assumption and compresses the gain to 1.6% at a 50% cure rate. Cost constants are **illustrative assumptions**, not estimates. There is **no out-of-time validation** (no application dates), the largest gap between these numbers and bank-grade evidence. Training data contains **accepted applicants only** ([reject inference](docs/METHODOLOGY.md#8-reject-inference-a-stated-limitation-not-a-solved-problem)). Fairness covers gender only. Regulatory framing is *"inspired by"* ECOA/GDPR/SR 11-7 practice — never a compliance claim.

## Quickstart

```bash
pip install -r requirements-dev.txt   # Python 3.12
make demo                             # Streamlit + API locally
make check                            # ruff + 83 tests
```

Full pipeline commands, API contract, Docker and deployment: [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

---

*Dataset: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (307,511 applications, 8.07% default rate). Raw data is not redistributed; download it into `data/raw/`.*
