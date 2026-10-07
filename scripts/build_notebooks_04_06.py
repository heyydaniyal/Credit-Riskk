"""
Build teaching notebooks 04 (calibration + decisioning), 05 (constraints,
explainability, fairness) and 06 (serving + monitoring).

House standard: every code cell is preceded by markdown that explains, in
plain language and from scratch, WHAT the cell does, WHY it matters, WHO
uses the result, WHEN in the lifecycle it happens, WHERE the logic lives
(always src/ — notebooks call functions and narrate, they hold no logic),
and HOW to read the output. Numbers are computed live from src/ where it
is cheap; long-running results (the sensitivity sweep, the fairness audit)
are loaded from the tables their scripts write, and the cell says so.

Run:  python scripts/build_notebooks_04_06.py   then   make notebooks
"""

import nbformat as nbf

SETUP = """\
import os, sys, json
sys.path.insert(0, "..")
import numpy as np
import pandas as pd
from IPython.display import Image
from config.constants import TABLES_DIR, FIGURES_DIR, DATA_PROCESSED, ARTIFACTS_DIR
pd.set_option("display.width", 140)"""


def md(nb, text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))


def code(nb, text):
    nb.cells.append(nbf.v4.new_code_cell(text))


# ═══════════════════════════════════════════════════════════════════════════
def build_04():
    nb = nbf.v4.new_notebook()
    md(nb, """\
# 04 · Calibration and cost-based decisioning — the centerpiece

**The question this notebook answers:** once the model gives every applicant a
probability of default, *how should the lender turn that number into
approve / reject?*

Two jobs happen here, in order:

1. **Calibration** — make the probability *mean what it says*. If the model
   says 10% for a group of applicants, about 10% of them should default.
2. **Decisioning** — compare that probability with a per-applicant threshold
   `t*(x)` derived from the money at stake on *that* loan.

Everything below is computed on **development data** (the 80% we are
allowed to look at). The final quoted numbers came from the one-shot holdout
(see the README) and are not touched here.""")

    md(nb, """\
## 0 · Setup

**What:** imports and the project paths. **Where:** every path comes from
`config/constants.py` — no file paths are typed in notebooks.""")
    code(nb, SETUP)

    md(nb, """\
## 1 · Load the out-of-fold predictions

**What:** one row per development applicant with three numbers:
`p_raw` (the model's raw score), `p_cal_crossfit` (the calibrated
probability) and `TARGET` (1 = had payment difficulties).

**Why "out-of-fold" (OOF):** the 246,008 development rows were split into 5
folds. Each fold was scored by a model trained on the *other four* folds, so
every prediction here was made on data the model had never seen. That is the
only honest way to grade a model on development data.

**Why "cross-fitted" calibration:** the calibrator (isotonic regression,
explained in section 2) is itself fitted to data. If we fitted it on all rows
and graded it on the same rows, it would be marking its own homework. So for
these *measurements* it was fitted on 4 folds and applied to the 5th, rotating.
The calibrator that actually ships is fitted on all OOF rows.

**Who uses it:** every number in this notebook.""")
    code(nb, """\
oof = pd.read_parquet(os.path.join(DATA_PROCESSED, "oof_constrained_calibrated.parquet"))
print(f"{len(oof):,} development applicants, default rate {oof.TARGET.mean():.2%}")
oof.head()""")

    md(nb, """\
## 2 · Is the probability honest? Brier score and ECE

**What:** two calibration measures, computed live with
`src.models.calibration`.

- **Brier score** = the average of (probability − outcome)². Lower is better.
  A "know-nothing" forecaster that says 8.07% (the base rate) to everyone
  scores **0.0736** — that is the bar to beat ("climatology").
- **ECE (expected calibration error)** = sort applicants into 10 equal-size
  groups by predicted probability, then measure the average gap between
  *predicted* and *observed* default rate in each group.

**Why it matters here more than in most projects:** the decision rule
multiplies the probability by money. A probability that says 20% when the
truth is 12% would make every threshold wrong.

**How isotonic regression works (from scratch):** it learns a step function
that maps raw score → probability, with one rule: it can never go down (a
higher raw score never gets a lower probability). It only *re-labels* scores;
it never re-orders applicants, so it cannot change AUC (a ranking metric).""")
    code(nb, """\
from src.models.calibration import brier, ece_and_reliability

y = oof.TARGET.values
b_raw, b_cal = brier(y, oof.p_raw.values), brier(y, oof.p_cal_crossfit.values)
e_raw, _ = ece_and_reliability(y, oof.p_raw.values)
e_cal, rel = ece_and_reliability(y, oof.p_cal_crossfit.values)
print(f"Brier  raw {b_raw:.5f} → calibrated {b_cal:.5f}   (climatology 0.0736)")
print(f"ECE    raw {e_raw:.5f} → calibrated {e_cal:.5f}")
rel.round(4)""")

    md(nb, """\
**How to read it:** in the table, `mean_pred` and `obs_rate` should be close
in every row — that is what "calibrated" means. The reliability diagram below
is the same table as a picture: points on the diagonal = honest probabilities.""")
    code(nb, """Image(os.path.join(FIGURES_DIR, "41_reliability.png"))""")

    md(nb, """\
## 3 · Does the calibrator fit the model that actually ships?

**What:** the deployed scorer is the **average of the 5 fold models**, not any
single one. The calibrator was fitted on OOF scores (each from *one* model).
Averaging five models can squeeze or stretch the score distribution, so we
*measure* how different the two distributions are rather than assuming they
match.

**How to read it:** PSI (population stability index) near 0 means the two
distributions are practically the same; below 0.10 is "stable" by industry
convention. The overlay figure shows the two histograms on top of each other.""")
    code(nb, """\
al = json.load(open(os.path.join(TABLES_DIR, "final_pipeline.json")))["alignment"]
print(f"PSI {al['psi']:.5f}   KS {al['ks_statistic']:.4f}   std ratio {al['compression_ratio_std']:.3f}")
print(al["note"])
Image(os.path.join(FIGURES_DIR, "42_alignment_overlay.png"))""")

    md(nb, """\
## 4 · The decision rule, worked by hand for one loan

**What:** the threshold formula, derived from expected profit.

For one applicant with probability of default `p` and loan amount `A`:

| | repays (1 − p) | defaults (p) |
|---|---|---|
| **approve** | earn `m(x)·A` (net margin) | lose `LGD·EAD·A` |
| **reject** | 0 | 0 |

Approve when the expected value is positive:
`(1 − p)·m·A − p·LGD·EAD·A > 0`. Divide by `A` and rearrange:

**approve iff  p < t\\*(x) = m / (m + LGD·EAD)**

- `m(x)` — net margin as a share of the loan. For a cash loan:
  `5% a year × term ÷ 2`. The **÷ 2** is the amortization correction: the
  borrower repays principal every month, so on average only half the loan is
  outstanding.
- `LGD` — share of the exposure lost if the borrower defaults (0.70 cash).
- `EAD` — share of the original amount still exposed at default (0.85 cash).

**These constants are illustrative assumptions, not estimates** — the dataset
has no recovery, exposure or pricing data. They were frozen before any result
existed so they could not be tuned to flatter the outcome.

**Where:** `src.models.decision.attach_cost_params` is the single implementation.
Below, the spec's sanity anchor — a 3-year cash loan — by hand and by the code.""")
    code(nb, """\
from src.features.stateless import implied_term_years
from src.models.decision import attach_cost_params

m = 0.05 * 3 / 2
L = 0.70 * 0.85
print(f"by hand: m = {m:.3f}, LGD·EAD = {L:.3f}, t* = {m / (m + L):.4f}")

A = 100_000.0
loan = pd.DataFrame({"NAME_CONTRACT_TYPE": ["Cash loans"], "AMT_CREDIT": [A],
                     "term_years": implied_term_years(pd.Series([A]), pd.Series([A / 36]))})
print(f"by code: t* = {attach_cost_params(loan).t_star.iloc[0]:.4f}")""")

    md(nb, """\
**Why one threshold per applicant:** short loans earn little margin, so they
get *strict* thresholds; long loans earn more, so they get more lenient ones;
revolving credit has its own economics. The histogram shows the spread of
`t*(x)` across the book (roughly 0.03 to 0.17).""")
    code(nb, """Image(os.path.join(FIGURES_DIR, "43_t_star_hist.png"))""")

    md(nb, """\
## 5 · Is the per-applicant rule better than one global threshold?

**What:** the honest comparison. The naive 0.5 threshold is a straw man (it
approves almost everyone). The real competitor is the **best possible single
threshold**, found by trying 99 cut-offs from 0.01 to 0.50.

**How profit is counted (the frozen "ledger"):** approved and repaid → +margin;
approved and defaulted → −loss; rejected → 0. Realized outcomes, never
probabilities.

**Why cross-fitted here:** picking the best flat threshold on the same rows
you score it on flatters the flat policy (it gets to see the answers). Below,
each fold's flat threshold is chosen on the *other four* folds —
`src.models.decision.crossfit_flat_approvals`.""")
    code(nb, """\
from src.data.load import load_modeling_frame
from src.models.decision import crossfit_flat_approvals, realized_profit

cols = load_modeling_frame("dev")[["SK_ID_CURR", "NAME_CONTRACT_TYPE", "term_years", "AMT_CREDIT"]]
d = attach_cost_params(oof.merge(cols, on="SK_ID_CURR", validate="one_to_one"))
p, y, folds = d.p_cal_crossfit.values, d.TARGET.values, d.fold.values
per10k = 1e4 / len(d)

profit = {
    "naive 0.5": realized_profit(p < 0.5, y, d),
    "best single flat (cross-fitted)": realized_profit(crossfit_flat_approvals(p, y, d, folds), y, d),
    "per-applicant t*(x)": realized_profit(p < d.t_star.values, y, d),
}
for k, v in profit.items():
    print(f"{k:>32}: {v * per10k:>14,.0f} currency units per 10k applications")
flat = profit["best single flat (cross-fitted)"]
print(f"\\ngain over best flat: {(profit['per-applicant t*(x)'] - flat) / abs(flat):+.2%}")""")

    md(nb, """\
**Currency:** the dataset never states one (median income 147,150; median
loan 513,531 — not plausibly euros), so amounts are in the dataset's own units.
The percentage is the currency-free claim.""")
    code(nb, """Image(os.path.join(FIGURES_DIR, "44_flat_sweep.png"))""")

    md(nb, """\
## 6 · Who changes sides? The swap-set

**What:** compare the per-applicant rule with the naive 0.5 rule and count the
applicants each one approves that the other rejects, with their *observed*
default rates.

**How to read it:** the "swapped OUT" row is the people the cost rule turns
away that 0.5 would have approved. If they default far above the 8% base
rate, the rule is rejecting the right people.""")
    code(nb, """pd.read_csv(os.path.join(TABLES_DIR, "swap_set_oof.csv"))""")

    md(nb, """\
## 7 · How much of the gain is just segmentation?

**What (added after the October 2026 review, development data only):** a
fairer, stronger competitor. Instead of *one* flat threshold, fit a separate
flat threshold for each **segment** — revolving loans, and cash loans split
into 5 term buckets (6 thresholds), cross-fitted the same way. This
competitor knows contract type and term but nothing about the loan economics.

**Why it matters:** if a fitted 6-threshold policy did as well, the cost
formula would be decorative. It captures most of the gain — so most of the
value is *term and contract segmentation* — but the closed-form `t*(x)`
still wins, and it needs no fitting at all.

**Where:** `scripts/run_sensitivity_dev.py` (≈ 90 s) writes the table; this
cell loads it.""")
    code(nb, """\
sens = json.load(open(os.path.join(TABLES_DIR, "sensitivity_dev_full.json")))
print(sens["label"], "\\n")
for k, v in sens["segmentation_benchmark"].items():
    print(f"{k:>58}: {v['gain_vs_single_flat']:+.2%}")
print(f"\\nshare of the t*(x) gain captured by the 6-segment policy: "
      f"{sens['share_of_gain_captured_by_6_segment_policy']:.0%}")""")

    md(nb, """\
## 8 · Do the conclusions survive different assumptions?

**What:** re-run the comparison under every value in the frozen sensitivity
grids — all six cost parameters — plus the **cure rate**.

**Cure rate, from scratch:** `TARGET = 1` means *early* payment difficulty,
not a written-off loan. Many early-delinquent borrowers catch up ("cure").
LGD and EAD describe *written-off* loans. If a share `c` of TARGET=1 cases
cure with no loss, the expected loss shrinks to `(1 − c)·LGD·EAD` and every
threshold loosens: `t* = m / (m + (1 − c)·LGD·EAD)`. The deployed system uses
`c = 0`; the cell below shows what a 30% or 50% cure rate does.""")
    code(nb, """\
for c in (0.0, 0.30, 0.50):
    t = attach_cost_params(loan, overrides={"cure_rate": c}).t_star.iloc[0]
    print(f"cure rate {c:.0%}: 3-year cash t* = {t:.4f}")
table = pd.read_csv(os.path.join(TABLES_DIR, "sensitivity_dev_full.csv"))
table[~table.duplicate_of_frozen][["grid", "value", "gap_vs_flat_relative",
                                   "approval_rate_instance"]].round(4)""")

    md(nb, """\
**How to read it:** the gain over the best flat threshold is positive at all
14 distinct grid points (1.6% to 10.9%). The **cure rate shrinks it the most**
— at 50% cure the gain falls to 1.6% — because when losses are smaller, being
choosy matters less. The event-definition question (what TARGET really
measures) is therefore the biggest open assumption in the cost model.""")
    code(nb, """Image(os.path.join(FIGURES_DIR, "46_sensitivity_dev.png"))""")

    md(nb, """\
## 9 · The view a credit committee uses: profit vs approval rate

**What:** scale every applicant's threshold up and down together and plot
expected profit against the approval rate it produces. Lenders operate to a
risk appetite ("we approve 60%"), so this curve answers "what does each extra
point of approval cost or earn?".""")
    code(nb, """Image(os.path.join(FIGURES_DIR, "45_profit_vs_approval.png"))""")

    nbf.write(nb, "notebooks/04_calibration_decision.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
def build_05():
    nb = nbf.v4.new_notebook()
    md(nb, """\
# 05 · Monotonic constraints, explanations, and the fairness audit

**The question this notebook answers:** can we *trust and explain* the model's
decisions — to a regulator, to a rejected applicant, and across groups?

Three parts: (1) force the model to move in sensible directions,
(2) explain each decision in plain language, (3) measure how decisions differ
by gender — without using gender in the model.""")
    md(nb, "## 0 · Setup")
    code(nb, SETUP)

    md(nb, """\
## 1 · Monotonic constraints — and proving they hold

**What:** a *monotonic constraint* tells LightGBM that, all else equal, risk
may only go **up** as a feature goes up (e.g. payment burden) or only go
**down** (e.g. external credit score). Nine features carry such a constraint.

**Why:** an unconstrained model can learn small wiggles — "a slightly better
credit score slightly *increases* risk here" — that no credit officer could
defend. Real risk teams accept a tiny accuracy cost for defensibility.

**Why we *test* it:** the review found LightGBM's `advanced` constraint mode
violating monotonicity by up to 0.166 log-odds on these features, so the
pipeline uses `basic` mode and a **gate** checks every fold model.

**How the check works (`src.models.monotonicity.verify_monotonicity`):** take
real applicants, sweep one constrained feature across its range while holding
everything else fixed, and record the worst step in the wrong direction.
Zero = the guarantee holds. Below: a live check on a deployed fold model.""")
    code(nb, """\
import lightgbm as lgb
from config.constants import MONOTONIC_FEATURES
from src.features.serving import cast_with_contract
from src.models.monotonicity import verify_monotonicity

levels = json.load(open(os.path.join(ARTIFACTS_DIR, "categorical_levels.json")))["levels"]
feats = json.load(open(os.path.join(ARTIFACTS_DIR, "lgbm_features.json")))["features"]
pool, _ = cast_with_contract(pd.read_parquet(os.path.join(ARTIFACTS_DIR, "demo_pool.parquet")), levels)
booster = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, "lgbm_constrained_fold0.txt"))
report = verify_monotonicity(booster, pool[feats].sample(200, random_state=0), MONOTONIC_FEATURES)
pd.Series(report, name="worst violation (log-odds)")""")

    md(nb, """\
**The cost of the guarantee:** compare out-of-fold AUC with and without the
constraints. AUC = the probability that a randomly chosen defaulter is scored
riskier than a randomly chosen repayer (0.5 = coin flip, 1.0 = perfect).""")
    code(nb, """\
fp = json.load(open(os.path.join(TABLES_DIR, "final_pipeline.json")))
print(f"constrained OOF AUC   {fp['constrained_oof_auc']:.5f}")
print(f"unconstrained OOF AUC {fp['unconstrained_oof_auc']:.5f}")
print(f"cost of the guarantee {fp['auc_cost_of_constraints']:.5f} AUC")
Image(os.path.join(FIGURES_DIR, "51_constraint_cost.png"))""")

    md(nb, """\
## 2 · Explaining a decision: SHAP and reason codes

**What SHAP is, from scratch:** a model's score for one applicant can be split
into contributions — "the base rate, plus this much because of the external
score, plus this much because of the payment burden, …" — that add up exactly
to the score. TreeSHAP computes these exactly for tree models.

**Reason codes:** the three features pushing *this* applicant's risk up the
most, translated to plain language ("low average external credit score").
This is modeled on adverse-action notices (US ECOA/Reg B; EU GDPR Art. 22) —
*inspired by*, never a compliance claim.

**One subtlety:** SHAP explains the raw score; the decision uses the
calibrated probability. Calibration never re-orders applicants, so the
explanation stays valid for the decision.

**Where:** `DeployedPipeline.score_frame` — the same code path the API uses.""")
    code(nb, """\
from src.models.pipeline import DeployedPipeline

pipe = DeployedPipeline.from_artifacts()
raw_pool = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "demo_pool.parquet"))
scored = pipe.score_frame(raw_pool.head(300))
for i in scored[scored.decision == "REJECT"].index[:2]:
    r = scored.loc[i]
    print(f"applicant {raw_pool.SK_ID_CURR[i]}: PD {r.pd_calibrated:.1%}, "
          f"threshold {r.threshold_applied:.3f} → {r.decision}")
    for rc in r.reason_codes:
        print(f"   • {rc['plain_language']}  (+{rc['contribution']:.2f} log-odds)")""")

    md(nb, """\
**The global picture:** each dot is one applicant; position = how much that
feature moved their risk; colour = the feature's value. External credit
scores dominate, as the EDA predicted.""")
    code(nb, """Image(os.path.join(FIGURES_DIR, "52_shap_beeswarm.png"))""")

    md(nb, """\
## 3 · The fairness audit

**What:** gender (`CODE_GENDER`) is **excluded from the model** and kept only
as an audit column. We then ask three questions.

**(a) Is gender still encoded in the other features? (proxy check)** Train a
model to predict gender from the model's features. AUC near 0.5 would mean
"no gender signal left". We *pre-registered* — wrote down before running —
that it would land at 0.75–0.85, because occupation, income type and family
status carry gender signal.

**(b) Do decisions differ by group?** The **disparate impact ratio** = the
lower group's approval rate ÷ the higher group's. Below 0.80 (the
"four-fifths rule") is the conventional flag.

**(c) Is any gap coming from the model or from the policy?** Because
thresholds differ per applicant, one group could face stricter thresholds
through its product mix. We compare the average `t*(x)` by group.

**Where:** `scripts/run_phase5_fairness.py` computes the audit (proxy model
included) and writes `fairness_audit.json`; this cell loads it.""")
    code(nb, """\
fa = json.load(open(os.path.join(TABLES_DIR, "fairness_audit.json")))
pc, dm = fa["proxy_check"], fa["decision_metrics"]
print(f"(a) proxy AUC {pc['proxy_oof_auc']:.3f}  (pre-registered {pc['preregistered_expectation']}, "
      f"flag {pc['flag_threshold']})")
print("    strongest gender carriers:", list(pc["top10_proxy_carriers_fold1"])[:4])
groups = pd.DataFrame({g: dm[g] for g in ("F", "M")}).T
groups = groups[["n", "base_default_rate", "approval_rate", "mean_t_star",
                 "FPR_goods_rejected", "FNR_defaulters_approved"]]
dir_ = groups.approval_rate.min() / groups.approval_rate.max()
print(f"(b) disparate impact ratio {dir_:.3f}  (flag below 0.80)")
print(f"(c) mean t*(x): F {groups.loc['F', 'mean_t_star']:.4f} vs M {groups.loc['M', 'mean_t_star']:.4f}")
groups.round(3)""")

    md(nb, """\
**How to read it:**

- **(a)** The proxy AUC came out *above* the pre-registered band. Dropping the
  gender column did not remove gender from the data — which is why the audit
  looks at *decisions*, not at the feature list.
- **(b)** The ratio passes the four-fifths flag, but narrowly. It is reported
  as narrow, not clean.
- **(c)** Average thresholds barely differ between groups, so the approval
  gap is **score-driven** — it follows the different observed default rates
  (base_default_rate column) — not threshold-driven.
- **Error rates** (FPR = good customers rejected; FNR = defaulters approved)
  differ between groups. With different base rates, a score that is
  calibrated *within each group* cannot also equalize both error rates
  (Kleinberg et al. 2016; Chouldechova 2017).
- **Why no "fix":** the textbook remedy — different thresholds by gender —
  uses the protected attribute in the decision, which is itself disparate
  treatment in credit. It is named, not implemented. Scope: gender only, the
  only protected attribute in the data.""")
    code(nb, """Image(os.path.join(FIGURES_DIR, "53_fairness.png"))""")

    md(nb, """\
## 4 · Is the probability honest *for each group*?

**What:** section 2 of notebook 04 showed the probabilities are honest *on
average over everyone*. That does not imply they are honest for each group
separately. Here we compare mean predicted PD with the observed default rate
by gender on the development predictions.

**Why it matters:** an earlier version of the docs said this system "chose
calibration" over equal error rates. Under audit that was wrong: it is
calibrated in **aggregate** only. The check below shows women's risk slightly
**over**-predicted and men's **under**-predicted — the same pattern the holdout
showed (METHODOLOGY §6). It is measured and disclosed; group-wise
recalibration would close it but trades against the error-rate gaps.""")
    code(nb, """\
audit = pd.read_parquet(os.path.join(DATA_PROCESSED, "audit_frame.parquet"))
oof = pd.read_parquet(os.path.join(DATA_PROCESSED, "oof_constrained_calibrated.parquet"))
g = (oof.merge(audit[["SK_ID_CURR", "CODE_GENDER"]], on="SK_ID_CURR")
        .query("CODE_GENDER in ['F', 'M']")
        .groupby("CODE_GENDER")
        .agg(n=("TARGET", "size"), observed=("TARGET", "mean"), predicted=("p_cal_crossfit", "mean")))
g["relative_error"] = g.predicted / g.observed - 1
g.round(4)""")
    nbf.write(nb, "notebooks/05_explain_fairness.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
def build_06():
    nb = nbf.v4.new_notebook()
    md(nb, """\
# 06 · Serving and monitoring — the model as a running system

**The question this notebook answers:** what happens between "a request
arrives" and "a decision leaves", what the service refuses to do, and how we
would notice the world changing underneath it.

Everything here runs **in-process**: FastAPI's `TestClient` sends real HTTP
requests to the app object without starting a server. It is the same app the
Docker container runs.""")
    md(nb, "## 0 · Setup")
    code(nb, SETUP)

    md(nb, """\
## 1 · Where the service gets its model: the hash-gated bundle

**What:** `artifacts/` holds exactly what serving needs — 5 fold models, the
calibrator, the feature list, the categorical contract, golden test cases,
the monitoring reference — and `MANIFEST.json` records a SHA-256 fingerprint
of each file.

**Why:** the Docker build, CI and this notebook all check those fingerprints.
If any file changes by one byte without a rebuild, the check fails *before* a
silently different model can serve decisions.

**Who:** the API and the demo load **only** from here
(`DeployedPipeline.from_artifacts()`).""")
    code(nb, """\
import hashlib
manifest = json.load(open(os.path.join(ARTIFACTS_DIR, "MANIFEST.json")))["sha256"]
ok = {f: hashlib.sha256(open(os.path.join(ARTIFACTS_DIR, f), "rb").read()).hexdigest() == h
      for f, h in manifest.items()}
print(f"{sum(ok.values())}/{len(ok)} artifact hashes verified")""")

    md(nb, """\
## 2 · One request, end to end

**What:** score one applicant through `POST /score`. The request is a
**precomputed feature vector** — the service does not re-run the bureau
joins; in production that is an upstream feature pipeline's job.

**How to read the response:** besides the decision it returns *every input
to the decision* — threshold, LGD, EAD, margin, cure rate, the term used —
plus three reason codes and the model version, so any decision can be
audited later.""")
    code(nb, """\
import tempfile
# this notebook logs to its own scratch file, never the service's request log
os.environ["CREDIT_RISK_REQUEST_LOG"] = os.path.join(tempfile.mkdtemp(), "notebook_log.sqlite")

from fastapi.testclient import TestClient
from src.api.app import API_FIELDS, CATS, app

client = TestClient(app)
rows = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "golden_inputs.parquet"))

def payload(row):
    return {f: (None if pd.isna(row[f]) else (str(row[f]) if f in CATS else float(row[f])))
            for f in API_FIELDS}

base = payload(rows.iloc[0])
response = client.post("/score", json=base).json()
{k: v for k, v in response.items() if k != "reason_codes"}""")

    md(nb, """\
## 3 · What the service refuses — and why each rule exists

A scoring service must not quietly score nonsense. Each probe below
reproduces a defect found in the October 2026 review.

1. **Client-chosen loan term.** `term_years` drives both the model and the
   threshold. A 43.8%-PD applicant flipped from REJECT to APPROVE by sending
   `term_years = 7`. Now the server *derives* term from
   `AMT_CREDIT / (12 × AMT_ANNUITY)` and refuses a contradicting value.
2. **Near-empty request.** Only the loan amount and contract type → the model
   would follow its "missing" branches and approve at 2.2%. No training
   applicant was ever that empty, so it is refused as outside the training
   support.
3. **Unknown fields.** Including `CODE_GENDER`: the protected attribute cannot
   even reach the scoring code.
4. **Economics outside the frozen grids.** The demo can explore
   sensitivity, but only inside the pre-registered ranges.""")
    code(nb, """\
probes = {
    "term_years = 7 (contradicts the amounts)": {**base, "term_years": 7.0},
    "near-empty payload": {"AMT_CREDIT": 1000.0, "NAME_CONTRACT_TYPE": "Cash loans"},
    "CODE_GENDER supplied": {**base, "CODE_GENDER": "M"},
    "LGD 0.99 (outside frozen grid)": {**base, "cost_overrides": {"lgd_cash": 0.99}},
    "cure rate 30% (inside grid)": {**base, "cost_overrides": {"cure_rate": 0.30}},
}
for name, body in probes.items():
    r = client.post("/score", json=body)
    detail = r.json().get("detail") if r.status_code != 200 else f"t* = {r.json()['threshold_applied']:.4f}"
    print(f"{r.status_code}  {name:<42} {str(detail)[:90]}")""")

    md(nb, """\
## 4 · Monitoring: noticing that the world changed

**The problem, from scratch:** a credit model is trained on the past. If the
economy turns, applicants start to look different, and the model's numbers
quietly stop meaning what they used to. We cannot wait for defaults to show it
— those arrive 12+ months later.

**What we watch instead — PSI (population stability index):** compare the
distribution of a feature (or of the score) today against the development
reference. PSI below 0.10 = stable, 0.10–0.25 = investigate, above 0.25 =
retrain trigger (Siddiqi 2006).

**What we simulate:** a severe recession on 1,000 applicants — external
credit scores −0.10, payment burden +25%. This is a **simulated** scenario to
prove the alarm works, not production telemetry.

**Where:** `src.monitoring.monitoring_report` and `simulate_recession`.""")
    code(nb, """\
from src.models.pipeline import DeployedPipeline
from src.monitoring import load_reference, monitoring_report, simulate_recession

pool = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "demo_pool.parquet")).sample(1000, random_state=1)
pipe = DeployedPipeline.from_artifacts()
ref = load_reference()
out = {}
for name, frame in (("baseline", pool), ("SIMULATED recession", simulate_recession(pool))):
    s = pipe.score_frame(frame, with_reason_codes=False)
    rep = monitoring_report(frame, s.pd_raw_ensemble.values, s.threshold_applied.values,
                            s.decision.values, ref)
    out[name] = {"score PSI": rep["score_psi"]["psi"],
                 "worst feature PSI": max(v["psi"] for v in rep["feature_psi"].values()),
                 "approval rate": rep["approval_rate"],
                 "mean t*(x)": rep["mean_threshold_t_star"]}
pd.DataFrame(out).round(3)""")

    md(nb, """\
**How to read it:** under the simulated recession the worst feature PSI jumps
into the retrain band and approvals fall — the alarm fires. The mean
threshold does **not** move: thresholds depend on contract type and term, not
on credit scores. If the *product mix* shifted instead (say, more short
loans), mean `t*(x)` would move with no score drift at all — that is why both
are tracked.

## 5 · The live request log

**What:** every `/score` call above was logged (prediction, threshold,
decision — no applicant features, which would be personal data). This
notebook pointed the log at a scratch file in section 2, so only its own
requests appear here.
`read_log()` turns it into rolling approval-rate and mean-threshold series;
the Streamlit Monitoring tab plots them.""")
    code(nb, """\
from src.monitoring import read_log
log = read_log(window=5)
print(f"{len(log)} requests logged")
log.tail()[["ts", "p_cal", "threshold", "decision", "rolling_approval_rate", "rolling_mean_t_star"]]""")
    nbf.write(nb, "notebooks/06_serving_monitoring.ipynb")


if __name__ == "__main__":
    build_04()
    build_05()
    build_06()
    print("wrote notebooks 04, 05, 06")
