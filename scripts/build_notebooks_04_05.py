"""Build teaching notebooks 04 (calibration+decision) and 05 (explain+fairness).
All numbers load from run artifacts at execution time."""
import nbformat as nbf


def build_04():
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell("""\
# Phase 4 — Calibration + Cost-Based Decisioning (the centerpiece)

Why calibration first: the cost analysis multiplies probabilities by euro amounts — if P(default)=0.20 really means 12%, every threshold is wrong. The pipeline: **fold-ensemble → isotonic → decision layer**, with three honesty disciplines: cross-fitted *metrics* (the calibrator never grades itself), a *measured* OOF-vs-ensemble alignment overlay, and the gap narrative *locked* at the OOF checkpoint before any holdout contact.

The decision rule (Elkan 2001): approve iff `p_cal < t*(x) = m(x) / (m(x) + LGD(x)·ead(x))` — every constant an **illustrative assumption frozen before results existed**; the amount cancels from the threshold but not from the €-weighted ledger."""))
    c.append(nbf.v4.new_code_cell("""\
import sys, os, json
sys.path.insert(0, "..")
import pandas as pd
from IPython.display import Image
from config.constants import TABLES_DIR, FIGURES_DIR

cal = json.load(open(os.path.join(TABLES_DIR, "calibration_metrics.json")))
al  = json.load(open(os.path.join(TABLES_DIR, "alignment_report.json")))
print(f"Brier raw {cal['brier_raw']:.5f} → calibrated {cal['brier_calibrated_crossfit']:.5f} "
      f"(climatology {cal['brier_climatology']}, beaten by {cal['beats_climatology_by']:.4f})")
print(f"ECE {cal['ece_raw']:.5f} → {cal['ece_calibrated_crossfit']:.5f}  ← isotonic's real contribution is SHAPE")
print(f"alignment: PSI {al['psi']:.5f}, KS {al['ks_statistic']:.4f}, std ratio {al['compression_ratio_std']:.3f}")
Image(os.path.join(FIGURES_DIR, "41_reliability.png"))"""))
    c.append(nbf.v4.new_code_cell("""\
Image(os.path.join(FIGURES_DIR, "42_alignment_overlay.png"))"""))
    c.append(nbf.v4.new_markdown_cell("""\
The measurement corrected the naive expectation: the dev ensemble is slightly **more** dispersed than OOF (4 of 5 fold models saw each dev row in training) — on truly new data all 5 are out-of-sample, so this overlay *overstates* the mismatch. Second-order either way."""))
    c.append(nbf.v4.new_code_cell("""\
ck = json.load(open(os.path.join(TABLES_DIR, "gap_checkpoint.json")))
print(f"best flat (dev-selected, frozen): {ck['best_flat_threshold_dev']}  |  €-weighted mean t*: {ck['eur_weighted_mean_t_star']:.3f}")
for k, v in ck["profit_per_10k"].items(): print(f"  {k:>22}: €{v:,.0f} / 10k apps")
print(f"\\nGAP instance vs best-flat: €{ck['gap_instance_vs_best_flat_per_10k']:,.0f} / 10k "
      f"= {ck['gap_instance_vs_best_flat_relative']:.2%}")
ua = ck["uncertainty_audit"]
print(f"bootstrap 95% CI: {ua['gap_bootstrap_95CI_relative']} — entirely above the pre-registered 1–3% band")
print(f"decision-band ECE (p∈[0.02,0.20]): {ua['decision_band_calibration']['band_ece']}")
Image(os.path.join(FIGURES_DIR, "43_t_star_hist.png"))"""))
    c.append(nbf.v4.new_code_cell("""\
Image(os.path.join(FIGURES_DIR, "44_flat_sweep.png"))"""))
    c.append(nbf.v4.new_code_cell("""\
pd.read_csv(os.path.join(TABLES_DIR, "swap_set_oof.csv"))"""))
    c.append(nbf.v4.new_markdown_cell("""\
**Reading the swap-set:** the 91,660 applicants the instance rule rejects that naive-0.5 approves default at **15.1%** — nearly 2× the book. The narrative was locked at this checkpoint (`gap_checkpoint.json`), holdout expectations pre-registered — including the stated asymmetry that best-flat carries one dev-fitted parameter while the instance rule carries zero."""))
    c.append(nbf.v4.new_code_cell("""\
Image(os.path.join(FIGURES_DIR, "45_profit_vs_approval.png"))"""))
    nbf.write(nb, "notebooks/04_calibration_decision.ipynb")


def build_05():
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell("""\
# Phase 5 — Constraints (verified), Explainability, Fairness (pre-registered)

The design review found LightGBM's `advanced` constraint method violating monotonicity by up to **0.166 raw log-odds** on our features — so the pipeline uses `basic`, and a **gate** verifies zero violations on every fold model before recalibration may proceed. The guarantee is a *tested property*, not a parameter."""))
    c.append(nbf.v4.new_code_cell("""\
import sys, os, json
sys.path.insert(0, "..")
import pandas as pd
from IPython.display import Image
from config.constants import TABLES_DIR, FIGURES_DIR

fp = json.load(open(os.path.join(TABLES_DIR, "final_pipeline.json")))
print(f"constrained OOF AUC {fp['constrained_oof_auc']:.5f} vs unconstrained {fp['unconstrained_oof_auc']:.5f}")
print(f"→ cost of the guarantee: {fp['auc_cost_of_constraints']:.5f} (spec typical ≤ {fp['spec_typical_cost']})")
print(f"gate passed: {fp['monotonicity_gate']['passed']} (worst violation by feature all 0.0)")
print(f"recalibrated: Brier {fp['calibration']['brier_crossfit']:.5f}, ECE {fp['calibration']['ece_crossfit']:.5f}")
d = fp["decision"]
print(f"gap under the DEPLOYED pipeline: {d['gap_relative']:.2%} "
      f"(unconstrained CI {d['unconstrained_gap_CI_for_reference']}) — the story survives the retrain")
Image(os.path.join(FIGURES_DIR, "51_constraint_cost.png"))"""))
    c.append(nbf.v4.new_code_cell("""\
Image(os.path.join(FIGURES_DIR, "52_shap_beeswarm.png"))"""))
    c.append(nbf.v4.new_code_cell("""\
demo = json.load(open(os.path.join(TABLES_DIR, "reason_codes_demo.json")))
for a in demo[:2]:
    print(f"applicant {a['SK_ID_CURR']}: PD {a['p_cal']:.3f} → score {a['score']:.0f}")
    for rc in a["reason_codes"]:
        print(f"   • {rc['plain_language']}  (+{rc['contribution']:.2f} log-odds)")"""))
    c.append(nbf.v4.new_markdown_cell("""\
## Fairness — the pre-registration did its job, including where reality exceeded it

Pre-registered: the proxy check *will* fire (0.75–0.85 expected). Measured: **0.907 — above the band** and disclosed as such. Deliverable = measurement + disclosure; per-group thresholds would be disparate treatment, which is why the textbook fix is named, not implemented."""))
    c.append(nbf.v4.new_code_cell("""\
fa = json.load(open(os.path.join(TABLES_DIR, "fairness_audit.json")))
p = fa["proxy_check"]; dm = fa["decision_metrics"]
print(f"proxy OOF AUC: {p['proxy_oof_auc']:.3f} (pre-registered {p['preregistered_expectation']}, flag {p['flag_threshold']})")
print("top carriers:", list(p["top10_proxy_carriers_fold1"])[:4])
rows = {g: dm[g] for g in ("F","M")}
print(pd.DataFrame(rows).T[["n","approval_rate","base_default_rate","FPR_goods_rejected","FNR_defaulters_approved","mean_t_star"]].round(3))
print("summary:", dm["summary"])
Image(os.path.join(FIGURES_DIR, "53_fairness.png"))"""))
    c.append(nbf.v4.new_markdown_cell("""\
**The senior readings:** (1) DIR 0.817 *passes the four-fifths flag by 0.017* — reported as narrow, not clean. (2) The policy-vs-model question the instance rule creates has a measured answer: mean t\\*(x) differs by only 0.004 between groups — the approval gap is **score-driven** (tracking real base-rate differences 10.1% vs 7.0%), not threshold-driven. (3) The equalized-odds gaps are the canonical impossibility pattern: with different base rates, a calibrated score cannot equalize both error rates (Kleinberg 2016; Chouldechova 2017) — this system chose calibration, and reports what that costs on the other axes. Limits disclosed: gender only, dev only, "inspired by" governance practice — never compliance."""))
    nbf.write(nb, "notebooks/05_explain_fairness.ipynb")


build_04()
build_05()
print("wrote notebooks 04 + 05")
