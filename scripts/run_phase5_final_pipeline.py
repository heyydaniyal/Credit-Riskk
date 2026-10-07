"""
Phase 5a — `make final-pipeline`: the sequence that cannot be half-run.

  1. retrain 5 CONSTRAINED folds (method='basic' per frozen design; trial-20
     params reused verbatim — no tuning loop)
  2. GATE: verify_monotonicity on every fold model — zero violations or halt
  3. new OOF -> refit deployed isotonic -> cross-fitted metrics
  4. re-verify OOF-vs-ensemble alignment (PSI/KS)
  5. decision layer: t*(x) unchanged (frozen economics); best-flat RE-SELECTED
     once on the constrained cross-fitted OOF (frozen convention amendment);
     gap reported next to the unconstrained CI
  6. AUC cost vs unconstrained reported

Outputs: models/lgbm_constrained_fold{k}.txt, models/isotonic_final.pkl,
data/processed/oof_constrained_calibrated.parquet,
reports/tables/final_pipeline.json (+ alignment/gap blocks), figures 51-52.
"""

import json
import logging
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import (
    DATA_PROCESSED,
    FIGURES_DIR,
    ID_COL,
    MODELS_DIR,
    MONOTONIC_FEATURES,
    TABLES_DIR,
    TARGET_COL,
)
from src.data.load import load_modeling_frame
from src.models.calibration import (
    alignment_report,
    brier,
    cross_fitted_calibrated,
    ece_and_reliability,
    fit_deployed_calibrator,
)
from src.models.decision import (
    attach_cost_params,
    best_flat_threshold,
    flat_policy,
    instance_policy,
    realized_profit,
)
from src.models.lgbm import ensemble_predict, train_cv
from src.models.monotonicity import assert_monotone, verify_monotonicity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("final-pipeline")


def main() -> None:
    dev = load_modeling_frame("dev")
    with open(os.path.join(DATA_PROCESSED, "lgbm_features.json")) as f:
        feats = json.load(f)["features"]
    tuned = json.load(open(os.path.join(TABLES_DIR, "lgbm_final.json")))["params"]

    # ── 1. constrained retrain ─────────────────────────────────────────────
    vec = [MONOTONIC_FEATURES.get(f, 0) for f in feats]
    params = {**tuned, "monotone_constraints": vec,
              "monotone_constraints_method": "basic"}
    logger.info("training 5 constrained folds (basic; trial-20 params)…")
    out = train_cv(dev, feats, params)
    for k, b in enumerate(out["boosters"]):
        b.save_model(os.path.join(MODELS_DIR, f"lgbm_constrained_fold{k}.txt"),
                     num_iteration=b.best_iteration)

    # ── 2. THE GATE ────────────────────────────────────────────────────────
    gate_sample = dev[feats].sample(300, random_state=1)
    worst = {}
    for _k, b in enumerate(out["boosters"]):
        rep = verify_monotonicity(b, gate_sample, MONOTONIC_FEATURES)
        assert_monotone(rep)
        for f, v in rep.items():
            worst[f] = max(worst.get(f, 0.0), v)
    logger.info("GATE PASSED: zero monotonicity violations on all 5 folds")

    # ── 3. recalibration ───────────────────────────────────────────────────
    oof = dev[[ID_COL, "fold", TARGET_COL]].copy()
    oof["p_raw"] = out["oof"]
    iso = fit_deployed_calibrator(oof["p_raw"].values, oof[TARGET_COL].values)
    with open(os.path.join(MODELS_DIR, "isotonic_final.pkl"), "wb") as f:
        pickle.dump(iso, f)
    p_cal_x = cross_fitted_calibrated(oof)
    oof["p_cal_crossfit"] = p_cal_x
    oof.to_parquet(os.path.join(DATA_PROCESSED, "oof_constrained_calibrated.parquet"),
                   index=False)
    y = oof[TARGET_COL].values
    ece_c, rel_c = ece_and_reliability(y, p_cal_x)

    # ── 4. alignment re-verification ───────────────────────────────────────
    logger.info("scoring dev with the constrained ensemble (alignment)…")
    p_ens = ensemble_predict(out["boosters"], dev[feats])
    align = alignment_report(oof["p_raw"].values, p_ens)
    pd.DataFrame({ID_COL: dev[ID_COL], "p_ensemble_raw": p_ens}).to_parquet(
        os.path.join(DATA_PROCESSED, "dev_ensemble_scores_constrained.parquet"),
        index=False)

    # ── 5. decision layer recompute ────────────────────────────────────────
    cols = pd.read_parquet(
        os.path.join(DATA_PROCESSED, "features_full.parquet"),
        columns=[ID_COL, "NAME_CONTRACT_TYPE", "term_years", "AMT_CREDIT"])
    d = oof.merge(cols, on=ID_COL, validate="one_to_one")
    d = attach_cost_params(d)
    p = d["p_cal_crossfit"].values
    scale = 10_000 / len(d)
    t_best, sweep = best_flat_threshold(p, y, d)
    profits = {
        "best_flat_constrained": realized_profit(flat_policy(p, t_best), y, d),
        "instance_t_star": realized_profit(instance_policy(p, d), y, d),
        "naive_0.5": realized_profit(flat_policy(p, 0.5), y, d),
    }
    gap = profits["instance_t_star"] - profits["best_flat_constrained"]
    rel_gap = gap / abs(profits["best_flat_constrained"])

    # ── 6. report ──────────────────────────────────────────────────────────
    unc = json.load(open(os.path.join(TABLES_DIR, "lgbm_final.json")))
    final = {
        "constrained_oof_auc": out["oof_auc"],
        "unconstrained_oof_auc": unc["oof_auc_pooled"],
        "auc_cost_of_constraints": unc["oof_auc_pooled"] - out["oof_auc"],
        "spec_typical_cost": 0.003,
        "best_iterations": out["best_iterations"],
        "monotonicity_gate": {"passed": True, "worst_raw_violation_by_feature": worst},
        "calibration": {"brier_crossfit": brier(y, p_cal_x), "ece_crossfit": ece_c,
                        "climatology": 0.0736},
        "alignment": align,
        "decision": {
            "best_flat_threshold_constrained": t_best,
            "profit_per_10k": {k: v * scale for k, v in profits.items()},
            "gap_instance_vs_best_flat_per_10k": gap * scale,
            "gap_relative": rel_gap,
            "unconstrained_gap_CI_for_reference": [0.0421, 0.0583],
        },
        "deployed_artifact": "constrained ensemble + isotonic_final.pkl (Phase 4 calibrator retired)",
    }
    with open(os.path.join(TABLES_DIR, "final_pipeline.json"), "w") as f:
        json.dump(final, f, indent=2)
    rel_c.to_csv(os.path.join(TABLES_DIR, "reliability_table_constrained.csv"),
                 index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["unconstrained", "constrained\n(deployed)"],
           [unc["oof_auc_pooled"], out["oof_auc"]],
           color=["#a0aec0", "#2b6cb0"], width=0.5)
    for i, v in enumerate([unc["oof_auc_pooled"], out["oof_auc"]]):
        ax.text(i, v + 0.0002, f"{v:.5f}", ha="center")
    ax.set_ylim(0.770, 0.780)
    ax.set_ylabel("OOF AUC")
    ax.set_title(f"The defensibility trade: monotonic constraints cost "
                 f"{final['auc_cost_of_constraints']:.5f} AUC\n(guarantee verified: "
                 f"zero violations on all 5 folds)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "51_constraint_cost.png"), dpi=120)
    plt.close(fig)

    logger.info("FINAL PIPELINE: AUC %.5f (cost %.5f) | gate PASSED | Brier %.5f | "
                "gap %.2f%% (unconstrained CI [4.2%%, 5.8%%]) | best-flat %.3f",
                out["oof_auc"], final["auc_cost_of_constraints"],
                final["calibration"]["brier_crossfit"], 100 * rel_gap, t_best)


if __name__ == "__main__":
    main()
