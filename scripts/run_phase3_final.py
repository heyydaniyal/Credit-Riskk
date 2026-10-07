"""
Phase 3 finalizer — after tuning converges.

1. Pull best params from the Optuna study (journal storage).
2. Retrain the 5 fold models with them (full early stopping), save:
     models/lgbm_fold{0..4}.txt          — the deployment fold ensemble
     data/processed/oof_lgbm.parquet     — (ID, fold, y, p_raw): Phase 4 input
     reports/tables/lgbm_final.json      — params + AUCs + diagnostics
     reports/tables/model_comparison.csv — scorecard vs LightGBM, honestly
     reports/figures/31/32_*.png
3. Run the 3c diagnostics playbook check (train−OOF gap vs 0.025).

The scorecard OOF must already exist (run_phase3_scorecard.py).
"""

import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import (
    DATA_PROCESSED,
    FIGURES_DIR,
    ID_COL,
    MODELS_DIR,
    TABLES_DIR,
    TARGET_COL,
)
from src.data.load import load_modeling_frame
from src.models.lgbm import train_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("phase3-final")

GAP_THRESHOLD = 0.025  # spec 3c: overfitting alarm


def main() -> None:
    study = optuna.load_study(
        study_name="lgbm_oof_auc",
        storage=JournalStorage(JournalFileBackend(
            os.path.join(MODELS_DIR, "optuna_lgbm_journal.log"))),
    )
    n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
    best = study.best_params
    logger.info("study: %d complete trials | best OOF AUC %.5f | params %s",
                n_complete, study.best_value, best)

    dev = load_modeling_frame("dev")
    with open(os.path.join(DATA_PROCESSED, "lgbm_features.json")) as f:
        features = json.load(f)["features"]

    out = train_cv(dev, features, best)

    for k, booster in enumerate(out["boosters"]):
        booster.save_model(
            os.path.join(MODELS_DIR, f"lgbm_fold{k}.txt"),
            num_iteration=booster.best_iteration,
        )

    oof = dev[[ID_COL, "fold", TARGET_COL]].copy()
    oof["p_raw"] = out["oof"]
    oof.to_parquet(os.path.join(DATA_PROCESSED, "oof_lgbm.parquet"), index=False)

    # ── diagnostics (spec 3c) ──────────────────────────────────────────────
    gap = out["train_oof_gap"]
    diagnosis = (
        "overfitting — apply playbook (halve num_leaves, double min_data_in_leaf, …)"
        if gap > GAP_THRESHOLD
        else "healthy — train−OOF gap under the 0.025 alarm"
    )

    final = {
        "params": best,
        "n_optuna_trials_complete": n_complete,
        "oof_auc_pooled": out["oof_auc"],
        "val_auc_mean": out["val_auc_mean"],
        "val_aucs_per_fold": out["val_aucs"],
        "train_auc_mean": out["train_auc_mean"],
        "train_oof_gap": gap,
        "gap_threshold": GAP_THRESHOLD,
        "diagnosis": diagnosis,
        "best_iterations": out["best_iterations"],
        "expected_range": [0.775, 0.790],
        "leakage_alarm_above": 0.81,
    }
    with open(os.path.join(TABLES_DIR, "lgbm_final.json"), "w") as f:
        json.dump(final, f, indent=2)

    # ── honest model comparison ────────────────────────────────────────────
    sc = json.load(open(os.path.join(TABLES_DIR, "scorecard_cv.json")))
    comparison = pd.DataFrame(
        [
            {
                "model": "WOE logistic scorecard (30 feats)",
                "oof_auc": sc["oof_auc"],
                "train_oof_gap": sc["train_oof_gap"],
                "notes": f"best_C={sc['best_C']}, class_weight=balanced, "
                         "binner refit per fold",
            },
            {
                "model": "LightGBM tuned (58 feats)",
                "oof_auc": out["oof_auc"],
                "train_oof_gap": gap,
                "notes": f"{n_complete} Optuna trials, no class weights, "
                         f"ES iters {out['best_iterations']}",
            },
        ]
    )
    comparison["auc_gap_vs_scorecard"] = comparison["oof_auc"] - sc["oof_auc"]
    comparison.to_csv(os.path.join(TABLES_DIR, "model_comparison.csv"), index=False)

    # ── figures ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["scorecard\n(logistic, WOE)", "LightGBM\n(tuned)"],
           [sc["oof_auc"], out["oof_auc"]], color=["#a0aec0", "#2b6cb0"], width=0.5)
    for i, v in enumerate([sc["oof_auc"], out["oof_auc"]]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)
    ax.scatter([1] * 5, out["val_aucs"], color="#e53e3e", zorder=3, s=18,
               label="per-fold AUC")
    ax.set_ylim(0.72, max(out["val_aucs"]) + 0.01)
    ax.set_ylabel("OOF AUC")
    ax.set_title(f"Model comparison — GBM buys +{out['oof_auc']-sc['oof_auc']:.4f} AUC "
                 "over the linear scorecard")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "31_model_comparison.png"), dpi=120)
    plt.close(fig)

    hist = [t for t in study.trials if t.state.name == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [t.number for t in hist]
    ys = [t.value for t in hist]
    ax.plot(xs, np.maximum.accumulate(ys), "-", color="#2b6cb0", label="best so far")
    ax.scatter(xs, ys, s=14, color="#a0aec0", label="trial OOF AUC")
    ax.set_xlabel("trial")
    ax.set_ylabel("mean fold AUC")
    ax.set_title("Optuna study — convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "32_optuna_history.png"), dpi=120)
    plt.close(fig)

    logger.info("FINAL: pooled OOF AUC=%.5f (expected 0.775–0.790) | gap=%.4f (%s) | "
                "vs scorecard +%.4f",
                out["oof_auc"], gap, diagnosis.split(" — ")[0],
                out["oof_auc"] - sc["oof_auc"])
    if out["oof_auc"] > 0.81:
        logger.warning("⚠ AUC > 0.81 — LEAKAGE ALARM (spec): audit before celebrating")


if __name__ == "__main__":
    main()
