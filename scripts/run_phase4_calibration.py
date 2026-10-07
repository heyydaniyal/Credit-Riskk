"""
Phase 4a/4b — fit the deployed calibrator, measure honestly, verify alignment.

Outputs:
  models/isotonic_dev.pkl                     deployed calibrator (ships)
  data/processed/oof_calibrated.parquet       OOF + cross-fitted p_cal (measurement grade)
  data/processed/dev_ensemble_scores.parquet  fold-ensemble p on dev (alignment + Phase 5b reuse)
  reports/tables/calibration_metrics.json     Brier/ECE, raw vs calibrated, vs climatology
  reports/tables/alignment_report.json        KS/PSI OOF vs ensemble (the 4b verification)
  reports/figures/41_reliability.png, 42_alignment_overlay.png
"""

import json
import logging
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import DATA_PROCESSED, FIGURES_DIR, ID_COL, MODELS_DIR, TABLES_DIR
from src.data.load import load_modeling_frame
from src.models.calibration import (
    alignment_report,
    brier,
    cross_fitted_calibrated,
    ece_and_reliability,
    fit_deployed_calibrator,
)
from src.models.lgbm import ensemble_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase4-cal")

CLIMATOLOGY_BRIER = 0.0736  # always-predict-8% baseline (spec 4b)


def main() -> None:
    oof = pd.read_parquet(os.path.join(DATA_PROCESSED, "oof_lgbm.parquet"))
    y = oof["TARGET"].values

    # ── deployed calibrator (ships) ────────────────────────────────────────
    iso = fit_deployed_calibrator(oof["p_raw"].values, y)
    with open(os.path.join(MODELS_DIR, "isotonic_dev.pkl"), "wb") as f:
        pickle.dump(iso, f)

    # ── cross-fitted measurement ───────────────────────────────────────────
    p_cal_x = cross_fitted_calibrated(oof)
    oof_out = oof.copy()
    oof_out["p_cal_crossfit"] = p_cal_x
    oof_out.to_parquet(os.path.join(DATA_PROCESSED, "oof_calibrated.parquet"), index=False)

    ece_raw, rel_raw = ece_and_reliability(y, oof["p_raw"].values)
    ece_cal, rel_cal = ece_and_reliability(y, p_cal_x)
    metrics = {
        "brier_raw": brier(y, oof["p_raw"].values),
        "brier_calibrated_crossfit": brier(y, p_cal_x),
        "brier_climatology": CLIMATOLOGY_BRIER,
        "beats_climatology_by": CLIMATOLOGY_BRIER - brier(y, p_cal_x),
        "ece_raw": ece_raw,
        "ece_calibrated_crossfit": ece_cal,
        "target_range_brier": [0.065, 0.070],
        "measurement": "cross-fitted (fit on 4 folds' OOF, scored on the 5th, rotated) — "
                       "the deployed calibrator is fit on all OOF and is NOT what these "
                       "numbers grade",
    }
    with open(os.path.join(TABLES_DIR, "calibration_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    rel_cal.to_csv(os.path.join(TABLES_DIR, "reliability_table.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 0.35], [0, 0.35], "--", c="gray", label="perfect")
    ax.plot(rel_raw["mean_pred"], rel_raw["obs_rate"], "o-", c="#a0aec0", label="raw")
    ax.plot(rel_cal["mean_pred"], rel_cal["obs_rate"], "o-", c="#2b6cb0",
            label="calibrated (cross-fit)")
    ax.set_xlabel("mean predicted PD (bin)")
    ax.set_ylabel("observed default rate (bin)")
    ax.set_title(f"Reliability (10 quantile bins)\n"
                 f"Brier {metrics['brier_raw']:.4f} → {metrics['brier_calibrated_crossfit']:.4f} "
                 f"(climatology {CLIMATOLOGY_BRIER})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "41_reliability.png"), dpi=120)
    plt.close(fig)
    logger.info("calibration metrics: %s", {k: round(v, 5) for k, v in metrics.items()
                                            if isinstance(v, float)})

    # ── alignment: OOF vs deployed fold-ensemble on dev ────────────────────
    dev = load_modeling_frame("dev")
    with open(os.path.join(DATA_PROCESSED, "lgbm_features.json")) as f:
        feats = json.load(f)["features"]
    boosters = [lgb.Booster(model_file=os.path.join(MODELS_DIR, f"lgbm_fold{k}.txt"))
                for k in range(5)]
    logger.info("scoring dev with the 5-fold ensemble (alignment check)…")
    p_ens = ensemble_predict(boosters, dev[feats])
    pd.DataFrame({ID_COL: dev[ID_COL], "p_ensemble_raw": p_ens}).to_parquet(
        os.path.join(DATA_PROCESSED, "dev_ensemble_scores.parquet"), index=False)

    align = alignment_report(oof["p_raw"].values, p_ens)
    with open(os.path.join(TABLES_DIR, "alignment_report.json"), "w") as f:
        json.dump(align, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(0, 0.5, 80)
    ax.hist(oof["p_raw"], bins=bins, density=True, alpha=0.55, color="#a0aec0",
            label="OOF scores (calibrator's training dist.)")
    ax.hist(p_ens, bins=bins, density=True, alpha=0.55, color="#2b6cb0",
            label="fold-ensemble scores (deployed dist.)")
    ax.set_xlabel("raw score")
    ax.set_ylabel("density")
    ax.set_title(f"4b alignment — quantified, not asserted\n"
                 f"PSI={align['psi']:.4f}, KS={align['ks_statistic']:.4f}, "
                 f"std ratio={align['compression_ratio_std']:.3f} "
                 f"(mild compression expected)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "42_alignment_overlay.png"), dpi=120)
    plt.close(fig)
    logger.info("alignment: PSI=%.4f KS=%.4f std-ratio=%.3f",
                align["psi"], align["ks_statistic"], align["compression_ratio_std"])


if __name__ == "__main__":
    main()
