"""
Phase 3a — run the WOE scorecard baseline.

Outputs:
  data/processed/oof_scorecard.parquet   (ID, fold, y, p) — Phase 4 raw material
  models/scorecard_folds.pkl             5 × (binner, logistic) fold artifacts
  reports/tables/scorecard_cv.json       C grid results + diagnostics
"""

import json
import logging
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import DATA_PROCESSED, ID_COL, MODELS_DIR, TABLES_DIR, TARGET_COL
from src.data.load import load_modeling_frame
from src.models.scorecard import cv_scorecard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase3a")

if __name__ == "__main__":
    dev = load_modeling_frame("dev")
    with open(os.path.join(DATA_PROCESSED, "woe_features.json")) as f:
        features = json.load(f)["features"]

    out = cv_scorecard(dev, features)

    oof = dev[[ID_COL, "fold", TARGET_COL]].copy()
    oof["p_raw"] = out["oof"]
    oof.to_parquet(os.path.join(DATA_PROCESSED, "oof_scorecard.parquet"), index=False)

    with open(os.path.join(MODELS_DIR, "scorecard_folds.pkl"), "wb") as f:
        pickle.dump(out["fold_artifacts"], f)

    report = {
        "best_C": out["best_C"],
        "oof_auc": out["oof_auc"],
        "train_oof_gap": out["train_oof_gap"],
        "grid_results": out["grid_results"],
        "n_features": len(features),
        "expected_range": [0.74, 0.76],
        "note": "class_weight=balanced distorts probabilities (recalibrated in Phase 4); "
                "WOE binner refit inside every fold",
    }
    with open(os.path.join(TABLES_DIR, "scorecard_cv.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info("scorecard: best_C=%g, OOF AUC=%.5f (expected 0.74–0.76), gap=%.4f",
                out["best_C"], out["oof_auc"], out["train_oof_gap"])
