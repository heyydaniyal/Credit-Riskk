"""
Phase 0 — Validation Protocol
==============================

Everything downstream depends on getting this right.
- 20% stratified holdout (fixed seed), touched exactly once at the end.
- Stratified 5-fold CV on the 80%, same folds reused everywhere.
- OOF predictions for calibration, threshold optimization, model comparison.
- Pipeline-integrity check (dev vs. holdout distinguishability → leak detector).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.constants import (
    RANDOM_STATE, N_FOLDS, TEST_SIZE, TARGET_COL, ID_COL
)

logger = logging.getLogger(__name__)


def create_holdout_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve 20% stratified holdout.  Returns (dev, holdout).
    The holdout is touched exactly once at the very end (Phase 7).
    """
    dev, holdout = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )
    dev = dev.reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)

    logger.info(
        f"Split: dev={len(dev):,} ({dev[TARGET_COL].mean():.4f} default rate), "
        f"holdout={len(holdout):,} ({holdout[TARGET_COL].mean():.4f} default rate)"
    )
    return dev, holdout


def get_cv_folds(dev: pd.DataFrame) -> StratifiedKFold:
    """
    Stratified 5-fold CV object.  Same folds reused for every model
    so comparisons are paired, not noisy.
    """
    return StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def pipeline_integrity_check(
    dev: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.55,
) -> dict:
    """
    Pipeline-integrity check (formerly 'adversarial validation').

    Train a classifier to distinguish dev vs. holdout rows.
    Because the holdout is a random stratified split of the same dataset,
    the expected AUC is ≈ 0.5 BY CONSTRUCTION.

    What it catches: processing bugs — imputation statistics, WOE bins,
    or encoders accidentally fit across the split boundary show up as
    AUC > ~0.55.

    This is a LEAK DETECTOR for the pipeline, not a covariate-shift test.
    """
    # Label: 0 = dev, 1 = holdout
    dev_subset = dev[feature_cols].copy()
    dev_subset["_is_holdout"] = 0
    hold_subset = holdout[feature_cols].copy()
    hold_subset["_is_holdout"] = 1

    combined = pd.concat([dev_subset, hold_subset], ignore_index=True)
    y_adv = combined["_is_holdout"].values
    X_adv = combined.drop(columns=["_is_holdout"])

    # Quick LightGBM — no tuning needed, just checking for signal
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for train_idx, val_idx in cv.split(X_adv, y_adv):
        dtrain = lgb.Dataset(X_adv.iloc[train_idx], label=y_adv[train_idx])
        dval = lgb.Dataset(X_adv.iloc[val_idx], label=y_adv[val_idx], reference=dtrain)
        model = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "n_estimators": 200,
            },
            dtrain,
            num_boost_round=200,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = model.predict(X_adv.iloc[val_idx])
        aucs.append(roc_auc_score(y_adv[val_idx], preds))

    mean_auc = np.mean(aucs)
    passed = mean_auc < threshold

    result = {
        "mean_auc": mean_auc,
        "fold_aucs": aucs,
        "threshold": threshold,
        "passed": passed,
        "interpretation": (
            "PASS — no pipeline leakage detected (AUC ≈ 0.50 as expected)."
            if passed
            else f"FAIL — AUC={mean_auc:.3f} > {threshold}. "
            "Imputation/encoding may have been fit across the split boundary. Audit Phase 0."
        ),
    }
    logger.info(f"Pipeline integrity: AUC={mean_auc:.4f} — {'PASS' if passed else 'FAIL'}")
    return result


def leakage_audit_report(feature_names: list[str], target_col: str = TARGET_COL) -> str:
    """
    Return the leakage-audit checklist as a formatted string.
    The actual checks are enforced in the feature-engineering code.
    """
    return f"""
    LEAKAGE AUDIT CHECKLIST
    =======================
    1. Aggregate auxiliary tables per {ID_COL} only — never let target
       information cross the join.
    2. Any target encoding fit INSIDE each CV fold (fit on 4 folds,
       transform the 5th), never on the full training set.
    3. Imputation statistics (medians, WOE bins) computed on training
       folds only, applied to validation folds.
    4. {target_col} is never in the feature set.
    5. {ID_COL} is never in the feature set.
    6. Features count: {len(feature_names)}
    """
