"""
Phase 3a — WOE logistic scorecard baseline (sklearn, spec v4.1).

The leakage-critical detail: the WOEBinner is REFIT INSIDE EVERY FOLD —
fit on the 4 training folds, transform the held-out fold.  Fitting WOE
once on all dev data would leak each fold's own target rate into its
validation transform (Phase 0 checklist: "WOE bins computed on training
folds only").

class_weight='balanced' compensates the 11.4:1 imbalance for the linear
model; the spec notes this distorts predicted probabilities — one of the
reasons Phase 4 recalibrates.  AUC (rank-based) is unaffected.

Interview-ready math (understood, not re-implemented):
- log-likelihood ℓ(β) = Σ y·log p + (1−y)·log(1−p),  p = σ(Xβ)
- gradient ∇ℓ = Xᵀ(p − y) — which lbfgs drives to ~0
- L2 (1/C) shrinks coefficients toward 0, trading variance for bias
- WOE + linear model ≈ a GAM: each feature's bin-step function is a
  learned univariate shape, combined linearly on the log-odds scale.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.features.woe import WOEBinner

logger = logging.getLogger(__name__)

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]  # spec: log grid {1e-2 … 1e2}


def _fit_fold(X_tr, y_tr, X_va, C: float, seed: int):
    """Fit binner + logistic on training folds; return validation scores."""
    binner = WOEBinner(n_bins=5, min_bin_frac=0.01).fit(X_tr, y_tr)
    clf = LogisticRegression(
        C=C, class_weight="balanced", solver="lbfgs", max_iter=1000,
        random_state=seed,
    ).fit(binner.transform(X_tr), y_tr)
    return clf.predict_proba(binner.transform(X_va))[:, 1], binner, clf


def cv_scorecard(
    dev: pd.DataFrame,
    features: list[str],
    target_col: str = "TARGET",
    fold_col: str = "fold",
    seed: int = 42,
) -> dict:
    """
    Full CV over the C grid with per-fold WOE refits.

    Returns dict with: best_C, per-C OOF AUCs, OOF predictions (best C),
    per-fold train/val AUCs (diagnostics), and per-fold fitted artifacts
    for the best C (binner + model — the scorecard "fold ensemble").
    """
    folds = sorted(dev[fold_col].unique())
    results = {}
    oof_by_C = {}

    for C in C_GRID:
        oof = np.full(len(dev), np.nan)
        train_aucs, val_aucs = [], []
        artifacts = []
        for k in folds:
            tr_idx = dev[fold_col] != k
            va_idx = ~tr_idx
            X_tr, y_tr = dev.loc[tr_idx, features], dev.loc[tr_idx, target_col]
            X_va, y_va = dev.loc[va_idx, features], dev.loc[va_idx, target_col]

            p_va, binner, clf = _fit_fold(X_tr, y_tr, X_va, C, seed)
            oof[va_idx.values] = p_va
            p_tr = clf.predict_proba(binner.transform(X_tr))[:, 1]
            train_aucs.append(roc_auc_score(y_tr, p_tr))
            val_aucs.append(roc_auc_score(y_va, p_va))
            artifacts.append({"fold": int(k), "binner": binner, "model": clf})

        assert not np.isnan(oof).any(), "OOF has gaps — fold assembly bug"
        auc = roc_auc_score(dev[target_col], oof)
        results[C] = {
            "oof_auc": float(auc),
            "train_auc_mean": float(np.mean(train_aucs)),
            "val_auc_mean": float(np.mean(val_aucs)),
            "val_auc_std": float(np.std(val_aucs)),
        }
        oof_by_C[C] = (oof, artifacts)
        logger.info("C=%g → OOF AUC %.5f (train %.5f)", C, auc,
                    results[C]["train_auc_mean"])

    best_C = max(results, key=lambda c: results[c]["oof_auc"])
    oof_best, artifacts_best = oof_by_C[best_C]
    return {
        "best_C": best_C,
        "grid_results": results,
        "oof": oof_best,
        "fold_artifacts": artifacts_best,
        "oof_auc": results[best_C]["oof_auc"],
        "train_oof_gap": results[best_C]["train_auc_mean"]
        - results[best_C]["val_auc_mean"],
    }
