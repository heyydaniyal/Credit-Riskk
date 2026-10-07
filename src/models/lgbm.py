"""
Phase 3b — LightGBM benchmark: CV trainer + Optuna objective.

Spec decisions encoded here:
- Search space exactly as the frozen table (learning_rate log 0.01–0.1,
  num_leaves 16–128, min_data_in_leaf 50–500, feature/bagging fractions,
  L1/L2 log 1e-8–10, max_depth UNSET — leaf-wise growth).
- NO class weights / is_unbalance / scale_pos_weight: AUC is invariant to
  them, and unweighted models yield less-distorted probabilities → easier
  calibration (Phase 4).
- n_estimators=10,000 with early stopping (200 rounds) per fold — the
  data decides model size, not the grid.
- Objective = OOF AUC across the 5 frozen folds (same folds as every
  other model → paired comparisons).
- Pruning: fold-by-fold median pruning kills hopeless trials early —
  a compute optimization that cannot bias the winner (completed trials
  are always fully evaluated on all 5 folds).
"""

import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

FIXED_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "bagging_freq": 1,
    "num_threads": 0,
    "seed": 42,
}
NUM_BOOST_ROUND = 10_000
EARLY_STOPPING = 200


def suggest_params(trial: optuna.Trial) -> dict:
    """The spec's frozen search space — do not widen after results exist."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def train_cv(
    dev: pd.DataFrame,
    features: list[str],
    params: dict,
    target_col: str = "TARGET",
    fold_col: str = "fold",
    trial: optuna.Trial | None = None,
    fold_callback=None,
    extra_lgb_callbacks: list | None = None,
) -> dict:
    """
    Train 5 fold models with early stopping; return OOF predictions,
    per-fold train/val AUCs, best iterations, and the boosters.

    If `trial` is given, reports intermediate fold AUCs for pruning.
    """
    cat_cols = [c for c in features if isinstance(dev[c].dtype, pd.CategoricalDtype)]
    full = {**FIXED_PARAMS, **params}
    oof = np.full(len(dev), np.nan)
    boosters, train_aucs, val_aucs, best_iters = [], [], [], []

    for k in sorted(dev[fold_col].unique()):
        tr = dev[dev[fold_col] != k]
        va = dev[dev[fold_col] == k]
        dtrain = lgb.Dataset(tr[features], label=tr[target_col],
                             categorical_feature=cat_cols, free_raw_data=False)
        dval = lgb.Dataset(va[features], label=va[target_col], reference=dtrain)
        booster = lgb.train(
            full, dtrain, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)]
            + (extra_lgb_callbacks or []),
        )
        p_va = booster.predict(va[features], num_iteration=booster.best_iteration)
        oof[(dev[fold_col] == k).values] = p_va
        p_tr = booster.predict(tr[features], num_iteration=booster.best_iteration)
        train_aucs.append(roc_auc_score(tr[target_col], p_tr))
        val_aucs.append(roc_auc_score(va[target_col], p_va))
        best_iters.append(booster.best_iteration)
        boosters.append(booster)

        if trial is not None:
            # Report running mean fold AUC; prune hopeless configurations.
            trial.report(float(np.mean(val_aucs)), step=int(k))
            if trial.should_prune():
                raise optuna.TrialPruned()
        if fold_callback is not None:
            fold_callback(int(k))  # may raise TrialPruned (time cap)

    assert not np.isnan(oof).any(), "OOF has gaps — fold assembly bug"
    return {
        "oof": oof,
        "oof_auc": float(roc_auc_score(dev[target_col], oof)),
        "train_auc_mean": float(np.mean(train_aucs)),
        "val_auc_mean": float(np.mean(val_aucs)),
        "val_aucs": [float(a) for a in val_aucs],
        "best_iterations": best_iters,
        "boosters": boosters,
        "train_oof_gap": float(np.mean(train_aucs) - np.mean(val_aucs)),
    }


def make_objective(dev: pd.DataFrame, features: list[str],
                   time_cap_seconds: float | None = None):
    """
    Optuna objective: mean OOF AUC across the 5 frozen folds.

    time_cap_seconds: if set, a trial exceeding the cap is PRUNED — the
    check runs INSIDE the boosting loop (LightGBM iteration callback), so
    even a single slow fold cannot outlive the cap.  Pruned means recorded:
    TPE steers away from configurations this machine cannot afford.  Stated compute constraint: this biases the search
    against very slow configs (lowest learning rates with large forests).
    Documented rather than hidden; the spec's own guidance is that trials
    beyond ~150 move AUC less than features do — and the winner is judged
    by full 5-fold OOF either way.
    """
    import time as _time

    def objective(trial: optuna.Trial) -> float:
        t0 = _time.monotonic()

        def time_guard(env) -> None:  # LightGBM iteration callback
            if (time_cap_seconds and env.iteration % 50 == 0
                    and _time.monotonic() - t0 > time_cap_seconds):
                trial.set_user_attr("pruned_reason", "time_cap")
                raise optuna.TrialPruned()

        params = suggest_params(trial)
        out = train_cv(dev, features, params, trial=trial,
                       extra_lgb_callbacks=[time_guard])
        trial.set_user_attr("best_iterations", out["best_iterations"])
        trial.set_user_attr("train_oof_gap", out["train_oof_gap"])
        trial.set_user_attr("fit_seconds", round(_time.monotonic() - t0, 1))
        return out["oof_auc"]

    return objective


def ensemble_predict(boosters: list, X: pd.DataFrame) -> np.ndarray:
    """Deployment target (Phase 0 #4): mean of the 5 fold models' probabilities."""
    preds = [b.predict(X, num_iteration=b.best_iteration) for b in boosters]
    return np.mean(preds, axis=0)
