"""
Phase 3 tests — model-layer contracts on tiny synthetic data.

What each block protects against:
  scorecard   WOE leakage across folds (the Phase 0 checklist item),
              probability validity
  lgbm        OOF assembly bugs (row predicted by a model that saw it),
              ensemble != mean-of-folds (breaks Phase 4 calibrator alignment)
"""

import numpy as np
import pandas as pd
import pytest

from src.models.lgbm import ensemble_predict, train_cv
from src.models.scorecard import cv_scorecard


@pytest.fixture(scope="module")
def toy():
    rng = np.random.RandomState(0)
    n = 4000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-2.2 + 1.1 * x1 - 0.7 * x2)))
    return pd.DataFrame(
        {
            "f1": x1,
            "f2": x2,
            "TARGET": rng.binomial(1, p),
            "fold": rng.randint(0, 5, n),
        }
    )


# ── scorecard ──────────────────────────────────────────────────────────────
def test_scorecard_refits_binner_per_fold(toy):
    """Five folds must yield five DIFFERENT fitted binners — a single shared
    binner (the leakage bug) would make all WOE tables identical."""
    out = cv_scorecard(toy, ["f1", "f2"])
    woes = [
        tuple(a["binner"].numeric_bins_["f1"]["woes"]) for a in out["fold_artifacts"]
    ]
    assert len(set(woes)) == 5, "identical WOE tables across folds — binner not refit"


def test_scorecard_probabilities_valid_and_learnable(toy):
    out = cv_scorecard(toy, ["f1", "f2"])
    assert np.all((out["oof"] >= 0) & (out["oof"] <= 1))
    assert out["oof_auc"] > 0.7  # real signal must be found


# ── lgbm ───────────────────────────────────────────────────────────────────
def test_oof_rows_predicted_by_unseen_model(toy):
    """Each row's OOF value must equal the prediction of the booster that
    did NOT train on that row's fold."""
    out = train_cv(toy, ["f1", "f2"], {"learning_rate": 0.1, "num_leaves": 15})
    for k in range(5):
        mask = (toy["fold"] == k).values
        direct = out["boosters"][k].predict(
            toy.loc[mask, ["f1", "f2"]],
            num_iteration=out["boosters"][k].best_iteration,
        )
        assert np.allclose(out["oof"][mask], direct), f"fold {k} OOF mismatch"


def test_ensemble_is_mean_of_folds(toy):
    """Phase 0 #4 / Phase 6 test: deployed score = mean of fold predictions.
    Any drift here breaks the Phase 4 calibrator-alignment argument."""
    out = train_cv(toy, ["f1", "f2"], {"learning_rate": 0.1, "num_leaves": 15})
    X = toy[["f1", "f2"]].head(50)
    manual = np.mean(
        [b.predict(X, num_iteration=b.best_iteration) for b in out["boosters"]],
        axis=0,
    )
    assert np.allclose(ensemble_predict(out["boosters"], X), manual)


def test_no_class_weights_in_fixed_params():
    """Spec 3b: AUC is weight-invariant and unweighted probabilities
    calibrate better — the GBM must never be class-weighted."""
    from src.models.lgbm import FIXED_PARAMS

    forbidden = {"is_unbalance", "scale_pos_weight", "class_weight"}
    assert not forbidden & set(FIXED_PARAMS)
