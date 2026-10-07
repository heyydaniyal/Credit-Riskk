"""
Phase 6b monitoring tests — fast, synthetic, no run artifacts required.

What each block protects against:
  psi delegation    the dashboard drifting from the tested PSI path
  categorical PSI   the gap the first drift-gate run exposed (a watched
                    feature was categorical and crashed the numeric path)
  bands             mislabeled severity on a live dashboard
  scenario          the simulated-drift transformation silently becoming a
                    no-op (an alarm demo that cannot fire is decoration)
  report            missing keys / mislabeled framework claims
"""

import numpy as np
import pandas as pd

from src.models.calibration import psi as canonical_psi
from src.monitoring import (
    PSI_INVESTIGATE,
    PSI_RETRAIN,
    monitoring_report,
    simulate_recession,
)
from src.monitoring.psi import band, compute_psi


def test_legacy_psi_delegates_to_canonical():
    rng = np.random.RandomState(0)
    a, b = rng.normal(size=5000), rng.normal(0.5, 1.2, 5000)
    assert compute_psi(a, b) == canonical_psi(a, b)
    assert compute_psi(a, a) == 0.0


def test_bands_match_standard_thresholds():
    assert band(0.05) == "stable"
    assert band(PSI_INVESTIGATE + 0.01) == "investigate"
    assert band(PSI_RETRAIN + 0.01) == "retrain"
    assert band(float("nan")) == "n/a"


def _reference(numeric_vals, cat_vals):
    return {
        "top_features": ["num", "cat"],
        "feature_samples": {"num": list(map(float, numeric_vals))},
        "categorical_shares": pd.Series(cat_vals).value_counts(normalize=True).to_dict(),
        "score_sample": list(np.linspace(0.01, 0.3, 2000)),
    }


def test_categorical_psi_path_runs_and_detects_mix_shift():
    """A watched CATEGORICAL feature must be scored, not crash — the exact
    failure the first drift-gate run surfaced."""
    rng = np.random.RandomState(0)
    ref = _reference(rng.normal(size=5000), ["A"] * 700 + ["B"] * 300)
    ref["categorical_shares"] = {"cat": ref["categorical_shares"]}

    same = pd.DataFrame({"num": rng.normal(size=1000),
                         "cat": ["A"] * 700 + ["B"] * 300})
    shifted = pd.DataFrame({"num": rng.normal(size=1000),
                            "cat": ["A"] * 200 + ["B"] * 800})
    scores = np.linspace(0.01, 0.3, 1000)
    thr = np.full(1000, 0.08)
    dec = np.array(["APPROVE"] * 600 + ["REJECT"] * 400)

    r_same = monitoring_report(same, scores, thr, dec, ref)
    r_shift = monitoring_report(shifted, scores, thr, dec, ref)
    assert np.isfinite(r_same["feature_psi"]["cat"]["psi"])
    assert r_shift["feature_psi"]["cat"]["psi"] > r_same["feature_psi"]["cat"]["psi"]
    assert r_shift["feature_psi"]["cat"]["psi"] > PSI_RETRAIN


def test_report_structure_and_honest_label():
    rng = np.random.RandomState(1)
    ref = _reference(rng.normal(size=3000), ["A"] * 900 + ["B"] * 100)
    ref["categorical_shares"] = {"cat": ref["categorical_shares"]}
    df = pd.DataFrame({"num": rng.normal(size=500), "cat": ["A"] * 450 + ["B"] * 50})
    rep = monitoring_report(df, np.linspace(0.01, 0.3, 500), np.full(500, 0.08),
                            np.array(["APPROVE"] * 300 + ["REJECT"] * 200), ref)
    for key in ("feature_psi", "score_psi", "approval_rate",
                "mean_threshold_t_star", "bands", "label"):
        assert key in rep
    assert rep["approval_rate"] == 0.6
    assert "SIMULATED" in rep["label"] and "not production telemetry" in rep["label"]


def test_simulate_recession_moves_watched_features_in_the_right_direction():
    """The scenario is defined once and shared by the UI button and the
    validation gate; it must actually degrade credit quality."""
    df = pd.DataFrame(
        {
            "EXT_SOURCE_1": [0.5, 0.6, 0.05],
            "ext_source_mean": [0.5, 0.6, 0.05],
            "annuity_income_ratio": [0.2, 0.3, 0.4],
            "income_per_person": [1000.0, 2000.0, 3000.0],
        }
    )
    out = simulate_recession(df)
    assert (out["EXT_SOURCE_1"] < df["EXT_SOURCE_1"]).all()      # scores deteriorate
    assert (out["EXT_SOURCE_1"] >= 0).all()                       # floored, not negative
    assert (out["annuity_income_ratio"] > df["annuity_income_ratio"]).all()  # burden up
    assert (out["income_per_person"] < df["income_per_person"]).all()        # income down


def test_request_log_stores_no_pii():
    """F9.2: the request log must persist monitoring signals (prediction,
    threshold, decision) but NEVER the raw applicant feature vector."""
    import os
    import sqlite3
    import tempfile

    from src.monitoring import init_log, log_request

    db = os.path.join(tempfile.mkdtemp(), "t.sqlite")
    init_log(db)
    log_request({"AMT_INCOME_TOTAL": 999999, "CODE_GENDER": "F"},
                0.1, 0.1, 0.08, "APPROVE", db_path=db)
    row = sqlite3.connect(db).execute(
        "SELECT features_json, p_cal, decision FROM requests").fetchone()
    assert row[0] is None            # no features persisted
    assert row[1] == 0.1 and row[2] == "APPROVE"   # monitoring signals present
