"""
Phase 6b — monitoring framework (demonstrated with SIMULATED drift).

Reuses the tested PSI implementation from src.models.calibration (quantile
bins fit on the reference, <5%-mass merge).  The watchlist is the frozen
design's: PSI on the top-10 SHAP features, SCORE PSI (the most-watched
number on real risk dashboards), rolling approval rate, and mean t*(x) —
the metric instance-dependent thresholds make necessary, because a
product-mix shift moves approvals without any score drift.

What this deliberately is NOT: production telemetry.  Realized default
rates cannot be monitored here — labels arrive 12+ months late, which is
exactly why credit monitoring leans on PSI rather than live AUC.

Bands (industry-standard, cite Siddiqi 2006): < 0.10 stable ·
0.10–0.25 investigate · > 0.25 retrain trigger.
"""

import json
import os
import sqlite3

import numpy as np
import pandas as pd

from config.constants import (  # single definition of bands and paths
    ARTIFACTS_DIR,
    PSI_INVESTIGATE,
    PSI_RETRAIN,
    REQUEST_LOG_DB,
)

# NOTE: aliased deliberately. This package contains a submodule named
# `psi`, and importing it anywhere rebinds the name `psi` on this package,
# shadowing a bare `from ... import psi`. Caught by test_monitoring.py.
from src.models.calibration import psi as _psi

REFERENCE_PATH = os.path.join(ARTIFACTS_DIR, "monitoring_reference.json")
LOG_DB_ENV = "CREDIT_RISK_REQUEST_LOG"


def log_db_path() -> str:
    """Request-log location, resolved at CALL time: $CREDIT_RISK_REQUEST_LOG
    if set (tests and notebooks point it at a scratch file so they never
    write into the service's log), else config REQUEST_LOG_DB."""
    return os.environ.get(LOG_DB_ENV) or REQUEST_LOG_DB


# ── reference (built once from dev artifacts) ──────────────────────────────
def build_reference(dev_features: pd.DataFrame, dev_scores: np.ndarray,
                    top_features: list[str], out_path: str = REFERENCE_PATH) -> dict:
    """Persist the dev reference DISTRIBUTIONS (raw samples, downsampled)
    so PSI at serve time re-fits bins on the same reference every call —
    identical to the tested psi() path, no bespoke bin logic to drift."""
    rng = np.random.RandomState(42)
    feature_samples, categorical_shares = {}, {}
    for f in top_features:
        col = dev_features[f]
        if pd.api.types.is_numeric_dtype(col):
            feature_samples[f] = col.dropna().sample(
                min(20_000, col.notna().sum()), random_state=42).astype(float).tolist()
        else:  # categorical: PSI over category SHARES (incl. missing as a class)
            categorical_shares[f] = (
                col.astype(object).where(col.notna(), "__MISSING__")
                .value_counts(normalize=True).to_dict())
    ref = {
        "top_features": top_features,
        "feature_samples": feature_samples,
        "categorical_shares": categorical_shares,
        "score_sample": rng.choice(dev_scores, 20_000, replace=False).tolist(),
        "dev_approval_rate": None,  # filled by caller if desired
    }
    json.dump(ref, open(out_path, "w"))
    return ref


def load_reference(path: str = REFERENCE_PATH) -> dict:
    return json.load(open(path))


# ── request logging (SQLite: append-friendly, single file) ─────────────────
def init_log(db_path: str | None = None) -> None:
    db_path = db_path or log_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS requests (
             ts TEXT, features_json TEXT, p_raw REAL, p_cal REAL,
             threshold REAL, decision TEXT)"""
    )
    con.commit()
    con.close()


def log_request(features: dict, p_raw: float, p_cal: float, threshold: float,
                decision: str, db_path: str | None = None) -> None:
    """
    Log ONLY the non-PII monitoring signals: the prediction, threshold,
    decision, and timestamp. The raw 59-field feature vector is deliberately
    NOT persisted — storing it would be plaintext applicant PII with no
    retention or access control (F9.2). Score-PSI, approval-rate, and
    threshold-drift monitoring all work from these fields alone; feature-PSI
    in production would come from a separate, access-controlled feature store,
    not this request log.
    """
    import datetime

    db_path = db_path or log_db_path()
    init_log(db_path)  # idempotent; survives the file being removed mid-run
    con = sqlite3.connect(db_path)
    con.execute(
        "INSERT INTO requests VALUES (?,?,?,?,?,?)",
        (datetime.datetime.now().isoformat(), None,  # features column intentionally NULL
         float(p_raw), float(p_cal), float(threshold), decision),
    )
    con.commit()
    con.close()


def read_log(db_path: str | None = None, window: int = 50) -> pd.DataFrame:
    """
    Decision drift from LIVE traffic: every logged /score request, oldest
    first, with rolling approval rate and rolling mean t*(x) over the last
    `window` requests. This is what makes log_request() more than a
    write-only sink — the monitoring tab plots these series. (A product-mix
    shift moves approvals and mean t* with no score drift; tracking both
    separates the causes.)
    """
    db_path = db_path or log_db_path()
    cols = ["ts", "p_raw", "p_cal", "threshold", "decision"]
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=[*cols, "rolling_approval_rate", "rolling_mean_t_star"])
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM requests ORDER BY ts", con)
    finally:
        con.close()
    approved = (df["decision"] == "APPROVE").astype(float)
    df["rolling_approval_rate"] = approved.rolling(window, min_periods=1).mean()
    df["rolling_mean_t_star"] = df["threshold"].rolling(window, min_periods=1).mean()
    return df


# ── the report ─────────────────────────────────────────────────────────────
def monitoring_report(actual_features: pd.DataFrame, actual_scores: np.ndarray,
                      actual_thresholds: np.ndarray, actual_decisions: np.ndarray,
                      reference: dict | None = None) -> dict:
    """PSI per watched feature + score PSI + decision-drift stats,
    each labeled with its band."""
    ref = reference or load_reference()

    def band(v: float) -> str:
        return ("RETRAIN" if v > PSI_RETRAIN
                else "INVESTIGATE" if v > PSI_INVESTIGATE else "stable")

    feats = {}
    for f in ref["top_features"]:
        if f in ref.get("categorical_shares", {}):
            # categorical PSI: sum over categories of (a-e)·ln(a/e), floors 1e-6
            e = ref["categorical_shares"][f]
            s_act = (actual_features[f].astype(object)
                     .where(actual_features[f].notna(), "__MISSING__")
                     .value_counts(normalize=True))
            cats_all = set(e) | set(s_act.index)
            ev = np.array([max(e.get(c, 0.0), 1e-6) for c in cats_all])
            av = np.array([max(float(s_act.get(c, 0.0)), 1e-6) for c in cats_all])
            ev, av = ev / ev.sum(), av / av.sum()
            v = float(np.sum((av - ev) * np.log(av / ev)))
        else:
            exp = np.asarray(ref["feature_samples"][f], dtype=float)
            act = actual_features[f].dropna().values.astype(float)
            v = _psi(exp, act) if len(act) >= 50 else float("nan")
        feats[f] = {"psi": v, "band": band(v) if np.isfinite(v) else "n/a"}

    score_psi = _psi(np.asarray(ref["score_sample"]), np.asarray(actual_scores))
    return {
        "feature_psi": feats,
        "score_psi": {"psi": score_psi, "band": band(score_psi)},
        "approval_rate": float((np.asarray(actual_decisions) == "APPROVE").mean()),
        "mean_threshold_t_star": float(np.mean(actual_thresholds)),
        "mean_p_cal": float(np.mean(actual_scores)),
        "bands": {"investigate": PSI_INVESTIGATE, "retrain": PSI_RETRAIN},
        "label": "monitoring framework with SIMULATED drift — not production telemetry; "
                 "realized default rates unobservable (12+ month label latency)",
    }


# ── the simulated-drift scenario (single definition: UI button and the
#    validation gate both call THIS, so they cannot diverge) ───────────────
def simulate_recession(df):
    """SEVERE recession stress: external bureau scores deteriorate (-0.10,
    and payment burden rises (+25%).  Bureau-score deterioration is the
    economically realistic downturn signature AND hits the top of the
    monitored watchlist — a scenario that drifts unwatched features would
    (correctly) never fire the alarm, as the first gate run proved."""
    out = df.copy()
    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                "ext_source_mean", "ext_source_min"):
        if col in out:
            out[col] = (out[col] - 0.10).clip(lower=0.0)
    for col in ("annuity_income_ratio", "credit_income_ratio"):
        if col in out:
            out[col] = out[col] * 1.25
    if "income_per_person" in out:
        out["income_per_person"] = out["income_per_person"] * 0.8
    return out
