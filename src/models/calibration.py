"""
Phase 4a/4b — probability calibration (isotonic) with honest measurement.

Three separate concerns, kept separate on purpose:

1. DEPLOYED calibrator: isotonic fit on ALL OOF predictions.  This ships.
   ~19,860 positive OOF examples — far beyond the ~1k where isotonic's
   flexibility stops overfitting (Niculescu-Mizil & Caruana 2005), which
   is why isotonic over Platt.  out_of_bounds='clip' is mandatory: holdout
   scores outside the OOF range (design pin) must clip, not NaN.

2. CROSS-FITTED measurement: isotonic fit on OOF and scored on the same
   OOF is in-sample — it grades its own homework.  Every calibration
   METRIC (Brier, ECE, reliability) comes from a rotation: fit on 4 folds'
   OOF, score the 5th, concatenate.  The deployed calibrator is unchanged.

3. ALIGNMENT verification: the deployed scorer is the mean of 5 fold
   models, but each OOF score came from ONE fold model.  Averaging five
   correlated models mildly compresses the distribution — a second-order
   mismatch the spec requires us to MEASURE (overlay + KS + PSI), not
   assert away.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression


def fit_deployed_calibrator(p_raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """The shipping calibrator: isotonic on all OOF predictions."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p_raw, y)
    return iso


def cross_fitted_calibrated(oof: pd.DataFrame) -> np.ndarray:
    """
    Out-of-sample calibrated probabilities for MEASUREMENT ONLY.

    For each fold k: fit isotonic on the OOF predictions of the other 4
    folds, apply to fold k.  Returns p_cal aligned to oof's row order.
    """
    p_cal = np.full(len(oof), np.nan)
    for k in sorted(oof["fold"].unique()):
        mask = (oof["fold"] == k).values
        iso = fit_deployed_calibrator(
            oof.loc[~mask, "p_raw"].values, oof.loc[~mask, "TARGET"].values
        )
        p_cal[mask] = iso.predict(oof.loc[mask, "p_raw"].values)
    assert not np.isnan(p_cal).any(), "cross-fit left gaps"
    return p_cal


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece_and_reliability(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> tuple[float, pd.DataFrame]:
    """
    Expected Calibration Error over QUANTILE bins (equal-count, per spec),
    plus the reliability table (mean predicted vs observed rate per bin).
    """
    df = pd.DataFrame({"y": y, "p": p})
    # rank-based quantile bins; duplicates='drop' guards heavy ties from isotonic steps
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=False)
    rel = (
        df.groupby("bin")
        .agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
        .reset_index()
    )
    weights = rel["n"] / rel["n"].sum()
    ece = float((weights * (rel["mean_pred"] - rel["obs_rate"]).abs()).sum())
    return ece, rel


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index on quantile bins fit on `expected`,
    merging bins with <5% expected mass (spec 6b: avoids log blowups).
    Also used here for the OOF-vs-ensemble alignment check.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts = np.histogram(expected, edges)[0].astype(float)
    a_counts = np.histogram(actual, edges)[0].astype(float)

    # merge small expected bins into their left neighbor
    i = 1
    while i < len(e_counts):
        if e_counts[i] / e_counts.sum() < 0.05 and len(e_counts) > 2:
            e_counts[i - 1] += e_counts[i]
            a_counts[i - 1] += a_counts[i]
            e_counts = np.delete(e_counts, i)
            a_counts = np.delete(a_counts, i)
        else:
            i += 1

    e_pct = e_counts / e_counts.sum()
    a_pct = np.clip(a_counts / a_counts.sum(), 1e-6, None)
    e_pct = np.clip(e_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def alignment_report(p_oof: np.ndarray, p_ensemble: np.ndarray) -> dict:
    """KS + PSI between the calibrator's training distribution (OOF) and
    what it sees in deployment (fold-ensemble scores) — the quantified
    second-order approximation, per spec 4b."""
    ks = ks_2samp(p_oof, p_ensemble)
    return {
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "psi": psi(p_oof, p_ensemble),
        "oof_std": float(np.std(p_oof)),
        "ensemble_std": float(np.std(p_ensemble)),
        "compression_ratio_std": float(np.std(p_ensemble) / np.std(p_oof)),
        "note": "averaging 5 correlated fold models mildly compresses dispersion; "
                "expected: ensemble_std slightly < oof_std, PSI small",
    }
