"""
Phase 6b — PSI helpers (thin wrappers over the ONE canonical implementation).

DRY note (resolved during the Phase 6 build): this module originally carried
a second, independent PSI implementation alongside
`src.models.calibration.psi`.  They were verified to agree numerically
(0.238286 vs 0.238286 on a shifted normal), but two implementations of the
same metric is a divergence waiting to happen — the live dashboard could
drift from the tested path silently.  Both now delegate to the canonical
function, the one covered by `test_psi_identity_and_shift`.

Bands (Siddiqi 2006): PSI < 0.10 stable · 0.10–0.25 investigate ·
> 0.25 retrain trigger.
"""

import numpy as np
import pandas as pd

from src.models.calibration import psi as _canonical_psi

PSI_INVESTIGATE = 0.10
PSI_RETRAIN = 0.25


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    min_bin_pct: float = 0.05,  # noqa: ARG001 - kept for signature compatibility
) -> float:
    """
    Population Stability Index — quantile bins fit on `expected`, bins with
    small expected mass merged (avoids log blowups).  Returns 0.0 for
    identical distributions.  Delegates to the canonical tested path.
    """
    return _canonical_psi(np.asarray(expected, dtype=float),
                          np.asarray(actual, dtype=float),
                          bins=bins)


def band(value: float) -> str:
    """Map a PSI value to its standard band."""
    if not np.isfinite(value):
        return "n/a"
    return ("retrain" if value > PSI_RETRAIN
            else "investigate" if value > PSI_INVESTIGATE else "stable")


def compute_feature_psi(
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    feature_cols: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    """PSI per numeric feature -> DataFrame(feature, psi, status), worst first."""
    results = []
    for col in feature_cols:
        if col not in train_df.columns or col not in score_df.columns:
            continue
        train_vals = train_df[col].dropna().values
        score_vals = score_df[col].dropna().values
        if len(train_vals) == 0 or len(score_vals) == 0:
            continue
        value = compute_psi(train_vals, score_vals, bins=bins)
        results.append({"feature": col, "psi": value, "status": band(value)})
    return (pd.DataFrame(results)
            .sort_values("psi", ascending=False)
            .reset_index(drop=True))
