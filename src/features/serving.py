"""
Serving-side categorical casting — the EXECUTABLE form of the contract in
data/processed/categorical_levels.json.

Why this module exists (audit trail):
1. Audit experiment #1: a serving frame whose categorical dtype is missing
   levels silently corrupted 90/500 predictions — so serving MUST cast with
   the exact training levels.
2. Audit experiment #2: the naive cast
   `s.astype(pd.CategoricalDtype(categories=levels))` is deprecated and
   WILL RAISE under pandas 4 precisely when a value is outside the levels —
   i.e., on the exact case the contract exists to handle.

cast_with_contract() therefore maps unseen values to NaN EXPLICITLY first
(deliberate: NaN routes down LightGBM's learned missing branch), then
casts — future-proof, and it returns the unseen counts so serving can log
them (an unseen-category spike is itself a drift signal for Phase 6b).
"""

import numpy as np
import pandas as pd


def cast_with_contract(
    df: pd.DataFrame, levels: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Cast every contracted column to its training categorical levels.

    Returns (casted_copy, unseen_counts) where unseen_counts[col] is the
    number of values that were not in the training levels and became NaN.
    """
    out = df.copy()
    unseen_counts: dict[str, int] = {}
    for col, lv in levels.items():
        if col not in out.columns:
            continue
        s = out[col].astype(object)
        allowed = set(lv)
        mask_unseen = s.notna() & ~s.isin(allowed)
        unseen_counts[col] = int(mask_unseen.sum())
        s = s.where(~mask_unseen, other=pd.NA)
        out[col] = pd.Categorical(s, categories=lv)
    return out, unseen_counts


# ── derived-feature consistency (serving input gate) ──────────────────────
# /score accepts a precomputed feature vector, so a client can send derived
# features that contradict their own sources. The dangerous case found in
# review: term_years is BOTH a model feature and the main driver of t*(x);
# a 43.8%-PD reject became an approve by sending term_years=7 (t* 0.067 ->
# 0.227), and term_years=1000 pushed t* to 0.98. Derived fields whose
# sources are all in the payload are therefore recomputed here with the
# SAME stateless functions used to build the training matrix, and a
# supplied value that disagrees is rejected rather than silently trusted.

def _same(a: float, b: float, rtol: float = 1e-9) -> bool:
    a_nan, b_nan = (a is None or np.isnan(a)), (b is None or np.isnan(b))
    if a_nan or b_nan:
        return a_nan and b_nan
    return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))


def derive_and_check(row: dict) -> tuple[dict, list[str]]:
    """
    Recompute derivable features from their sources in `row`.

    Returns (row_with_derived_values, problems). `problems` lists every
    supplied derived value that disagrees with its sources; an empty list
    means the payload is internally consistent. Missing (None) supplied
    derived values are filled from the sources rather than flagged.
    """
    from src.features.stateless import implied_term_years

    def num(k):
        v = row.get(k)
        return np.nan if v is None else float(v)

    ext = [num("EXT_SOURCE_1"), num("EXT_SOURCE_2"), num("EXT_SOURCE_3")]
    ext_s = pd.Series(ext, dtype=float)
    derived = {
        "term_years": float(implied_term_years(pd.Series([num("AMT_CREDIT")]),
                                               pd.Series([num("AMT_ANNUITY")])).iloc[0]),
        "credit_goods_ratio": (num("AMT_CREDIT") / num("AMT_GOODS_PRICE")
                               if num("AMT_GOODS_PRICE") not in (0.0,) else np.nan),
        "ext_source_mean": float(ext_s.mean()) if ext_s.notna().any() else np.nan,
        "ext_source_min": float(ext_s.min()) if ext_s.notna().any() else np.nan,
        "ext_source_n_missing": float(ext_s.isna().sum()),
        "EXT_SOURCE_1_is_missing": float(np.isnan(ext[0])),
        "EXT_SOURCE_3_is_missing": float(np.isnan(ext[2])),
    }
    out, problems = dict(row), []
    for k, v in derived.items():
        supplied = row.get(k)
        if supplied is not None:
            if not _same(float(supplied), v):
                problems.append(f"{k}={supplied} contradicts its sources (derived {v})")
            # consistent supplied value is kept bit-for-bit (golden parity)
        else:
            out[k] = None if np.isnan(v) else v
    return out, problems
