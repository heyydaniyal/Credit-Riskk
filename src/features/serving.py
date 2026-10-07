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
