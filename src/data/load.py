"""
Data loading — the ONLY place in the project that reads raw files.

Why this module exists (DRY): every notebook and script needs the same
tables and the same dev/holdout logic. If loading lives in one place,
a path change or a dtype fix happens once, not in ten notebooks.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.constants import DATA_PROCESSED, DATA_RAW, DATA_SPLITS, ID_COL

SENTINEL_DAYS_EMPLOYED = 365243  # placeholder Home Credit uses for pensioners


def load_application() -> pd.DataFrame:
    """Load the main application table (307,511 rows × 122 cols)."""
    return pd.read_csv(os.path.join(DATA_RAW, "application_train.csv"))


def load_bureau() -> pd.DataFrame:
    """Load bureau.csv — one row per prior loan at OTHER institutions."""
    return pd.read_csv(os.path.join(DATA_RAW, "bureau.csv"))


def load_bureau_balance() -> pd.DataFrame:
    """Load bureau_balance.csv — monthly status history of bureau loans (27M rows)."""
    return pd.read_csv(os.path.join(DATA_RAW, "bureau_balance.csv"))


def load_previous_application() -> pd.DataFrame:
    """Load previous_application.csv — prior applications at Home Credit itself."""
    return pd.read_csv(os.path.join(DATA_RAW, "previous_application.csv"))


def load_dev_ids() -> pd.Series:
    """IDs of the 80% development set (frozen in Phase 0)."""
    return pd.read_parquet(os.path.join(DATA_SPLITS, "dev_ids.parquet"))[ID_COL]


def load_holdout_ids() -> pd.Series:
    """IDs of the 20% holdout — touched exactly once, in Phase 7."""
    return pd.read_parquet(os.path.join(DATA_SPLITS, "holdout_ids.parquet"))[ID_COL]


def load_fold_assignment() -> pd.DataFrame:
    """Frozen 5-fold assignment for the dev set (SK_ID_CURR, fold, TARGET)."""
    return pd.read_parquet(os.path.join(DATA_SPLITS, "dev_folds.parquet"))


def load_dev_application(fix_sentinel: bool = True) -> pd.DataFrame:
    """
    Application rows for the DEV set only — the safe default for all
    development work.  Never returns holdout rows.

    fix_sentinel: replace DAYS_EMPLOYED==365243 (pensioners) with NaN.
    """
    app = load_application()
    dev = app[app[ID_COL].isin(set(load_dev_ids()))].reset_index(drop=True)
    if fix_sentinel:
        dev["DAYS_EMPLOYED"] = dev["DAYS_EMPLOYED"].replace(
            SENTINEL_DAYS_EMPLOYED, pd.NA
        ).astype("Float64")
    return dev


def load_modeling_frame(split: str = "dev") -> pd.DataFrame:
    """
    THE canonical way to load the engineered feature matrix for modeling
    (Phase 3+).  Returns ID + TARGET + all engineered columns, plus the
    frozen 'fold' column when split='dev'.

    Why this exists (DRY): assembling dev rows + fold assignment + features
    in every notebook/script invites subtle inconsistencies (wrong join,
    stale filter).  One loader, one behavior, asserted every time.

    split: 'dev' (with folds), 'holdout' (Phase 7 ONLY), or 'all'.
    """
    features = pd.read_parquet(os.path.join(DATA_PROCESSED, "features_full.parquet"))
    if split == "all":
        return features
    if split == "holdout":
        out = features[features[ID_COL].isin(set(load_holdout_ids()))]
        return out.reset_index(drop=True)
    if split != "dev":
        raise ValueError(f"split must be 'dev', 'holdout', or 'all' — got {split!r}")
    out = features[features[ID_COL].isin(set(load_dev_ids()))].reset_index(drop=True)
    folds = load_fold_assignment()[[ID_COL, "fold"]]
    out = out.merge(folds, on=ID_COL, how="left")
    assert out["fold"].notna().all(), "dev rows missing fold assignment — splits corrupted"
    assert len(out) == len(folds), "dev row count != fold table — splits corrupted"
    return out
