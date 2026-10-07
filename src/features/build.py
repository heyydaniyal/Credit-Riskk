"""
Feature-matrix builder — raw tables in, one clean matrix + manifest out.

Order of operations (and why it matters, spec Phase 2 interdependency #4):
  1. aggregations   (bases from auxiliary tables)
  2. stateless      (sentinels, flags, ratios — computed from bases)
  3. quarantine     (protected attribute out of the model, into audit df)
No stateful/learned step happens here — selection and WOE fitting live in
run_phase2.py where the dev/holdout boundary is enforced.

Protected attribute: CODE_GENDER is removed from the modeling matrix at
build time (not at training time) so no downstream code can accidentally
use it.  It is preserved in a separate audit frame for Phase 5b.
"""

import logging

import pandas as pd

from config.constants import ID_COL, PROTECTED_ATTRIBUTE, TARGET_COL
from src.features.aggregations import (
    aggregate_bureau,
    aggregate_bureau_balance,
    aggregate_previous_application,
    join_aggregates,
)
from src.features.stateless import apply_all_stateless

logger = logging.getLogger(__name__)

# Columns that must NEVER appear in the modeling feature set
NON_FEATURE_COLS = [ID_COL, TARGET_COL, PROTECTED_ATTRIBUTE]


def build_feature_matrix(
    app: pd.DataFrame,
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    prev: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (features, audit):
      features — ID + TARGET + modeling features (gender excluded)
      audit    — ID + TARGET + CODE_GENDER (+ contract type) for Phase 5b
    """
    logger.info("Aggregating bureau_balance (27M rows → per-loan)…")
    bb_per_loan = aggregate_bureau_balance(bureau_balance)

    logger.info("Aggregating bureau → per-applicant…")
    bureau_agg = aggregate_bureau(bureau, bb_per_loan)

    logger.info("Aggregating previous_application → per-applicant…")
    prev_agg = aggregate_previous_application(prev)

    logger.info("Joining aggregates onto application table…")
    df = join_aggregates(app, bureau_agg, prev_agg)

    logger.info("Applying stateless transforms (sentinels, flags, ratios)…")
    df = apply_all_stateless(df)

    # Audit frame: everything Phase 5b needs, nothing the model may see
    audit = df[[ID_COL, TARGET_COL, PROTECTED_ATTRIBUTE, "NAME_CONTRACT_TYPE"]].copy()

    features = df.drop(columns=[PROTECTED_ATTRIBUTE])

    # Categorical dtype for LightGBM native handling (no one-hot)
    obj_cols = [
        c for c in features.columns
        if not pd.api.types.is_numeric_dtype(features[c])
        and not isinstance(features[c].dtype, pd.CategoricalDtype)
    ]
    for c in obj_cols:
        features[c] = features[c].astype("category")

    assert PROTECTED_ATTRIBUTE not in features.columns
    assert features[ID_COL].is_unique, "Duplicate SK_ID_CURR after joins — aggregation bug"
    assert len(features) == len(app), "Row count changed by joins — aggregation bug"

    logger.info("Feature matrix: %d rows × %d cols", *features.shape)
    return features, audit


def make_manifest(features: pd.DataFrame) -> pd.DataFrame:
    """Feature manifest: name, dtype, source, missing share — feeds Phase 6 pydantic."""
    rows = []
    for col in features.columns:
        if col in NON_FEATURE_COLS:
            continue
        if col.startswith("bureau_"):
            source = "bureau/bureau_balance aggregation"
        elif col.startswith("prev_"):
            source = "previous_application aggregation"
        elif col in (
            "credit_income_ratio", "annuity_income_ratio", "payment_rate",
            "credit_goods_ratio", "employment_stability", "income_per_person",
            "age_years", "employment_years", "term_years",
            "ext_source_mean", "ext_source_min", "ext_source_n_missing",
        ):
            source = "domain ratio"
        elif col.endswith("_is_missing") or col in (
            "is_pensioner", "annuity_missing", "goods_price_missing",
            "has_bureau_history", "has_prev_history",
        ):
            source = "missingness/sentinel flag"
        else:
            source = "application_train raw"
        rows.append(
            {
                "feature": col,
                "dtype": str(features[col].dtype),
                "source": source,
                "missing_share": float(features[col].isna().mean()),
            }
        )
    return pd.DataFrame(rows)
