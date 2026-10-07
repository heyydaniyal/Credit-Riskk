"""
Stateless transformations — sentinel fixes, missing flags, domain ratios.

STATELESS means: no statistics are learned from data.  Every function here
maps a row to a value using only that row.  These transforms carry ZERO
leakage risk and may be applied to any split (dev, holdout, live traffic)
identically.

Anything that learns parameters (bins, thresholds, encodings) lives in
selection.py / woe.py as fit/transform classes instead.
"""

import numpy as np
import pandas as pd

from config.constants import TERM_YEARS_MAX, TERM_YEARS_MIN

# Home Credit uses this placeholder for pensioners (no employment record)
SENTINEL_DAYS_EMPLOYED = 365243


def fix_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace known sentinel values with NaN + an explicit flag.

    Why a flag: EDA finding #4 — the sentinel marks pensioners (17.9% of
    rows).  Being a pensioner is signal; silently converting to NaN would
    keep the signal only for models with native NaN routing.  The flag
    makes it available to every model.
    """
    out = df.copy()
    out["is_pensioner"] = (out["DAYS_EMPLOYED"] == SENTINEL_DAYS_EMPLOYED).astype("int8")
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(SENTINEL_DAYS_EMPLOYED, np.nan)
    return out


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit _is_missing flags where missingness itself is predictive.

    EDA finding #3: EXT_SOURCE_1 missing for 56% of applicants; missing
    defaults at 8.5% vs 7.5% present.  Being *unscored by a bureau* is
    information — we surface it as a feature instead of hiding it behind
    imputation.
    """
    out = df.copy()
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        out[f"{col}_is_missing"] = out[col].isna().astype("int8")
    out["annuity_missing"] = out["AMT_ANNUITY"].isna().astype("int8")
    out["goods_price_missing"] = out["AMT_GOODS_PRICE"].isna().astype("int8")
    return out


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division that returns NaN (never inf) when the denominator is 0 or NaN."""
    den = den.replace(0, np.nan)
    return num / den


def implied_term_years(amt_credit: pd.Series, amt_annuity: pd.Series) -> pd.Series:
    """
    term = AMT_CREDIT / (12 × AMT_ANNUITY), capped [0.5, 7] years.

    Zero-interest approximation (spec 4c): slightly understates term and so
    keeps m(x) conservative. NaN when the annuity is missing or zero — the
    frozen term-fallback rule is a decision-layer concern. Single
    definition: feature building AND the API (which derives term
    server-side rather than trusting a client-supplied value) call this.
    """
    return _safe_div(amt_credit, 12.0 * amt_annuity).clip(lower=TERM_YEARS_MIN,
                                                          upper=TERM_YEARS_MAX)


def add_domain_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Credit-officer features.  One line per feature on why an underwriter cares:

    credit_income_ratio   — leverage: years of gross income the loan represents.
    annuity_income_ratio  — payment burden (DTI): repayment vs income flow.
    payment_rate          — annuity/credit: inverse of term; fast-amortizing
                            loans behave differently from stretched ones.
    credit_goods_ratio    — financed amount vs price: >1 means financing
                            more than the goods (no skin in the game).
    employment_stability  — share of life spent in current job; job-hoppers
                            and fresh hires carry more income risk.
    income_per_person     — household purchasing power after family size.
    age_years / employment_years — human-readable versions of DAYS_* fields.
    term_years            — implied term from the annuity, capped [0.5, 7]
                            (cost-model input; zero-interest approximation
                            per spec Phase 4c — NaN preserved when annuity
                            is missing, the fallback rule is a Phase 4
                            decision-layer concern).
    ext_source_mean/min   — consensus and worst-case of the three external
                            scores; an underwriter reads all bureaus, not one.
    ext_source_n_missing  — thin-file depth: how many bureaus have no score.
    """
    out = df.copy()

    out["credit_income_ratio"] = _safe_div(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["annuity_income_ratio"] = _safe_div(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["payment_rate"] = _safe_div(out["AMT_ANNUITY"], out["AMT_CREDIT"])
    out["credit_goods_ratio"] = _safe_div(out["AMT_CREDIT"], out["AMT_GOODS_PRICE"])
    out["employment_stability"] = _safe_div(out["DAYS_EMPLOYED"], out["DAYS_BIRTH"])
    out["income_per_person"] = _safe_div(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"])

    out["age_years"] = -out["DAYS_BIRTH"] / 365.25
    out["employment_years"] = -out["DAYS_EMPLOYED"] / 365.25

    # Cost-model input (spec Phase 4c): term = AMT_CREDIT / (12 × AMT_ANNUITY)
    out["term_years"] = implied_term_years(out["AMT_CREDIT"], out["AMT_ANNUITY"])

    ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    out["ext_source_mean"] = ext.mean(axis=1)
    out["ext_source_min"] = ext.min(axis=1)
    out["ext_source_n_missing"] = ext.isna().sum(axis=1).astype("int8")

    return out


def apply_all_stateless(df: pd.DataFrame) -> pd.DataFrame:
    """Full stateless pipeline in the canonical order."""
    return add_domain_ratios(add_missing_flags(fix_sentinels(df)))
