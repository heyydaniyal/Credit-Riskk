"""
Supplementary EDA tables — the analyses that don't need figures.

These close the gaps found in the pre-Phase-2 review:
  1. bureau_balance (27M rows) was never opened — coverage + status codes
  2. bureau side-fields never checked — active share, currency, recency
  3. DTI (payment burden) — a listed domain ratio, never tested
  4. Down-payment ratio — credit vs goods price

Each function returns a tidy DataFrame and saves it to reports/tables/.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.constants import TABLES_DIR

os.makedirs(TABLES_DIR, exist_ok=True)


def _save(df: pd.DataFrame, name: str, save: bool) -> pd.DataFrame:
    if save:
        df.to_csv(os.path.join(TABLES_DIR, name))
    return df


def bureau_balance_overview(bureau_dev: pd.DataFrame, bureau_balance: pd.DataFrame,
                            save: bool = True) -> pd.DataFrame:
    """
    Coverage and status mix of the monthly bureau_balance table.

    STATUS codes: C=closed, X=unknown, 0=no days past due (DPD),
    1..5 = escalating DPD buckets (5 = 120+ days or written off).
    """
    status = bureau_balance["STATUS"].value_counts(normalize=True).rename("share")
    coverage = bureau_dev["SK_ID_BUREAU"].isin(
        set(bureau_balance["SK_ID_BUREAU"])).mean()
    out = status.to_frame()
    out.loc["— dev bureau loans with any balance history —", "share"] = coverage
    return _save(out.round(4), "bureau_balance_overview.csv", save)


def bureau_side_fields(bureau_dev: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Active share, currency mix, and record recency — join-planning facts."""
    rows = {}
    rows["share_active"] = (bureau_dev["CREDIT_ACTIVE"] == "Active").mean()
    rows["share_closed"] = (bureau_dev["CREDIT_ACTIVE"] == "Closed").mean()
    rows["share_currency_1"] = (bureau_dev["CREDIT_CURRENCY"] == "currency 1").mean()
    recency_years = -bureau_dev["DAYS_CREDIT"] / 365
    rows["recency_median_years"] = recency_years.median()
    rows["recency_p90_years"] = recency_years.quantile(0.9)
    out = pd.Series(rows, name="value").round(4).to_frame()
    return _save(out, "bureau_side_fields.csv", save)


def dti_default_table(dev: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Default rate by payment-burden (DTI) bucket: AMT_ANNUITY / AMT_INCOME_TOTAL."""
    dti = dev["AMT_ANNUITY"] / dev["AMT_INCOME_TOTAL"]
    buckets = pd.cut(dti, [0, 0.1, 0.2, 0.3, 0.5, 10])
    out = dev.groupby(buckets, observed=True)["TARGET"] \
        .agg(default_rate="mean", n="count").round(4)
    return _save(out, "dti_default_rates.csv", save)


def downpayment_default_table(dev: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Default rate by credit/goods-price ratio (cash loans).
    Ratio > 1 means the loan exceeds the purchased item's price
    (fees/insurance financed into the principal).
    """
    cash = dev[dev["NAME_CONTRACT_TYPE"] == "Cash loans"]
    ratio = cash["AMT_CREDIT"] / cash["AMT_GOODS_PRICE"]
    buckets = pd.cut(ratio, [0, 0.8, 1.0, 1.15, 1.3, 5])
    out = cash.groupby(buckets, observed=True)["TARGET"] \
        .agg(default_rate="mean", n="count").round(4)
    out.attrs["share_above_1"] = float((ratio > 1.001).mean())
    return _save(out, "downpayment_default_rates.csv", save)


def dataset_overview(dev: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Section-1-style overview: shape, memory, dtypes, duplicates, missingness."""
    n_num = dev.select_dtypes("number").shape[1]
    n_cat = dev.select_dtypes("object").shape[1]
    miss = dev.isna().mean()
    rows = {
        "rows": len(dev),
        "columns": dev.shape[1],
        "memory_mb": round(dev.memory_usage(deep=True).sum() / 1e6, 1),
        "duplicate_rows": int(dev.duplicated().sum()),
        "numeric_features": n_num,
        "categorical_features": n_cat,
        "cols_with_any_missing": int((miss > 0).sum()),
        "cols_missing_gt_50pct": int((miss > 0.5).sum()),
        "cols_missing_gt_80pct": int((miss > 0.8).sum()),
        "default_rate": round(dev["TARGET"].mean(), 4),
    }
    out = pd.Series(rows, name="value").to_frame()
    return _save(out, "dataset_overview.csv", save)


def univariate_auc_scan(dev: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Leakage screen + signal ranking: AUC of each numeric feature ALONE
    against the target (direction-corrected: max(auc, 1-auc)).

    Flags: >0.90 = likely leakage (investigate before modeling);
           >0.55 = individually informative.
    """
    from sklearn.metrics import roc_auc_score

    y = dev["TARGET"].values
    rows = []
    num_cols = [c for c in dev.select_dtypes("number").columns
                if c not in ("SK_ID_CURR", "TARGET")]
    for col in num_cols:
        x = pd.to_numeric(dev[col], errors="coerce")
        mask = x.notna().values
        if mask.sum() < 1000 or y[mask].mean() in (0, 1):
            continue
        auc = roc_auc_score(y[mask], x[mask])
        rows.append({"feature": col,
                     "auc_strength": max(auc, 1 - auc),
                     "direction": "risk_up" if auc > 0.5 else "risk_down",
                     "coverage": round(mask.mean(), 3)})
    out = (pd.DataFrame(rows)
           .sort_values("auc_strength", ascending=False)
           .reset_index(drop=True))
    out["leakage_flag"] = out["auc_strength"] > 0.90
    return _save(out.round(4), "univariate_auc_scan.csv", save)


def skewness_table(dev: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Skewness of the core monetary/temporal variables -> transformation rules."""
    from scipy.stats import skew

    cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2",
            "EXT_SOURCE_3"]
    rows = []
    for col in cols:
        v = pd.to_numeric(dev[col], errors="coerce").dropna()
        rows.append({"feature": col, "skewness": round(float(skew(v)), 1),
                     "action": ("log-transform for logistic"
                                if abs(skew(v)) > 3 else "none needed")})
    return _save(pd.DataFrame(rows).set_index("feature"), "skewness.csv", save)


def term_formula_validation(prev: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Validate the Phase-4 term approximation against ACTUAL terms.

    previous_application records CNT_PAYMENT (true number of monthly
    payments) for prior loans.  On those same loans we compute our
    Phase-4 implied term  AMT_CREDIT / (12 x AMT_ANNUITY)  and measure
    the ratio implied/actual by term bucket.

    Ratio < 1 means the formula UNDERSTATES the true term (because the
    annuity contains interest, so credit/annuity < number of payments).
    """
    p = prev[(prev["NAME_CONTRACT_TYPE"] == "Cash loans")
             & prev["CNT_PAYMENT"].notna() & (prev["CNT_PAYMENT"] > 0)
             & prev["AMT_ANNUITY"].notna() & (prev["AMT_ANNUITY"] > 0)
             & prev["AMT_CREDIT"].notna() & (prev["AMT_CREDIT"] > 0)
             & (prev["NAME_CONTRACT_STATUS"] == "Approved")].copy()
    p["term_actual"] = (p["CNT_PAYMENT"] / 12).clip(0.5, 7)
    p["term_implied"] = (p["AMT_CREDIT"] / (12 * p["AMT_ANNUITY"])).clip(0.5, 7)
    p["ratio"] = p["term_implied"] / p["term_actual"]

    buckets = pd.cut(p["term_actual"], [0.5, 1, 2, 3, 5, 7])
    out = p.groupby(buckets, observed=True)["ratio"] \
        .agg(median_ratio="median", p25=lambda s: s.quantile(0.25),
             p75=lambda s: s.quantile(0.75), n="count").round(3)
    out.attrs["overall_median_ratio"] = float(p["ratio"].median())
    out.attrs["share_understated"] = float((p["ratio"] < 1).mean())
    return _save(out, "term_formula_validation.csv", save)


def gender_product_mix(dev: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Gender x product mix — pre-registers the Phase-5b t*(x) disparity
    direction.  Because t*(x) depends on contract type and term, any
    gender difference in product mix creates a POLICY-level threshold
    disparity distinct from model-level bias.
    """
    d = dev[dev["CODE_GENDER"].isin(["F", "M"])].copy()
    d["term_years"] = (d["AMT_CREDIT"] / (12 * d["AMT_ANNUITY"])).clip(0.5, 7)
    cash = d[d["NAME_CONTRACT_TYPE"] == "Cash loans"]
    out = pd.DataFrame({
        "share_revolving": d.groupby("CODE_GENDER")["NAME_CONTRACT_TYPE"]
            .apply(lambda s: (s == "Revolving loans").mean()),
        "cash_term_median": cash.groupby("CODE_GENDER")["term_years"].median(),
        "avg_credit": d.groupby("CODE_GENDER")["AMT_CREDIT"].mean(),
        "n": d.groupby("CODE_GENDER").size(),
    }).round(4)
    return _save(out, "gender_product_mix.csv", save)
