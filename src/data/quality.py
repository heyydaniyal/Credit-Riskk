"""
Data-quality audit — the systematic "what's broken or weird" pass.

Why a dedicated module: quality issues found in EDA become preprocessing
rules in Phase 2.  Keeping the checks as code (not ad-hoc notebook cells)
means the same audit can re-run on any future data slice.

Output: one tidy table (check, result, severity, action_for_phase2),
saved to reports/tables/data_quality_audit.csv.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.constants import TABLES_DIR


def data_quality_audit(dev: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Run all quality checks on the dev application table."""
    rows = []

    def add(check, result, severity, action):
        rows.append({"check": check, "result": result,
                     "severity": severity, "action_for_phase2": action})

    # 1. Duplicate applicant IDs
    n_dup = dev["SK_ID_CURR"].duplicated().sum()
    add("Duplicate SK_ID_CURR", f"{n_dup} duplicates",
        "ok" if n_dup == 0 else "CRITICAL",
        "none needed" if n_dup == 0 else "deduplicate before any join")

    # 2. XNA gender rows
    n_xna = (dev["CODE_GENDER"] == "XNA").sum()
    add("CODE_GENDER == 'XNA'", f"{n_xna} rows",
        "minor", "exclude from gender fairness audit groups (keep in model data)")

    # 3. Income outliers
    inc = dev["AMT_INCOME_TOTAL"]
    add("AMT_INCOME_TOTAL max vs p99",
        f"max={inc.max():,.0f} vs p99={inc.quantile(0.99):,.0f} "
        f"({inc.max() / inc.quantile(0.99):.0f}x)",
        "major", "log-transform for logistic; LightGBM splits are rank-based (robust)")

    # 4. Missing AMT_ANNUITY (breaks the term formula for t*(x))
    n_ann = dev["AMT_ANNUITY"].isna().sum()
    add("AMT_ANNUITY missing", f"{n_ann} rows",
        "major (cost model input)",
        "impute with fold-median for term calc; flag; cap term to [0.5, 7] anyway")

    # 5. Missing AMT_GOODS_PRICE (breaks the down-payment ratio)
    n_gp = dev["AMT_GOODS_PRICE"].isna().sum()
    add("AMT_GOODS_PRICE missing", f"{n_gp} rows",
        "minor", "ratio -> NaN + _is_missing flag (LightGBM handles NaN)")

    # 6. Pensioner sentinel (already fixed in loader — verify)
    n_sent = (pd.to_numeric(dev["DAYS_EMPLOYED"], errors="coerce") == 365243).sum()
    add("DAYS_EMPLOYED sentinel remaining", f"{n_sent} rows (0 = loader fixed it)",
        "ok" if n_sent == 0 else "CRITICAL",
        "add is_pensioner flag from NAME_INCOME_TYPE=='Pensioner'")

    # 7. Constant / near-constant columns (dead weight)
    nunique = dev.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    near_constant = [c for c in dev.columns
                     if dev[c].value_counts(normalize=True, dropna=False).iloc[0] > 0.999]
    add("Constant columns", f"{constant_cols or 'none'}",
        "minor", "drop")
    add("Near-constant columns (>99.9% one value)",
        f"{len(near_constant)}: {near_constant[:5]}",
        "minor", "drop (no signal, wastes selection budget)")

    # 8. Negative values where impossible
    neg_credit = (dev["AMT_CREDIT"] <= 0).sum()
    add("Non-positive AMT_CREDIT", f"{neg_credit} rows",
        "ok" if neg_credit == 0 else "major",
        "none needed" if neg_credit == 0 else "investigate")

    # 9. Family members anomaly
    fam_max = dev["CNT_FAM_MEMBERS"].max()
    add("CNT_FAM_MEMBERS max", f"{fam_max:.0f}",
        "minor", "cap at p99.9 for logistic; leave for LightGBM")

    # 10. DAYS_* columns future/zero anomalies
    n_phone_zero = (dev["DAYS_LAST_PHONE_CHANGE"] == 0).sum()
    add("DAYS_LAST_PHONE_CHANGE == 0", f"{n_phone_zero:,} rows",
        "minor", "0 means 'changed today' — plausible at application; keep")

    audit = pd.DataFrame(rows)
    if save:
        os.makedirs(TABLES_DIR, exist_ok=True)
        audit.to_csv(os.path.join(TABLES_DIR, "data_quality_audit.csv"), index=False)
    return audit
