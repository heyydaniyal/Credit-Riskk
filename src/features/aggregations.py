"""
Per-applicant aggregations from the auxiliary tables.

Leakage note (why these are safe to compute on ALL rows at once):
every statistic here is computed WITHIN one applicant's own history —
grouped by SK_ID_CURR, never across applicants and never touching TARGET.
Spec Phase 0 checklist: "Aggregate auxiliary tables per SK_ID_CURR only —
never let target information cross the join."

Aggregation functions are the boring five — min, max, mean, sum, count —
per spec Phase 2 ("nothing exotic").

Naming convention: <table>_<field>_<stat> so the feature manifest is
self-documenting.
"""

import numpy as np
import pandas as pd

# bureau_balance STATUS → worst delinquency bucket carried by that month.
# C (closed) and X (unknown) carry no delinquency information → 0.
_STATUS_TO_DPD = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}


def aggregate_bureau_balance(bb: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the 27M-row monthly history to one row per SK_ID_BUREAU.

    Two-stage design: bureau_balance knows nothing about applicants, so we
    first summarize each bureau *loan*, then let aggregate_bureau() roll
    loans up to the applicant.
    """
    bb = bb.copy()
    bb["dpd_bucket"] = bb["STATUS"].map(_STATUS_TO_DPD).astype("int8")
    bb["is_dpd_month"] = (bb["dpd_bucket"] > 0).astype("int8")

    per_loan = bb.groupby("SK_ID_BUREAU").agg(
        bb_months_recorded=("MONTHS_BALANCE", "count"),
        bb_max_dpd_bucket=("dpd_bucket", "max"),
        bb_share_dpd_months=("is_dpd_month", "mean"),
    )
    return per_loan.reset_index()


def aggregate_bureau(bureau: pd.DataFrame, bb_per_loan: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    One row per SK_ID_CURR from bureau (+ optional bureau_balance summary).

    What an underwriter asks of a bureau file, feature by feature:
    - how much history?           (count, months recorded)
    - how much is still open?     (active count/share, open debt)
    - how leveraged elsewhere?    (total debt, utilization vs limits)
    - ever late? how late?        (overdue days/amounts, max DPD bucket)
    - how recent is the history?  (most recent DAYS_CREDIT)
    """
    b = bureau.copy()
    if bb_per_loan is not None:
        b = b.merge(bb_per_loan, on="SK_ID_BUREAU", how="left")

    b["is_active"] = (b["CREDIT_ACTIVE"] == "Active").astype("int8")
    b["is_card"] = (b["CREDIT_TYPE"] == "Credit card").astype("int8")
    b["is_overdue_now"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype("int8")

    agg = b.groupby("SK_ID_CURR").agg(
        bureau_n_loans=("SK_ID_BUREAU", "count"),
        bureau_n_active=("is_active", "sum"),
        bureau_share_active=("is_active", "mean"),
        bureau_n_card=("is_card", "sum"),
        bureau_share_card=("is_card", "mean"),
        bureau_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_debt_max=("AMT_CREDIT_SUM_DEBT", "max"),
        bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
        bureau_credit_max=("AMT_CREDIT_SUM", "max"),
        bureau_limit_sum=("AMT_CREDIT_SUM_LIMIT", "sum"),
        bureau_overdue_amt_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_overdue_amt_max=("AMT_CREDIT_SUM_OVERDUE", "max"),
        bureau_max_overdue_amt=("AMT_CREDIT_MAX_OVERDUE", "max"),
        bureau_days_overdue_mean=("CREDIT_DAY_OVERDUE", "mean"),
        bureau_days_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
        bureau_n_overdue_now=("is_overdue_now", "sum"),
        bureau_n_prolonged=("CNT_CREDIT_PROLONG", "sum"),
        bureau_days_credit_max=("DAYS_CREDIT", "max"),   # most recent loan
        bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
        bureau_days_credit_min=("DAYS_CREDIT", "min"),   # oldest loan
        bureau_annuity_sum=("AMT_ANNUITY", "sum"),
        **(
            dict(
                bureau_bb_months_sum=("bb_months_recorded", "sum"),
                bureau_bb_max_dpd=("bb_max_dpd_bucket", "max"),
                bureau_bb_share_dpd_mean=("bb_share_dpd_months", "mean"),
                bureau_bb_share_dpd_max=("bb_share_dpd_months", "max"),
            )
            if bb_per_loan is not None
            else {}
        ),
    )

    # Utilization: open debt vs total granted credit (guard div-by-0 → NaN)
    denom = agg["bureau_credit_sum"].replace(0, np.nan)
    agg["bureau_utilization"] = agg["bureau_debt_sum"] / denom

    return agg.reset_index()


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """
    One row per SK_ID_CURR from prior Home Credit applications.

    EDA finding #12: refusal share is the strongest non-EXT_SOURCE signal
    (2.2× monotone).  The requested-vs-granted haircut and application
    recency round out the relationship picture.
    """
    p = prev.copy()
    p["is_approved"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype("int8")
    p["is_refused"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype("int8")
    p["is_revolving"] = (p["NAME_CONTRACT_TYPE"] == "Revolving loans").astype("int8")

    # Requested vs granted: >1 means the bank granted less than asked.
    granted = p["AMT_CREDIT"].replace(0, np.nan)
    p["requested_vs_granted"] = p["AMT_APPLICATION"] / granted

    agg = p.groupby("SK_ID_CURR").agg(
        prev_n_applications=("SK_ID_PREV", "count"),
        prev_n_approved=("is_approved", "sum"),
        prev_n_refused=("is_refused", "sum"),
        prev_approval_rate=("is_approved", "mean"),
        prev_refusal_share=("is_refused", "mean"),
        prev_share_revolving=("is_revolving", "mean"),
        prev_req_vs_granted_mean=("requested_vs_granted", "mean"),
        prev_req_vs_granted_max=("requested_vs_granted", "max"),
        prev_days_decision_max=("DAYS_DECISION", "max"),   # most recent application
        prev_days_decision_min=("DAYS_DECISION", "min"),   # first-ever application
        prev_cnt_payment_mean=("CNT_PAYMENT", "mean"),
        prev_cnt_payment_max=("CNT_PAYMENT", "max"),
        prev_amt_application_mean=("AMT_APPLICATION", "mean"),
        prev_amt_application_max=("AMT_APPLICATION", "max"),
        prev_down_payment_rate_mean=("RATE_DOWN_PAYMENT", "mean"),
    )
    return agg.reset_index()


def join_aggregates(
    app: pd.DataFrame,
    bureau_agg: pd.DataFrame,
    prev_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join aggregates onto the application table and add history flags.

    EDA finding #11 vs #12: the two "no history" groups point in OPPOSITE
    directions (no bureau file = riskiest 10.1%; no prior HC application =
    safest 6.0%) — hence two separate flags, never one merged indicator.

    NaN policy after the join: counts/sums for applicants with no history
    become 0 (an applicant with no bureau file has, factually, zero prior
    loans); means/maxes/ratios stay NaN (the mean of an empty history is
    undefined, and the flag carries the signal).
    """
    out = app.merge(bureau_agg, on="SK_ID_CURR", how="left")
    out = out.merge(prev_agg, on="SK_ID_CURR", how="left")

    out["has_bureau_history"] = out["bureau_n_loans"].notna().astype("int8")
    out["has_prev_history"] = out["prev_n_applications"].notna().astype("int8")

    count_sum_cols = [
        c
        for c in out.columns
        if (c.startswith(("bureau_n_", "prev_n_")))
        or c in (
            "bureau_debt_sum", "bureau_credit_sum", "bureau_limit_sum",
            "bureau_overdue_amt_sum", "bureau_annuity_sum",
            "bureau_bb_months_sum",
        )
    ]
    out[count_sum_cols] = out[count_sum_cols].fillna(0)

    return out
