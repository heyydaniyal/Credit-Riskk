"""
Phase 5a — reason codes and score scaling.

Sign convention (PINNED in phase5_design_frozen.json before implementation):
reason codes = the features with the LARGEST POSITIVE contribution to the
DEFAULT log-odds — adverse-to-applicant.  pred_contrib returns contributions
to the raw score where higher = more likely to default, so "reasons for
rejection" are the biggest positive entries.  SHAP explains the raw score;
isotonic is monotone, so the ranking carries to the calibrated decision.
"""

import numpy as np
import pandas as pd

# Plain-language mapping for reason codes (extend as features demand)
PLAIN_LANGUAGE = {
    "ext_source_mean": "low average external credit score",
    "ext_source_min": "low external credit score (worst bureau)",
    "EXT_SOURCE_1": "low external credit score (source 1)",
    "EXT_SOURCE_2": "low external credit score (source 2)",
    "EXT_SOURCE_3": "low external credit score (source 3)",
    "annuity_income_ratio": "high payment burden relative to income",
    "payment_rate": "loan structure (payment-to-credit profile)",
    "credit_goods_ratio": "financed amount high relative to goods price",
    "prev_refusal_share": "history of refused applications",
    "prev_approval_rate": "weak prior application outcomes",
    "bureau_utilization": "high utilization of existing credit",
    "bureau_bb_share_dpd_max": "past-due months on bureau accounts",
    "bureau_max_overdue_amt": "large past overdue amounts",
    "days_since_last_application": "recent application activity",
    "age_years": "age-related risk profile",
    "employment_years": "short employment history",
    "employment_stability": "short employment relative to age",
    "term_years": "loan term profile",
    "AMT_GOODS_PRICE": "financed goods value profile",
    # completed under audit (F11.2): every deployed feature now has reviewed
    # plain language rather than a de-underscored column name
    "ORGANIZATION_TYPE": "employer industry profile",
    "DAYS_BIRTH": "age-related risk profile",
    "OCCUPATION_TYPE": "occupation profile",
    "bureau_days_credit_max": "recency of most recent bureau credit",
    "AMT_ANNUITY": "loan payment amount",
    "DAYS_EMPLOYED": "length of employment",
    "prev_amt_application_max": "size of largest prior application",
    "prev_cnt_payment_mean": "typical term of prior loans",
    "prev_req_vs_granted_mean": "prior requested-vs-granted amount pattern",
    "bureau_credit_max": "largest bureau credit amount",
    "prev_req_vs_granted_max": "largest prior requested-vs-granted gap",
    "DAYS_ID_PUBLISH": "recency of identity document update",
    "OWN_CAR_AGE": "age of owned vehicle",
    "prev_cnt_payment_max": "longest term among prior loans",
    "NAME_EDUCATION_TYPE": "education level",
    "bureau_debt_max": "largest outstanding bureau debt",
    "bureau_credit_sum": "total bureau credit amount",
    "prev_share_revolving": "share of prior loans that were revolving",
    "bureau_share_active": "share of bureau accounts currently active",
    "bureau_n_active": "number of active bureau accounts",
    "prev_n_approved": "number of previously approved applications",
    "bureau_limit_sum": "total bureau credit limit",
    "REGION_RATING_CLIENT_W_CITY": "regional risk rating",
    "NAME_FAMILY_STATUS": "family status",  # PROHIBITED_BASIS — see PROHIBITED_REASON_FEATURES
    "DEF_30_CNT_SOCIAL_CIRCLE": "defaults among social contacts (30-day)",
    "FLAG_DOCUMENT_3": "document-submission profile",
    "bureau_overdue_amt_max": "largest past overdue amount on bureau accounts",
    "FLAG_WORK_PHONE": "work-phone-on-file profile",
    "REG_CITY_NOT_LIVE_CITY": "registered-vs-living city mismatch",
    "DEF_60_CNT_SOCIAL_CIRCLE": "defaults among social contacts (60-day)",
    "NAME_CONTRACT_TYPE": "loan product type",
    "AMT_REQ_CREDIT_BUREAU_QRT": "recent quarterly bureau enquiries",
    "ext_source_n_missing": "number of missing external credit scores",
    "bureau_n_card": "number of bureau credit cards",
    "prev_n_refused": "number of previously refused applications",
    "EXT_SOURCE_1_is_missing": "external credit score 1 unavailable",
    "WALLSMATERIAL_MODE": "dwelling construction profile",
    "NAME_INCOME_TYPE": "income source type",
    "NAME_HOUSING_TYPE": "housing situation",
    "FLAG_DOCUMENT_18": "document-submission profile",
    "FLAG_DOCUMENT_16": "document-submission profile",
    "AMT_REQ_CREDIT_BUREAU_DAY": "recent daily bureau enquiries",
    "EXT_SOURCE_3_is_missing": "external credit score 3 unavailable",
}

# Features that are a prohibited basis under ECOA/Reg B (or a direct proxy so
# strong that surfacing it as an adverse reason is legally fraught). If any of
# these appears as a top-3 adverse reason, reason_codes() suppresses it and
# substitutes the next admissible feature — an adverse-action notice must never
# cite marital status. (F11.3)
PROHIBITED_REASON_FEATURES = {"NAME_FAMILY_STATUS"}

BIAS_COL = -1  # pred_contrib's last column is the bias term


def reason_codes(
    booster, X: pd.DataFrame, feature_names: list[str], top_k: int = 3
) -> list[list[dict]]:
    """
    Top-k adverse reason codes per row from exact TreeSHAP (pred_contrib).

    Returns, per row, a list of {feature, plain_language, contribution}
    sorted by descending positive contribution to the default log-odds.
    Rows with fewer than k positive contributions return fewer codes
    (an applicant pushed DOWN by everything has no adverse reasons).
    """
    contrib = booster.predict(X[feature_names], pred_contrib=True)[:, :BIAS_COL]
    out = []
    for row in contrib:
        order = np.argsort(-row)
        codes = []
        for idx in order:  # scan all, in descending adverse order
            if row[idx] <= 0 or len(codes) == top_k:
                break
            f = feature_names[idx]
            if f in PROHIBITED_REASON_FEATURES:
                continue  # never cite a prohibited basis; backfill with the next feature
            codes.append(
                {
                    "feature": f,
                    "plain_language": PLAIN_LANGUAGE.get(f, f.replace("_", " ")),
                    "contribution": float(row[idx]),
                }
            )
        out.append(codes)
    return out


def pd_to_score(p: np.ndarray) -> np.ndarray:
    """
    Points-to-double-odds scaling (frozen): 600 at good:bad odds 19:1,
    PDO = 50.  score = 600 + 50/ln2 · ln(odds_good / 19), clipped [300, 850].
    Lower PD → higher score.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    odds_good = (1 - p) / p
    score = 600 + 50 / np.log(2) * np.log(odds_good / 19.0)
    return np.clip(score, 300, 850)
