"""
Phase 4c — Instance-Dependent Cost-Based Thresholding
=====================================================

Corrected loan economics on ILLUSTRATIVE ASSUMPTIONS.

The per-applicant decision rule:
    approve iff  p_cal × LGD(x) × ead_factor(x)  <  (1 − p_cal) × m(x)

    t*(x) = m(x) / (m(x) + LGD(x) × ead_factor(x))

Derivation: Bayes-optimal cost-sensitive decision theory (Elkan, IJCAI 2001).

Two traps closed:
  1. Constant LGD+margin → AMT_CREDIT cancels → flat threshold.
     Fix: per-contract LGD, EAD, and term-driven margin.
  2. Naive FP cost = rate × term × AMT_CREDIT assumes full principal stays
     outstanding for whole term.  Cash loans amortize → avg balance ≈ AMT_CREDIT/2.
     Fix: amortization factor = 0.5.

All constants frozen in config/constants.py BEFORE any results exist.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.constants import (
    LGD_CASH, LGD_REVOLVING,
    EAD_FACTOR_CASH, EAD_FACTOR_REVOLVING,
    R_NET_CASH, M_REVOLVING,
    AMORTIZATION_FACTOR,
    TERM_YEARS_MIN, TERM_YEARS_MAX,
    CONTRACT_CASH, CONTRACT_REVOLVING,
)


def compute_term_years(amt_credit: pd.Series, amt_annuity: pd.Series) -> pd.Series:
    """
    Implied term in years: AMT_CREDIT / (12 × AMT_ANNUITY), capped [0.5, 7].
    Zero-interest approximation — slightly understates term, keeps m(x) conservative.
    """
    term = amt_credit / (12 * amt_annuity)
    return term.clip(lower=TERM_YEARS_MIN, upper=TERM_YEARS_MAX)


def compute_margin(
    contract_type: pd.Series,
    amt_credit: pd.Series,
    amt_annuity: pd.Series,
) -> pd.Series:
    """
    Foregone net margin m(x) per applicant.

    Cash (amortizing):  m(x) = r_net × term_years(x) / 2  (×AMT_CREDIT applied later)
       The /2 is the amortization correction.
    Revolving:          m(x) = 0.18 (composite frozen multiplier)
    """
    is_revolving = contract_type == CONTRACT_REVOLVING
    term_years = compute_term_years(amt_credit, amt_annuity)

    # Cash: m = r_net × term / 2  (as fraction of AMT_CREDIT)
    m_cash = R_NET_CASH * term_years * AMORTIZATION_FACTOR
    # Revolving: flat composite
    margin = np.where(is_revolving, M_REVOLVING, m_cash)
    return pd.Series(margin, index=contract_type.index)


def compute_loss(contract_type: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    LGD(x) and ead_factor(x) per applicant, by contract type.
    Returns (lgd, ead_factor).
    """
    is_revolving = contract_type == CONTRACT_REVOLVING
    lgd = np.where(is_revolving, LGD_REVOLVING, LGD_CASH)
    ead = np.where(is_revolving, EAD_FACTOR_REVOLVING, EAD_FACTOR_CASH)
    return (
        pd.Series(lgd, index=contract_type.index),
        pd.Series(ead, index=contract_type.index),
    )


def compute_threshold(
    contract_type: pd.Series,
    amt_credit: pd.Series,
    amt_annuity: pd.Series,
) -> pd.Series:
    """
    Per-applicant optimal threshold:
        t*(x) = m(x) / (m(x) + LGD(x) × ead_factor(x))

    Sanity anchor: 3-year cash → m=0.075, L=0.595, t*≈0.112.
    """
    margin = compute_margin(contract_type, amt_credit, amt_annuity)
    lgd, ead_factor = compute_loss(contract_type)
    loss = lgd * ead_factor

    threshold = margin / (margin + loss)
    return threshold


def make_decision(
    p_calibrated: np.ndarray,
    contract_type: pd.Series,
    amt_credit: pd.Series,
    amt_annuity: pd.Series,
) -> pd.DataFrame:
    """
    Full decision layer: given calibrated PD, return decision + all inputs.
    Approve iff p_cal < t*(x).

    Returns DataFrame with columns:
        pd_calibrated, threshold, decision, margin, lgd, ead_factor,
        fn_cost, fp_cost
    """
    threshold = compute_threshold(contract_type, amt_credit, amt_annuity)
    margin = compute_margin(contract_type, amt_credit, amt_annuity)
    lgd, ead_factor = compute_loss(contract_type)

    decision = (p_calibrated < threshold.values).astype(int)  # 1=approve, 0=reject

    # Cost components (per AMT_CREDIT unit — multiply by AMT_CREDIT for €)
    fn_cost_rate = lgd * ead_factor           # loss if default (fraction of AMT_CREDIT)
    fp_cost_rate = margin                      # lost profit if reject good (fraction of AMT_CREDIT)

    return pd.DataFrame({
        "pd_calibrated": p_calibrated,
        "threshold": threshold.values,
        "decision": decision,
        "margin": margin.values,
        "lgd": lgd.values,
        "ead_factor": ead_factor.values,
        "fn_cost_rate": fn_cost_rate.values,
        "fp_cost_rate": fp_cost_rate.values,
    })


def compute_expected_cost(
    p_calibrated: np.ndarray,
    y_true: np.ndarray,
    contract_type: pd.Series,
    amt_credit: pd.Series,
    amt_annuity: pd.Series,
    flat_threshold: float | None = None,
) -> float:
    """
    Total expected cost over the population (in € terms).

    If flat_threshold is given, use that for all applicants instead of t*(x).
    Cost = sum over rejected good applicants of fp_cost
         + sum over approved defaulters of fn_cost
    """
    margin = compute_margin(contract_type, amt_credit, amt_annuity)
    lgd, ead_factor = compute_loss(contract_type)

    if flat_threshold is not None:
        approve = (p_calibrated < flat_threshold).astype(int)
    else:
        threshold = compute_threshold(contract_type, amt_credit, amt_annuity)
        approve = (p_calibrated < threshold.values).astype(int)

    # False negatives: approved but defaulted
    fn_mask = (approve == 1) & (y_true == 1)
    fn_cost = (lgd * ead_factor * amt_credit)[fn_mask].sum()

    # False positives: rejected but would have been good
    fp_mask = (approve == 0) & (y_true == 0)
    fp_cost = (margin * amt_credit)[fp_mask].sum()

    return fn_cost + fp_cost


def sanity_check_3yr_cash():
    """
    Verify the anchor: 3-year cash loan → t* ≈ 0.112.
    Called in tests.
    """
    # 3-year cash: term=3, m = 0.05 × 3/2 = 0.075
    # L = LGD × EAD = 0.70 × 0.85 = 0.595
    # t* = 0.075 / (0.075 + 0.595) = 0.075 / 0.670 ≈ 0.1119
    t = compute_threshold(
        contract_type=pd.Series([CONTRACT_CASH]),
        amt_credit=pd.Series([100_000]),  # cancels out
        amt_annuity=pd.Series([100_000 / (12 * 3)]),  # → term = 3 years
    )
    expected = 0.075 / (0.075 + 0.70 * 0.85)
    assert abs(t.iloc[0] - expected) < 1e-6, f"Sanity check failed: {t.iloc[0]} != {expected}"
    return t.iloc[0]


if __name__ == "__main__":
    anchor = sanity_check_3yr_cash()
    print(f"✓ 3-year cash anchor: t* = {anchor:.4f} (expected ≈ 0.112)")
