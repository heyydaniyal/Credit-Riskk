"""
Phase 2 tests — every test uses tiny synthetic data with hand-computable
answers.  What each block protects against:

  stateless     silent sentinel/ratio bugs (inf, wrong flags, bad caps)
  aggregations  join bugs — the most expensive silent failure in the project
  woe           wrong WOE math, NaN/unseen-category crashes, IV screening
  selection     dropping the wrong twin of a correlated pair
  contracts     protected-attribute leakage, transform-before-fit misuse
"""

import numpy as np
import pandas as pd
import pytest

from src.features.aggregations import (
    aggregate_bureau,
    aggregate_bureau_balance,
    aggregate_previous_application,
    join_aggregates,
)
from src.features.selection import CorrelationDropper, NearConstantDropper
from src.features.stateless import SENTINEL_DAYS_EMPLOYED, apply_all_stateless
from src.features.woe import WOEBinner


# ── fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture
def tiny_app():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "CODE_GENDER": ["F", "M", "F"],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", "Cash loans"],
            "DAYS_EMPLOYED": [SENTINEL_DAYS_EMPLOYED, -1000, -3000],
            "DAYS_BIRTH": [-15000, -12000, -20000],
            "AMT_CREDIT": [120000.0, 60000.0, 300000.0],
            "AMT_INCOME_TOTAL": [60000.0, 0.0, 100000.0],
            "AMT_ANNUITY": [5000.0, np.nan, 2000.0],
            "AMT_GOODS_PRICE": [110000.0, np.nan, 280000.0],
            "CNT_FAM_MEMBERS": [2.0, 1.0, 4.0],
            "EXT_SOURCE_1": [np.nan, 0.5, 0.7],
            "EXT_SOURCE_2": [0.3, np.nan, 0.6],
            "EXT_SOURCE_3": [0.4, 0.5, np.nan],
        }
    )


# ── stateless ──────────────────────────────────────────────────────────────
def test_sentinel_becomes_nan_plus_flag(tiny_app):
    out = apply_all_stateless(tiny_app)
    assert out["is_pensioner"].tolist() == [1, 0, 0]
    assert pd.isna(out["DAYS_EMPLOYED"].iloc[0])
    assert out["DAYS_EMPLOYED"].iloc[1] == -1000


def test_ratios_never_produce_inf(tiny_app):
    out = apply_all_stateless(tiny_app)
    num = out.select_dtypes("number").to_numpy(na_value=0.0)
    assert not np.isinf(num).any()
    assert pd.isna(out["credit_income_ratio"].iloc[1])  # income 0 → NaN, not inf


def test_domain_ratio_values(tiny_app):
    out = apply_all_stateless(tiny_app)
    assert out["credit_income_ratio"].iloc[0] == pytest.approx(2.0)
    assert out["payment_rate"].iloc[0] == pytest.approx(5000 / 120000)
    # term = 120000/(12*5000) = 2y; row 3: 300000/(12*2000)=12.5 → capped at 7
    assert out["term_years"].iloc[0] == pytest.approx(2.0)
    assert out["term_years"].iloc[2] == pytest.approx(7.0)


def test_missing_flags_and_ext_source_combos(tiny_app):
    out = apply_all_stateless(tiny_app)
    assert out["EXT_SOURCE_1_is_missing"].tolist() == [1, 0, 0]
    assert out["ext_source_n_missing"].tolist() == [1, 1, 1]
    assert out["ext_source_mean"].iloc[0] == pytest.approx((0.3 + 0.4) / 2)
    assert out["ext_source_min"].iloc[2] == pytest.approx(0.6)


# ── aggregations ───────────────────────────────────────────────────────────
def test_bureau_aggregation_hand_computed():
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [11, 12, 21],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "CREDIT_TYPE": ["Credit card", "Consumer credit", "Consumer credit"],
            "CREDIT_DAY_OVERDUE": [0, 30, 0],
            "AMT_CREDIT_SUM": [10000.0, 20000.0, 5000.0],
            "AMT_CREDIT_SUM_DEBT": [4000.0, 0.0, np.nan],
            "AMT_CREDIT_SUM_LIMIT": [10000.0, 0.0, 0.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 0.0],
            "AMT_CREDIT_MAX_OVERDUE": [np.nan, 500.0, np.nan],
            "CNT_CREDIT_PROLONG": [0, 1, 0],
            "DAYS_CREDIT": [-100, -900, -50],
            "AMT_ANNUITY": [np.nan, np.nan, np.nan],
        }
    )
    agg = aggregate_bureau(bureau).set_index("SK_ID_CURR")
    assert agg.loc[1, "bureau_n_loans"] == 2
    assert agg.loc[1, "bureau_n_active"] == 1
    assert agg.loc[1, "bureau_debt_sum"] == pytest.approx(4000.0)
    assert agg.loc[1, "bureau_days_overdue_max"] == 30
    assert agg.loc[1, "bureau_days_credit_max"] == -100  # most recent
    assert agg.loc[1, "bureau_utilization"] == pytest.approx(4000 / 30000)
    assert agg.loc[1, "bureau_share_card"] == pytest.approx(0.5)


def test_bureau_balance_dpd_mapping():
    bb = pd.DataFrame(
        {
            "SK_ID_BUREAU": [11, 11, 11, 12],
            "MONTHS_BALANCE": [0, -1, -2, 0],
            "STATUS": ["0", "2", "C", "X"],
        }
    )
    per_loan = aggregate_bureau_balance(bb).set_index("SK_ID_BUREAU")
    assert per_loan.loc[11, "bb_max_dpd_bucket"] == 2
    assert per_loan.loc[11, "bb_share_dpd_months"] == pytest.approx(1 / 3)
    assert per_loan.loc[12, "bb_max_dpd_bucket"] == 0  # X carries no DPD info


def test_prev_application_refusal_share():
    prev = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 1, 2],
            "SK_ID_PREV": [101, 102, 103, 201],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Refused", "Approved"],
            "NAME_CONTRACT_TYPE": ["Cash loans"] * 4,
            "AMT_APPLICATION": [10000.0, 20000.0, 15000.0, 8000.0],
            "AMT_CREDIT": [9000.0, 0.0, 15000.0, 8000.0],
            "DAYS_DECISION": [-100, -400, -700, -30],
            "CNT_PAYMENT": [12.0, 24.0, np.nan, 6.0],
            "RATE_DOWN_PAYMENT": [0.1, np.nan, 0.2, 0.0],
        }
    )
    agg = aggregate_previous_application(prev).set_index("SK_ID_CURR")
    assert agg.loc[1, "prev_refusal_share"] == pytest.approx(2 / 3)
    assert agg.loc[1, "prev_days_decision_max"] == -100
    # AMT_CREDIT==0 → requested_vs_granted NaN (not inf), excluded from mean
    assert agg.loc[1, "prev_req_vs_granted_mean"] == pytest.approx(
        np.nanmean([10000 / 9000, np.nan, 1.0])
    )


def test_join_nan_policy_opposite_history_flags(tiny_app):
    """EDA #11 vs #12: 'no history' means opposite things per table."""
    bureau_agg = aggregate_bureau(
        pd.DataFrame(
            {
                "SK_ID_CURR": [1], "SK_ID_BUREAU": [11], "CREDIT_ACTIVE": ["Active"],
                "CREDIT_TYPE": ["Consumer credit"], "CREDIT_DAY_OVERDUE": [0],
                "AMT_CREDIT_SUM": [1000.0], "AMT_CREDIT_SUM_DEBT": [500.0],
                "AMT_CREDIT_SUM_LIMIT": [0.0], "AMT_CREDIT_SUM_OVERDUE": [0.0],
                "AMT_CREDIT_MAX_OVERDUE": [np.nan], "CNT_CREDIT_PROLONG": [0],
                "DAYS_CREDIT": [-10], "AMT_ANNUITY": [np.nan],
            }
        )
    )
    prev_agg = aggregate_previous_application(
        pd.DataFrame(
            {
                "SK_ID_CURR": [2], "SK_ID_PREV": [201],
                "NAME_CONTRACT_STATUS": ["Approved"], "NAME_CONTRACT_TYPE": ["Cash loans"],
                "AMT_APPLICATION": [1.0], "AMT_CREDIT": [1.0], "DAYS_DECISION": [-1],
                "CNT_PAYMENT": [1.0], "RATE_DOWN_PAYMENT": [0.0],
            }
        )
    )
    out = join_aggregates(tiny_app, bureau_agg, prev_agg)
    r3 = out[out.SK_ID_CURR == 3].iloc[0]  # no history in either table
    assert r3["has_bureau_history"] == 0 and r3["has_prev_history"] == 0
    assert r3["bureau_n_loans"] == 0          # count → 0 (factually zero loans)
    assert pd.isna(r3["bureau_days_credit_max"])  # max of nothing → NaN
    r1 = out[out.SK_ID_CURR == 1].iloc[0]
    assert r1["has_bureau_history"] == 1 and r1["has_prev_history"] == 0


# ── WOE ────────────────────────────────────────────────────────────────────
def test_woe_hand_computed_two_bins():
    """2 bins, counts chosen so WOE is checkable on paper (with 0.5 smoothing)."""
    x = pd.Series([1.0] * 50 + [10.0] * 50)
    y = pd.Series([0] * 45 + [1] * 5 + [0] * 25 + [1] * 25)  # bin1: 45g/5b, bin2: 25g/25b
    b = WOEBinner(n_bins=2).fit(pd.DataFrame({"x": x}), y)
    g1, b1, g2, b2 = 45.5, 5.5, 25.5, 25.5
    G, B = g1 + g2, b1 + b2
    expected_woe1 = np.log((g1 / G) / (b1 / B))
    assert b.numeric_bins_["x"]["woes"][0] == pytest.approx(expected_woe1, rel=1e-6)
    expected_iv = ((g1 / G - b1 / B) * expected_woe1
                   + (g2 / G - b2 / B) * np.log((g2 / G) / (b2 / B)))
    assert b.iv_["x"] == pytest.approx(expected_iv, rel=1e-6)


def test_woe_nan_gets_own_bin_and_transform_never_nan():
    rng = np.random.RandomState(0)
    x = pd.Series(rng.uniform(0, 1, 1000))
    x.iloc[:200] = np.nan
    y = pd.Series((rng.uniform(0, 1, 1000) < 0.2).astype(int))
    b = WOEBinner(n_bins=4).fit(pd.DataFrame({"x": x}), y)
    t = b.transform(pd.DataFrame({"x": x}))
    assert t["woe_x"].notna().all()


def test_woe_unseen_category_maps_to_other_not_crash():
    x = pd.Series(["a"] * 500 + ["b"] * 490 + ["rare"] * 10)
    y = pd.Series([0] * 450 + [1] * 50 + [0] * 480 + [1] * 10 + [0] * 10)
    b = WOEBinner(min_bin_frac=0.02).fit(pd.DataFrame({"x": x}), y)
    t = b.transform(pd.DataFrame({"x": pd.Series(["a", "NEVER_SEEN", None])}))
    assert t["woe_x"].notna().all()
    assert t["woe_x"].iloc[1] == pytest.approx(b.categorical_bins_["x"]["other_woe"])


def test_iv_screen_flags_leaky_feature():
    """A near-perfect separator must land ABOVE the 0.5 audit ceiling."""
    rng = np.random.RandomState(1)
    y = pd.Series((rng.uniform(0, 1, 3000) < 0.08).astype(int))
    leak = y * 10 + rng.normal(0, 0.1, 3000)  # basically the target
    b = WOEBinner(n_bins=5).fit(pd.DataFrame({"leak": leak}), y)
    assert b.iv_["leak"] > 0.5
    assert "leak" not in b.select_by_iv(0.02, 0.5)


# ── selection ──────────────────────────────────────────────────────────────
def test_correlation_dropper_keeps_predictive_twin():
    rng = np.random.RandomState(0)
    n = 5000
    y = pd.Series((rng.uniform(size=n) < 0.2).astype(int))
    strong = y + rng.normal(0, 0.5, n)          # predictive
    clone = strong * 3 + 1                       # rank-identical copy (|spearman| = 1)
    noise = pd.Series(rng.normal(size=n))
    X = pd.DataFrame({"strong": strong, "clone": clone, "noise": noise})
    cd = CorrelationDropper(threshold=0.98).fit(X, y)
    assert len(cd.dropped_) == 1
    assert cd.dropped_[0] in ("strong", "clone")  # one twin gone, one kept
    assert "noise" not in cd.dropped_


def test_near_constant_dropper():
    X = pd.DataFrame({"const": [1] * 999 + [2], "ok": list(range(1000))})
    nc = NearConstantDropper(max_dominant_share=0.995).fit(X)
    assert nc.dropped_ == ["const"]
    assert list(nc.transform(X).columns) == ["ok"]


# ── contracts ──────────────────────────────────────────────────────────────
def test_build_excludes_protected_attribute(tiny_app):
    """Gender must be quarantined at BUILD time, not at training time."""
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1], "SK_ID_BUREAU": [11], "CREDIT_ACTIVE": ["Active"],
            "CREDIT_TYPE": ["Consumer credit"], "CREDIT_DAY_OVERDUE": [0],
            "AMT_CREDIT_SUM": [1.0], "AMT_CREDIT_SUM_DEBT": [0.0],
            "AMT_CREDIT_SUM_LIMIT": [0.0], "AMT_CREDIT_SUM_OVERDUE": [0.0],
            "AMT_CREDIT_MAX_OVERDUE": [np.nan], "CNT_CREDIT_PROLONG": [0],
            "DAYS_CREDIT": [-1], "AMT_ANNUITY": [np.nan],
        }
    )
    bb = pd.DataFrame({"SK_ID_BUREAU": [11], "MONTHS_BALANCE": [0], "STATUS": ["0"]})
    prev = pd.DataFrame(
        {
            "SK_ID_CURR": [1], "SK_ID_PREV": [101],
            "NAME_CONTRACT_STATUS": ["Approved"], "NAME_CONTRACT_TYPE": ["Cash loans"],
            "AMT_APPLICATION": [1.0], "AMT_CREDIT": [1.0], "DAYS_DECISION": [-1],
            "CNT_PAYMENT": [1.0], "RATE_DOWN_PAYMENT": [0.0],
        }
    )
    from src.features.build import build_feature_matrix

    features, audit = build_feature_matrix(tiny_app, bureau, bb, prev)
    assert "CODE_GENDER" not in features.columns
    assert "CODE_GENDER" in audit.columns
    assert len(features) == len(tiny_app)


# ── serving cast (pandas-4-safe categorical contract) ──────────────────────
def test_cast_with_contract_unseen_to_nan_no_deprecation():
    """The naive astype(CategoricalDtype) raises under pandas 4 on unseen
    values; the helper must handle them silently (unseen → NaN) and count
    them — with deprecations promoted to errors."""
    import warnings

    from src.features.serving import cast_with_contract

    df = pd.DataFrame({"OCC": ["Laborers", "NEVER_SEEN", None, "Managers"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, unseen = cast_with_contract(df, {"OCC": ["Laborers", "Managers"]})
    assert unseen == {"OCC": 1}
    assert list(out["OCC"].cat.categories) == ["Laborers", "Managers"]
    assert out["OCC"].isna().tolist() == [False, True, True, False]


def test_cast_with_contract_preserves_seen_values_and_level_order():
    from src.features.serving import cast_with_contract

    df = pd.DataFrame({"OCC": ["Managers", "Laborers"]})
    out, unseen = cast_with_contract(df, {"OCC": ["Laborers", "Managers"]})
    assert unseen["OCC"] == 0
    assert out["OCC"].tolist() == ["Managers", "Laborers"]
    assert list(out["OCC"].cat.categories) == ["Laborers", "Managers"]
