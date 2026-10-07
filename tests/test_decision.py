"""
Phase 4 tests — calibration and decision-layer contracts.

What each block protects against:
  calibration   non-monotone or out-of-range calibrated probabilities,
                NaN on out-of-domain scores (the clip pin), in-sample
                metric grading, broken PSI
  decision      wrong t*(x) math (incl. the spec's 0.112 anchor), the
                amortization factor leaking into revolving, fallback rule
                not applied, ledger sign/magnitude errors, best-flat
                selected off-convention
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.models.calibration import (
    brier,
    cross_fitted_calibrated,
    ece_and_reliability,
    fit_deployed_calibrator,
    psi,
)
from src.models.decision import (
    attach_cost_params,
    best_flat_threshold,
    flat_policy,
    realized_profit,
)


@pytest.fixture(scope="module")
def miscal():
    rng = np.random.RandomState(0)
    p_raw = rng.uniform(0, 0.5, 5000)
    y = rng.binomial(1, np.clip(p_raw * 1.5, 0, 1))  # miscalibrated by design
    return p_raw, y


# ── calibration ────────────────────────────────────────────────────────────
def test_isotonic_monotone_bounded_and_clips(miscal):
    p_raw, y = miscal
    iso = fit_deployed_calibrator(p_raw, y)
    pc = iso.predict(p_raw)
    assert (pc >= 0).all() and (pc <= 1).all()
    order = np.argsort(p_raw)
    assert (np.diff(pc[order]) >= -1e-12).all()
    # the clip pin: scores beyond the training range must NOT become NaN
    out = iso.predict(np.array([0.99, -0.5]))
    assert not np.isnan(out).any()


def test_cross_fit_improves_brier_on_miscalibrated_input(miscal):
    p_raw, y = miscal
    oof = pd.DataFrame(
        {"p_raw": p_raw, "TARGET": y,
         "fold": np.random.RandomState(1).randint(0, 5, len(y))}
    )
    pcx = cross_fitted_calibrated(oof)
    assert brier(y, pcx) < brier(y, p_raw)
    ece, rel = ece_and_reliability(y, pcx)
    assert len(rel) == 10 and 0 <= ece < 1


def test_psi_identity_and_shift():
    rng = np.random.RandomState(0)
    x = rng.normal(size=20000)
    assert psi(x, x) == 0.0
    assert psi(x, x + 1.0) > 0.25  # a full-sigma shift must trip the retrain band


# ── decision layer ─────────────────────────────────────────────────────────
@pytest.fixture
def three_loans():
    return attach_cost_params(
        pd.DataFrame(
            {
                "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans", "Revolving loans"],
                "term_years": [3.0, np.nan, 5.0],
                "AMT_CREDIT": [100000.0, 50000.0, 20000.0],
            }
        )
    )


def test_t_star_anchor_and_revolving(three_loans):
    c = three_loans
    assert c.t_star.iloc[0] == pytest.approx(0.075 / 0.670, abs=1e-9)  # ≈ 0.1119
    # revolving ignores term entirely: t* = 0.18 / (0.18 + 0.85)
    assert c.t_star.iloc[2] == pytest.approx(0.18 / 1.03, abs=1e-9)


def test_term_fallback_applied_and_flagged(three_loans):
    c = three_loans
    assert c.term_fallback_used.tolist() == [0, 1, 0]
    assert c.term_years_eff.notna().all()


def test_amortization_only_for_cash(three_loans):
    c = three_loans
    assert c.margin.iloc[0] == pytest.approx(0.05 * 3.0 * 0.5)   # /2 applied
    assert c.margin.iloc[2] == pytest.approx(0.18)               # not applied


def test_ledger_hand_computed(three_loans):
    c = three_loans
    y = np.array([0, 1, 0])
    expected = 0.075 * 100000 - 0.70 * 0.85 * 50000 + 0.18 * 20000
    assert realized_profit(np.array([True] * 3), y, c) == pytest.approx(expected)
    assert realized_profit(np.array([False] * 3), y, c) == 0.0  # rejected → 0


def test_best_flat_rejects_the_defaulter(three_loans):
    c = three_loans
    y = np.array([0, 1, 0])
    p_cal = np.array([0.05, 0.9, 0.05])
    t_best, sweep = best_flat_threshold(p_cal, y, c)
    profit_at_best = realized_profit(flat_policy(p_cal, t_best), y, c)
    assert profit_at_best == pytest.approx(0.075 * 100000 + 0.18 * 20000)
    assert (sweep["profit"] <= profit_at_best + 1e-9).all()


# ── monotonicity gate ──────────────────────────────────────────────────────
def test_monotonicity_verifier_catches_violation_and_passes_clean():
    """The gate must catch a deliberately non-monotone model and pass a
    genuinely constrained one (basic method)."""
    import lightgbm as lgb

    from src.models.monotonicity import assert_monotone, verify_monotonicity

    rng = np.random.RandomState(0)
    n = 6000
    x = rng.uniform(-2, 2, n)
    z = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(np.sin(2 * x) + 0.5 * z)))).astype(int)
    df = pd.DataFrame({"x": x, "z": z})

    unconstrained = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 31, "seed": 0},
        lgb.Dataset(df, label=y), num_boost_round=150)
    rep_u = verify_monotonicity(unconstrained, df.sample(200, random_state=1), {"x": 1})
    assert rep_u["x"] > 1e-3, "sin(2x) target must violate monotonicity unconstrained"
    with pytest.raises(AssertionError):
        assert_monotone(rep_u)

    constrained = lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 31, "seed": 0,
         "monotone_constraints": [1, 0], "monotone_constraints_method": "basic"},
        lgb.Dataset(df, label=y), num_boost_round=150)
    rep_c = verify_monotonicity(constrained, df.sample(200, random_state=1), {"x": 1})
    assert_monotone(rep_c)  # zero violations — passes


# ── explainability contracts ───────────────────────────────────────────────
def test_score_scale_anchors_and_monotone():
    """Frozen scale: p at 19:1 good odds (p=0.05) → exactly 600; doubling
    the odds adds exactly PDO=50; lower PD → higher score; clipped."""
    from src.models.explain import pd_to_score

    assert pd_to_score(np.array([0.05]))[0] == pytest.approx(600.0, abs=1e-6)
    p_double_odds = 1 / (1 + 38.0)  # odds 38:1 = 2×19 → 650
    assert pd_to_score(np.array([p_double_odds]))[0] == pytest.approx(650.0, abs=1e-6)
    ps = np.linspace(0.001, 0.99, 50)
    s = pd_to_score(ps)
    assert (np.diff(s) <= 1e-9).all()          # higher PD → lower score
    assert s.min() >= 300 and s.max() <= 850


def test_reason_codes_sign_convention():
    """Pinned convention: reasons = largest POSITIVE contributions to the
    default log-odds. A feature that strongly increases risk must appear;
    a protective feature must not."""
    import lightgbm as lgb

    from src.models.explain import reason_codes

    rng = np.random.RandomState(0)
    n = 5000
    risky = rng.normal(size=n)        # ↑ risky → ↑ default
    protect = rng.normal(size=n)      # ↑ protect → ↓ default
    logit = -2.0 + 1.5 * risky - 1.5 * protect
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"risky": risky, "protect": protect})
    b = lgb.train({"objective": "binary", "verbosity": -1, "num_leaves": 15, "seed": 0},
                  lgb.Dataset(df, label=y), num_boost_round=120)

    hi = pd.DataFrame({"risky": [2.5], "protect": [-2.5]})  # both push toward default
    codes = reason_codes(b, hi, ["risky", "protect"], top_k=2)[0]
    assert {c["feature"] for c in codes} == {"risky", "protect"}
    assert all(c["contribution"] > 0 for c in codes)

    safe = pd.DataFrame({"risky": [-2.5], "protect": [2.5]})  # both push away
    codes_safe = reason_codes(b, safe, ["risky", "protect"], top_k=3)[0]
    assert codes_safe == [] or all(c["contribution"] > 0 for c in codes_safe)


def test_decision_layer_sources_config():
    """F1.1/F8.2 guard: the decision layer's constants ARE config's, so
    'frozen in config' is structurally true (one implementation, one source)."""
    from config import constants as C
    from src.models import decision as D

    assert D.LGD["Cash loans"] == C.LGD_CASH
    assert D.M_REVOLVING == C.M_REVOLVING
    assert D.SENSITIVITY_GRID["ead_revolving"] == C.SENSITIVITY_EAD_REVOLVING
    assert D.SENSITIVITY_GRID["cure_rate"] == C.SENSITIVITY_CURE_RATE
    assert D.OVERRIDE_BOUNDS["lgd_cash"] == (min(C.SENSITIVITY_LGD_CASH),
                                             max(C.SENSITIVITY_LGD_CASH))


def test_reason_codes_complete_mapping_and_no_prohibited_basis():
    """F11.2/F11.3: every deployed feature has reviewed plain language (no
    raw-column-name fallback), and a prohibited basis (marital status) is
    never emitted as an adverse reason even when it ranks in the top 3."""
    import json

    import lightgbm as lgb

    from config.constants import ARTIFACTS_DIR
    from src.features.serving import cast_with_contract
    from src.models.explain import (
        PLAIN_LANGUAGE,
        PROHIBITED_REASON_FEATURES,
        reason_codes,
    )

    # runs from the deployment bundle (in CI and inside the image); the demo
    # pool is 3,000 dev rows with every model feature
    feats = json.load(open(os.path.join(ARTIFACTS_DIR, "lgbm_features.json")))["features"]
    assert [f for f in feats if f not in PLAIN_LANGUAGE] == []  # full coverage

    levels = json.load(open(os.path.join(ARTIFACTS_DIR, "categorical_levels.json")))["levels"]
    pool = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "demo_pool.parquet"))
    dev, _ = cast_with_contract(pool.sample(500, random_state=0), levels)
    b = lgb.Booster(model_file=os.path.join(ARTIFACTS_DIR, "lgbm_constrained_fold0.txt"))
    codes = reason_codes(b, dev, feats, top_k=3)
    emitted = [c for row in codes for c in row]
    assert all(c["feature"] not in PROHIBITED_REASON_FEATURES for c in emitted)
    assert all(c["plain_language"] != c["feature"].replace("_", " ") for c in emitted)
    assert all(len(row) == 3 for row in codes)


def test_crossfit_flat_is_not_an_oracle():
    """A cross-fitted flat policy chooses each fold's threshold on the OTHER
    folds, so it can never beat the in-sample (oracle) best flat; with
    segments it can only add flexibility on the training folds."""
    from src.models.decision import (
        best_flat_threshold,
        cash_term_segments,
        crossfit_flat_approvals,
    )

    rng = np.random.RandomState(0)
    n = 4000
    p = rng.beta(1, 10, n)
    y = (rng.rand(n) < p).astype(int)
    df = pd.DataFrame({"NAME_CONTRACT_TYPE": np.where(rng.rand(n) < 0.1, "Revolving loans",
                                                      "Cash loans"),
                       "term_years": rng.uniform(0.7, 4.0, n),
                       "AMT_CREDIT": rng.uniform(5e4, 5e5, n)})
    cost = attach_cost_params(df)
    folds = np.arange(n) % 5
    _, sweep = best_flat_threshold(p, y, cost)
    cf = realized_profit(crossfit_flat_approvals(p, y, cost, folds), y, cost)
    assert cf <= sweep["profit"].max() + 1e-6
    seg = cash_term_segments(cost)
    assert set(np.unique(seg)) <= set(range(6)) and (seg[df.NAME_CONTRACT_TYPE == "Revolving loans"] == 0).all()
    a = crossfit_flat_approvals(p, y, cost, folds, seg)
    assert a.dtype == bool and len(a) == n
