"""
Core tests — the spec's mandated contracts for the cost model, threshold
math and PSI. They exercise THE production decision layer
(src/models/decision.py) and the single term definition
(src/features/stateless.implied_term_years) — the former parallel
src/decisioning/cost_model.py was deleted, so there is nothing for a
duplicate implementation to drift from.
"""

import numpy as np
import pandas as pd
import pytest

from config.constants import (
    CONTRACT_CASH,
    CONTRACT_REVOLVING,
    EAD_FACTOR_CASH,
    EAD_FACTOR_REVOLVING,
    LGD_CASH,
    LGD_REVOLVING,
)
from src.features.stateless import implied_term_years
from src.models.decision import attach_cost_params


def _cost(contract: list[str], amt: list[float], annuity: list[float], **ov) -> pd.DataFrame:
    a, n = pd.Series(amt, dtype=float), pd.Series(annuity, dtype=float)
    df = pd.DataFrame({"NAME_CONTRACT_TYPE": contract, "AMT_CREDIT": a,
                       "term_years": implied_term_years(a, n)})
    return attach_cost_params(df, overrides=ov or None)


class TestCostModel:
    """Threshold math reproduces t*(x) = m(x)/(m(x)+LGD·ead)."""

    def test_sanity_anchor_3yr_cash(self):
        """Spec anchor: 3-year cash loan → t* = 0.075/0.670 ≈ 0.1119."""
        t = _cost([CONTRACT_CASH], [100_000], [100_000 / (12 * 3)])["t_star"].iloc[0]
        assert abs(t - 0.075 / (0.075 + 0.70 * 0.85)) < 1e-12

    def test_threshold_formula(self):
        """t*(x) = m(x) / (m(x) + LGD(x) × ead_factor(x)) for arbitrary inputs."""
        c = _cost([CONTRACT_CASH, CONTRACT_REVOLVING], [100_000, 50_000],
                  [100_000 / (12 * 5), 50_000 / (12 * 2)])
        expected = c["margin"] / (c["margin"] + c["lgd"] * c["ead_factor"])
        assert np.allclose(c["t_star"], expected, atol=1e-12)

    def test_amortization_applied_cash(self):
        """Cash margin uses the /2 amortization factor: 0.05 × 4 / 2 = 0.10."""
        m = _cost([CONTRACT_CASH], [120_000], [120_000 / (12 * 4)])["margin"].iloc[0]
        assert abs(m - 0.10) < 1e-12

    def test_amortization_not_applied_revolving(self):
        """Revolving uses the flat composite 0.10 × 3 × 0.6 = 0.18 (term irrelevant)."""
        m = _cost([CONTRACT_REVOLVING], [50_000], [50_000 / (12 * 3)])["margin"].iloc[0]
        assert m == 0.18

    def test_term_capping(self):
        """Term is capped to [0.5, 7] years; zero/missing annuity → NaN."""
        t = implied_term_years(pd.Series([1_000.0, 1_000_000.0, 1_000.0, 1_000.0]),
                               pd.Series([10_000.0, 1_000.0, 0.0, np.nan]))
        assert t.iloc[0] == 0.5 and t.iloc[1] == 7.0
        assert t.iloc[2:].isna().all()

    def test_lgd_ead_by_contract(self):
        """LGD/EAD differ by contract type and come from config."""
        c = _cost([CONTRACT_CASH, CONTRACT_REVOLVING], [1e5, 1e5], [1e4, 1e4])
        assert c["lgd"].tolist() == [LGD_CASH, LGD_REVOLVING]
        assert c["ead_factor"].tolist() == [EAD_FACTOR_CASH, EAD_FACTOR_REVOLVING]

    def test_threshold_range(self):
        """Spec range: ~0.02 (short cash) to ~0.23 (7-year cash); revolving ≈ 0.17."""
        c = _cost([CONTRACT_CASH, CONTRACT_CASH, CONTRACT_REVOLVING],
                  [10_000, 100_000, 50_000],
                  [10_000 / (12 * 0.5), 100_000 / (12 * 7), 50_000 / (12 * 3)])
        lo, hi, rev = c["t_star"].tolist()
        assert 0.01 < lo < 0.05 and 0.15 < hi < 0.30 and 0.10 < rev < 0.25

    def test_cure_rate_axis(self):
        """Cure rate c scales the loss term: t* = m / (m + (1-c)·LGD·EAD).
        METHODOLOGY's table: 3y cash anchor 0.1119 → 0.1526 (30%) → 0.2013 (50%)."""
        args = ([CONTRACT_CASH], [100_000], [100_000 / (12 * 3)])
        t = [_cost(*args, cure_rate=c)["t_star"].iloc[0] for c in (0.0, 0.30, 0.50)]
        assert [round(v, 4) for v in t] == [0.1119, 0.1526, 0.2013]

    def test_overrides_bounded_by_frozen_grids(self):
        """Overrides inside the grid move t*; outside it (or unknown) they raise."""
        args = ([CONTRACT_CASH], [100_000], [100_000 / (12 * 3)])
        base = _cost(*args)["t_star"].iloc[0]
        assert _cost(*args, r_net_cash=0.08)["t_star"].iloc[0] > base
        with pytest.raises(ValueError):
            _cost(*args, lgd_cash=0.99)
        with pytest.raises(ValueError):
            _cost(*args, made_up_param=0.5)


class TestPSI:
    """PSI = 0 for identical distributions."""

    def test_psi_identical(self):
        from src.monitoring.psi import compute_psi

        x = np.random.RandomState(42).normal(0, 1, 10_000)
        assert abs(compute_psi(x, x, bins=10)) < 1e-10
