"""
Core tests — spec mandates 7-9 tests (Phase 6).
These cover the cost model, threshold math, and pipeline invariants.
More tests added as modules are built.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.constants import (
    CONTRACT_CASH, CONTRACT_REVOLVING,
    LGD_CASH, LGD_REVOLVING,
    EAD_FACTOR_CASH, EAD_FACTOR_REVOLVING,
)
from src.decisioning.cost_model import (
    compute_threshold,
    compute_term_years,
    compute_margin,
    compute_loss,
    sanity_check_3yr_cash,
)


class TestCostModel:
    """Threshold math reproduces t*(x) = m(x)/(m(x)+LGD·ead)."""

    def test_sanity_anchor_3yr_cash(self):
        """3-year cash loan → t* ≈ 0.112."""
        t = sanity_check_3yr_cash()
        assert abs(t - 0.1119) < 0.001

    def test_threshold_formula(self):
        """t*(x) = m(x) / (m(x) + LGD(x) × ead_factor(x)) for arbitrary inputs."""
        ct = pd.Series([CONTRACT_CASH, CONTRACT_REVOLVING])
        amt = pd.Series([100_000, 50_000])
        ann = pd.Series([100_000 / (12 * 5), 50_000 / (12 * 2)])  # 5yr, 2yr

        thresholds = compute_threshold(ct, amt, ann)
        margins = compute_margin(ct, amt, ann)
        lgds, eads = compute_loss(ct)

        for i in range(len(ct)):
            expected = margins.iloc[i] / (margins.iloc[i] + lgds.iloc[i] * eads.iloc[i])
            assert abs(thresholds.iloc[i] - expected) < 1e-8

    def test_amortization_applied_cash(self):
        """Cash margin uses /2 amortization factor."""
        m = compute_margin(
            pd.Series([CONTRACT_CASH]),
            pd.Series([120_000]),
            pd.Series([120_000 / (12 * 4)]),  # 4-year term
        )
        # m = 0.05 × 4 / 2 = 0.10
        assert abs(m.iloc[0] - 0.10) < 1e-8

    def test_amortization_not_applied_revolving(self):
        """Revolving uses flat composite (no amortization division)."""
        m = compute_margin(
            pd.Series([CONTRACT_REVOLVING]),
            pd.Series([50_000]),
            pd.Series([50_000 / (12 * 3)]),  # term irrelevant for revolving
        )
        # m_rev = 0.10 × 3 × 0.6 = 0.18
        assert abs(m.iloc[0] - 0.18) < 1e-8

    def test_term_capping(self):
        """Term is capped to [0.5, 7] years."""
        # Very short term
        t_short = compute_term_years(pd.Series([1_000]), pd.Series([10_000]))
        assert t_short.iloc[0] == 0.5
        # Very long term
        t_long = compute_term_years(pd.Series([1_000_000]), pd.Series([1_000]))
        assert t_long.iloc[0] == 7.0

    def test_lgd_ead_by_contract(self):
        """LGD/EAD differ by contract type."""
        lgd, ead = compute_loss(pd.Series([CONTRACT_CASH, CONTRACT_REVOLVING]))
        assert lgd.iloc[0] == LGD_CASH
        assert lgd.iloc[1] == LGD_REVOLVING
        assert ead.iloc[0] == EAD_FACTOR_CASH
        assert ead.iloc[1] == EAD_FACTOR_REVOLVING

    def test_threshold_range(self):
        """Thresholds fall in expected range ~0.02 to ~0.23."""
        # Short cash (0.5yr) → lowest threshold
        t_low = compute_threshold(
            pd.Series([CONTRACT_CASH]),
            pd.Series([10_000]),
            pd.Series([10_000 / (12 * 0.5)]),
        )
        # 7-year cash → highest cash threshold
        t_high = compute_threshold(
            pd.Series([CONTRACT_CASH]),
            pd.Series([100_000]),
            pd.Series([100_000 / (12 * 7)]),
        )
        # Revolving
        t_rev = compute_threshold(
            pd.Series([CONTRACT_REVOLVING]),
            pd.Series([50_000]),
            pd.Series([50_000 / (12 * 3)]),
        )

        assert 0.01 < t_low.iloc[0] < 0.05, f"Short cash threshold {t_low.iloc[0]} out of range"
        assert 0.15 < t_high.iloc[0] < 0.30, f"Long cash threshold {t_high.iloc[0]} out of range"
        assert 0.10 < t_rev.iloc[0] < 0.25, f"Revolving threshold {t_rev.iloc[0]} out of range"


class TestPSI:
    """PSI = 0 for identical distributions."""

    def test_psi_identical(self):
        """PSI of a distribution against itself should be 0."""
        from src.monitoring.psi import compute_psi

        x = np.random.RandomState(42).normal(0, 1, 10_000)
        psi = compute_psi(x, x, bins=10)
        assert abs(psi) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
