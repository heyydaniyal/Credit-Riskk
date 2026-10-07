"""
Phase 4c/4d — the decision layer: frozen cost model, t*(x), profit ledger.

Every constant here is an ILLUSTRATIVE ECONOMIC ASSUMPTION, frozen in the
spec before any result existed (see spec 4c and
config/cost_evaluation_convention.json).  They are not estimated from
data — this dataset cannot estimate them (no recovery data, no
balance-at-default, no pricing).  They occupy the slots where a bank's
LGD/EAD/pricing models plug in.  DO NOT revise them after seeing results.

Decision rule (Elkan 2001, derived in METHODOLOGY):
    approve  iff  p_cal * LGD(x) * ead(x)  <  (1 - p_cal) * m(x)
    t*(x) = m(x) / (m(x) + LGD(x) * ead(x))
AMT_CREDIT cancels from the threshold but NOT from the ledger — costs are
euro-weighted.

Ledger (frozen convention — realized outcomes, never p_cal):
    approved good       -> + m(x) * AMT_CREDIT
    approved defaulter  -> - LGD(x) * ead(x) * AMT_CREDIT
    rejected            ->   0
"""

import json
import os

import numpy as np
import pandas as pd

# ── frozen illustrative assumptions (spec 4c) ──────────────────────────────
# SOURCED FROM config/constants.py so the "frozen in config" claim is
# structurally true: editing the config now changes the deployed system.
# Verified behaviour-preserving — every value equals the former hardcoded
# literal (config M_REVOLVING was rounded to the exact 0.18 to match).
from config.constants import (  # noqa: E402
    AMORTIZATION_FACTOR,
    EAD_FACTOR_CASH,
    EAD_FACTOR_REVOLVING,
    LGD_CASH,
    LGD_REVOLVING,
    M_REVOLVING,
    R_NET_CASH,
    SENSITIVITY_EAD_CASH,
    SENSITIVITY_EAD_REVOLVING,
    SENSITIVITY_LGD_CASH,
    SENSITIVITY_LGD_REVOLVING,
    SENSITIVITY_M_REVOLVING,
    SENSITIVITY_R_NET_CASH,
)

LGD = {"Cash loans": LGD_CASH, "Revolving loans": LGD_REVOLVING}
EAD_FACTOR = {"Cash loans": EAD_FACTOR_CASH, "Revolving loans": EAD_FACTOR_REVOLVING}

SENSITIVITY_GRID = {
    "LGD_cash": SENSITIVITY_LGD_CASH,
    "LGD_revolving": SENSITIVITY_LGD_REVOLVING,
    "ead_cash": SENSITIVITY_EAD_CASH,
    "ead_revolving": SENSITIVITY_EAD_REVOLVING,
    "r_net_cash": SENSITIVITY_R_NET_CASH,
    "m_revolving": SENSITIVITY_M_REVOLVING,
}

_RULE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config",
                          "term_fallback_rule.json")


def attach_cost_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-applicant m(x), LGD(x), ead(x), t*(x) from contract type and term.

    Applies the FROZEN term fallback (config/term_fallback_rule.json) for
    rows with missing term_years, flagging them (term_fallback_used).
    Requires columns: NAME_CONTRACT_TYPE, term_years, AMT_CREDIT.
    """
    out = df.copy()
    ct = out["NAME_CONTRACT_TYPE"].astype(str)

    rule = json.load(open(_RULE_PATH))["dev_median_term_years_by_contract"]
    out["term_fallback_used"] = out["term_years"].isna().astype("int8")
    out["term_years_eff"] = out["term_years"].fillna(ct.map(rule))
    assert out["term_years_eff"].notna().all(), "term fallback failed to cover a row"

    out["lgd"] = ct.map(LGD).astype(float)
    out["ead_factor"] = ct.map(EAD_FACTOR).astype(float)

    is_cash = (ct == "Cash loans").values
    m = np.where(
        is_cash,
        R_NET_CASH * out["term_years_eff"].values * AMORTIZATION_FACTOR,
        M_REVOLVING,
    )
    out["margin"] = m
    out["t_star"] = m / (m + out["lgd"].values * out["ead_factor"].values)
    return out


def realized_profit(
    approved: np.ndarray, y: np.ndarray, cost_df: pd.DataFrame
) -> float:
    """
    Frozen ledger, realized outcomes:
      approved & good      -> + margin * AMT_CREDIT
      approved & defaulted -> - lgd * ead_factor * AMT_CREDIT
      rejected             ->   0
    """
    a = np.asarray(approved, dtype=bool)
    amt = cost_df["AMT_CREDIT"].values
    gain = cost_df["margin"].values * amt
    loss = cost_df["lgd"].values * cost_df["ead_factor"].values * amt
    return float(np.where(a, np.where(y == 1, -loss, gain), 0.0).sum())


def flat_policy(p_cal: np.ndarray, threshold: float) -> np.ndarray:
    return p_cal < threshold


def instance_policy(p_cal: np.ndarray, cost_df: pd.DataFrame) -> np.ndarray:
    return p_cal < cost_df["t_star"].values


def best_flat_threshold(
    p_cal: np.ndarray, y: np.ndarray, cost_df: pd.DataFrame,
    grid: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    The strongest flat competitor: single threshold maximizing the realized
    ledger on DEV OOF (frozen convention: selected once here, never
    re-optimized on holdout).  Returns (threshold, full sweep table).
    """
    if grid is None:
        grid = np.round(np.arange(0.01, 0.501, 0.005), 3)
    rows = [
        {"threshold": float(t),
         "profit": realized_profit(flat_policy(p_cal, t), y, cost_df),
         "approval_rate": float((p_cal < t).mean())}
        for t in grid
    ]
    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep["profit"].idxmax()]
    return float(best["threshold"]), sweep


def swap_set(
    a_new: np.ndarray, a_base: np.ndarray, y: np.ndarray, cost_df: pd.DataFrame
) -> pd.DataFrame:
    """2×2 swap-set vs a baseline policy: who moved, their default rates,
    and the net € effect per cell (spec 4d)."""
    rows = []
    for name, mask in [
        ("approved by both", a_new & a_base),
        ("swapped IN (new approves, base rejects)", a_new & ~a_base),
        ("swapped OUT (new rejects, base approves)", ~a_new & a_base),
        ("rejected by both", ~a_new & ~a_base),
    ]:
        n = int(mask.sum())
        rows.append({
            "cell": name,
            "n": n,
            "default_rate": float(y[mask].mean()) if n else np.nan,
            "net_eur_under_new": realized_profit(a_new & mask, y, cost_df),
            "net_eur_under_base": realized_profit(a_base & mask, y, cost_df),
        })
    return pd.DataFrame(rows)
