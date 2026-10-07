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
    CURE_RATE,
    EAD_FACTOR_CASH,
    EAD_FACTOR_REVOLVING,
    LGD_CASH,
    LGD_REVOLVING,
    M_REVOLVING,
    R_NET_CASH,
    SENSITIVITY_CURE_RATE,
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
    "cure_rate": SENSITIVITY_CURE_RATE,
}

# Admissible override ranges = the span of each frozen sensitivity grid.
# The demo sliders and /score cost_overrides may explore inside these and
# nowhere else (sensitivity exploration, never new economics).
OVERRIDE_BOUNDS = {
    "lgd_cash": (min(SENSITIVITY_LGD_CASH), max(SENSITIVITY_LGD_CASH)),
    "lgd_revolving": (min(SENSITIVITY_LGD_REVOLVING), max(SENSITIVITY_LGD_REVOLVING)),
    "ead_cash": (min(SENSITIVITY_EAD_CASH), max(SENSITIVITY_EAD_CASH)),
    "ead_revolving": (min(SENSITIVITY_EAD_REVOLVING), max(SENSITIVITY_EAD_REVOLVING)),
    "r_net_cash": (min(SENSITIVITY_R_NET_CASH), max(SENSITIVITY_R_NET_CASH)),
    "m_revolving": (min(SENSITIVITY_M_REVOLVING), max(SENSITIVITY_M_REVOLVING)),
    "cure_rate": (min(SENSITIVITY_CURE_RATE), max(SENSITIVITY_CURE_RATE)),
}


def validate_overrides(overrides: dict | None) -> dict:
    """Drop None entries; raise ValueError for unknown names or values
    outside the frozen grid span. Returns the overrides actually applied."""
    used = {}
    for name, val in (overrides or {}).items():
        if val is None:
            continue
        if name not in OVERRIDE_BOUNDS:
            raise ValueError(f"unknown cost override '{name}' "
                             f"(known: {sorted(OVERRIDE_BOUNDS)})")
        lo, hi = OVERRIDE_BOUNDS[name]
        if not (lo <= float(val) <= hi):
            raise ValueError(f"{name}={val} outside the frozen sensitivity grid [{lo}, {hi}]")
        used[name] = float(val)
    return used

_RULE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config",
                          "term_fallback_rule.json")


def attach_cost_params(df: pd.DataFrame, overrides: dict | None = None) -> pd.DataFrame:
    """
    Per-applicant m(x), LGD(x), ead(x), t*(x) from contract type and term.

    THE single implementation of the cost model (the former parallel
    src/decisioning/cost_model.py was deleted — two implementations kept in
    step by a test is a divergence waiting to happen).

    Applies the FROZEN term fallback (config/term_fallback_rule.json) for
    rows with missing term_years, flagging them (term_fallback_used).
    Requires columns: NAME_CONTRACT_TYPE, term_years, AMT_CREDIT.

    overrides: optional {lgd_cash, lgd_revolving, ead_cash, ead_revolving,
    r_net_cash, m_revolving, cure_rate}, each bounded by its frozen grid
    (validate_overrides). None/{} reproduces the deployed constants exactly.

    Cure rate c (METHODOLOGY §1): TARGET is early delinquency, LGD/EAD are
    charge-off severities, so the expected loss on a TARGET=1 approval is
    (1-c)·LGD·EAD·A and t*(x) = m / (m + (1-c)·LGD·EAD). Cured borrowers are
    booked at zero (conservative: they earn no margin). Deployed c = 0.
    """
    ov = validate_overrides(overrides)
    out = df.copy()
    ct = out["NAME_CONTRACT_TYPE"].astype(str)
    is_cash = (ct == "Cash loans").values

    rule = json.load(open(_RULE_PATH))["dev_median_term_years_by_contract"]
    out["term_fallback_used"] = out["term_years"].isna().astype("int8")
    out["term_years_eff"] = out["term_years"].fillna(ct.map(rule))
    assert out["term_years_eff"].notna().all(), "term fallback failed to cover a row"

    lgd_map = {"Cash loans": ov.get("lgd_cash", LGD["Cash loans"]),
               "Revolving loans": ov.get("lgd_revolving", LGD["Revolving loans"])}
    ead_map = {"Cash loans": ov.get("ead_cash", EAD_FACTOR["Cash loans"]),
               "Revolving loans": ov.get("ead_revolving", EAD_FACTOR["Revolving loans"])}
    out["lgd"] = ct.map(lgd_map).astype(float)
    out["ead_factor"] = ct.map(ead_map).astype(float)

    m = np.where(
        is_cash,
        ov.get("r_net_cash", R_NET_CASH) * out["term_years_eff"].values * AMORTIZATION_FACTOR,
        ov.get("m_revolving", M_REVOLVING),
    )
    cure = ov.get("cure_rate", CURE_RATE)
    out["margin"] = m
    out["cure_rate"] = cure
    out["loss_rate"] = out["lgd"].values * out["ead_factor"].values * (1.0 - cure)
    out["t_star"] = m / (m + out["loss_rate"].values)
    return out


def realized_profit(
    approved: np.ndarray, y: np.ndarray, cost_df: pd.DataFrame
) -> float:
    """
    Frozen ledger, realized outcomes:
      approved & good      -> + margin * AMT_CREDIT
      approved & defaulted -> - lgd * ead_factor * (1 - cure) * AMT_CREDIT
      rejected             ->   0
    """
    a = np.asarray(approved, dtype=bool)
    amt = cost_df["AMT_CREDIT"].values
    gain = cost_df["margin"].values * amt
    loss = cost_df["loss_rate"].values * amt  # LGD·EAD·(1-cure); cure=0 deployed
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


# ── cross-fitted comparators (development evidence, never holdout) ─────────
FLAT_GRID = np.round(np.arange(0.01, 0.501, 0.005), 3)


def crossfit_flat_approvals(p_cal: np.ndarray, y: np.ndarray, cost_df: pd.DataFrame,
                            folds: np.ndarray, segments: np.ndarray | None = None,
                            grid: np.ndarray = FLAT_GRID) -> np.ndarray:
    """
    Approvals from a FITTED flat-threshold policy, scored honestly: for each
    fold k the threshold(s) are chosen to maximise the ledger on the other
    folds, then applied to fold k. With `segments` (an integer label per
    row) one threshold is fitted per segment — the data-driven competitor
    to t*(x) that knows contract type / term bucket but no loan economics.

    Without cross-fitting, the best flat threshold is selected on the same
    rows it is scored on, which flatters the flat policy (an oracle).
    """
    seg = np.zeros(len(p_cal), dtype=int) if segments is None else np.asarray(segments)
    approved = np.zeros(len(p_cal), dtype=bool)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        for s in np.unique(seg):
            in_s_tr = tr & (seg == s)
            best = max(grid, key=lambda t, m=in_s_tr: realized_profit(
                (p_cal < t) & m, y, cost_df))
            approved[te & (seg == s)] = p_cal[te & (seg == s)] < best
    return approved


def cash_term_segments(cost_df: pd.DataFrame, n_buckets: int = 5) -> np.ndarray:
    """Segment label: revolving = 0, cash split into term quantile buckets 1..n.
    (Quantile edges from all rows' term — labels only, no outcome used.)"""
    rev = (cost_df["NAME_CONTRACT_TYPE"].astype(str) == "Revolving loans").values
    term = cost_df["term_years_eff"].values
    edges = np.quantile(term[~rev], np.linspace(0, 1, n_buckets + 1)[1:-1])
    return np.where(rev, 0, 1 + np.searchsorted(edges, term))
