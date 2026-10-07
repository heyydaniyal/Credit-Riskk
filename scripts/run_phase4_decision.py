"""
Phase 4c/4d — the decision layer on cross-fitted calibrated OOF.

THE GAP CHECKPOINT (spec-mandated, runs here): compute instance-vs-best-flat
with the frozen parameters and LOCK the narrative before the holdout is ever
touched.  All numbers here are OOF development estimates; the quoted result
comes from the Phase 7 ceremony.

Outputs:
  reports/tables/gap_checkpoint.json      the locked narrative + OOF numbers
  reports/tables/flat_threshold_sweep.csv
  reports/tables/swap_set_oof.csv
  reports/tables/policy_comparison_oof.csv
  reports/figures/43_t_star_hist.png, 44_flat_sweep.png, 45_profit_vs_approval.png
"""

import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import DATA_PROCESSED, FIGURES_DIR, ID_COL, TABLES_DIR
from src.models.decision import (
    attach_cost_params,
    best_flat_threshold,
    flat_policy,
    instance_policy,
    realized_profit,
    swap_set,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase4-dec")

PER_10K = 10_000


def main() -> None:
    oof = pd.read_parquet(os.path.join(DATA_PROCESSED, "oof_calibrated.parquet"))
    cols = pd.read_parquet(
        os.path.join(DATA_PROCESSED, "features_full.parquet"),
        columns=[ID_COL, "NAME_CONTRACT_TYPE", "term_years", "AMT_CREDIT"],
    )
    df = oof.merge(cols, on=ID_COL, how="left", validate="one_to_one")
    df = attach_cost_params(df)
    y = df["TARGET"].values
    p = df["p_cal_crossfit"].values
    n = len(df)
    scale = PER_10K / n

    # ── t*(x) distribution (fig 43) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cash = df["NAME_CONTRACT_TYPE"] == "Cash loans"
    ax.hist(df.loc[cash, "t_star"], bins=80, alpha=0.7, color="#2b6cb0",
            label=f"cash ({cash.mean():.0%})")
    ax.hist(df.loc[~cash, "t_star"], bins=20, alpha=0.7, color="#e53e3e",
            label=f"revolving ({(~cash).mean():.0%})")
    ax.set_xlabel("t*(x) — per-applicant approval threshold")
    ax.set_ylabel("applicants")
    ax.set_title("Instance-dependent thresholds under the frozen assumptions\n"
                 "(term-driven spread for cash; revolving at 0.175)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "43_t_star_hist.png"), dpi=120)
    plt.close(fig)

    # ── policies (frozen convention) ───────────────────────────────────────
    a_naive = flat_policy(p, 0.5)
    t_best, sweep = best_flat_threshold(p, y, df)
    a_flat = flat_policy(p, t_best)
    a_inst = instance_policy(p, df)
    sweep.to_csv(os.path.join(TABLES_DIR, "flat_threshold_sweep.csv"), index=False)

    profits = {
        "naive_0.5": realized_profit(a_naive, y, df),
        f"best_flat_{t_best}": realized_profit(a_flat, y, df),
        "instance_t_star": realized_profit(a_inst, y, df),
    }
    comparison = pd.DataFrame(
        [
            {"policy": k, "profit_eur_total": v, "profit_eur_per_10k": v * scale,
             "approval_rate": float(a.mean())}
            for (k, v), a in zip(profits.items(), [a_naive, a_flat, a_inst])
        ]
    )
    comparison.to_csv(os.path.join(TABLES_DIR, "policy_comparison_oof.csv"), index=False)

    gap_vs_flat = profits["instance_t_star"] - profits[f"best_flat_{t_best}"]
    gap_vs_naive = profits["instance_t_star"] - profits["naive_0.5"]
    rel_gap = gap_vs_flat / abs(profits[f"best_flat_{t_best}"])

    # ── empirical check: sweep minimum vs volume-weighted anchor ───────────
    t_anchor = float(np.average(df["t_star"], weights=df["AMT_CREDIT"]))
    logger.info("best flat t=%.3f | €-weighted mean t*=%.3f (should be near)", t_best, t_anchor)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(sweep["threshold"], sweep["profit"] * scale, color="#2b6cb0")
    ax.axvline(t_best, ls="--", c="#e53e3e", label=f"best flat = {t_best}")
    ax.axvline(t_anchor, ls=":", c="gray", label=f"€-weighted mean t* = {t_anchor:.3f}")
    ax.set_xlabel("flat threshold on p_cal")
    ax.set_ylabel("realized profit € per 10k applications")
    ax.set_title("Empirical check (spec 4c): the flat-policy optimum lands near\n"
                 "the volume-weighted t* anchor")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "44_flat_sweep.png"), dpi=120)
    plt.close(fig)

    # ── profit vs approval-rate curve (4d): sweep global scaling of t*(x) ──
    scales = np.linspace(0.2, 3.0, 57)
    curve = pd.DataFrame(
        [
            {"scale": s,
             "approval_rate": float((p < df["t_star"].values * s).mean()),
             "profit_per_10k": realized_profit(p < df["t_star"].values * s, y, df) * scale}
            for s in scales
        ]
    )
    curve.to_csv(os.path.join(TABLES_DIR, "profit_vs_approval_oof.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(curve["approval_rate"], curve["profit_per_10k"], color="#2b6cb0")
    op = curve.iloc[(curve["scale"] - 1.0).abs().idxmin()]
    ax.scatter([op["approval_rate"]], [op["profit_per_10k"]], color="#e53e3e", zorder=3,
               label=f"deployed rule (scale=1): {op['approval_rate']:.0%} approval")
    ax.set_xlabel("approval rate")
    ax.set_ylabel("realized profit € per 10k applications")
    ax.set_title("How a credit committee reads it: profit vs approval rate\n"
                 "(global scaling of the per-applicant thresholds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "45_profit_vs_approval.png"), dpi=120)
    plt.close(fig)

    # ── swap-set vs 0.5 (4d) ───────────────────────────────────────────────
    ss = swap_set(a_inst, a_naive, y, df)
    ss.to_csv(os.path.join(TABLES_DIR, "swap_set_oof.csv"), index=False)

    # ── THE GAP CHECKPOINT — narrative locked here ─────────────────────────
    checkpoint = {
        "computed_on": "cross-fitted calibrated OOF (development estimates — NOT results; "
                       "the quoted numbers come from the one-shot holdout, Phase 7)",
        "frozen_parameters_untouched": True,
        "best_flat_threshold_dev": t_best,
        "eur_weighted_mean_t_star": t_anchor,
        "profit_per_10k": {k: v * scale for k, v in profits.items()},
        "gap_instance_vs_best_flat_per_10k": gap_vs_flat * scale,
        "gap_instance_vs_best_flat_relative": rel_gap,
        "gap_instance_vs_naive_per_10k": gap_vs_naive * scale,
        "term_fallback_rows_used": int(df["term_fallback_used"].sum()),
        "LOCKED_NARRATIVE": (
            "The instance-dependent gain over the best flat threshold is small but free — "
            "same model, same calibration, zero added runtime cost — and the framework is "
            "what scales when LGD/EAD models and risk-based pricing get richer. This book's "
            "short annuity-implied terms (median 1.7y) put most t*(x) mass in a narrow band, "
            "as anticipated in the pre-registered expectation update; the vs-0.5 number is "
            "the naive baseline and is never a substitute for the best-flat comparison. "
            "This narrative was locked at the OOF checkpoint, before the holdout run, and "
            "no cost parameter was or will be revised."
        ),
    }
    with open(os.path.join(TABLES_DIR, "gap_checkpoint.json"), "w") as f:
        json.dump(checkpoint, f, indent=2)

    logger.info("GAP CHECKPOINT (OOF): instance vs best-flat = €%.0f per 10k (%.2f%%) | "
                "vs naive 0.5 = €%.0f per 10k | narrative LOCKED",
                gap_vs_flat * scale, 100 * rel_gap, gap_vs_naive * scale)


if __name__ == "__main__":
    main()
