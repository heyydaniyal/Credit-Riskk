"""
Post-holdout sensitivity analysis — DEVELOPMENT DATA ONLY.

Why this exists (review findings, October 2026):
  1. The holdout ceremony's sensitivity table covered 4 of the 6 frozen
     grids (LGD_revolving and ead_revolving were missing), counted the
     central point four times ("12 grid points" = 9 distinct), and re-picked
     the best flat threshold ON THE HOLDOUT at every grid point — so its
     central row (5.92M) does not reconcile with the headline (6.04M, flat
     threshold frozen from dev). The holdout is spent; it is not re-run.
  2. CURE_RATE was documented as a "sensitivity axis" but nothing computed
     it. It is now wired into the cost model and swept here.
  3. How much of the instance-dependent gain is just segmentation? A
     fitted per-segment flat policy (contract type × cash-term quintile) is
     the strongest data-driven competitor that uses no loan economics.

Protocol: calibrated cross-fitted OOF PDs on the 246,008 dev rows; every
competing flat policy is CROSS-FITTED (threshold chosen on 4 folds, scored
on the 5th), so no comparator is an oracle. Ledger = the frozen realized
convention. These are development numbers and are labelled as such; they
add evidence, they do not replace the pre-registered holdout results.

Outputs: reports/tables/sensitivity_dev_full.csv, sensitivity_dev_full.json,
         reports/figures/46_sensitivity_dev.png
"""

import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.constants import DATA_PROCESSED, FIGURES_DIR, ID_COL, TABLES_DIR  # noqa: E402
from src.data.load import load_modeling_frame  # noqa: E402
from src.models.decision import (  # noqa: E402
    SENSITIVITY_GRID,
    attach_cost_params,
    cash_term_segments,
    crossfit_flat_approvals,
    realized_profit,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sensitivity_dev")

# SENSITIVITY_GRID key -> attach_cost_params override name
OVERRIDE_NAME = {"LGD_cash": "lgd_cash", "LGD_revolving": "lgd_revolving",
                 "ead_cash": "ead_cash", "ead_revolving": "ead_revolving",
                 "r_net_cash": "r_net_cash", "m_revolving": "m_revolving",
                 "cure_rate": "cure_rate"}


def main() -> None:
    oof = pd.read_parquet(os.path.join(DATA_PROCESSED, "oof_constrained_calibrated.parquet"))
    f = load_modeling_frame("dev")[[ID_COL, "NAME_CONTRACT_TYPE", "term_years", "AMT_CREDIT"]]
    d = oof.merge(f, on=ID_COL, how="inner", validate="one_to_one")
    assert len(d) == len(oof)
    p, y, folds = d["p_cal_crossfit"].values, d["TARGET"].values, d["fold"].values
    per10k = 1e4 / len(d)

    rows = []
    points = [("frozen (deployed)", None, None)]
    for grid, values in SENSITIVITY_GRID.items():
        for v in values:
            points.append((grid, v, {OVERRIDE_NAME[grid]: v}))
    seen = set()
    for grid, value, ov in points:
        cost = attach_cost_params(d, overrides=ov)
        key = cost["t_star"].values.tobytes()  # full vector: identical policy ⇔ identical t*
        central_dup = ov is not None and key in seen
        seen.add(key)
        inst = realized_profit(p < cost["t_star"].values, y, cost)
        flat = realized_profit(crossfit_flat_approvals(p, y, cost, folds), y, cost)
        naive = realized_profit(p < 0.5, y, cost)
        rows.append({"grid": grid, "value": value, "duplicate_of_frozen": central_dup,
                     "instance_per_10k": inst * per10k,
                     "crossfit_best_flat_per_10k": flat * per10k,
                     "gap_vs_flat_per_10k": (inst - flat) * per10k,
                     "gap_vs_flat_relative": (inst - flat) / abs(flat),
                     "gap_vs_naive_per_10k": (inst - naive) * per10k,
                     "approval_rate_instance": float((p < cost["t_star"].values).mean())})
        log.info("%-18s %-6s gap vs cross-fitted flat %+.2f%%", grid, value,
                 100 * rows[-1]["gap_vs_flat_relative"])
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLES_DIR, "sensitivity_dev_full.csv"), index=False)

    # segmentation benchmark at the frozen assumptions
    cost = attach_cost_params(d)
    inst = realized_profit(p < cost["t_star"].values, y, cost)
    flat = realized_profit(crossfit_flat_approvals(p, y, cost, folds), y, cost)
    seg_ct = (cost["NAME_CONTRACT_TYPE"].astype(str) == "Revolving loans").astype(int).values
    two = realized_profit(crossfit_flat_approvals(p, y, cost, folds, seg_ct), y, cost)
    six = realized_profit(crossfit_flat_approvals(p, y, cost, folds,
                                                  cash_term_segments(cost)), y, cost)
    bench = {name: {"profit_per_10k": v * per10k, "gain_vs_single_flat": (v - flat) / abs(flat)}
             for name, v in [("single flat (cross-fitted)", flat),
                             ("flat per contract type (2 thresholds, cross-fitted)", two),
                             ("flat per contract × cash-term quintile (6, cross-fitted)", six),
                             ("instance-dependent t*(x) (no fitting)", inst)]}
    share = (six - flat) / (inst - flat)

    distinct = table[~table["duplicate_of_frozen"]]
    summary = {
        "label": "DEVELOPMENT evidence (dev OOF, cross-fitted comparators) — added after "
                 "the holdout ceremony; does not revise any pre-registered holdout number",
        "n_rows": int(len(d)),
        "distinct_grid_points": int(len(distinct)),
        "gap_positive_at_every_point": bool((distinct["gap_vs_flat_per_10k"] > 0).all()),
        "gap_relative_range": [float(distinct["gap_vs_flat_relative"].min()),
                               float(distinct["gap_vs_flat_relative"].max())],
        "frozen_point_gap_relative": float(table.iloc[0]["gap_vs_flat_relative"]),
        "segmentation_benchmark": bench,
        "share_of_gain_captured_by_6_segment_policy": float(share),
        "reading": "most of the gain is term/contract segmentation; the closed-form "
                   "t*(x) still beats a fitted segment policy without fitting anything",
    }
    json.dump(summary, open(os.path.join(TABLES_DIR, "sensitivity_dev_full.json"), "w"),
              indent=2)

    # figure: relative gap at each distinct grid point
    fig, ax = plt.subplots(figsize=(9, 4.2))
    lab = [f"{g}={v}" if g != "frozen (deployed)" else "frozen" for g, v in
           zip(distinct["grid"], distinct["value"], strict=True)]
    colors = ["#c05621" if g == "frozen (deployed)" else "#2b6cb0" for g in distinct["grid"]]
    ax.barh(range(len(distinct)), 100 * distinct["gap_vs_flat_relative"], color=colors)
    ax.axvline(100 * summary["frozen_point_gap_relative"], color="#c05621", ls="--", lw=1)
    ax.set_yticks(range(len(distinct)), lab, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("instance-dependent gain over cross-fitted best flat threshold (%)")
    ax.set_title("Sensitivity over all frozen grids + cure rate (development OOF)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "46_sensitivity_dev.png"), dpi=150)
    log.info("segmentation benchmark: %s", json.dumps(bench, indent=1))
    log.info("6-segment policy captures %.0f%% of the instance-dependent gain", 100 * share)


if __name__ == "__main__":
    main()
