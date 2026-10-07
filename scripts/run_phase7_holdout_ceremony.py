"""
Phase 7 — the one-shot holdout ceremony.

Implements EXACTLY config/holdout_analysis_plan.json (frozen at Phase 3):
the eight outputs, in order, nothing else.  Every score comes from the
canonical DeployedPipeline — the same code path the API uses, guarded by
the golden regression test.

DRY-RUN MODE (--dry-run): executes the ENTIRE mechanism on a labeled dev
sample so the ceremony code is fully tested BEFORE it ever sees holdout
rows.  The dry run never loads holdout data, and its outputs are written
to *_DRYRUN files that are not results.

The real run:  python scripts/run_phase7_holdout_ceremony.py
It logs every invocation to the ceremony output (plan rule: runs logged).
"""

import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import DATA_PROCESSED, ID_COL, TABLES_DIR, TARGET_COL
from src.data.load import load_modeling_frame
from src.models.calibration import brier, ece_and_reliability
from src.models.decision import (
    attach_cost_params,
    flat_policy,
    instance_policy,
    realized_profit,
    swap_set,
)
from src.models.decision import LGD, EAD_FACTOR, M_REVOLVING, R_NET_CASH
from src.models.pipeline import DeployedPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ceremony")

PER_10K = 10_000


def preflight(frame: pd.DataFrame) -> dict:
    """
    Label-free pre-flight (runs BEFORE any TARGET is read).

    Why it exists: an unknown NAME_CONTRACT_TYPE would map to NaN LGD/EAD,
    every `p < NaN` comparison would be False, and the ceremony would
    silently reject 100% of applicants while producing plausible-looking
    euro figures — on the one dataset that cannot be re-run.  Validated on
    holdout FEATURES during the design review (contract types clean, zero
    NaN cost params, zero unseen categories); this gate makes the check
    permanent rather than a one-time inspection.
    """
    from src.models.decision import LGD as _LGD

    unknown = set(frame["NAME_CONTRACT_TYPE"].astype(str).unique()) - set(_LGD)
    if unknown:
        raise SystemExit(f"PREFLIGHT FAILED — unknown contract types {unknown}: "
                         "cost params would be NaN and every applicant silently rejected")
    cost = attach_cost_params(frame[["NAME_CONTRACT_TYPE", "term_years",
                                     "AMT_CREDIT"]].copy())
    for col in ("lgd", "ead_factor", "margin", "t_star"):
        if cost[col].isna().any():
            raise SystemExit(f"PREFLIGHT FAILED — NaN in {col} "
                             f"({int(cost[col].isna().sum())} rows)")
    if frame["AMT_CREDIT"].isna().any():
        raise SystemExit("PREFLIGHT FAILED — NaN AMT_CREDIT (ledger input)")
    return {"rows": int(len(frame)),
            "term_fallback_rows": int(cost["term_fallback_used"].sum()),
            "t_star_range": [float(cost["t_star"].min()), float(cost["t_star"].max())]}


def sensitivity_rows(d: pd.DataFrame, p: np.ndarray, y: np.ndarray) -> list[dict]:
    """Frozen grids (spec 4c): recompute gap + vs-naive per grid point."""
    grids = {
        "LGD_cash": [0.60, 0.70, 0.80],
        "r_net_cash": [0.03, 0.05, 0.08],
        "ead_cash": [0.75, 0.85, 0.95],
        "m_revolving": [0.10, 0.18, 0.30],
    }
    cash = (d["NAME_CONTRACT_TYPE"].astype(str) == "Cash loans").values
    term = d["term_years_eff"].values
    A = d["AMT_CREDIT"].values
    scale = PER_10K / len(d)
    rows = []
    for param, values in grids.items():
        for v in values:
            lgd = np.where(cash, v if param == "LGD_cash" else LGD["Cash loans"],
                           LGD["Revolving loans"])
            ead = np.where(cash, v if param == "ead_cash" else EAD_FACTOR["Cash loans"],
                           EAD_FACTOR["Revolving loans"])
            r = v if param == "r_net_cash" else R_NET_CASH
            m_rev = v if param == "m_revolving" else M_REVOLVING
            m = np.where(cash, r * term / 2, m_rev)
            t_star = m / (m + lgd * ead)
            def ledger(a, lgd=lgd, ead=ead, m=m):
                return np.where(a, np.where(y == 1, -lgd * ead * A, m * A), 0.0).sum()
            grid_flats = {t: ledger(p < t) for t in np.round(np.arange(0.01, 0.501, 0.005), 3)}
            tb = max(grid_flats, key=grid_flats.get)
            rows.append({"param": param, "value": v,
                         "gap_vs_best_flat_per_10k": (ledger(p < t_star) - grid_flats[tb]) * scale,
                         "gap_vs_naive_per_10k": (ledger(p < t_star) - ledger(p < 0.5)) * scale})
    return rows


def main(dry_run: bool, force: bool = False) -> None:
    tag = "_DRYRUN" if dry_run else ""
    out_path = os.path.join(TABLES_DIR, f"holdout_ceremony{tag}.json")
    run_log = os.path.join(TABLES_DIR, "ceremony_run_log.jsonl")

    # RUN-ONCE GUARD: the plan says the ceremony runs once and every run is
    # logged — enforce it in code, not prose.  Re-running requires --force
    # AND is appended to an immutable log, so a second run can never pass
    # unnoticed as a first.
    if (not dry_run) and os.path.exists(out_path) and not force:
        raise SystemExit(
            f"CEREMONY ALREADY RUN — {out_path} exists.\n"
            "The holdout is one-shot. Re-running requires --force and is "
            "recorded in ceremony_run_log.jsonl; a re-run is only legitimate "
            "if no prior result influenced the change being made.")
    with open(run_log, "a") as f:
        f.write(json.dumps({"ts": datetime.datetime.now().isoformat(),
                            "mode": "dry_run" if dry_run else "ceremony",
                            "forced": bool(force)}) + "\n")

    pipe = DeployedPipeline()

    if dry_run:
        frame = load_modeling_frame("dev").sample(20_000, random_state=0).reset_index(drop=True)
        logger.info("DRY RUN on 20k dev rows — mechanism test, outputs are NOT results")
    else:
        frame = load_modeling_frame("holdout")
        logger.info("CEREMONY: scoring the %d-row holdout ONCE", len(frame))

    pre = preflight(frame)   # label-free: must pass before any TARGET is read
    logger.info("preflight OK: %s", pre)

    scored = pipe.score_frame(frame, with_reason_codes=False)
    p_raw = scored["pd_raw_ensemble"].values
    p = pipe.calibrator.predict(p_raw)  # undisplayed (unclipped) for decisions/metrics
    y = frame[TARGET_COL].values
    d = attach_cost_params(frame[[ID_COL, "NAME_CONTRACT_TYPE", "term_years",
                                  "AMT_CREDIT"]].copy())
    scale = PER_10K / len(frame)

    # frozen best-flat of the DEPLOYED pipeline (selected on dev, never re-fit here)
    t_best = json.load(open(os.path.join(TABLES_DIR, "final_pipeline.json")))[
        "decision"]["best_flat_threshold_constrained"]

    # 1. AUC
    auc = roc_auc_score(y, p_raw)
    # 2. calibration
    br = brier(y, p)
    ece, rel = ece_and_reliability(y, p)
    # 3-4. policies
    a_inst, a_flat, a_naive = (instance_policy(p, d), flat_policy(p, t_best),
                               flat_policy(p, 0.5))
    profits = {k: realized_profit(a, y, d) for k, a in
               [("instance", a_inst), (f"best_flat_{t_best}", a_flat), ("naive_0.5", a_naive)]}
    gap_flat = profits["instance"] - profits[f"best_flat_{t_best}"]
    # 5. sensitivity
    sens = sensitivity_rows(d, p, y)
    # 6. curves + swap-set
    ss = swap_set(a_inst, a_naive, y, d)
    scales = np.linspace(0.2, 3.0, 57)
    curve = [{"scale": float(s), "approval_rate": float((p < d.t_star.values * s).mean()),
              "profit_per_10k": realized_profit(p < d.t_star.values * s, y, d) * scale}
             for s in scales]
    # 7. fairness at deployed decisions
    audit = pd.read_parquet(os.path.join(DATA_PROCESSED, "audit_frame.parquet"))
    fd = frame[[ID_COL]].merge(audit[[ID_COL, "CODE_GENDER"]], on=ID_COL)
    fd = fd.assign(approved=a_inst, y=y, t_star=d["t_star"].values)
    fd = fd[fd.CODE_GENDER.isin(["F", "M"])]
    fair = {}
    for g, sub in fd.groupby("CODE_GENDER"):
        fair[g] = {"n": int(len(sub)), "approval_rate": float(sub.approved.mean()),
                   "FPR_goods_rejected": float((~sub.approved[sub.y == 0]).mean()),
                   "FNR_defaulters_approved": float(sub.approved[sub.y == 1].mean()),
                   "mean_t_star": float(sub.t_star.mean())}
    rates = {g: v["approval_rate"] for g, v in fair.items()}
    fair["disparate_impact_ratio"] = min(rates.values()) / max(rates.values())
    # 8. scorecard AUC (one line)
    # scorecard AUC runs in BOTH modes: the dry run must exercise every
    # line the ceremony will execute — no branch may run first on holdout
    if True:
        import pickle
        arts = pickle.load(open("models/scorecard_folds.pkl", "rb"))
        woe_feats = json.load(open(os.path.join(DATA_PROCESSED, "woe_features.json")))["features"]
        ps = np.mean([a["model"].predict_proba(
            a["binner"].transform(frame[woe_feats]))[:, 1] for a in arts], axis=0)
        sc_holdout_auc = float(roc_auc_score(y, ps))

    # consistency vs the PRE-REGISTERED bands (power analysis, dev-only)
    consistency = {}
    power_path = os.path.join(TABLES_DIR, "holdout_power_analysis.json")
    if os.path.exists(power_path):
        pw = json.load(open(power_path))
        g_rel = gap_flat / abs(profits[f"best_flat_{t_best}"])
        glo, ghi = pw["expected_holdout_gap_95"]
        alo, ahi = pw["expected_holdout_auc_95_sampling_only"]
        consistency = {
            "gap_in_preregistered_band": bool(glo <= g_rel <= ghi),
            "gap_band": [glo, ghi],
            "auc_in_sampling_band": bool(alo <= auc <= ahi),
            "auc_band": [alo, ahi],
            "verdict": "consistent with development — holdout fills in numbers, "
                       "story unchanged"
                       if (glo <= g_rel <= ghi and alo <= auc <= ahi)
                       else "OUTSIDE a pre-registered band — a FINDING to report and "
                            "investigate (playbook step 5: large AUC disagreement "
                            "means the CV leaked), never a number to accept quietly",
        }

    result = {
        "run_logged_at": datetime.datetime.now().isoformat(),
        "artifact_manifest_sha": __import__("hashlib").sha256(
            open("artifacts/MANIFEST.json").read().encode()).hexdigest()[:12]
            if os.path.exists("artifacts/MANIFEST.json") else "unversioned",
        "preflight": pre,
        "consistency_vs_preregistered": consistency,
        "mode": "DRY_RUN (dev sample — NOT results)" if dry_run else "CEREMONY (one-shot holdout)",
        "n_rows": int(len(frame)),
        "1_auc_deployed_ensemble": float(auc),
        "2_calibration": {"brier": br, "ece": ece, "climatology": 0.0736},
        "3_gap_instance_vs_best_flat": {"per_10k": gap_flat * scale,
                                        "relative": gap_flat / abs(profits[f"best_flat_{t_best}"]),
                                        "best_flat_frozen_from_dev": t_best},
        "4_eur_savings_vs_naive_per_10k": (profits["instance"] - profits["naive_0.5"]) * scale,
        "4_qualifier": "under the stated ILLUSTRATIVE economic assumptions",
        "5_sensitivity": sens,
        "7_fairness_at_deployed_decisions": fair,
        "8_scorecard_holdout_auc": sc_holdout_auc,
        "profit_per_10k": {k: v * scale for k, v in profits.items()},
    }
    with open(os.path.join(TABLES_DIR, f"holdout_ceremony{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    ss.to_csv(os.path.join(TABLES_DIR, f"holdout_swap_set{tag}.csv"), index=False)
    pd.DataFrame(curve).to_csv(os.path.join(TABLES_DIR,
                                            f"holdout_profit_curve{tag}.csv"), index=False)
    rel.to_csv(os.path.join(TABLES_DIR, f"holdout_reliability{tag}.csv"), index=False)
    logger.info("%s complete: AUC=%.5f Brier=%.5f gap=%.2f%% savings-vs-naive=€%.0f/10k",
                "DRY RUN" if dry_run else "CEREMONY", auc, br,
                100 * result["3_gap_instance_vs_best_flat"]["relative"],
                result["4_eur_savings_vs_naive_per_10k"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="mechanism test on dev sample; never touches holdout")
    ap.add_argument("--force", action="store_true",
                    help="re-run the ceremony (logged); only legitimate if no prior "
                         "result influenced the change being made")
    a = ap.parse_args()
    main(a.dry_run, force=a.force)
