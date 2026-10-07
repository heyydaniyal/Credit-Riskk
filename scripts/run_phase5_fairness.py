"""
Phase 5b — fairness audit, executed per config/phase5_design_frozen.json.

Everything here follows the FROZEN plan: proxy on dev only (same folds, no
tuning), pinned metric definitions, decisions = constrained pipeline's
cross-fitted p_cal vs t*(x).  The pre-registered expectation is that the
proxy check FIRES (AUC 0.75-0.85) — the deliverable is measurement +
disclosure, never remediation.
"""

import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import DATA_PROCESSED, FIGURES_DIR, ID_COL, TABLES_DIR
from src.data.load import load_modeling_frame
from src.models.decision import attach_cost_params, instance_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase5-fairness")

PROXY_PARAMS = {"objective": "binary", "verbosity": -1, "learning_rate": 0.05,
                "num_leaves": 63, "seed": 42}


def proxy_check(dev: pd.DataFrame, feats: list[str], gender: pd.Series) -> dict:
    """OOF AUC of LightGBM predicting gender from the modeling features
    (frozen plan: same 5 folds, no tuning, XNA excluded upstream)."""
    cat_cols = [c for c in feats if isinstance(dev[c].dtype, pd.CategoricalDtype)]
    g = (gender == "F").astype(int).values
    oof = np.full(len(dev), np.nan)
    contrib_fold1 = None
    for k in sorted(dev["fold"].unique()):
        tr, va = dev["fold"] != k, dev["fold"] == k
        dtrain = lgb.Dataset(dev.loc[tr, feats], label=g[tr.values],
                             categorical_feature=cat_cols)
        dval = lgb.Dataset(dev.loc[va, feats], label=g[va.values], reference=dtrain)
        b = lgb.train(PROXY_PARAMS, dtrain, num_boost_round=300, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va.values] = b.predict(dev.loc[va, feats],
                                   num_iteration=b.best_iteration)
        if k == 0:  # top proxy carriers from fold-1's model (stated)
            sample = dev.loc[va, feats].sample(5000, random_state=0)
            c = b.predict(sample, pred_contrib=True)[:, :-1]
            contrib_fold1 = pd.Series(np.abs(c).mean(0), index=feats)
    auc = roc_auc_score(g, oof)
    top10 = contrib_fold1.sort_values(ascending=False).head(10)
    return {"proxy_oof_auc": float(auc),
            "preregistered_expectation": [0.75, 0.85],
            "flag_threshold": 0.65,
            "fired": bool(auc > 0.65),
            "top10_proxy_carriers_fold1": {k: float(v) for k, v in top10.items()}}


def decision_metrics(d: pd.DataFrame) -> dict:
    """Pinned definitions: FPR = share of goods rejected; FNR = share of
    defaulters approved; DIR = min/max approval, flag < 0.8."""
    out = {}
    for grp, sub in d.groupby("CODE_GENDER"):
        appr = sub["approved"].values
        y = sub["TARGET"].values
        out[grp] = {
            "n": int(len(sub)),
            "approval_rate": float(appr.mean()),
            "FPR_goods_rejected": float((~appr[y == 0]).mean()),
            "FNR_defaulters_approved": float(appr[y == 1].mean()),
            "mean_t_star": float(sub["t_star"].mean()),
            "t_star_quantiles": {q: float(sub["t_star"].quantile(q))
                                 for q in (0.1, 0.5, 0.9)},
            "share_revolving": float((sub["NAME_CONTRACT_TYPE"]
                                      == "Revolving loans").mean()),
            "base_default_rate": float(y.mean()),
        }
    rates = {g: v["approval_rate"] for g, v in out.items()}
    dir_ = min(rates.values()) / max(rates.values())
    out["summary"] = {
        "disparate_impact_ratio": float(dir_),
        "four_fifths_flag": bool(dir_ < 0.8),
        "abs_FPR_gap": float(abs(out["F"]["FPR_goods_rejected"]
                                 - out["M"]["FPR_goods_rejected"])),
        "abs_FNR_gap": float(abs(out["F"]["FNR_defaulters_approved"]
                                 - out["M"]["FNR_defaulters_approved"])),
    }
    return out


def main() -> None:
    dev = load_modeling_frame("dev")
    with open(os.path.join(DATA_PROCESSED, "lgbm_features.json")) as f:
        feats = json.load(f)["features"]
    audit = pd.read_parquet(os.path.join(DATA_PROCESSED, "audit_frame.parquet"))
    dev = dev.merge(audit[[ID_COL, "CODE_GENDER"]], on=ID_COL, validate="one_to_one")
    dev = dev[dev["CODE_GENDER"].isin(["F", "M"])].reset_index(drop=True)  # XNA excluded (plan)

    logger.info("proxy check (5 folds, no tuning)…")
    proxy = proxy_check(dev, feats, dev["CODE_GENDER"])
    logger.info("proxy OOF AUC = %.4f (pre-registered 0.75-0.85; flag 0.65)",
                proxy["proxy_oof_auc"])

    # decisions of the DEPLOYED (constrained) pipeline
    oofc = pd.read_parquet(os.path.join(DATA_PROCESSED,
                                        "oof_constrained_calibrated.parquet"))
    cols = pd.read_parquet(os.path.join(DATA_PROCESSED, "features_full.parquet"),
                           columns=[ID_COL, "NAME_CONTRACT_TYPE", "term_years",
                                    "AMT_CREDIT"])
    d = oofc.merge(cols, on=ID_COL).merge(audit[[ID_COL, "CODE_GENDER"]], on=ID_COL)
    d = d[d["CODE_GENDER"].isin(["F", "M"])].reset_index(drop=True)
    d = attach_cost_params(d)
    d["approved"] = instance_policy(d["p_cal_crossfit"].values, d)

    metrics = decision_metrics(d)
    report = {"proxy_check": proxy, "decision_metrics": metrics,
              "scope": "dev only; gender only (no ethnicity in data — audit is partial); "
                       "framing: inspired by governance practice, not compliance"}
    with open(os.path.join(TABLES_DIR, "fairness_audit.json"), "w") as f:
        json.dump(report, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for g, color in [("F", "#2b6cb0"), ("M", "#e53e3e")]:
        axes[0].hist(d.loc[d.CODE_GENDER == g, "t_star"], bins=60, alpha=0.55,
                     density=True, color=color, label=g)
    axes[0].set_title("t*(x) distribution by gender\n(policy-level check the "
                      "instance rule creates)")
    axes[0].set_xlabel("t*(x)")
    axes[0].legend()
    carriers = pd.Series(proxy["top10_proxy_carriers_fold1"])
    axes[1].barh(range(len(carriers)), carriers.values, color="#2b6cb0")
    axes[1].set_yticks(range(len(carriers)), carriers.index, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_title(f"Top proxy carriers (fold-1)\nproxy OOF AUC = "
                      f"{proxy['proxy_oof_auc']:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "53_fairness.png"), dpi=120)
    plt.close(fig)
    logger.info("fairness audit written: DIR=%.3f (flag<0.8: %s) |ΔFPR|=%.4f |ΔFNR|=%.4f",
                metrics["summary"]["disparate_impact_ratio"],
                metrics["summary"]["four_fifths_flag"],
                metrics["summary"]["abs_FPR_gap"], metrics["summary"]["abs_FNR_gap"])


if __name__ == "__main__":
    main()
