"""
Phase 2 — Feature Engineering (scripted, restartable).

Stages (each writes its artifacts, so any stage can be re-run alone):
  build   raw tables → full feature matrix (ALL rows; stateless + per-
          applicant aggregations only — safe on all rows by construction)
  select  near-constant → Spearman-correlation → null-importance filters,
          ALL fit on the DEV rows only
  woe     WOE binner fit on DEV, IV screening → scorecard feature list
  report  tables, figures, FEATURE_ENGINEERING.md inputs
  check   pipeline-integrity (dev vs holdout on final features) + sanity AUC

Usage:  python scripts/run_phase2.py [stage ...]   (default: all stages)
"""

import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import (
    CORR_THRESHOLD,
    DATA_PROCESSED,
    DATA_RAW,
    FIGURES_DIR,
    ID_COL,
    IV_MAX,
    IV_MIN,
    MODELS_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TARGET_COL,
)
from src.data.load import load_dev_ids, load_holdout_ids
from src.features.build import NON_FEATURE_COLS, build_feature_matrix, make_manifest
from src.features.selection import (
    CorrelationDropper,
    NearConstantDropper,
    univariate_auc,
)
from src.features.woe import WOEBinner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase2")

FEATURES_PATH = os.path.join(DATA_PROCESSED, "features_full.parquet")
AUDIT_PATH = os.path.join(DATA_PROCESSED, "audit_frame.parquet")
MANIFEST_PATH = os.path.join(DATA_PROCESSED, "feature_manifest.csv")
LGBM_FEATURES_PATH = os.path.join(DATA_PROCESSED, "lgbm_features.json")
WOE_FEATURES_PATH = os.path.join(DATA_PROCESSED, "woe_features.json")
WOE_BINNER_PATH = os.path.join(MODELS_DIR, "woe_binner_dev.pkl")

# Bureau-balance dtypes: 27M rows on a 3 GB box needs explicit narrow types
BB_DTYPES = {"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16", "STATUS": "category"}


# ────────────────────────────────────────────────────────────────────────────
def stage_build() -> None:
    app = pd.read_csv(os.path.join(DATA_RAW, "application_train.csv"))
    bureau = pd.read_csv(os.path.join(DATA_RAW, "bureau.csv"))
    bb = pd.read_csv(os.path.join(DATA_RAW, "bureau_balance.csv"), dtype=BB_DTYPES)
    prev = pd.read_csv(os.path.join(DATA_RAW, "previous_application.csv"))

    features, audit = build_feature_matrix(app, bureau, bb, prev)
    del app, bureau, bb, prev

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)
    audit.to_parquet(AUDIT_PATH, index=False)
    make_manifest(features).to_csv(MANIFEST_PATH, index=False)
    logger.info("build: wrote %s (%d×%d)", FEATURES_PATH, *features.shape)


# ────────────────────────────────────────────────────────────────────────────
def _load_dev_features() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH)
    dev_ids = set(load_dev_ids())
    return features[features[ID_COL].isin(dev_ids)].reset_index(drop=True)


def stage_select() -> None:
    dev = _load_dev_features()
    y = dev[TARGET_COL]
    X = dev.drop(columns=[c for c in NON_FEATURE_COLS if c in dev.columns])

    # 1) near-constant (fit: dev)
    ncd = NearConstantDropper().fit(X)
    ncd.report_.to_csv(os.path.join(TABLES_DIR, "near_constant_drops.csv"), index=False)
    X = ncd.transform(X)

    # 2) Spearman correlation > 0.98 (numeric only; fit: dev)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cd = CorrelationDropper(threshold=CORR_THRESHOLD).fit(X[num_cols], y)
    cd.report_.to_csv(os.path.join(TABLES_DIR, "correlation_drops.csv"), index=False)
    X = cd.transform(X)

    # 3) null-importance runs in checkpointed chunks (single-CPU box):
    #    save the post-correlation column set, then run
    #    scripts/run_phase2_null_chunks.py until it prints DONE.
    with open(os.path.join(DATA_PROCESSED, "post_corr_columns.json"), "w") as f:
        json.dump({"columns": list(X.columns), "fitted_on": "dev only"}, f, indent=2)
    logger.info(
        "select: fast filters done (%d → %d cols). Now run "
        "scripts/run_phase2_null_chunks.py until DONE.",
        dev.shape[1] - len(NON_FEATURE_COLS), X.shape[1],
    )


# ────────────────────────────────────────────────────────────────────────────
def stage_woe() -> None:
    dev = _load_dev_features()
    y = dev[TARGET_COL]

    with open(LGBM_FEATURES_PATH) as f:
        lgbm_features = json.load(f)["features"]

    # Candidates: features that survived selection (already de-duplicated,
    # already signal-bearing) — WOE-bin those, screen by IV.
    X = dev[lgbm_features]
    binner = WOEBinner(n_bins=5, min_bin_frac=0.01).fit(X, y)
    iv_table = binner.iv_table()
    iv_table.to_csv(os.path.join(TABLES_DIR, "woe_iv_table.csv"), index=False)

    in_range = binner.select_by_iv(IV_MIN, IV_MAX)
    suspicious = iv_table[iv_table["iv"] > IV_MAX]["feature"].tolist()
    if suspicious:
        logger.warning("IV > %.2f (audit for leakage): %s", IV_MAX, suspicious)

    # Scorecard list: cap at 40 by IV rank (spec target 20–40)
    ranked = [f for f in iv_table["feature"] if f in in_range]
    scorecard = ranked[:40]

    # Refit the binner on exactly the scorecard features (smaller artifact).
    # NOTE: this dev-fit binner is for reporting/holdout-transform only —
    # Phase 3 CV must refit per fold (see WOEBinner docstring).
    binner_final = WOEBinner(n_bins=5, min_bin_frac=0.01).fit(dev[scorecard], y)
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(WOE_BINNER_PATH, "wb") as f:
        pickle.dump(binner_final, f)

    with open(WOE_FEATURES_PATH, "w") as f:
        json.dump(
            {
                "features": scorecard,
                "n_features": len(scorecard),
                "iv_range": [IV_MIN, IV_MAX],
                "suspicious_iv_excluded": suspicious,
                "note": "binner artifact is dev-fit; Phase 3 CV refits per fold",
            },
            f, indent=2,
        )
    logger.info("woe: %d scorecard features (of %d in IV range)", len(scorecard), len(in_range))


# ────────────────────────────────────────────────────────────────────────────
def stage_report() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = _load_dev_features()
    y = dev[TARGET_COL]
    manifest = pd.read_csv(MANIFEST_PATH)
    with open(LGBM_FEATURES_PATH) as f:
        lgbm_features = json.load(f)["features"]
    null_rep = pd.read_csv(os.path.join(TABLES_DIR, "null_importance_report.csv"))
    iv_table = pd.read_csv(os.path.join(TABLES_DIR, "woe_iv_table.csv"))

    # Aggregation summary table
    agg_cols = [c for c in lgbm_features if c.startswith(("bureau_", "prev_"))]
    agg_summary = dev[agg_cols].describe().T[["count", "mean", "std", "min", "max"]]
    agg_summary["missing_share"] = dev[agg_cols].isna().mean()
    agg_summary.to_csv(os.path.join(TABLES_DIR, "aggregation_summary.csv"))

    # Univariate AUC of the new engineered features (top table for the docs)
    eng_cols = [c for c in lgbm_features
                if c.startswith(("bureau_", "prev_")) or manifest.set_index("feature")
                .loc[c, "source"] in ("domain ratio", "missingness/sentinel flag")]
    num_eng = [c for c in eng_cols if pd.api.types.is_numeric_dtype(dev[c])]
    ua = univariate_auc(dev[num_eng], y).sort_values(ascending=False)
    ua.to_frame().to_csv(os.path.join(TABLES_DIR, "engineered_feature_auc.csv"))

    # Figure 1: real vs null gain for top 20 features
    top20 = null_rep.head(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    ypos = np.arange(len(top20))
    ax.barh(ypos, top20["real_gain"], color="#2b6cb0", label="real gain")
    ax.barh(ypos, top20["null_gain_p95"], color="#e53e3e", alpha=0.6,
            label="null 95th pct", height=0.4)
    ax.set_yticks(ypos, top20["feature"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_title("Null-importance filter — top 20 features\n(real gain vs its own noise floor)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "21_null_importance_top20.png"), dpi=120)
    plt.close(fig)

    # Figure 2: IV ranking of scorecard candidates
    top_iv = iv_table.head(40)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#e53e3e" if v > IV_MAX else "#2b6cb0" for v in top_iv["iv"]]
    ax.barh(np.arange(len(top_iv)), top_iv["iv"], color=colors)
    ax.set_yticks(np.arange(len(top_iv)), top_iv["feature"], fontsize=6)
    ax.invert_yaxis()
    ax.axvline(IV_MIN, ls="--", c="gray")
    ax.axvline(IV_MAX, ls="--", c="red")
    ax.set_title("Information Value — top 40 (red line = 0.5 leakage-audit threshold)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "22_iv_ranking.png"), dpi=120)
    plt.close(fig)

    # Figure 3: WOE monotonicity examples (6 features)
    with open(WOE_BINNER_PATH, "rb") as f:
        binner = pickle.load(f)
    examples = [c for c in ["ext_source_mean", "EXT_SOURCE_2", "prev_refusal_share",
                            "credit_income_ratio", "annuity_income_ratio", "age_years"]
                if c in binner.numeric_bins_][:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.ravel(), examples):
        woes = binner.numeric_bins_[col]["woes"]
        ax.bar(range(len(woes)), woes, color="#2b6cb0")
        if binner.numeric_bins_[col]["nan_woe"]:
            ax.bar(len(woes), binner.numeric_bins_[col]["nan_woe"], color="#a0aec0")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("bin (last gray = NaN)", fontsize=7)
        ax.set_ylabel("WOE", fontsize=7)
    fig.suptitle("WOE by bin — gray bar = missing-value bin (own empirical WOE)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "23_woe_bins_examples.png"), dpi=120)
    plt.close(fig)

    logger.info("report: tables + figures written")


# ────────────────────────────────────────────────────────────────────────────
def stage_check() -> None:
    """Validation gate: integrity check on FINAL features + quick sanity AUC."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    from src.validation.protocol import pipeline_integrity_check

    features = pd.read_parquet(FEATURES_PATH)
    with open(LGBM_FEATURES_PATH) as f:
        lgbm_features = json.load(f)["features"]

    dev_ids, hold_ids = set(load_dev_ids()), set(load_holdout_ids())
    dev = features[features[ID_COL].isin(dev_ids)].reset_index(drop=True)
    hold = features[features[ID_COL].isin(hold_ids)].reset_index(drop=True)

    # 1) Pipeline integrity on the engineered features (uses features only,
    #    never TARGET — the holdout labels stay untouched).
    numeric_final = [c for c in lgbm_features if pd.api.types.is_numeric_dtype(features[c])]
    integrity = pipeline_integrity_check(dev, hold, numeric_final)
    with open(os.path.join(TABLES_DIR, "integrity_check_phase2.json"), "w") as f:
        json.dump(integrity, f, indent=2, default=float)

    # 2) Sanity AUC — fold 1 only, default params.  NOT a result; only a
    #    tripwire ("did Phase 2 produce learnable features in the expected
    #    range, and not a leak?").  Phase 3 owns real numbers.
    from src.data.load import load_modeling_frame

    dev = load_modeling_frame("dev")
    tr, va = dev[dev["fold"] != 0], dev[dev["fold"] == 0]
    cat_cols = [c for c in lgbm_features if isinstance(features[c].dtype, pd.CategoricalDtype)]
    dtrain = lgb.Dataset(tr[lgbm_features], label=tr[TARGET_COL], categorical_feature=cat_cols)
    dval = lgb.Dataset(va[lgbm_features], label=va[TARGET_COL], reference=dtrain)
    booster = lgb.train(
        {"objective": "binary", "metric": "auc", "verbosity": -1,
         "learning_rate": 0.05, "num_leaves": 31, "seed": RANDOM_STATE},
        dtrain, num_boost_round=2000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    auc = roc_auc_score(va[TARGET_COL], booster.predict(va[lgbm_features]))
    sanity = {
        "fold1_auc": float(auc),
        "expected_range": [0.76, 0.80],
        "leakage_alarm_above": 0.81,
        "note": "sanity tripwire only — Phase 3 owns real numbers",
        "alarm": bool(auc > 0.81),
    }
    with open(os.path.join(TABLES_DIR, "sanity_auc_phase2.json"), "w") as f:
        json.dump(sanity, f, indent=2)
    logger.info("check: integrity AUC=%.4f (%s) | sanity fold-1 AUC=%.4f%s",
                integrity["mean_auc"], "PASS" if integrity["passed"] else "FAIL",
                auc, "  ⚠ LEAKAGE ALARM" if sanity["alarm"] else "")


# ────────────────────────────────────────────────────────────────────────────
STAGES = {"build": stage_build, "select": stage_select, "woe": stage_woe,
          "report": stage_report, "check": stage_check}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    for name in wanted:
        logger.info("═══ stage: %s ═══", name)
        STAGES[name]()
