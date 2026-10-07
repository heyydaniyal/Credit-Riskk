"""
Checkpointed null-importance computation (resumable).

Same math as NullImportanceSelector, split into restartable chunks:
each invocation runs up to --chunk fits and appends them to a checkpoint
file.  When 3 real-seed fits + NULL_IMPORTANCE_RUNS null fits exist, it
writes the final report + feature list, identical to what the selector
would produce in one shot.

Why this exists: a 25-minute unresumable job is fragile by design; a
checkpointed one survives interruptions and proves determinism (seeds are
a function of run index, so re-running any chunk reproduces it exactly).

Usage:  python scripts/run_phase2_null_chunks.py --chunk 8
Repeat until it prints DONE.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import (
    DATA_PROCESSED,
    ID_COL,
    NULL_IMPORTANCE_PCTL,
    NULL_IMPORTANCE_RUNS,
    RANDOM_STATE,
    TABLES_DIR,
    TARGET_COL,
)
from src.data.load import load_dev_ids
from src.features.selection import NullImportanceSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("null-chunks")

N_REAL_SEEDS = 3
CKPT = os.path.join(DATA_PROCESSED, "null_importance_ckpt.npz")
COLS_JSON = os.path.join(DATA_PROCESSED, "post_corr_columns.json")
LGBM_FEATURES_PATH = os.path.join(DATA_PROCESSED, "lgbm_features.json")


def load_X_y():
    features = pd.read_parquet(os.path.join(DATA_PROCESSED, "features_full.parquet"))
    dev = features[features[ID_COL].isin(set(load_dev_ids()))].reset_index(drop=True)
    with open(COLS_JSON) as f:
        cols = json.load(f)["columns"]
    return dev[cols], dev[TARGET_COL]


def main(chunk: int) -> None:
    X, y = load_X_y()
    sel = NullImportanceSelector(random_state=RANDOM_STATE)
    yv = y.values

    if os.path.exists(CKPT):
        d = np.load(CKPT, allow_pickle=True)
        real, nulls = list(d["real"]), list(d["nulls"])
        assert list(d["columns"]) == list(X.columns), "column set changed mid-checkpoint"
    else:
        real, nulls = [], []

    done = 0
    while done < chunk:
        if len(real) < N_REAL_SEEDS:
            s = len(real)
            logger.info("real fit %d/%d", s + 1, N_REAL_SEEDS)
            real.append(sel._gain(X, yv, seed=RANDOM_STATE + s))
        elif len(nulls) < NULL_IMPORTANCE_RUNS:
            run = len(nulls)
            logger.info("null fit %d/%d", run + 1, NULL_IMPORTANCE_RUNS)
            # rng seeded by run index → chunking cannot change the shuffles
            rng = np.random.RandomState(RANDOM_STATE + 5000 + run)
            nulls.append(sel._gain(X, rng.permutation(yv), seed=RANDOM_STATE + 1000 + run))
        else:
            break
        done += 1
        np.savez_compressed(CKPT, real=np.array(real), nulls=np.array(nulls),
                            columns=np.array(X.columns, dtype=object))

    if len(real) == N_REAL_SEEDS and len(nulls) == NULL_IMPORTANCE_RUNS:
        real_m = np.mean(real, axis=0)
        nl = np.array(nulls)
        thresh = np.percentile(nl, NULL_IMPORTANCE_PCTL, axis=0)
        keep = real_m > thresh
        report = pd.DataFrame(
            {
                "feature": X.columns,
                "real_gain": real_m,
                "null_gain_p95": thresh,
                "null_gain_max": nl.max(axis=0),
                "gain_ratio": real_m / np.where(thresh > 0, thresh, np.nan),
                "kept": keep,
            }
        ).sort_values("real_gain", ascending=False)
        report.to_csv(os.path.join(TABLES_DIR, "null_importance_report.csv"), index=False)
        kept = report.loc[report["kept"], "feature"].tolist()
        with open(LGBM_FEATURES_PATH, "w") as f:
            json.dump(
                {
                    "features": kept,
                    "n_features": len(kept),
                    "pipeline": ["near_constant>0.995", "spearman>0.98",
                                 f"null_importance_p{NULL_IMPORTANCE_PCTL}_{NULL_IMPORTANCE_RUNS}runs"],
                    "fitted_on": "dev only",
                },
                f, indent=2,
            )
        logger.info("DONE — kept %d / %d features", len(kept), X.shape[1])
    else:
        logger.info("progress: real %d/%d, null %d/%d — re-run to continue",
                    len(real), N_REAL_SEEDS, len(nulls), NULL_IMPORTANCE_RUNS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=8)
    main(ap.parse_args().chunk)
