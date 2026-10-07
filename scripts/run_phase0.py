"""
Phase 0 Runner — Create the frozen holdout split.
Run once. The holdout is touched again only in Phase 7.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.constants import DATA_RAW, DATA_SPLITS, ID_COL, N_FOLDS, TARGET_COL
from src.validation.protocol import create_holdout_split, get_cv_folds

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    # Load main application table
    app = pd.read_csv(os.path.join(DATA_RAW, "application_train.csv"))
    logger.info(f"Loaded application_train: {app.shape}")

    # Create the 20% stratified holdout (fixed seed)
    dev, holdout = create_holdout_split(app)

    # Persist splits — store only IDs + target for reproducibility,
    # plus full frames for convenience
    os.makedirs(DATA_SPLITS, exist_ok=True)

    # Save the ID lists — these define the frozen split forever
    dev[[ID_COL, TARGET_COL]].to_parquet(
        os.path.join(DATA_SPLITS, "dev_ids.parquet"), index=False
    )
    holdout[[ID_COL, TARGET_COL]].to_parquet(
        os.path.join(DATA_SPLITS, "holdout_ids.parquet"), index=False
    )

    # Assign and save fold indices for the dev set (reused by every model)
    skf = get_cv_folds(dev)
    fold_assignment = np.full(len(dev), -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(dev, dev[TARGET_COL])):
        fold_assignment[val_idx] = fold_idx

    fold_df = pd.DataFrame({
        ID_COL: dev[ID_COL].values,
        "fold": fold_assignment,
        TARGET_COL: dev[TARGET_COL].values,
    })
    fold_df.to_parquet(os.path.join(DATA_SPLITS, "dev_folds.parquet"), index=False)

    # Verify fold stratification
    logger.info("\nFold distribution (should be balanced):")
    for f in range(N_FOLDS):
        mask = fold_df["fold"] == f
        n = mask.sum()
        rate = fold_df.loc[mask, TARGET_COL].mean()
        logger.info(f"  Fold {f}: {n:,} rows, default rate {rate:.4f}")

    logger.info(f"\n✓ Phase 0 splits frozen in {DATA_SPLITS}")
    logger.info(f"  dev_ids.parquet    — {len(dev):,} rows")
    logger.info(f"  holdout_ids.parquet — {len(holdout):,} rows (DO NOT TOUCH until Phase 7)")
    logger.info(f"  dev_folds.parquet  — {N_FOLDS}-fold assignment")


if __name__ == "__main__":
    main()
