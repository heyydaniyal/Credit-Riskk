"""
Feature selection — STATEFUL transforms, fit on dev data only.

Every class here learns parameters from data (thresholds, correlations,
importance distributions).  Per the Phase 0 leakage discipline they are
fit on the dev set and merely APPLIED to the holdout.  None of them may
ever see holdout rows during fit.

Selection caveat (documented, not hidden): fitting selection on the full
dev set — rather than inside each CV fold — makes later OOF estimates
very slightly optimistic (the selector saw each fold's labels).  This is
the standard trade-off for a stable, single feature list; the untouched
holdout is what keeps the FINAL numbers honest.
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class NearConstantDropper:
    """Drop columns where one value dominates (default > 99.5% of non-null rows)."""

    def __init__(self, max_dominant_share: float = 0.995):
        self.max_dominant_share = max_dominant_share
        self.dropped_: list[str] = []
        self.report_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame) -> "NearConstantDropper":
        rows = []
        for col in X.columns:
            vc = X[col].value_counts(dropna=True, normalize=True)
            share = float(vc.iloc[0]) if len(vc) else 1.0
            if share > self.max_dominant_share:
                rows.append({"feature": col, "dominant_share": share})
        self.report_ = pd.DataFrame(rows)
        self.dropped_ = self.report_["feature"].tolist() if len(rows) else []
        logger.info("NearConstantDropper: dropping %d columns", len(self.dropped_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.dropped_ if c in X.columns])


def univariate_auc(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Per-feature |AUC - 0.5| + 0.5 on non-null rows (direction-free strength).
    Used as the tiebreaker in correlation dropping: keep the predictive one.
    """
    out = {}
    yv = y.values
    for col in X.columns:
        x = X[col].values.astype(float)
        mask = ~np.isnan(x)
        if mask.sum() < 100 or len(np.unique(yv[mask])) < 2:
            out[col] = 0.5
            continue
        auc = roc_auc_score(yv[mask], x[mask])
        out[col] = max(auc, 1 - auc)
    return pd.Series(out, name="univariate_auc")


class CorrelationDropper:
    """
    Drop one feature from every pair with |Spearman| > threshold (0.98),
    keeping the one with higher univariate AUC.

    Spearman (rank) correlation is used because it is robust to the heavy
    tails this dataset carries (income max = 248× p99) and detects any
    monotone redundancy, which is exactly what makes two features
    interchangeable for a tree model.
    """

    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold
        self.dropped_: list[str] = []
        self.report_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CorrelationDropper":
        cols = list(X.columns)
        aucs = univariate_auc(X, y)

        # Rank-transform (NaN kept out of ranks), then Pearson on ranks = Spearman.
        # float32 halves memory; with n≈250k the estimate is exact for our purpose.
        ranks = np.empty((len(X), len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            x = X[c].values.astype(float)
            mask = ~np.isnan(x)
            r = np.full(len(x), np.nan, dtype=np.float32)
            r[mask] = rankdata(x[mask])
            ranks[:, j] = r

        # Pairwise complete-obs correlation via masked arrays is O(p² n) —
        # too slow.  Instead: mean-impute ranks per column (only for the
        # correlation ESTIMATE, never for modeling), which biases pairs
        # toward 0 correlation — conservative for a "drop if > 0.98" rule.
        col_means = np.nanmean(ranks, axis=0)
        inds = np.where(np.isnan(ranks))
        ranks[inds] = np.take(col_means, inds[1])
        corr = np.corrcoef(ranks, rowvar=False)

        drop: set[str] = set()
        pairs = []
        iu = np.triu_indices(len(cols), k=1)
        hits = np.where(np.abs(corr[iu]) > self.threshold)[0]
        for h in hits:
            i, j = iu[0][h], iu[1][h]
            a, b = cols[i], cols[j]
            if a in drop or b in drop:
                continue
            loser = b if aucs[a] >= aucs[b] else a
            keeper = a if loser == b else b
            drop.add(loser)
            pairs.append(
                {
                    "kept": keeper, "dropped": loser,
                    "spearman": float(corr[i, j]),
                    "kept_auc": float(aucs[keeper]), "dropped_auc": float(aucs[loser]),
                }
            )
        self.dropped_ = sorted(drop)
        self.report_ = pd.DataFrame(pairs)
        logger.info("CorrelationDropper: dropping %d of %d features", len(drop), len(cols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.dropped_ if c in X.columns])


class NullImportanceSelector:
    """
    Null-importance feature selection (shuffled-target benchmark).

    Real importance must beat the 95th percentile of that SAME feature's
    importance distribution under a shuffled target.  A feature that can't
    outperform its own noise floor is memorizing, not predicting.

    Gain importance is used (split counts reward high-cardinality noise).
    The real importance is averaged over `n_real_seeds` fits so a single
    lucky/unlucky seed can't decide a feature's fate.
    """

    _PARAMS = {
        "objective": "binary",
        "verbosity": -1,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "num_threads": 0,
    }

    def __init__(self, n_null_runs: int = 50, percentile: float = 95,
                 n_real_seeds: int = 3, num_boost_round: int = 200,
                 random_state: int = 42):
        self.n_null_runs = n_null_runs
        self.percentile = percentile
        self.n_real_seeds = n_real_seeds
        self.num_boost_round = num_boost_round
        self.random_state = random_state
        self.kept_: list[str] = []
        self.report_: pd.DataFrame | None = None

    def _gain(self, X: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
        params = {**self._PARAMS, "seed": seed}
        cat_cols = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        dtrain = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=True)
        booster = lgb.train(params, dtrain, num_boost_round=self.num_boost_round)
        return booster.feature_importance(importance_type="gain")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NullImportanceSelector":
        rng = np.random.RandomState(self.random_state)
        yv = y.values

        real = np.mean(
            [self._gain(X, yv, seed=self.random_state + s) for s in range(self.n_real_seeds)],
            axis=0,
        )

        nulls = np.zeros((self.n_null_runs, X.shape[1]))
        for run in range(self.n_null_runs):
            y_shuf = rng.permutation(yv)
            nulls[run] = self._gain(X, y_shuf, seed=self.random_state + 1000 + run)
            if (run + 1) % 10 == 0:
                logger.info("Null-importance: %d/%d runs", run + 1, self.n_null_runs)

        thresh = np.percentile(nulls, self.percentile, axis=0)
        keep_mask = real > thresh

        self.report_ = pd.DataFrame(
            {
                "feature": X.columns,
                "real_gain": real,
                "null_gain_p95": thresh,
                "null_gain_max": nulls.max(axis=0),
                "gain_ratio": real / np.where(thresh > 0, thresh, np.nan),
                "kept": keep_mask,
            }
        ).sort_values("real_gain", ascending=False)
        self.kept_ = self.report_.loc[self.report_["kept"], "feature"].tolist()
        logger.info(
            "NullImportance: keeping %d / %d features", len(self.kept_), X.shape[1]
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[[c for c in self.kept_ if c in X.columns]]
