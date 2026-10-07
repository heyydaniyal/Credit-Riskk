"""
WOE/IV binning for the logistic scorecard baseline (Siddiqi 2006).

Weight of Evidence per bin:  WOE = ln(%good_in_bin / %bad_in_bin)
Information Value:           IV  = Σ (%good − %bad) × WOE

Why WOE *is* the preprocessing for the scorecard (spec Phase 2):
- every feature lands on the same log-odds scale → no StandardScaler;
- NaN gets its own bin → no imputation, missingness signal preserved;
- outliers land in an edge bin → no clipping rules;
- non-monotone relationships (e.g. default vs term, EDA #7) become
  linear in WOE space because each bin carries its own empirical rate.

LEAKAGE CONTRACT: fit() learns bin edges and WOE values from the data it
is given.  Phase 3 must fit this binner INSIDE each CV fold (fit on 4
folds, transform the 5th) — never once on the full training set.  The
class is deliberately a fit/transform object so that discipline is
possible; a module-level function would invite leaking.
"""

import numpy as np
import pandas as pd

_SMOOTH = 0.5  # Laplace smoothing on good/bad counts: no bin ever yields ±inf


class WOEBinner:
    """
    Quantile bins for numeric features (NaN = own bin), category grouping
    for categoricals (rare < min_bin_frac → '__OTHER__', NaN → '__MISSING__').
    Unseen categories at transform time map to the '__OTHER__' WOE (or 0.0
    if no other-bin exists) — live traffic will contain values training
    never saw, and a scorecard must score them, not crash.
    """

    def __init__(self, n_bins: int = 5, min_bin_frac: float = 0.01):
        self.n_bins = n_bins
        self.min_bin_frac = min_bin_frac
        self.numeric_bins_: dict[str, dict] = {}   # col -> {edges, woes[list], nan_woe}
        self.categorical_bins_: dict[str, dict] = {}  # col -> {mapping, other_woe, nan_woe}
        self.iv_: dict[str, float] = {}

    # ── internals ──────────────────────────────────────────────────────────
    @staticmethod
    def _woe_iv(goods: np.ndarray, bads: np.ndarray) -> tuple[np.ndarray, float]:
        """Vector of per-bin WOE + total IV, with Laplace smoothing."""
        g = goods + _SMOOTH
        b = bads + _SMOOTH
        pg = g / g.sum()
        pb = b / b.sum()
        woe = np.log(pg / pb)
        iv = float(((pg - pb) * woe).sum())
        return woe, iv

    def _fit_numeric(self, x: pd.Series, y: pd.Series) -> tuple[dict, float]:
        mask = x.notna()
        xv, yv = x[mask].values.astype(float), y[mask].values
        edges = np.unique(np.quantile(xv, np.linspace(0, 1, self.n_bins + 1)))
        if len(edges) < 3:  # near-constant → single bin, IV ≈ 0
            edges = np.array([-np.inf, np.inf])
        edges[0], edges[-1] = -np.inf, np.inf

        bin_idx = np.digitize(xv, edges[1:-1], right=True)
        n_bins = len(edges) - 1
        goods = np.array([(yv[bin_idx == i] == 0).sum() for i in range(n_bins)], dtype=float)
        bads = np.array([(yv[bin_idx == i] == 1).sum() for i in range(n_bins)], dtype=float)

        # NaN bin appended as one extra good/bad pair so IV counts it
        n_nan = int((~mask).sum())
        if n_nan > 0:
            goods = np.append(goods, float((y[~mask] == 0).sum()))
            bads = np.append(bads, float((y[~mask] == 1).sum()))

        woes, iv = self._woe_iv(goods, bads)
        nan_woe = float(woes[-1]) if n_nan > 0 else 0.0
        bin_woes = woes[:n_bins] if n_nan > 0 else woes
        return {"edges": edges, "woes": bin_woes.tolist(), "nan_woe": nan_woe}, iv

    def _fit_categorical(self, x: pd.Series, y: pd.Series) -> tuple[dict, float]:
        xs = x.astype(object).where(x.notna(), "__MISSING__")
        freq = xs.value_counts(normalize=True)
        rare = set(freq[freq < self.min_bin_frac].index)
        xs = xs.where(~xs.isin(rare), "__OTHER__")

        cats = pd.Index(xs.unique())
        goods = np.array([((xs == c) & (y == 0)).sum() for c in cats], dtype=float)
        bads = np.array([((xs == c) & (y == 1)).sum() for c in cats], dtype=float)
        woes, iv = self._woe_iv(goods, bads)

        mapping = dict(zip(cats, woes.astype(float), strict=True))
        return {
            "mapping": mapping,
            "other_woe": mapping.get("__OTHER__", 0.0),
            "nan_woe": mapping.get("__MISSING__", 0.0),
        }, iv

    # ── public API ─────────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEBinner":
        self.numeric_bins_.clear()
        self.categorical_bins_.clear()
        self.iv_.clear()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                bins, iv = self._fit_numeric(X[col], y)
                self.numeric_bins_[col] = bins
            else:
                bins, iv = self._fit_categorical(X[col], y)
                self.categorical_bins_[col] = bins
            self.iv_[col] = iv
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col, bins in self.numeric_bins_.items():
            x = X[col].values.astype(float)
            idx = np.digitize(x, bins["edges"][1:-1], right=True)
            woes = np.asarray(bins["woes"])
            vals = woes[np.clip(idx, 0, len(woes) - 1)]
            vals = np.where(np.isnan(x), bins["nan_woe"], vals)
            out[f"woe_{col}"] = vals
        for col, bins in self.categorical_bins_.items():
            xs = X[col].astype(object).where(X[col].notna(), "__MISSING__")
            vals = xs.map(bins["mapping"])
            vals = vals.fillna(bins["other_woe"]).astype(float)  # unseen → other
            out[f"woe_{col}"] = vals.values
        return pd.DataFrame(out, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        rows = [
            {
                "feature": c,
                "iv": iv,
                "type": "numeric" if c in self.numeric_bins_ else "categorical",
                "n_bins": (
                    len(self.numeric_bins_[c]["woes"]) + (1 if self.numeric_bins_[c]["nan_woe"] else 0)
                    if c in self.numeric_bins_
                    else len(self.categorical_bins_[c]["mapping"])
                ),
            }
            for c, iv in self.iv_.items()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def select_by_iv(self, iv_min: float = 0.02, iv_max: float = 0.50) -> list[str]:
        """Scorecard rule: IV < 0.02 useless; IV > 0.5 suspicious (audit for leakage)."""
        return [c for c, iv in self.iv_.items() if iv_min <= iv <= iv_max]
