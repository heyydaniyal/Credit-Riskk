"""
THE canonical deployed scoring path.  Phase 6's API and Phase 7's holdout
ceremony MUST call this — never re-assemble the chain themselves.

Why this module exists (audit finding): the pipeline lived as five separate
pieces (fold ensemble, calibrator, cost params, reason codes, categorical
contract).  Two independent re-assemblies of a five-step chain WILL diverge
somewhere — a forgotten clip, a different cast order, the wrong calibrator.
One code path + a golden regression test closes that failure class.

Chain: contract cast -> constrained fold-ensemble mean -> isotonic (clip)
       -> t*(x) decision -> reason codes -> score scale.
"""

import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from config.constants import ARTIFACTS_DIR, DATA_PROCESSED, MODELS_DIR
from src.features.serving import cast_with_contract
from src.models.decision import attach_cost_params, instance_policy
from src.models.explain import pd_to_score, reason_codes

# Displayed-probability clip: isotonic's edge bins emit exactly 0.0 / 1.0,
# which is epistemically wrong to DISPLAY (no applicant is a certain
# default).  Decisions are unaffected (t*(x) <= ~0.23 everywhere).
DISPLAY_PD_CLIP = (1e-4, 0.999)


class DeployedPipeline:
    """Loads the deployed artifacts once; scores feature-frames end-to-end."""

    def __init__(self, models_dir: str = MODELS_DIR, data_dir: str = DATA_PROCESSED):
        """Default dirs = the training outputs (models/, data/processed/), used
        by the offline scripts. Serving uses from_artifacts()."""
        with open(os.path.join(data_dir, "lgbm_features.json")) as f:
            self.features: list[str] = json.load(f)["features"]
        self.levels: dict = json.load(
            open(os.path.join(data_dir, "categorical_levels.json")))["levels"]
        self.boosters = [
            lgb.Booster(model_file=os.path.join(models_dir,
                                                f"lgbm_constrained_fold{k}.txt"))
            for k in range(5)
        ]
        with open(os.path.join(models_dir, "isotonic_final.pkl"), "rb") as f:
            self.calibrator = pickle.load(f)

        # Fast path precondition (checked, not assumed): the serving contract's
        # categorical levels are exactly the categories every booster was
        # trained with, in order — so integer codes from the contract ARE the
        # codes LightGBM would compute from a pandas frame.
        self._cat_cols = [c for c in self.features if c in self.levels]
        for b in self.boosters:
            trained = [list(c) for c in (b.pandas_categorical or [])]
            contract = [list(self.levels[c]) for c in self._cat_cols]
            if trained != contract:
                raise ValueError("categorical contract != booster categories; "
                                 "the serving artifacts are inconsistent")

    @classmethod
    def from_artifacts(cls, artifacts_dir: str = ARTIFACTS_DIR) -> "DeployedPipeline":
        """The serving constructor: load from the hash-gated deployment bundle
        (artifacts/ + MANIFEST.json) — the same bytes CI tests and the image
        verifies. Byte-identical to models/ + data/processed/ when
        `make artifacts` has been run (asserted by the manifest test)."""
        return cls(models_dir=artifacts_dir, data_dir=artifacts_dir)

    # ── internals ──────────────────────────────────────────────────────────
    def _matrix(self, casted: pd.DataFrame) -> np.ndarray:
        """Contract-cast frame -> float64 matrix in model-feature order, with
        categoricals as their (verified) training codes and NaN for missing.
        Built ONCE per request and shared by all 5 fold models and the
        reason-code pass: LightGBM's per-call pandas conversion was over half
        of single-row latency. Bit-identical predictions and contributions
        (verified on 3,000 rows and by the golden tests)."""
        X = casted[self.features].copy()
        for c in self._cat_cols:
            X[c] = X[c].cat.codes.astype("float64").replace(-1, np.nan)
        return X.to_numpy(dtype=np.float64)

    def _raw_ensemble(self, M: np.ndarray) -> np.ndarray:
        return np.mean([b.predict(M) for b in self.boosters], axis=0)

    # ── public API ─────────────────────────────────────────────────────────
    def score_frame(self, df: pd.DataFrame, with_reason_codes: bool = True,
                    cost_overrides: dict | None = None) -> pd.DataFrame:
        """
        Score a feature frame (must contain the 58 model features plus
        NAME_CONTRACT_TYPE, term_years, AMT_CREDIT for the decision layer).

        Returns one row per applicant with every input to the decision —
        the spec's auditability requirement: pd_calibrated, decision,
        threshold_applied, lgd/ead/margin used, score_scaled, unseen-
        category counts (drift signal), and optionally reason codes.

        cost_overrides: optional sensitivity exploration inside the frozen
        grids (decision.validate_overrides); the PD is unaffected, only
        t*(x) and the decision move. None reproduces the deployed decision.
        """
        casted, unseen = cast_with_contract(df, self.levels)
        M = self._matrix(casted)
        p_raw = self._raw_ensemble(M)
        p_cal = self.calibrator.predict(p_raw)
        p_disp = np.clip(p_cal, *DISPLAY_PD_CLIP)

        cost = attach_cost_params(casted, overrides=cost_overrides)
        approved = instance_policy(p_cal, cost)

        out = pd.DataFrame(
            {
                "pd_calibrated": p_disp,
                "pd_raw_ensemble": p_raw,
                "decision": np.where(approved, "APPROVE", "REJECT"),
                "threshold_applied": cost["t_star"].values,
                "lgd_used": cost["lgd"].values,
                "ead_factor_used": cost["ead_factor"].values,
                "margin_used": cost["margin"].values,
                "cure_rate_used": cost["cure_rate"].values,
                "term_years_used": cost["term_years_eff"].values,
                "term_fallback_used": cost["term_fallback_used"].values,
                "score_scaled": pd_to_score(p_cal),
            },
            index=df.index,
        )
        out.attrs["unseen_category_counts"] = unseen
        if with_reason_codes:
            # fold-0 booster for explanations (stated in docs); ranking is
            # stable across folds and isotonic-monotone to the decision
            out["reason_codes"] = reason_codes(self.boosters[0], M,
                                               self.features, top_k=3)
        return out
