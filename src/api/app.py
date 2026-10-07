"""
Phase 6 — the scoring service.  A thin, validated transport around the
canonical DeployedPipeline (never re-assembles the chain — that is the
whole point of the golden-guarded pipeline).

Frozen-design decisions implemented here:
- pydantic model generated programmatically from the feature manifest
  (59 fields); numeric = Optional[float] (JSON null → NaN); categoricals =
  UNCONSTRAINED Optional[str] — unseen values are NaN-routed by the
  contract WITH counts in the response, because drift observability beats
  strict rejection.
- hand-written range checks on the top SHAP features: reject only the
  impossible (negative amounts, EXT_SOURCE outside [0,1]).
- /score accepts optional cost_overrides BOUNDED BY THE FROZEN SENSITIVITY
  GRIDS — the demo can explore sensitivity, never invent economics.
- every input to the decision is returned; cost params labeled illustrative.
- log_request() writes to SQLite for the monitoring tab.
"""

import os
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.decision import EAD_FACTOR, LGD, M_REVOLVING, R_NET_CASH
from src.models.pipeline import DeployedPipeline
from src.monitoring import init_log, log_request

# ── load once at startup ───────────────────────────────────────────────────

# Model provenance: hash of the artifact manifest, so every scored decision
# can be tied to the exact model build that produced it (F1.9/F9.3).
import hashlib as _hashlib
try:
    _manifest_txt = open(os.path.join("artifacts", "MANIFEST.json")).read()
    MODEL_VERSION = _hashlib.sha256(_manifest_txt.encode()).hexdigest()[:12]
except FileNotFoundError:
    MODEL_VERSION = "unversioned"

PIPE = DeployedPipeline()
CATS = set(PIPE.levels)
API_FIELDS = list(dict.fromkeys(PIPE.features
                                + ["NAME_CONTRACT_TYPE", "term_years", "AMT_CREDIT"]))

_fields = {f: ((str | None), None) if f in CATS else ((float | None), None)
           for f in API_FIELDS}

# frozen sensitivity grids — the ONLY admissible override ranges
OVERRIDE_BOUNDS = {
    "lgd_cash": (0.60, 0.80), "lgd_revolving": (0.75, 0.95),
    "ead_cash": (0.75, 0.95), "ead_revolving": (0.85, 1.00),
    "r_net_cash": (0.03, 0.08), "m_revolving": (0.10, 0.30),
}


class CostOverrides(BaseModel):
    lgd_cash: float | None = None
    lgd_revolving: float | None = None
    ead_cash: float | None = None
    ead_revolving: float | None = None
    r_net_cash: float | None = None
    m_revolving: float | None = None


# single flat body: 59 feature fields + optional nested cost_overrides
ScoreRequest = create_model("ScoreRequest",
                            cost_overrides=(CostOverrides | None, None), **_fields)


# impossible-value checks on the top features (reject) — everything else
# is the model's job, not the transport's
def _sanity(row: dict) -> None:
    for f in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "ext_source_mean",
              "ext_source_min"):
        v = row.get(f)
        if v is not None and not (0.0 <= v <= 1.0):
            raise HTTPException(422, f"{f} must be in [0,1], got {v}")
    for f in ("AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY"):
        v = row.get(f)
        if v is not None and v < 0:
            raise HTTPException(422, f"{f} must be non-negative, got {v}")
    if row.get("AMT_CREDIT") in (None,):
        raise HTTPException(422, "AMT_CREDIT is required (decision-layer input)")

    # Decision-layer pre-flight (ports the ceremony's guard into serving).
    # These two fields DRIVE t*(x); an unhandled bad value here yields either a
    # 500 (NaN cost params -> non-JSON-compliant response) or, worse, a valid
    # 200 with a nonsensical threshold. Validate them explicitly.
    ct = row.get("NAME_CONTRACT_TYPE")
    if ct is None:
        raise HTTPException(422, "NAME_CONTRACT_TYPE is required (drives LGD/EAD/threshold)")
    if str(ct) not in LGD:
        raise HTTPException(
            422, f"NAME_CONTRACT_TYPE '{ct}' is not a scored product "
                 f"(known: {sorted(LGD)}); cost parameters would be undefined")
    term = row.get("term_years")
    if term is not None and term <= 0:
        raise HTTPException(422, f"term_years must be positive when provided, got {term}")


def _apply_overrides(cost_df: pd.DataFrame, ov: CostOverrides) -> tuple[pd.DataFrame, dict]:
    used = {}
    d = cost_df.copy()
    cash = (d["NAME_CONTRACT_TYPE"].astype(str) == "Cash loans").values
    for name, val in ov.model_dump().items():
        if val is None:
            continue
        lo, hi = OVERRIDE_BOUNDS[name]
        if not (lo <= val <= hi):
            raise HTTPException(
                422, f"{name}={val} outside the frozen sensitivity grid [{lo}, {hi}]")
        used[name] = val
    lgd = np.where(cash, used.get("lgd_cash", LGD["Cash loans"]),
                   used.get("lgd_revolving", LGD["Revolving loans"]))
    ead = np.where(cash, used.get("ead_cash", EAD_FACTOR["Cash loans"]),
                   used.get("ead_revolving", EAD_FACTOR["Revolving loans"]))
    m = np.where(cash,
                 used.get("r_net_cash", R_NET_CASH) * d["term_years_eff"].values / 2,
                 used.get("m_revolving", M_REVOLVING))
    d["lgd"], d["ead_factor"], d["margin"] = lgd, ead, m
    d["t_star"] = m / (m + lgd * ead)
    return d, used


app = FastAPI(title="Credit Risk Scoring Service",
              description="Calibrated PD + instance-dependent cost decisioning. "
                          "Cost parameters are ILLUSTRATIVE economic assumptions.")
init_log()


@app.get("/health")
def health():
    return {"status": "ok", "folds": len(PIPE.boosters),
            "n_features": len(PIPE.features), "model_version": MODEL_VERSION}


@app.post("/score")
def score(req: ScoreRequest):
    row = req.model_dump()
    overrides = req.cost_overrides
    row.pop("cost_overrides", None)
    _sanity(row)
    df = pd.DataFrame([row])
    for f in API_FIELDS:  # JSON null → NaN for numerics
        if f not in CATS:
            df[f] = pd.to_numeric(df[f], errors="coerce")

    try:
        out = PIPE.score_frame(df, with_reason_codes=True)
    except Exception as _e:  # noqa: BLE001 - surface a clean error, not a stack trace
        raise HTTPException(500, f"scoring failed: {type(_e).__name__}") from _e
    r = out.iloc[0]
    threshold, lgd, ead, margin = (r.threshold_applied, r.lgd_used,
                                   r.ead_factor_used, r.margin_used)
    decision = r.decision
    overrides_used = {}
    if overrides is not None and any(v is not None
                                     for v in overrides.model_dump().values()):
        from src.models.decision import attach_cost_params

        base = attach_cost_params(df)
        cost2, overrides_used = _apply_overrides(base, overrides)
        threshold = float(cost2["t_star"].iloc[0])
        lgd, ead, margin = (float(cost2["lgd"].iloc[0]),
                            float(cost2["ead_factor"].iloc[0]),
                            float(cost2["margin"].iloc[0]))
        decision = "APPROVE" if r.pd_calibrated < threshold else "REJECT"

    # Invariant: a threshold outside (0,1) is never a valid decision. This
    # catches any residual path (e.g. override arithmetic) that _sanity missed.
    if not (0.0 < float(threshold) < 1.0):
        raise HTTPException(422, f"computed threshold {threshold:.4f} is outside (0,1); "
                                 "check term_years and cost_overrides")

    log_request(row, float(r.pd_raw_ensemble), float(r.pd_calibrated),
                float(threshold), str(decision))
    return {
        "pd_calibrated": float(r.pd_calibrated),
        "decision": str(decision),
        "threshold_applied": float(threshold),
        "lgd_used": float(lgd),
        "ead_factor_used": float(ead),
        "margin_used": float(margin),
        "term_fallback_used": bool(r.term_fallback_used),
        "reason_codes": r.reason_codes,
        "score_scaled": float(r.score_scaled),
        "unseen_category_counts": out.attrs.get("unseen_category_counts", {}),
        "cost_overrides_used": overrides_used,
        "model_version": MODEL_VERSION,
        "note": "cost parameters are illustrative economic assumptions; monetary amounts are in the dataset's unspecified currency units, not euros",
    }
