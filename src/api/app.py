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
- extra fields are FORBIDDEN (422): a protected attribute such as
  CODE_GENDER cannot even reach the scoring path.
- derived features are recomputed from their sources and a contradicting
  value is rejected (serving.derive_and_check) — term_years in particular,
  which drives both the PD and t*(x).
- training-support gate: more missing application-side or credit-history
  features than any training row had → 422, not a confident low PD.
- /score accepts optional cost_overrides BOUNDED BY THE FROZEN SENSITIVITY
  GRIDS — the demo can explore sensitivity, never invent economics.
- every input to the decision is returned; cost params labeled illustrative.
- log_request() writes to SQLite for the monitoring tab.
"""

import hashlib
import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, create_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.constants import (  # noqa: E402
    ARTIFACTS_DIR,
    HISTORY_FEATURE_PREFIXES,
    MAX_MISSING_APPLICATION_FEATURES,
    MAX_MISSING_HISTORY_FEATURES,
)
from src.features.serving import derive_and_check  # noqa: E402
from src.models.decision import LGD, OVERRIDE_BOUNDS  # noqa: E402
from src.models.pipeline import DeployedPipeline  # noqa: E402
from src.monitoring import init_log, log_request  # noqa: E402

# ── load once at startup (from the hash-gated bundle) ───────────────────────
# Model provenance: hash of the artifact manifest, so every scored decision
# can be tied to the exact model build that produced it (F1.9/F9.3).
try:
    with open(os.path.join(ARTIFACTS_DIR, "MANIFEST.json")) as _f:
        MODEL_VERSION = hashlib.sha256(_f.read().encode()).hexdigest()[:12]
except FileNotFoundError:
    MODEL_VERSION = "unversioned"

PIPE = DeployedPipeline.from_artifacts()
CATS = set(PIPE.levels)
API_FIELDS = list(dict.fromkeys(PIPE.features
                                + ["NAME_CONTRACT_TYPE", "term_years", "AMT_CREDIT"]))

_fields = {f: ((str | None), None) if f in CATS else ((float | None), None)
           for f in API_FIELDS}
HIST_FEATURES = [f for f in PIPE.features if f.startswith(HISTORY_FEATURE_PREFIXES)]
APP_FEATURES = [f for f in PIPE.features if f not in HIST_FEATURES]

_STRICT = ConfigDict(extra="forbid")


class CostOverrides(BaseModel):
    """Bounded by the frozen sensitivity grids (decision.OVERRIDE_BOUNDS)."""
    model_config = _STRICT
    lgd_cash: float | None = None
    lgd_revolving: float | None = None
    ead_cash: float | None = None
    ead_revolving: float | None = None
    r_net_cash: float | None = None
    m_revolving: float | None = None
    cure_rate: float | None = None


# single flat body: 59 feature fields + optional nested cost_overrides
ScoreRequest = create_model("ScoreRequest", __config__=_STRICT,
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
    if row.get("AMT_CREDIT") is None:
        raise HTTPException(422, "AMT_CREDIT is required (decision-layer input)")

    # Decision-layer pre-flight (ports the ceremony's guard into serving).
    ct = row.get("NAME_CONTRACT_TYPE")
    if ct is None:
        raise HTTPException(422, "NAME_CONTRACT_TYPE is required (drives LGD/EAD/threshold)")
    if str(ct) not in LGD:
        raise HTTPException(
            422, f"NAME_CONTRACT_TYPE '{ct}' is not a scored product "
                 f"(known: {sorted(LGD)}); cost parameters would be undefined")


def _gate(row: dict) -> dict:
    """Consistency + minimum-information gate. Returns the row to score."""
    row, problems = derive_and_check(row)
    if problems:
        raise HTTPException(422, "inconsistent derived features (they are recomputed "
                                 "server-side from their sources): " + "; ".join(problems))
    for block, feats, limit in (("application-side", APP_FEATURES, MAX_MISSING_APPLICATION_FEATURES),
                                ("credit-history", HIST_FEATURES, MAX_MISSING_HISTORY_FEATURES)):
        n_missing = sum(row.get(f) is None for f in feats)
        if n_missing > limit:
            raise HTTPException(
                422, f"{n_missing} of {len(feats)} {block} features missing; no training "
                     f"applicant had more than {limit}. Refusing to score an input outside "
                     "the training support"
                     + (" (a no-history applicant has zero-valued count aggregates, "
                        "not missing ones)" if block == "credit-history" else "") + ".")
    return row


app = FastAPI(title="Credit Risk Scoring Service",
              description="Calibrated PD + instance-dependent cost decisioning. "
                          "Cost parameters are ILLUSTRATIVE economic assumptions.")
init_log()


@app.get("/health")
def health():
    return {"status": "ok", "folds": len(PIPE.boosters),
            "n_features": len(PIPE.features), "model_version": MODEL_VERSION}


@app.get("/cost-bounds")
def cost_bounds():
    """The admissible override ranges (= frozen sensitivity-grid spans)."""
    return {"bounds": OVERRIDE_BOUNDS,
            "note": "illustrative economic assumptions — explore sensitivity only"}


@app.post("/score")
def score(req: ScoreRequest):
    row = req.model_dump()
    overrides = row.pop("cost_overrides", None)
    _sanity(row)
    row = _gate(row)
    df = pd.DataFrame([row])
    for f in API_FIELDS:  # JSON null → NaN for numerics
        if f not in CATS:
            df[f] = pd.to_numeric(df[f], errors="coerce")

    try:
        out = PIPE.score_frame(df, with_reason_codes=True, cost_overrides=overrides)
    except ValueError as e:  # override outside the frozen grid
        raise HTTPException(422, str(e)) from e
    except Exception as _e:  # noqa: BLE001 - surface a clean error, not a stack trace
        raise HTTPException(500, f"scoring failed: {type(_e).__name__}") from _e
    r = out.iloc[0]
    threshold = float(r.threshold_applied)

    # Invariant: a threshold outside (0,1) is never a valid decision.
    if not (0.0 < threshold < 1.0):
        raise HTTPException(422, f"computed threshold {threshold:.4f} is outside (0,1)")

    overrides_used = {k: v for k, v in (overrides or {}).items() if v is not None}
    log_request(row, float(r.pd_raw_ensemble), float(r.pd_calibrated),
                threshold, str(r.decision))
    return {
        "pd_calibrated": float(r.pd_calibrated),
        "decision": str(r.decision),
        "threshold_applied": threshold,
        "lgd_used": float(r.lgd_used),
        "ead_factor_used": float(r.ead_factor_used),
        "margin_used": float(r.margin_used),
        "cure_rate_used": float(r.cure_rate_used),
        "term_years_used": float(r.term_years_used),
        "term_fallback_used": bool(r.term_fallback_used),
        "reason_codes": r.reason_codes,
        "score_scaled": float(r.score_scaled),
        "unseen_category_counts": out.attrs.get("unseen_category_counts", {}),
        "cost_overrides_used": overrides_used,
        "model_version": MODEL_VERSION,
        "note": "cost parameters are illustrative economic assumptions; monetary amounts "
                "are in the dataset's unspecified currency units, not euros",
    }
