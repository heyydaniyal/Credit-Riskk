"""
Phase 6 API tests — the transport layer must add validation and nothing else.

  golden-through-HTTP   the 25 golden applicants scored via the HTTP layer
                        must reproduce the frozen outputs exactly (the
                        system-level skew guard, extended to transport)
  schema rejections     impossible values, bad types, missing AMT_CREDIT
  reason codes          exactly 3 for a rejected applicant (spec test)
  bounded overrides     in-grid moves the threshold; out-of-grid → 422
"""

import json
import os

import pandas as pd
import pytest

from config.constants import ARTIFACTS_DIR

# Everything here runs from the committed, hash-gated bundle — so it runs in
# CI and in a fresh clone (previously it read data/processed/, which is
# gitignored, and all five API tests silently skipped in CI).
DEPLOYED = os.path.join(ARTIFACTS_DIR, "lgbm_constrained_fold0.txt")
GOLDEN = os.path.join(ARTIFACTS_DIR, "golden_scoring.json")
GOLDEN_INPUTS = os.path.join(ARTIFACTS_DIR, "golden_inputs.parquet")

needs_deploy = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (DEPLOYED, GOLDEN, GOLDEN_INPUTS)),
    reason="deployment bundle absent (run `make artifacts`)",
)


@pytest.fixture(scope="module")
def client_and_payloads():
    from fastapi.testclient import TestClient

    from src.api.app import API_FIELDS, CATS, app

    golden = json.load(open(GOLDEN))
    rows = pd.read_parquet(GOLDEN_INPUTS).set_index("SK_ID_CURR")
    payloads = []
    for i in golden["ids"]:
        row = rows.loc[i, API_FIELDS]
        payloads.append(
            {f: (None if pd.isna(row[f])
                 else (str(row[f]) if f in CATS else float(row[f])))
             for f in API_FIELDS}
        )
    return TestClient(app), payloads, golden


@needs_deploy
def test_golden_through_http(client_and_payloads):
    client, payloads, golden = client_and_payloads
    for i, p in enumerate(payloads):
        r = client.post("/score", json=p)
        assert r.status_code == 200
        body = r.json()
        assert round(body["pd_calibrated"], 10) == golden["pd_calibrated"][i]
        assert body["decision"] == golden["decision"][i]
        assert round(body["score_scaled"], 6) == golden["score_scaled"][i]


@needs_deploy
def test_schema_rejections(client_and_payloads):
    client, payloads, _ = client_and_payloads
    base = payloads[0]
    assert client.post("/score", json={**base, "EXT_SOURCE_2": 5.0}).status_code == 422
    assert client.post("/score", json={**base, "AMT_CREDIT": -1.0}).status_code == 422
    assert client.post("/score", json={**base, "AMT_CREDIT": None}).status_code == 422
    assert client.post("/score", json={**base, "EXT_SOURCE_1": "text"}).status_code == 422


@needs_deploy
def test_rejected_applicant_gets_exactly_three_reason_codes(client_and_payloads):
    client, payloads, golden = client_and_payloads
    idx = golden["decision"].index("REJECT")
    body = client.post("/score", json=payloads[idx]).json()
    assert body["decision"] == "REJECT"
    assert len(body["reason_codes"]) == 3
    assert all(c["contribution"] > 0 for c in body["reason_codes"])


@needs_deploy
def test_cost_overrides_bounded_by_frozen_grids(client_and_payloads):
    client, payloads, _ = client_and_payloads
    base = payloads[0]
    ok = client.post("/score", json={**base, "cost_overrides": {"r_net_cash": 0.08}})
    assert ok.status_code == 200
    plain = client.post("/score", json=base).json()
    assert ok.json()["threshold_applied"] > plain["threshold_applied"]  # richer margin → lenient
    bad = client.post("/score", json={**base, "cost_overrides": {"lgd_cash": 0.99}})
    assert bad.status_code == 422  # outside the frozen sensitivity grid


@needs_deploy
def test_decision_layer_inputs_are_validated(client_and_payloads):
    """Regression for the F1.3 production defects: unknown/null contract type
    must 422 (not 500), and non-positive term_years must 422 (not return a
    negative threshold in a 200). Mirrors the ceremony's pre-flight gate."""
    client, payloads, _ = client_and_payloads
    base = payloads[0]
    assert client.post("/score", json={**base, "NAME_CONTRACT_TYPE": "MYSTERY"}).status_code == 422
    assert client.post("/score", json={**base, "NAME_CONTRACT_TYPE": None}).status_code == 422
    assert client.post("/score", json={**base, "term_years": -3.0}).status_code == 422
    assert client.post("/score", json={**base, "term_years": 0.0}).status_code == 422
    # the valid path is untouched
    assert client.post("/score", json=base).status_code == 200


# ── input-gate regressions (each reproduces a defect found in review) ──────
@needs_deploy
def test_term_years_cannot_be_chosen_by_the_client(client_and_payloads):
    """Review probe: a 43.8%-PD reject flipped to APPROVE by sending
    term_years=7 (t* 0.067 → 0.227); term_years=1000 gave t* = 0.98.
    term_years is derived server-side; a contradicting value is a 422, and
    omitting it yields exactly the supplied-and-consistent result."""
    client, payloads, golden = client_and_payloads
    idx = golden["decision"].index("REJECT")
    base = payloads[idx]
    for t in (7.0, 50.0, 1000.0):
        r = client.post("/score", json={**base, "term_years": t})
        assert r.status_code == 422 and "term_years" in r.json()["detail"]
    with_term = client.post("/score", json=base).json()
    without = client.post("/score", json={**base, "term_years": None}).json()
    assert without["threshold_applied"] == with_term["threshold_applied"]
    assert without["pd_calibrated"] == with_term["pd_calibrated"]


@needs_deploy
def test_other_derived_features_must_match_their_sources(client_and_payloads):
    client, payloads, _ = client_and_payloads
    base = payloads[0]
    for field, bad in (("ext_source_mean", 0.99), ("credit_goods_ratio", 9.0),
                       ("ext_source_n_missing", 3.0)):
        r = client.post("/score", json={**base, field: bad})
        assert r.status_code == 422 and field in r.json()["detail"], field


@needs_deploy
def test_near_empty_payload_is_refused_not_approved(client_and_payloads):
    """Review probe: AMT_CREDIT + contract type only (57/58 features null)
    returned 200 APPROVE at PD 2.2%. Such inputs are outside the training
    support → 422. Same for 'history unknown' (all aggregates null), which no
    training row has; a genuine no-history applicant (zero counts) scores."""
    client, payloads, _ = client_and_payloads
    r = client.post("/score", json={"AMT_CREDIT": 1000.0, "NAME_CONTRACT_TYPE": "Cash loans"})
    assert r.status_code == 422 and "application-side" in r.json()["detail"]

    from config.constants import HISTORY_FEATURE_PREFIXES, NO_HISTORY_ZERO_FEATURES

    hist = [f for f in payloads[0] if f.startswith(HISTORY_FEATURE_PREFIXES)]
    unknown = {**payloads[0], **{f: None for f in hist}}
    r = client.post("/score", json=unknown)
    assert r.status_code == 422 and "credit-history" in r.json()["detail"]
    no_history = {**unknown, **{f: 0.0 for f in NO_HISTORY_ZERO_FEATURES}}
    assert client.post("/score", json=no_history).status_code == 200


@needs_deploy
def test_unknown_fields_rejected_including_protected_attribute(client_and_payloads):
    client, payloads, _ = client_and_payloads
    base = payloads[0]
    assert client.post("/score", json={**base, "CODE_GENDER": "M"}).status_code == 422
    assert client.post("/score", json={**base, "cost_overrides": {"bogus": 0.1}}).status_code == 422


@needs_deploy
def test_cure_rate_override_and_response_audit_fields(client_and_payloads):
    client, payloads, _ = client_and_payloads
    base = payloads[0]
    plain = client.post("/score", json=base).json()
    cured = client.post("/score", json={**base, "cost_overrides": {"cure_rate": 0.5}}).json()
    assert cured["threshold_applied"] > plain["threshold_applied"]
    assert cured["cure_rate_used"] == 0.5 and plain["cure_rate_used"] == 0.0
    assert cured["pd_calibrated"] == plain["pd_calibrated"]  # economics never move the PD
    assert plain["term_years_used"] > 0 and plain["cost_overrides_used"] == {}
    assert client.post("/score", json={**base, "cost_overrides": {"cure_rate": 0.9}}).status_code == 422
    bounds = client.get("/cost-bounds").json()["bounds"]
    assert bounds["cure_rate"] == [0.0, 0.5]
