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

DEPLOYED = os.path.join("models", "lgbm_constrained_fold0.txt")
GOLDEN = os.path.join("data", "processed", "golden_scoring.json")

needs_deploy = pytest.mark.skipif(
    not (os.path.exists(DEPLOYED) and os.path.exists(GOLDEN)),
    reason="deployed artifacts absent",
)


@pytest.fixture(scope="module")
def client_and_payloads():
    from fastapi.testclient import TestClient

    from src.api.app import API_FIELDS, CATS, app
    from src.data.load import load_modeling_frame

    golden = json.load(open(GOLDEN))
    dev = load_modeling_frame("dev").set_index("SK_ID_CURR")
    payloads = []
    for i in golden["ids"]:
        row = dev.loc[i, API_FIELDS]
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
