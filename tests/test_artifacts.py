"""
Artifact-consistency guards (integration tests).

Failure mode these protect against: someone regenerates the feature matrix
with changed code, and the saved feature LISTS (lgbm/woe/monotonic) go
silently stale — Phase 3+ would then train on a mismatched contract.

These tests need the Phase 2 run artifacts, which are gitignored — so they
SKIP (not fail) when artifacts are absent, keeping CI green while still
guarding every local/dev environment where the pipeline has run.
"""

import json
import os
import pickle

import pandas as pd
import pytest

from config.constants import (
    ARTIFACTS_DIR,
    DATA_PROCESSED,
    HISTORY_FEATURE_PREFIXES,
    MAX_MISSING_APPLICATION_FEATURES,
    MAX_MISSING_HISTORY_FEATURES,
    MODELS_DIR,
    MONOTONIC_FEATURES,
    NO_HISTORY_ZERO_FEATURES,
)

FEATURES = os.path.join(DATA_PROCESSED, "features_full.parquet")
LGBM_JSON = os.path.join(DATA_PROCESSED, "lgbm_features.json")
WOE_JSON = os.path.join(DATA_PROCESSED, "woe_features.json")
BINNER = os.path.join(MODELS_DIR, "woe_binner_dev.pkl")

needs_artifacts = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (FEATURES, LGBM_JSON, WOE_JSON, BINNER)),
    reason="Phase 2 artifacts not present (run `make features` first)",
)


@pytest.fixture(scope="module")
def artifacts():
    return {
        "cols": set(pd.read_parquet(FEATURES).columns),
        "lgbm": json.load(open(LGBM_JSON))["features"],
        "woe": json.load(open(WOE_JSON))["features"],
    }


@needs_artifacts
def test_feature_lists_exist_in_matrix(artifacts):
    assert set(artifacts["lgbm"]) <= artifacts["cols"], "lgbm list stale vs matrix"
    assert set(artifacts["woe"]) <= set(artifacts["lgbm"]), "woe list not a subset of lgbm"


@needs_artifacts
def test_no_contamination_in_feature_lists(artifacts):
    forbidden = {"TARGET", "SK_ID_CURR", "CODE_GENDER"}
    assert not forbidden & set(artifacts["lgbm"])
    assert not forbidden & set(artifacts["woe"])


@needs_artifacts
def test_monotonic_features_all_survived_selection(artifacts):
    """A constraint on a feature not in the model silently no-ops in Phase 5."""
    missing = [f for f in MONOTONIC_FEATURES if f not in set(artifacts["lgbm"])]
    assert not missing, f"monotonic constraints reference dropped features: {missing}"


@needs_artifacts
def test_scorecard_size_in_spec_range(artifacts):
    assert 20 <= len(artifacts["woe"]) <= 40


@needs_artifacts
def test_cost_model_inputs_present_in_matrix(artifacts):
    """AMT_CREDIT may leave the FEATURE list but must stay in the MATRIX —
    Phase 4's t*(x) reads it from the applicant record."""
    needed = {"AMT_CREDIT", "AMT_ANNUITY", "NAME_CONTRACT_TYPE", "term_years"}
    assert needed <= artifacts["cols"]


@needs_artifacts
def test_binner_loads_and_covers_scorecard(artifacts):
    binner = pickle.load(open(BINNER, "rb"))
    fitted = set(binner.numeric_bins_) | set(binner.categorical_bins_)
    assert fitted == set(artifacts["woe"]), "binner fit on a different feature set"


LEVELS = os.path.join(DATA_PROCESSED, "categorical_levels.json")


@pytest.mark.skipif(not os.path.exists(LEVELS), reason="Phase 3 audit artifact absent")
def test_serving_categorical_contract_matches_matrix():
    """Train/serve skew guard: an INCOMPLETE category level set silently
    corrupts predictions (audit experiment: 90/500 rows changed, no error).
    The persisted serving contract must exactly match the training matrix."""
    contract = json.load(open(LEVELS))["levels"]
    f = pd.read_parquet(FEATURES)
    for col, saved in contract.items():
        actual = [str(x) for x in f[col].cat.categories]
        assert saved == actual, f"{col}: persisted levels drifted from matrix"


PLAN = os.path.join("config", "holdout_analysis_plan.json")
CONVENTION = os.path.join("config", "cost_evaluation_convention.json")


@pytest.mark.skipif(not os.path.exists(CONVENTION), reason="convention not frozen yet")
def test_cost_convention_frozen_and_referenced():
    """Pre-registration guard: the accounting convention exists, pins the
    ledger + best-flat selection rule, and the holdout plan references it —
    so no headline can be produced under an unpinned convention."""
    c = json.load(open(CONVENTION))
    assert c["ledger_convention"]["name"] == "realized-outcome profit ledger"
    assert "DEV OOF" in c["baseline_definitions"]["best_flat"]
    p = json.load(open(PLAN))
    assert "cost_evaluation_convention.json" in p["inputs"]["cost_accounting"]


GOLDEN = os.path.join(ARTIFACTS_DIR, "golden_scoring.json")
GOLDEN_INPUTS = os.path.join(ARTIFACTS_DIR, "golden_inputs.parquet")
CONSTRAINED = os.path.join(ARTIFACTS_DIR, "lgbm_constrained_fold0.txt")


@pytest.mark.skipif(not all(os.path.exists(p) for p in (GOLDEN, GOLDEN_INPUTS, CONSTRAINED)),
                    reason="deployment bundle absent")
def test_golden_scoring_regression():
    """THE system-level skew guard: the deployed pipeline, loaded from the
    hash-gated bundle exactly as the API loads it, must reproduce the frozen
    golden outputs. Runs in CI (inputs ship in artifacts/golden_inputs)."""
    from src.models.pipeline import DeployedPipeline

    golden = json.load(open(GOLDEN))
    rows = pd.read_parquet(GOLDEN_INPUTS).set_index("SK_ID_CURR").loc[golden["ids"]].reset_index()
    out = DeployedPipeline.from_artifacts().score_frame(rows)

    assert out["decision"].tolist() == golden["decision"]
    assert [round(float(v), 10) for v in out["pd_calibrated"]] == golden["pd_calibrated"]
    assert [round(float(v), 10) for v in out["threshold_applied"]] == golden["threshold_applied"]
    assert [round(float(v), 6) for v in out["score_scaled"]] == golden["score_scaled"]
    top1 = [c[0]["feature"] if c else None for c in out["reason_codes"]]
    assert top1 == golden["reason_top1"]


def test_manifest_hashes_match_bundle():
    """The same gate the Docker build runs, in CI: every manifest entry exists
    and hashes match (a gitignored or stale artifact fails here, not in the
    image build)."""
    import hashlib

    m = json.load(open(os.path.join(ARTIFACTS_DIR, "MANIFEST.json")))["sha256"]
    on_disk = {f for f in os.listdir(ARTIFACTS_DIR) if f != "MANIFEST.json"}
    assert set(m) == on_disk, f"manifest/bundle mismatch: {set(m) ^ on_disk}"
    for name, h in m.items():
        with open(os.path.join(ARTIFACTS_DIR, name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == h, f"ARTIFACT DRIFT: {name}"


@pytest.mark.skipif(not os.path.exists(os.path.join(MODELS_DIR, "isotonic_final.pkl")),
                    reason="training outputs absent (fresh clone)")
def test_training_outputs_equal_deployed_bundle():
    """Where training outputs exist, the bundle must be a byte copy of them —
    otherwise offline scripts (models/) and serving (artifacts/) disagree."""
    import filecmp

    pairs = [(os.path.join(MODELS_DIR, f), f) for f in
             [*[f"lgbm_constrained_fold{k}.txt" for k in range(5)],
              "isotonic_final.pkl", "scorecard_folds.pkl"]]
    pairs += [(os.path.join(DATA_PROCESSED, f), f) for f in
              ("lgbm_features.json", "categorical_levels.json", "golden_scoring.json")]
    stale = [b for a, b in pairs
             if not filecmp.cmp(a, os.path.join(ARTIFACTS_DIR, b), shallow=False)]
    assert not stale, f"artifacts/ stale vs training outputs — run `make artifacts`: {stale}"


@pytest.mark.skipif(not os.path.exists(FEATURES), reason="feature matrix absent")
def test_input_gate_limits_match_training_data():
    """The API's training-support gate constants are measured facts about the
    training matrix, not tuned numbers — and 'history unknown' (all 22
    aggregates missing) never occurs in training."""
    feats = json.load(open(os.path.join(ARTIFACTS_DIR, "lgbm_features.json")))["features"]
    hist = [f for f in feats if f.startswith(HISTORY_FEATURE_PREFIXES)]
    app = [f for f in feats if f not in hist]
    m = pd.read_parquet(FEATURES, columns=feats)
    assert int(m[app].isna().sum(axis=1).max()) == MAX_MISSING_APPLICATION_FEATURES
    n_hist = m[hist].isna().sum(axis=1)
    assert int(n_hist.max()) == MAX_MISSING_HISTORY_FEATURES < len(hist)
    no_hist = m.loc[n_hist == MAX_MISSING_HISTORY_FEATURES, hist]
    assert sorted(no_hist.columns[no_hist.notna().all()]) == sorted(NO_HISTORY_ZERO_FEATURES)
    assert (no_hist[NO_HISTORY_ZERO_FEATURES] == 0).all().all()


@pytest.mark.skipif(not os.path.exists(FEATURES), reason="feature matrix absent")
def test_stored_term_years_equals_the_single_definition():
    """term_years in the training matrix == implied_term_years(...) — the
    function the API now uses to derive term server-side."""
    import numpy as np

    from src.features.stateless import implied_term_years

    m = pd.read_parquet(FEATURES, columns=["term_years", "AMT_CREDIT", "AMT_ANNUITY"])
    t = implied_term_years(m["AMT_CREDIT"], m["AMT_ANNUITY"])
    assert np.array_equal(t.values, m["term_years"].values, equal_nan=True)


# ── Phase 7 ceremony guards ────────────────────────────────────────────────
def _load_ceremony_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ceremony", os.path.join("scripts", "run_phase7_holdout_ceremony.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not os.path.exists(CONSTRAINED), reason="deployed artifacts absent")
def test_preflight_aborts_on_unknown_contract_type():
    """An unknown contract type would make LGD/EAD NaN, every `p < NaN`
    False, and the ceremony would silently reject 100% of applicants while
    printing plausible euros — on data that cannot be re-run."""
    import pandas as pd

    mod = _load_ceremony_module()
    bad = pd.DataFrame({"NAME_CONTRACT_TYPE": ["Cash loans", "MYSTERY PRODUCT"],
                        "term_years": [3.0, 2.0], "AMT_CREDIT": [1e5, 5e4]})
    with pytest.raises(SystemExit):
        mod.preflight(bad)


@pytest.mark.skipif(not os.path.exists(CONSTRAINED), reason="deployed artifacts absent")
def test_preflight_passes_clean_frame_and_reports_scope():
    import pandas as pd

    mod = _load_ceremony_module()
    good = pd.DataFrame({"NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
                         "term_years": [3.0, None], "AMT_CREDIT": [1e5, 5e4]})
    rep = mod.preflight(good)
    assert rep["rows"] == 2
    assert rep["term_fallback_rows"] == 1          # NaN term → frozen fallback
    assert 0.0 < rep["t_star_range"][0] <= rep["t_star_range"][1] < 0.5


def test_ceremony_run_once_guard_is_implemented():
    """The plan says the ceremony runs once and every run is logged —
    enforced in code, not prose."""
    src = open(os.path.join("scripts", "run_phase7_holdout_ceremony.py")).read()
    assert "CEREMONY ALREADY RUN" in src
    assert "ceremony_run_log.jsonl" in src
    assert "--force" in src


def test_all_third_party_imports_are_in_requirements():
    """Caught in the final audit: `requests` (imported by the Streamlit demo)
    was missing from requirements.txt — the container would have installed
    fine and then crashed on import, taking down the demo deliverable."""
    import ast
    import sys

    stdlib = set(sys.stdlib_module_names)  # exact for the running interpreter
    local = {"src", "config", "scripts", "app", "tests"}
    alias = {"sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow"}
    req = (open("requirements.txt").read() + open("requirements-dev.txt").read()).lower()

    found = set()
    for root, _, files in os.walk("."):
        if any(s in root for s in (".git", "__pycache__", "notebooks", ".venv", "venv")):
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            try:
                tree = ast.parse(open(os.path.join(root, fn)).read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    found.add(node.module.split(".")[0])

    missing = [m for m in found
               if m and m not in stdlib and m not in local
               and alias.get(m, m).lower() not in req]
    assert not missing, f"imports missing from requirements.txt: {sorted(missing)}"


def test_no_unqualified_euro_claims_in_published_docs():
    """The dataset specifies NO currency (median income 147,150, median loan
    513,531 — not euros). Monetary figures must be labeled as currency units;
    a bare euro headline is an accuracy claim the data cannot support."""
    for doc in ("README.md", os.path.join("docs", "METHODOLOGY.md")):
        text = open(doc).read()
        if "\u20ac" in text:
            assert ("currency unit" in text.lower()), (
                f"{doc} uses a currency symbol without the units qualifier")


@pytest.mark.skipif(not os.path.exists(CONSTRAINED), reason="deployment bundle absent")
def test_matrix_fast_path_is_bit_identical_to_pandas_path():
    """The serving fast path (one float matrix shared by all boosters) must
    reproduce LightGBM's own pandas-categorical handling exactly — raw
    scores AND TreeSHAP contributions — or reason codes and PDs drift."""
    import numpy as np

    from src.features.serving import cast_with_contract
    from src.models.pipeline import DeployedPipeline

    p = DeployedPipeline.from_artifacts()
    pool = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "demo_pool.parquet")).head(500)
    casted, _ = cast_with_contract(pool, p.levels)
    M = p._matrix(casted)
    for b in p.boosters:
        assert np.array_equal(b.predict(M), b.predict(casted[p.features]))
    b0 = p.boosters[0]
    assert np.array_equal(b0.predict(M, pred_contrib=True),
                          b0.predict(casted[p.features], pred_contrib=True))
