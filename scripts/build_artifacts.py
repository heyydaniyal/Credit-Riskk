"""
Build the Docker deployment set: artifacts/ + sha256 MANIFEST.

Reproducible by design — the deployment whitelist must never be a one-off
hand-copy.  The image build re-verifies every hash and fails on drift, so
this script is the single place the shipped model set is defined.

Contents: 5 constrained fold models, the deployed isotonic calibrator, the
scorecard fold artifacts, feature list, categorical serving contract,
golden regression file, term-fallback rule, feature manifest, and the
monitoring reference distributions.
"""

import hashlib
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.load import load_modeling_frame  # noqa: E402
from src.monitoring import build_reference  # noqa: E402

ARTIFACT_DIR = "artifacts"
WHITELIST = [
    *[f"models/lgbm_constrained_fold{k}.txt" for k in range(5)],
    "models/isotonic_final.pkl",
    "models/scorecard_folds.pkl",
    "data/processed/lgbm_features.json",
    "data/processed/categorical_levels.json",
    "data/processed/golden_scoring.json",
    "data/processed/feature_manifest.csv",
    "config/term_fallback_rule.json",
]

# demo_pool is built fresh below (not copied from a source path)


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    missing = [p for p in WHITELIST if not os.path.exists(p)]
    if missing:
        sys.exit(f"missing deployment artifacts (run `make final-pipeline`): {missing}")

    for src in WHITELIST:
        shutil.copy2(src, os.path.join(ARTIFACT_DIR, os.path.basename(src)))

    # monitoring reference: top-10 SHAP features + deployed-ensemble scores
    top10 = (pd.read_csv("reports/tables/shap_mean_abs_top30.csv", index_col=0)
             .head(10).index.tolist())
    dev = load_modeling_frame("dev")
    scores = pd.read_parquet(
        "data/processed/dev_ensemble_scores_constrained.parquet")["p_ensemble_raw"].values
    build_reference(dev, scores, top10)  # writes artifacts/monitoring_reference.json

    # demo pool (dev rows only) so the Streamlit container is self-contained
    import pandas as _pd
    from src.data.load import load_dev_ids as _ldi
    _feats = json.load(open("data/processed/lgbm_features.json"))["features"]
    _fields = list(dict.fromkeys(_feats + ["NAME_CONTRACT_TYPE", "term_years",
                                           "AMT_CREDIT", "SK_ID_CURR"]))
    _m = _pd.read_parquet("data/processed/features_full.parquet", columns=_fields)
    _dev = set(_ldi())
    _m[_m.SK_ID_CURR.isin(_dev)].sample(3000, random_state=0).reset_index(drop=True) \
        .to_parquet(os.path.join(ARTIFACT_DIR, "demo_pool.parquet"))

    files = sorted(f for f in os.listdir(ARTIFACT_DIR) if f != "MANIFEST.json")
    manifest = {
        f: hashlib.sha256(open(os.path.join(ARTIFACT_DIR, f), "rb").read()).hexdigest()
        for f in files
    }
    json.dump(
        {"purpose": "Docker deployment set — the image build re-verifies these "
                    "hashes and FAILS on drift, so a stale model cannot ship",
         "sha256": manifest},
        open(os.path.join(ARTIFACT_DIR, "MANIFEST.json"), "w"), indent=2)
    print(f"artifacts/: {len(manifest)} files hashed "
          f"({sum(os.path.getsize(os.path.join(ARTIFACT_DIR, f)) for f in files)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
