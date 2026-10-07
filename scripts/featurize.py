"""
featurize CLI — bridges raw data to the API (spec Phase 6).

Modes:
  lookup   --id SK_ID_CURR      applicant's feature vector from the
                                precomputed matrix (the demo path; honest:
                                bureau joins are upstream-pipeline work)
  raw      --csv path --row N   application-table row alone: stateless
                                transforms only, aggregates NaN'd + flags
                                (degraded path, documented)

Output: JSON payload for POST /score on stdout (or --out file).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features.stateless import apply_all_stateless  # noqa: E402


def payload_from_series(row: pd.Series, api_fields: list[str], cats: set) -> dict:
    out = {}
    for f in api_fields:
        v = row.get(f, np.nan)
        if isinstance(v, float) and np.isnan(v):
            out[f] = None
        elif f in cats:
            out[f] = None if pd.isna(v) else str(v)
        else:
            out[f] = float(v) if pd.notna(v) else None
    return out


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=None)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("lookup", parents=[common])
    p1.add_argument("--id", type=int, required=True)
    p2 = sub.add_parser("raw", parents=[common])
    p2.add_argument("--csv", required=True)
    p2.add_argument("--row", type=int, default=0)
    args = ap.parse_args()

    feats = json.load(open("data/processed/lgbm_features.json"))["features"]
    cats = set(json.load(open("data/processed/categorical_levels.json"))["levels"])
    api_fields = list(dict.fromkeys(feats + ["NAME_CONTRACT_TYPE", "term_years",
                                             "AMT_CREDIT"]))

    if args.mode == "lookup":
        m = pd.read_parquet("data/processed/features_full.parquet")
        hit = m[m.SK_ID_CURR == args.id]
        if hit.empty:
            sys.exit(f"SK_ID_CURR {args.id} not found")
        payload = payload_from_series(hit.iloc[0], api_fields, cats)
    else:
        raw = pd.read_csv(args.csv).iloc[[args.row]]
        row = apply_all_stateless(raw).iloc[0]
        payload = payload_from_series(row, api_fields, cats)
        payload["_note"] = ("raw mode: bureau/prev aggregates unavailable → NaN "
                            "(model routes missing); production uses the upstream "
                            "feature pipeline")

    text = json.dumps(payload, indent=1)
    if args.out:
        open(args.out, "w").write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
