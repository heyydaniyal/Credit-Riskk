"""
featurize CLI — bridges raw data to the API (spec Phase 6).

Modes:
  lookup   --id SK_ID_CURR      applicant's feature vector from the
                                precomputed matrix (the demo path; honest:
                                bureau joins are upstream-pipeline work)
  raw      --csv path --row N   application-table row alone: stateless
                                transforms only. Credit-history aggregates
                                are unknown → all NaN, which the API REFUSES
                                (no training row lacks all history). Pass
                                --assume-no-history to state explicitly that
                                the applicant has no bureau/previous record:
                                count/sum aggregates become 0, exactly as for
                                the 2,470 no-history training applicants.

Output: JSON payload for POST /score on stdout (or --out file). The API
forbids unknown fields, so diagnostics (e.g. the raw-mode degradation note)
go to stderr, never into the payload.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.constants import ARTIFACTS_DIR, NO_HISTORY_ZERO_FEATURES  # noqa: E402
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
    p2.add_argument("--assume-no-history", action="store_true",
                    help="applicant has no bureau/previous record (explicit assumption)")
    args = ap.parse_args()

    feats = json.load(open(os.path.join(ARTIFACTS_DIR, "lgbm_features.json")))["features"]
    cats = set(json.load(open(os.path.join(ARTIFACTS_DIR, "categorical_levels.json")))["levels"])
    api_fields = list(dict.fromkeys(feats + ["NAME_CONTRACT_TYPE", "term_years",
                                             "AMT_CREDIT"]))

    if args.mode == "lookup":
        m = pd.read_parquet("data/processed/features_full.parquet")
        hit = m[args.id == m.SK_ID_CURR]
        if hit.empty:
            sys.exit(f"SK_ID_CURR {args.id} not found")
        payload = payload_from_series(hit.iloc[0], api_fields, cats)
    else:
        raw = pd.read_csv(args.csv).iloc[[args.row]]
        row = apply_all_stateless(raw).iloc[0]
        payload = payload_from_series(row, api_fields, cats)
        if args.assume_no_history:
            for f in NO_HISTORY_ZERO_FEATURES:
                payload[f] = 0.0
            print("featurize: raw mode, ASSUMING NO CREDIT HISTORY (count/sum aggregates "
                  "= 0); production uses the upstream feature pipeline", file=sys.stderr)
        else:
            print("featurize: raw mode — credit-history aggregates unknown; the API will "
                  "refuse this payload (outside training support). Use --assume-no-history "
                  "only if the applicant genuinely has no record.", file=sys.stderr)

    text = json.dumps(payload, indent=1)
    if args.out:
        open(args.out, "w").write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
