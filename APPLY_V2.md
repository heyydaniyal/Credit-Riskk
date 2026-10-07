# Applying the v2 overlay

This archive holds only the files that are new or changed relative to
`credit-risk-FINAL-audit-fixed`. `data/processed/`, `models/` and every
pre-existing artifact are byte-identical and are not included.

1. Unzip this archive **over** your existing `credit-risk/` folder (replace files).
2. Delete what v2 removed:

```bash
rm -rf src/decisioning
rm -rf scripts/build_notebooks_04_05.py
rm -rf data/request_log.sqlite
```

3. Verify:

```bash
pip install -r requirements-dev.txt     # Python 3.12
make check                              # expect: ruff clean, 83 passed
```

`artifacts/MANIFEST.json` now lists 15 files (`golden_inputs.parquet` added);
`test_manifest_hashes_match_bundle` fails loudly if anything is out of place.

Change log with evidence: `reports/AUDIT_RESPONSE.md`, "Second review (October 2026)".
