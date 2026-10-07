"""
Credit Risk Scoring System — Frozen Constants & Configuration
==============================================================

All economic assumptions are ILLUSTRATIVE — directionally plausible placeholders
occupying the slots where a bank's LGD/EAD/pricing models would plug in.
They are frozen BEFORE any results exist (per spec v4.1) and may NOT be revised
after seeing the gap. Pre-registered means pre-registered.

Reference: spec v4.1, Phase 4c.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_SPLITS = os.path.join(PROJECT_ROOT, "data", "splits")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# The hash-gated deployment bundle (scripts/build_artifacts.py writes it,
# artifacts/MANIFEST.json pins it). SERVING (API, demo) loads ONLY from here,
# so what is tested in CI, what the image verifies and what serves are the
# same bytes. Training scripts keep writing to models/ + data/processed/.
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
REQUEST_LOG_DB = os.path.join(PROJECT_ROOT, "data", "request_log.sqlite")

# All generated outputs live under reports/ — never inside notebooks/ or src/
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TABLES_DIR = os.path.join(REPORTS_DIR, "tables")

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.20  # 20% stratified holdout

# ── Target ─────────────────────────────────────────────────────────────────
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# ── Feature selection thresholds ───────────────────────────────────────────
CORR_THRESHOLD = 0.98        # drop pairs with |Spearman| > this
NULL_IMPORTANCE_RUNS = 50     # shuffled-target LightGBM runs
NULL_IMPORTANCE_PCTL = 95     # keep features beating this percentile
IV_MIN = 0.02                 # WOE: below = useless
IV_MAX = 0.50                 # WOE: above = suspicious, audit for leakage

# ── LightGBM tuning ───────────────────────────────────────────────────────
OPTUNA_N_TRIALS = 120         # 100-150 range per spec
EARLY_STOPPING_ROUNDS = 200
N_ESTIMATORS_MAX = 10_000

# ── Calibration ────────────────────────────────────────────────────────────
CALIBRATION_N_BINS = 10       # reliability diagram bins

# ══════════════════════════════════════════════════════════════════════════
# FROZEN ECONOMIC ASSUMPTIONS — ILLUSTRATIVE, NOT ESTIMATED
# Frozen in this file before any result exists.  Do NOT change after
# seeing the instance-vs-best-flat gap.
# ══════════════════════════════════════════════════════════════════════════

# ── LGD (Loss Given Default) ──────────────────────────────────────────────
# Unsecured consumer lending; no recovery data in this dataset.
LGD_CASH = 0.70           # sensitivity ±0.10
LGD_REVOLVING = 0.85      # unsecured revolving recovers worse

# ── EAD factors (Exposure At Default / AMT_CREDIT) ────────────────────────
# Cash: amortization repays principal, but consumer defaults cluster early.
# Revolving: drawdown before default; Basel CCF ~0.75-1.0, conservative end.
EAD_FACTOR_CASH = 0.85    # sensitivity ±0.10
EAD_FACTOR_REVOLVING = 1.00  # sensitivity 0.85-1.00

# ── Net margin ─────────────────────────────────────────────────────────────
# Net of funding and operating cost — bank loses profit, not revenue.
R_NET_CASH = 0.05          # 5% annual net margin (sensitivity 3-8%)
R_NET_REVOLVING = 0.10     # higher for revolving

# ── Revolving composite multiplier ────────────────────────────────────────
# m_rev = r_net_rev × E[life] × E[utilization] = 0.10 × 3y × 0.6
REVOLVING_EXPECTED_LIFE = 3.0   # years
REVOLVING_UTILIZATION = 0.60
# Composite revolving margin. Rounded to the exact deployed literal (0.18) rather
# than the raw product 0.10*3*0.6, which floats to 0.18000000000000002 and would
# perturb every revolving decision by an epsilon (breaking golden parity).
M_REVOLVING = round(R_NET_REVOLVING * REVOLVING_EXPECTED_LIFE * REVOLVING_UTILIZATION, 10)  # 0.18

# ── Cash term formula ─────────────────────────────────────────────────────
# term_years = AMT_CREDIT / (12 × AMT_ANNUITY), capped [0.5, 7]
# Zero-interest approximation — slightly understates term, keeps m(x) conservative.
# Amortization correction: average outstanding ≈ AMT_CREDIT / 2 → divide by 2.
TERM_YEARS_MIN = 0.5
TERM_YEARS_MAX = 7.0
AMORTIZATION_FACTOR = 0.5  # the /2 correction

# ── Sensitivity grids (frozen) ─────────────────────────────────────────────
SENSITIVITY_LGD_CASH = [0.60, 0.70, 0.80]
SENSITIVITY_LGD_REVOLVING = [0.75, 0.85, 0.95]
SENSITIVITY_EAD_CASH = [0.75, 0.85, 0.95]
SENSITIVITY_EAD_REVOLVING = [0.85, 1.00]  # reconciled to the deployed grid (was stray [0.85,0.90,1.00])
SENSITIVITY_R_NET_CASH = [0.03, 0.05, 0.08]
SENSITIVITY_M_REVOLVING = [0.10, 0.18, 0.30]

# Cure rate: fraction of TARGET=1 (early-delinquency) events that recover
# WITHOUT charge-off. The Home Credit TARGET is an early-delinquency label
# (late on the first installments), NOT a Basel charge-off. LGD/EAD are
# charge-off severities, so effective loss = LGD*EAD*(1-CURE_RATE). Set to 0
# in the DEPLOYED path (behaviour-preserving; the as-built assumption is an
# implicit 0% cure) and swept here because event-definition uncertainty
# dominates every other cost parameter (a 50% cure moves the 3y anchor +80%
# vs +/-13% for the whole LGD grid).
CURE_RATE = 0.0
SENSITIVITY_CURE_RATE = [0.0, 0.30, 0.50]

# ── PSI monitoring thresholds (Siddiqi 2006) ──────────────────────────────
# < 0.10 stable · 0.10-0.25 investigate · > 0.25 retrain trigger.
# Single definition: src/monitoring imports these (previously defined in
# three places, one of them with the names shifted by a band).
PSI_INVESTIGATE = 0.10
PSI_RETRAIN = 0.25

# ── Serving input gate (training support, measured — not tuned) ─────────────
# Measured on the full 307,511-row feature matrix, per block of the 58 model
# features:
#   application-side (36 features): at most 12 missing in any row
#   credit history (22 bureau_*/prev_* aggregates): at most 16 missing —
#     a no-history applicant has 16 NaN aggregates and 6 count/sum
#     aggregates equal to 0 (NO_HISTORY_ZERO_FEATURES); NO training row has
#     all 22 missing, i.e. "history unknown" never occurred in training.
# A request beyond either limit is outside the training support. LightGBM
# would still route it down NaN branches and return a confident-looking PD
# (review probe: AMT_CREDIT + contract type only → APPROVE at PD 2.2%), so
# the API refuses it. Verified against the matrix by
# tests/test_artifacts.py::test_input_gate_limits_match_training_data.
HISTORY_FEATURE_PREFIXES = ("bureau_", "prev_")
MAX_MISSING_APPLICATION_FEATURES = 12
MAX_MISSING_HISTORY_FEATURES = 16
NO_HISTORY_ZERO_FEATURES = ["bureau_credit_sum", "bureau_n_active", "bureau_limit_sum",
                            "bureau_n_card", "prev_n_approved", "prev_n_refused"]

# ── Fairness ───────────────────────────────────────────────────────────────
PROTECTED_ATTRIBUTE = "CODE_GENDER"
DISPARATE_IMPACT_THRESHOLD = 0.80  # four-fifths rule
PROXY_AUC_FLAG = 0.65

# ── Monotonic constraint features (Phase 5) ────────────────────────────────
# direction: 1 = risk increases with feature value, -1 = risk decreases
# Populated in Phase 2: only features that (a) SURVIVED selection and
# (b) have an unambiguous domain direction a regulator would agree with.
# credit_income_ratio is absent because it was dropped by null-importance
# (consistent with EDA #8: leverage deciles were flat).
MONOTONIC_FEATURES = {
    "EXT_SOURCE_1": -1,            # ↑ external score → ↓ risk
    "EXT_SOURCE_2": -1,
    "EXT_SOURCE_3": -1,
    "ext_source_mean": -1,         # composite of the three above
    "ext_source_min": -1,          # worst-case external score
    "annuity_income_ratio": 1,     # ↑ payment burden (DTI) → ↑ risk
    "prev_refusal_share": 1,       # ↑ share of past refusals → ↑ risk
    "bureau_bb_share_dpd_max": 1,  # ↑ share of delinquent months → ↑ risk
    "bureau_utilization": 1,       # ↑ debt vs granted credit → ↑ risk
}

# ── Contract type mapping ──────────────────────────────────────────────────
CONTRACT_CASH = "Cash loans"
CONTRACT_REVOLVING = "Revolving loans"
