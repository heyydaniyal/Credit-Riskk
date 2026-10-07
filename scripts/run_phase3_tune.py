"""
Phase 3b — resumable Optuna study for LightGBM (single-CPU discipline).

Design mirrors the null-importance runner: journal-file storage means any
interruption loses at most the in-flight trial; re-running continues the
same study.  Run chunks until the trial budget (or convergence) is hit:

    python scripts/run_phase3_tune.py --minutes 9

Pruning: MedianPruner on running mean fold AUC (after fold 2) kills
hopeless configurations early.  Pruned trials never become the winner, so
pruning saves compute without biasing the selection.

The first trial is enqueued at a sensible textbook configuration so TPE
starts from a known-good region rather than a random corner.
"""

import argparse
import json
import logging
import os
import sys
import time

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.constants import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE, TABLES_DIR
from src.data.load import load_modeling_frame
from src.models.lgbm import make_objective

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("phase3b-tune")

STUDY_PATH = os.path.join(MODELS_DIR, "optuna_lgbm_journal.log")
STUDY_NAME = "lgbm_oof_auc"
TRIAL_BUDGET = 150  # spec ceiling; convergence usually well before

SEED_TRIAL = {
    "learning_rate": 0.05, "num_leaves": 64, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "lambda_l1": 1e-3, "lambda_l2": 1e-3,
}


def get_study(no_pruning: bool = False) -> optuna.Study:
    storage = JournalStorage(JournalFileBackend(STUDY_PATH))
    return optuna.create_study(
        study_name=STUDY_NAME, storage=storage, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        # Candidate-evaluation mode (no_pruning): running-mean fold AUCs are
        # noisy, and the median pruner was observed killing a near-best
        # candidate — full evaluations must run all 5 folds, no exceptions.
        pruner=optuna.pruners.NopPruner() if no_pruning
        else optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )


def dump_state(study: optuna.Study) -> None:
    done = [t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")]
    complete = [t for t in done if t.state.name == "COMPLETE"]
    state = {
        "n_trials_total": len(done),
        "n_complete": len(complete),
        "n_pruned": len(done) - len(complete),
        "best_oof_auc": study.best_value if complete else None,
        "best_params": study.best_params if complete else None,
        "best_trial_number": study.best_trial.number if complete else None,
        "best_trial_user_attrs": dict(study.best_trial.user_attrs) if complete else None,
        "trial_budget": TRIAL_BUDGET,
    }
    with open(os.path.join(TABLES_DIR, "optuna_study_state.json"), "w") as f:
        json.dump(state, f, indent=2)
    logger.info("study state: %d done (%d complete, %d pruned) | best OOF AUC %s",
                state["n_trials_total"], state["n_complete"], state["n_pruned"],
                f"{state['best_oof_auc']:.5f}" if complete else "—")


def main(minutes: float, no_time_cap: bool = False) -> None:
    dev = load_modeling_frame("dev")
    with open(os.path.join(DATA_PROCESSED, "lgbm_features.json")) as f:
        features = json.load(f)["features"]

    study = get_study(no_pruning=no_time_cap)
    if len(study.trials) == 0:
        study.enqueue_trial(SEED_TRIAL)
        logger.info("fresh study — enqueued textbook seed trial")

    objective = make_objective(dev, features,
                               time_cap_seconds=None if no_time_cap else 600)
    deadline = time.time() + minutes * 60

    while time.time() < deadline:  # only START new trials before the deadline;
        done = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
        if done >= TRIAL_BUDGET:      # the 400s per-trial cap bounds overshoot
            logger.info("trial budget reached")
            break
        study.optimize(objective, n_trials=1)
    dump_state(study)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=3.0,
                    help="wall-clock budget for this chunk")
    ap.add_argument("--no-time-cap", action="store_true",
                    help="disable the per-trial time cap (final candidate evaluation)")
    args = ap.parse_args()
    main(args.minutes, no_time_cap=args.no_time_cap)
