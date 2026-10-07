"""
Monotonicity verification — the required gate of the final pipeline.

Why this exists (design-review evidence): LightGBM accepts the
monotone_constraints parameter silently, and the experiment on our real
feature structure showed method='advanced' violating the guarantee by up
to 0.166 raw log-odds while claiming to constrain.  A "constrained" model
that is not monotone is worse than an unconstrained one — it asserts a
property it lacks.  So the property is TESTED, per fold, before the
pipeline may proceed; zero violations or the gate fails.
"""

import numpy as np
import pandas as pd


def verify_monotonicity(
    booster,
    X_sample: pd.DataFrame,
    constraints: dict[str, int],
    quantile_range: tuple[float, float] = (0.02, 0.98),
    n_grid: int = 15,
    tol: float = 1e-9,
) -> dict[str, float]:
    """
    Sweep each constrained feature over its observed range on fixed rows,
    in RAW-SCORE space; return worst violation magnitude per feature.
    A violation is a step where prediction moves AGAINST the constraint
    direction by more than tol.
    """
    report = {}
    for feat, direction in constraints.items():
        lo, hi = X_sample[feat].quantile(list(quantile_range))
        if not np.isfinite(lo) or lo == hi:
            report[feat] = 0.0
            continue
        grid = np.linspace(lo, hi, n_grid)
        raw = np.stack(
            [booster.predict(X_sample.assign(**{feat: g}), raw_score=True) for g in grid]
        )
        diffs = np.diff(raw, axis=0) * direction
        report[feat] = float(max(0.0, -(diffs.min())))
    return report


def assert_monotone(report: dict[str, float], tol: float = 1e-9) -> None:
    bad = {f: v for f, v in report.items() if v > tol}
    assert not bad, f"monotonicity GATE FAILED — violations: {bad}"
