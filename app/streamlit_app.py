"""
Streamlit demo — two tabs.

Serving plumbing (single-container host): this process launches uvicorn on
an internal port and talks to it over localhost — the Score tab exercises
the REAL API, not a shortcut into the pipeline.

Score tab      pick an applicant, see PD / score / decision / threshold /
               reason codes from POST /score; sliders for LGD, net margin,
               EAD and cure rate — labeled "illustrative assumptions —
               explore sensitivity", BOUNDED by the frozen sensitivity grids
               (the API rejects anything outside them). The sliders move the
               whole t*(x) DISTRIBUTION over the 3,000-applicant demo pool,
               not just one applicant's threshold.
Monitoring tab (1) decision drift from the LIVE request log — rolling
               approval rate and rolling mean t*(x) over every /score call;
               (2) PSI bars with 0.10/0.25 bands and the SIMULATED-drift
               button — the alarm must visibly fire.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config.constants import ARTIFACTS_DIR, PSI_INVESTIGATE, PSI_RETRAIN  # noqa: E402
from src.models.decision import OVERRIDE_BOUNDS, attach_cost_params  # noqa: E402
from src.models.pipeline import DeployedPipeline  # noqa: E402
from src.monitoring import (  # noqa: E402
    load_reference,
    monitoring_report,
    read_log,
    simulate_recession,
)

API_PORT = int(os.environ.get("API_PORT", "8008"))
API = f"http://127.0.0.1:{API_PORT}"
DERIVED = ["term_years", "credit_goods_ratio", "ext_source_mean", "ext_source_min",
           "ext_source_n_missing", "EXT_SOURCE_1_is_missing", "EXT_SOURCE_3_is_missing"]


@st.cache_resource
def start_api() -> bool:
    try:
        requests.get(f"{API}/health", timeout=1)
        return True
    except requests.RequestException:
        pass
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:app",
         "--host", "127.0.0.1", "--port", str(API_PORT)],
        cwd=ROOT,
    )
    for _ in range(90):
        try:
            requests.get(f"{API}/health", timeout=1)
            return True
        except requests.RequestException:
            time.sleep(1)
    return False


@st.cache_resource
def pipeline() -> DeployedPipeline:
    return DeployedPipeline.from_artifacts()


@st.cache_data
def load_pool() -> tuple[pd.DataFrame, list[str], set]:
    feats = json.load(open(os.path.join(ARTIFACTS_DIR, "lgbm_features.json")))["features"]
    cats = set(json.load(open(os.path.join(ARTIFACTS_DIR, "categorical_levels.json")))["levels"])
    pool = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "demo_pool.parquet"))
    return pool.reset_index(drop=True), feats, cats


@st.cache_data
def pool_pd() -> np.ndarray:
    """Calibrated PD for every demo applicant (economics never move the PD)."""
    pool, _, _ = load_pool()
    s = pipeline().score_frame(pool, with_reason_codes=False)
    return pipeline().calibrator.predict(s["pd_raw_ensemble"].values)


def to_payload(row: pd.Series, cats: set, feats: list[str], omit_derived: bool = False) -> dict:
    fields = list(dict.fromkeys(feats + ["NAME_CONTRACT_TYPE", "term_years", "AMT_CREDIT"]))
    if omit_derived:  # let the server derive them from their sources
        fields = [f for f in fields if f not in DERIVED]
    return {f: (None if pd.isna(row[f]) else (str(row[f]) if f in cats else float(row[f])))
            for f in fields}


def post_score(payload: dict) -> tuple[bool, dict]:
    try:
        r = requests.post(f"{API}/score", json=payload, timeout=30)
    except requests.RequestException as e:
        return False, {"detail": f"API unreachable: {e}"}
    return r.ok, r.json()


def t_star_histogram(t: np.ndarray, bins: np.ndarray) -> pd.Series:
    counts, _ = np.histogram(t, bins=bins)
    return pd.Series(counts, index=[f"{b:.2f}" for b in bins[:-1]])


st.set_page_config(page_title="Credit Risk Scoring", layout="wide")
api_ok = start_api()
if not api_ok:
    st.error("The scoring API did not start — see the container logs.")
    st.stop()
pool, feats, cats = load_pool()

tab_score, tab_mon = st.tabs(["Score", "Monitoring"])

# ── Score ──────────────────────────────────────────────────────────────────
with tab_score:
    st.subheader("Calibrated PD + instance-dependent decision")
    st.caption("Cost sliders — **illustrative assumptions, explore sensitivity**. "
               "Bounded by the pre-registered sensitivity grids; they never revise "
               "the frozen parameters behind the reported results. Amounts are in the "
               "dataset's unspecified currency units.")

    def slider(label: str, key: str, default: float, step: float) -> float:
        lo, hi = OVERRIDE_BOUNDS[key]
        return st.slider(label, lo, hi, default, step, key=key)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lgd_cash = slider("LGD (cash)", "lgd_cash", 0.70, 0.01)
    with c2:
        r_net = slider("Net margin r (cash, annual)", "r_net_cash", 0.05, 0.005)
    with c3:
        ead_cash = slider("EAD factor (cash)", "ead_cash", 0.85, 0.01)
    with c4:
        cure = slider("Cure rate (early-delinquency → no loss)", "cure_rate", 0.0, 0.05)
    with st.expander("Revolving-loan assumptions"):
        r1, r2, r3 = st.columns(3)
        with r1:
            lgd_rev = slider("LGD (revolving)", "lgd_revolving", 0.85, 0.01)
        with r2:
            ead_rev = slider("EAD factor (revolving)", "ead_revolving", 1.00, 0.01)
        with r3:
            m_rev = slider("Margin multiplier (revolving)", "m_revolving", 0.18, 0.01)
    overrides = {"lgd_cash": lgd_cash, "r_net_cash": r_net, "ead_cash": ead_cash,
                 "cure_rate": cure, "lgd_revolving": lgd_rev, "ead_revolving": ead_rev,
                 "m_revolving": m_rev}

    # the whole threshold distribution moves with the sliders
    p_cal = pool_pd()
    frozen = attach_cost_params(pool)
    moved = attach_cost_params(pool, overrides=overrides)
    bins = np.round(np.arange(0.0, 0.42, 0.02), 2)
    a, b, c = st.columns(3)
    a.metric("Demo-pool approval rate", f"{(p_cal < moved.t_star.values).mean():.1%}",
             f"{(p_cal < moved.t_star.values).mean() - (p_cal < frozen.t_star.values).mean():+.1%}"
             " vs frozen")
    b.metric("Mean t*(x)", f"{moved.t_star.mean():.4f}",
             f"{moved.t_star.mean() - frozen.t_star.mean():+.4f} vs frozen")
    c.metric("t*(x) range", f"{moved.t_star.min():.3f} – {moved.t_star.max():.3f}")
    st.bar_chart(pd.DataFrame({"frozen assumptions": t_star_histogram(frozen.t_star, bins),
                               "slider assumptions": t_star_histogram(moved.t_star, bins)}),
                 x_label="t*(x) — per-applicant approval threshold",
                 y_label="applicants (of 3,000)", stack=False)

    st.divider()
    sid = st.selectbox("Applicant (SK_ID_CURR)", pool.SK_ID_CURR.head(200))
    row = pool[pool["SK_ID_CURR"] == sid].iloc[0]
    payload = to_payload(row, cats, feats)
    payload["cost_overrides"] = overrides
    ok, r = post_score(payload)
    if not ok:
        st.error(f"API rejected the request: {r.get('detail')}")
    else:
        a, b, c, d = st.columns(4)
        a.metric("PD (calibrated)", f"{r['pd_calibrated']:.2%}")
        b.metric("Score", f"{r['score_scaled']:.0f}")
        c.metric("Threshold t*(x)", f"{r['threshold_applied']:.3f}")
        d.metric("Decision", r["decision"])
        st.write("**Reason codes (adverse):**")
        if not r["reason_codes"]:
            st.write("none — every feature pushes this applicant's risk down")
        for rc in r["reason_codes"]:
            st.write(f"• {rc['plain_language']}  (+{rc['contribution']:.2f} log-odds)")
        st.caption(f"term used {r['term_years_used']:.2f}y · LGD {r['lgd_used']:.2f} · "
                   f"EAD {r['ead_factor_used']:.2f} · margin {r['margin_used']:.3f} · "
                   f"cure {r['cure_rate_used']:.2f} · model {r['model_version']} — {r['note']}")

# ── Monitoring ─────────────────────────────────────────────────────────────
with tab_mon:
    st.subheader("Monitoring framework — demonstrated with **simulated** drift")
    st.caption("Not production telemetry. Realized default rates are unobservable "
               "here (12+ month label latency) — which is why credit monitoring "
               "leans on PSI. Bands: <0.10 stable · 0.10–0.25 investigate · "
               ">0.25 retrain.")

    st.markdown("#### 1 · Decision drift from the live request log")
    st.caption("Every POST /score is logged (prediction, threshold, decision — no "
               "applicant features). Send demo traffic to see the rolling series move.")
    b1, b2 = st.columns(2)
    send_normal = b1.button("Send 200 demo applicants through the API")
    send_drift = b2.button("Send 200 SIMULATED-recession applicants through the API")
    if send_normal or send_drift:
        batch = pool.sample(200, random_state=int(time.time()) % 10_000)
        if send_drift:
            batch = simulate_recession(batch)
        progress = st.progress(0.0)
        refused = 0
        for i, (_, rr) in enumerate(batch.iterrows()):
            # derived fields omitted: the server recomputes them from sources
            ok, _ = post_score(to_payload(rr, cats, feats, omit_derived=True))
            refused += not ok
            progress.progress((i + 1) / len(batch))
        if refused:
            st.warning(f"{refused} request(s) refused by the input gate")

    log = read_log()
    if log.empty:
        st.info("No requests logged yet.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Requests logged", f"{len(log):,}")
        m2.metric("Rolling approval rate (last 50)", f"{log.rolling_approval_rate.iloc[-1]:.1%}")
        m3.metric("Rolling mean t*(x) (last 50)", f"{log.rolling_mean_t_star.iloc[-1]:.4f}")
        st.line_chart(log[["rolling_approval_rate"]].reset_index(drop=True),
                      x_label="request #", y_label="approval rate (rolling 50)")
        st.line_chart(log[["rolling_mean_t_star"]].reset_index(drop=True),
                      x_label="request #", y_label="mean t*(x) (rolling 50)")

    st.markdown("#### 2 · Feature and score PSI vs the development reference")
    if st.button("⚠ Simulate drift (severe recession on 1,000 applicants) — SIMULATED"):
        st.session_state["drift"] = True
    if st.session_state.get("drift") and st.button("Reset to no drift"):
        st.session_state["drift"] = False
    drifted = st.session_state.get("drift", False)

    sample = pool.sample(1000, random_state=1).copy()
    if drifted:
        sample = simulate_recession(sample)
    scored = pipeline().score_frame(sample, with_reason_codes=False)
    rep = monitoring_report(sample, scored["pd_raw_ensemble"].values,
                            scored["threshold_applied"].values,
                            scored["decision"].values, load_reference())

    cols = st.columns(3)
    cols[0].metric("Score PSI", f"{rep['score_psi']['psi']:.3f}", rep["score_psi"]["band"],
                   delta_color="off")
    cols[1].metric("Approval rate", f"{rep['approval_rate']:.1%}")
    cols[2].metric("Mean t*(x)", f"{rep['mean_threshold_t_star']:.4f}")

    fdf = pd.DataFrame([{"feature": f, "psi": v["psi"], "band": v["band"]}
                        for f, v in rep["feature_psi"].items()])
    st.bar_chart(fdf.set_index("feature")["psi"])
    st.write(f"Bands: investigate > {PSI_INVESTIGATE}, retrain > {PSI_RETRAIN}")
    st.dataframe(fdf, hide_index=True)
    worst = float(fdf.psi.max())
    if drifted and worst > PSI_INVESTIGATE:
        band = "RETRAIN band" if worst > PSI_RETRAIN else "investigate band"
        st.error(f"ALARM: max feature PSI = {worst:.2f} ({band}) — simulated recession detected")
    st.caption("Mean t*(x) does not move under this scenario: thresholds depend on "
               "contract type and term, not on bureau scores — a product-mix shift "
               "would move it with no score drift, which is why both are tracked.")
