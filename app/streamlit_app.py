"""
Streamlit demo — two tabs per the frozen design.

Serving plumbing (single-container host): this process launches uvicorn on
an internal port and talks to it over localhost — the demo exercises the
REAL API, not a shortcut into the pipeline.

Score tab      pick an applicant (or paste an ID), see PD / score /
               decision / threshold / reason codes; sliders for LGD, net
               rate, EAD — labeled "illustrative assumptions — explore
               sensitivity", BOUNDED by the frozen sensitivity grids (the
               API rejects anything outside them).
Monitoring tab PSI bars with 0.10/0.25 bands, approval-rate and mean-t*
               readouts, and the SIMULATED-drift button (1,000 rows,
               incomes −20%) — the alarm must visibly fire.
"""

import json
import os
import subprocess
import sys
import time

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.monitoring import PSI_INVESTIGATE, PSI_RETRAIN, load_reference, monitoring_report  # noqa: E402

API = "http://127.0.0.1:8008"


@st.cache_resource
def start_api() -> None:
    try:
        requests.get(f"{API}/health", timeout=1)
        return
    except Exception:
        pass
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:app",
         "--host", "127.0.0.1", "--port", "8008"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    for _ in range(60):
        try:
            requests.get(f"{API}/health", timeout=1)
            return
        except Exception:
            time.sleep(1)
    st.error("API failed to start")


@st.cache_data
def load_demo_pool():
    feats = json.load(open("data/processed/lgbm_features.json"))["features"]
    # demo pool is a prebuilt artifact (dev rows only) so the container is
    # self-contained and never needs the full 60MB feature matrix (F1.5)
    pool_path = ("artifacts/demo_pool.parquet"
                 if os.path.exists("artifacts/demo_pool.parquet")
                 else "data/processed/demo_pool.parquet")
    m = pd.read_parquet(pool_path)
    return m.reset_index(drop=True), feats


def to_payload(row: pd.Series, cats: set, feats: list[str]) -> dict:
    fields = list(dict.fromkeys(feats + ["NAME_CONTRACT_TYPE", "term_years",
                                         "AMT_CREDIT"]))
    return {f: (None if pd.isna(row[f])
                else (str(row[f]) if f in cats else float(row[f])))
            for f in fields}


st.set_page_config(page_title="Credit Risk Scoring", layout="wide")
start_api()
pool, feats = load_demo_pool()
cats = set(json.load(open("data/processed/categorical_levels.json"))["levels"])

tab_score, tab_mon = st.tabs(["Score", "Monitoring"])

with tab_score:
    st.subheader("Calibrated PD + instance-dependent decision")
    sid = st.selectbox("Applicant (SK_ID_CURR)", pool.SK_ID_CURR.head(200))
    st.caption("Cost sliders — **illustrative assumptions, explore sensitivity**. "
               "Bounded by the pre-registered sensitivity grids; they never revise "
               "the frozen parameters behind reported results.")
    c1, c2, c3 = st.columns(3)
    lgd_cash = c1.slider("LGD (cash)", 0.60, 0.80, 0.70, 0.01)
    r_net = c2.slider("Net margin r (cash, annual)", 0.03, 0.08, 0.05, 0.005)
    ead_cash = c3.slider("EAD factor (cash)", 0.75, 0.95, 0.85, 0.01)

    row = pool[pool.SK_ID_CURR == sid].iloc[0]
    payload = to_payload(row, cats, feats)
    payload["cost_overrides"] = {"lgd_cash": lgd_cash, "r_net_cash": r_net,
                                 "ead_cash": ead_cash}
    r = requests.post(f"{API}/score", json=payload).json()

    a, b, c, d = st.columns(4)
    a.metric("PD (calibrated)", f"{r['pd_calibrated']:.2%}")
    b.metric("Score", f"{r['score_scaled']:.0f}")
    c.metric("Threshold t*(x)", f"{r['threshold_applied']:.3f}")
    d.metric("Decision", r["decision"])
    st.write("**Reason codes (adverse):**")
    for rc in r["reason_codes"]:
        st.write(f"• {rc['plain_language']}  (+{rc['contribution']:.2f} log-odds)")
    st.caption(r["note"])

with tab_mon:
    st.subheader("Monitoring framework — demonstrated with **simulated** drift")
    st.caption("Not production telemetry. Realized default rates are unobservable "
               "here (12+ month label latency) — which is why credit monitoring "
               "leans on PSI. Bands: <0.10 stable · 0.10–0.25 investigate · "
               ">0.25 retrain.")

    if st.button("⚠ Simulate drift (recession: incomes −20% on 1,000 applicants) — SIMULATED"):
        st.session_state["drift"] = True
    drifted = st.session_state.get("drift", False)

    from src.monitoring import simulate_recession

    sample = pool.sample(1000, random_state=1).copy()
    if drifted:
        sample = simulate_recession(sample)

    from src.models.pipeline import DeployedPipeline

    @st.cache_resource
    def pipe():
        return DeployedPipeline()

    scored = pipe().score_frame(sample, with_reason_codes=False)
    rep = monitoring_report(sample, scored["pd_raw_ensemble"].values,
                            scored["threshold_applied"].values,
                            scored["decision"].values, load_reference())

    cols = st.columns(3)
    cols[0].metric("Score PSI", f"{rep['score_psi']['psi']:.3f}",
                   rep["score_psi"]["band"])
    cols[1].metric("Approval rate", f"{rep['approval_rate']:.1%}")
    cols[2].metric("Mean t*(x)", f"{rep['mean_threshold_t_star']:.4f}")

    fdf = pd.DataFrame([{"feature": f, "psi": v["psi"], "band": v["band"]}
                        for f, v in rep["feature_psi"].items()])
    st.bar_chart(fdf.set_index("feature")["psi"])
    st.write(f"Bands: investigate > {PSI_INVESTIGATE}, retrain > {PSI_RETRAIN}")
    st.dataframe(fdf)
    if drifted:
        worst = fdf.psi.max()
        st.error(f"ALARM: max feature PSI = {worst:.2f} "
                 f"({'RETRAIN band' if worst > PSI_RETRAIN else 'investigate band'}) "
                 "— simulated recession detected") if worst > PSI_INVESTIGATE else None
