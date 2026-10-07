"""Phase 5a — global beeswarm (stratified 25k sample, fold-0 constrained model,
stated on the figure) + reason-code demo + score scale figure."""
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.constants import DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, TABLES_DIR
from src.data.load import load_modeling_frame
from src.models.explain import pd_to_score, reason_codes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase5-explain")

dev = load_modeling_frame("dev")
feats = json.load(open(os.path.join(DATA_PROCESSED, "lgbm_features.json")))["features"]
b = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lgbm_constrained_fold0.txt"))

# stratified 25k sample (matches phase5_design_frozen.json + the figure title)
sample = dev.groupby("TARGET", group_keys=False).apply(
    lambda g: g.sample(int(round(25000 * len(g) / len(dev))), random_state=0))
log.info("computing exact TreeSHAP on 25k sample (fold-0 constrained, stated)…")
contrib = b.predict(sample[feats], pred_contrib=True)[:, :-1]

mean_abs = pd.Series(np.abs(contrib).mean(0), index=feats).sort_values(ascending=False)
top = mean_abs.head(20)
fig, ax = plt.subplots(figsize=(9, 7))
for i, f in enumerate(top.index):
    j = feats.index(f)
    vals = contrib[:, j]
    jitter = np.random.RandomState(i).uniform(-0.28, 0.28, len(vals))
    x = sample[f].values.astype(float) if pd.api.types.is_numeric_dtype(sample[f]) else np.zeros(len(vals))
    with np.errstate(all="ignore"):
        r = pd.Series(x).rank(pct=True).fillna(0.5).values
    ax.scatter(vals, i + jitter, c=r, cmap="coolwarm", s=2, alpha=0.4)
ax.set_yticks(range(len(top)), top.index, fontsize=7)
ax.invert_yaxis()
ax.axvline(0, c="gray", lw=0.5)
ax.set_xlabel("SHAP contribution to default log-odds (exact TreeSHAP)")
ax.set_title("Global feature effects — 25k sample, fold-0 constrained model (stated)\n"
             "color = feature value percentile (blue low → red high)")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "52_shap_beeswarm.png"), dpi=120)
plt.close(fig)
mean_abs.head(30).to_csv(os.path.join(TABLES_DIR, "shap_mean_abs_top30.csv"))

# reason-code demo on 5 high-risk applicants + score scale mapping
oofc = pd.read_parquet(os.path.join(DATA_PROCESSED, "oof_constrained_calibrated.parquet"))
hi = oofc.nlargest(5, "p_cal_crossfit").merge(dev, on=["SK_ID_CURR","fold","TARGET"])
codes = reason_codes(b, hi, feats, top_k=3)
demo = [{"SK_ID_CURR": int(r.SK_ID_CURR), "p_cal": float(r.p_cal_crossfit),
         "score": float(pd_to_score(np.array([r.p_cal_crossfit]))[0]),
         "reason_codes": c} for r, c in zip(hi.itertuples(), codes, strict=True)]
json.dump(demo, open(os.path.join(TABLES_DIR, "reason_codes_demo.json"), "w"), indent=2)
log.info("explain artifacts written: beeswarm, top30 table, reason-code demo")
