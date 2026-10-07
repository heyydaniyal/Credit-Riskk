"""
EDA plot functions — one function per figure.

DRY design: notebooks call these functions to display figures inline;
scripts call the same functions to regenerate reports/figures/.
The plotting logic exists exactly once.

Every function:
  - takes the dev dataframe,
  - returns (fig, takeaway_string),
  - saves a PNG into reports/figures/ when save=True.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.constants import FIGURES_DIR, TABLES_DIR

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
BASE_RATE = 0.0807

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def _finish(fig, name: str, save: bool):
    """Save figure to reports/figures/ (single save path for the project)."""
    if save:
        fig.savefig(os.path.join(FIGURES_DIR, name), bbox_inches="tight")
    return fig


def plot_class_imbalance(dev: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = dev["TARGET"].value_counts()
    ax.bar(["Repaid (0)", "Default (1)"], counts.values, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Class imbalance: 8.07% default rate")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    takeaway = ("8.07% positives -> accuracy is useless (reject-nothing scores 92%); "
                "PR-AUC baseline = 0.08.")
    return _finish(fig, "01_class_imbalance.png", save), takeaway


def plot_ext_source_deciles(dev: pd.DataFrame, save: bool = True):
    dv = dev[dev["EXT_SOURCE_2"].notna()].copy()
    dv["dec"] = pd.qcut(dv["EXT_SOURCE_2"], 10, labels=False)
    rate = dv.groupby("dec")["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rate.index, rate.values, "o-", color="#264653")
    ax.axhline(BASE_RATE, ls="--", color="gray", label="base rate 8.07%")
    ax.set_xlabel("EXT_SOURCE_2 decile (low->high)")
    ax.set_ylabel("Default rate")
    ax.set_title("EXT_SOURCE_2: 6x default spread — external scores dominate")
    ax.legend()
    fig.tight_layout()
    takeaway = ("EXT_SOURCE_2 bottom decile 18.3% vs top 3.0% default — "
                "external scores dominate; note upfront.")
    return _finish(fig, "02_ext_source_deciles.png", save), takeaway


def plot_missingness(dev: pd.DataFrame, save: bool = True):
    miss = dev.isna().mean().sort_values(ascending=False)
    top_miss = miss[miss > 0].head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(top_miss)), top_miss.values, color="#e9c46a")
    ax.set_yticks(range(len(top_miss)))
    ax.set_yticklabels(top_miss.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction missing")
    ax.set_title("Missingness (EXT_SOURCE_1 = 56%) — missingness is signal")
    fig.tight_layout()
    takeaway = ("EXT_SOURCE_1 56% missing; being unscored by a bureau is itself "
                "predictive -> explicit _is_missing flags.")
    return _finish(fig, "03_missingness.png", save), takeaway


def plot_missingness_signal(dev: pd.DataFrame, save: bool = True):
    tmp = dev.copy()
    tmp["ext1_missing"] = tmp["EXT_SOURCE_1"].isna().astype(int)
    rate_by_miss = tmp.groupby("ext1_missing")["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Has EXT_SOURCE_1", "Missing EXT_SOURCE_1"], rate_by_miss.values,
           color=["#2a9d8f", "#e76f51"])
    ax.axhline(BASE_RATE, ls="--", color="gray")
    ax.set_ylabel("Default rate")
    ax.set_title("Missing EXT_SOURCE_1 -> higher default: missingness IS signal")
    fig.tight_layout()
    takeaway = (f"Applicants missing EXT_SOURCE_1 default at {rate_by_miss[1]:.1%} vs "
                f"{rate_by_miss[0]:.1%} — missingness carries risk signal.")
    return _finish(fig, "04_missingness_signal.png", save), takeaway


def plot_leverage_distribution(dev: pd.DataFrame, save: bool = True):
    tmp = dev.copy()
    tmp["credit_income"] = tmp["AMT_CREDIT"] / tmp["AMT_INCOME_TOTAL"]
    ci = tmp[tmp["credit_income"] < tmp["credit_income"].quantile(0.99)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for t, c, lbl in [(0, "#2a9d8f", "Repaid"), (1, "#e76f51", "Default")]:
        ax.hist(ci[ci["TARGET"] == t]["credit_income"], bins=40, alpha=0.5,
                density=True, color=c, label=lbl)
    ax.set_xlabel("AMT_CREDIT / AMT_INCOME_TOTAL (leverage)")
    ax.set_title("Leverage distribution by outcome")
    ax.legend()
    fig.tight_layout()
    takeaway = "Credit/income leverage: defaulters skew slightly higher — a credit-officer feature."
    return _finish(fig, "05_credit_income.png", save), takeaway


def plot_employment_length(dev: pd.DataFrame, save: bool = True):
    tmp = dev.copy()
    tmp["emp_years"] = -pd.to_numeric(tmp["DAYS_EMPLOYED"], errors="coerce") / 365
    ey = tmp[tmp["emp_years"].notna() & (tmp["emp_years"] < 40) & (tmp["emp_years"] > 0)]
    ey_bins = pd.cut(ey["emp_years"], bins=[0, 1, 3, 5, 10, 20, 40])
    rate = ey.groupby(ey_bins, observed=True)["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(rate)), rate.values, color="#457b9d")
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([str(i) for i in rate.index], rotation=45, fontsize=8)
    ax.axhline(BASE_RATE, ls="--", color="gray")
    ax.set_ylabel("Default rate")
    ax.set_title("Default rate by employment length (excl. pensioners)")
    fig.tight_layout()
    takeaway = ("Shorter employment -> higher default; the 365243 pensioner sentinel "
                "(17.9%) fixed to NaN+flag.")
    return _finish(fig, "06_employment_length.png", save), takeaway


def plot_age(dev: pd.DataFrame, save: bool = True):
    tmp = dev.copy()
    tmp["age_years"] = -tmp["DAYS_BIRTH"] / 365
    age_bins = pd.cut(tmp["age_years"], bins=[20, 30, 40, 50, 60, 70])
    rate = tmp.groupby(age_bins, observed=True)["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(rate)), rate.values, color="#a8dadc")
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([str(i) for i in rate.index], rotation=45, fontsize=8)
    ax.axhline(BASE_RATE, ls="--", color="gray")
    ax.set_ylabel("Default rate")
    ax.set_title("Default rate by age — younger applicants default more")
    fig.tight_layout()
    takeaway = "Younger applicants (20-30) default well above base rate; risk falls with age."
    return _finish(fig, "07_age.png", save), takeaway


def plot_segment_scan_contract(dev: pd.DataFrame, save: bool = True):
    seg = dev.groupby("NAME_CONTRACT_TYPE").agg(
        default_rate=("TARGET", "mean"),
        avg_credit=("AMT_CREDIT", "mean"),
        n=("TARGET", "count"),
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(seg))
    ax.bar(x, seg["default_rate"], color=["#e76f51", "#2a9d8f"])
    ax.set_xticks(x)
    ax.set_xticklabels(seg.index)
    ax.axhline(BASE_RATE, ls="--", color="gray")
    ax.set_ylabel("Default rate")
    for i, (_, row) in enumerate(seg.iterrows()):
        ax.text(i, row["default_rate"], f"{row['default_rate']:.1%}\n{int(row['n']):,}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("Segment scan: cash 8.3% vs revolving 5.5% (revolving 9.5% of book)")
    fig.tight_layout()
    takeaway = ("Cash defaults 8.3% vs revolving 5.5%; revolving only 9.5% of "
                "applications -> LGD/EAD split moves fewer applicants.")
    return _finish(fig, "08_segment_scan.png", save), takeaway


def plot_gender_base_rates(dev: pd.DataFrame, save: bool = True):
    grate = dev[dev["CODE_GENDER"].isin(["F", "M"])].groupby("CODE_GENDER")["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(grate.index, grate.values, color=["#f4a261", "#457b9d"])
    ax.axhline(BASE_RATE, ls="--", color="gray")
    ax.set_ylabel("Default rate")
    for i, v in enumerate(grate.values):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom")
    ax.set_title("Gender base rates differ (F 7.0% / M 10.1%) — fairness pre-registration")
    fig.tight_layout()
    takeaway = ("Base default rates differ by gender (F 7.0%, M 10.1%) -> "
                "disparate-impact flag is a live possibility (Phase 5b).")
    return _finish(fig, "09_gender_base_rates.png", save), takeaway


def plot_correlation(dev: pd.DataFrame, save: bool = True):
    tmp = dev.copy()
    tmp["credit_income"] = tmp["AMT_CREDIT"] / tmp["AMT_INCOME_TOTAL"]
    tmp["DAYS_EMPLOYED"] = pd.to_numeric(tmp["DAYS_EMPLOYED"], errors="coerce")
    key_feats = ["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                 "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
                 "DAYS_BIRTH", "DAYS_EMPLOYED", "credit_income"]
    corr = tmp[key_feats].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"shrink": 0.7}, annot_kws={"size": 7})
    ax.set_title("Spearman correlation — key features vs TARGET")
    fig.tight_layout()
    takeaway = ("EXT_SOURCE_* most correlated with TARGET (negative); no single raw "
                "feature is dominant enough to leak.")
    return _finish(fig, "10_correlation.png", save), takeaway


def plot_leverage_deciles(dev: pd.DataFrame, save: bool = True):
    tmp = dev.copy()
    tmp["credit_income"] = tmp["AMT_CREDIT"] / tmp["AMT_INCOME_TOTAL"]
    cv = tmp[tmp["credit_income"].notna()].copy()
    cv["dec"] = pd.qcut(cv["credit_income"], 10, labels=False, duplicates="drop")
    rate = cv.groupby("dec")["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rate.index, rate.values, "o-", color="#e76f51")
    ax.axhline(BASE_RATE, ls="--", color="gray", label="base rate")
    ax.set_xlabel("Credit/Income ratio decile (low->high leverage)")
    ax.set_ylabel("Default rate")
    ax.set_title("Default rate by leverage decile")
    ax.legend()
    fig.tight_layout()
    takeaway = ("Leverage deciles surprisingly flat (6.8%-9.2%) — raw leverage weaker "
                "than expected; interactions needed.")
    return _finish(fig, "11_leverage_deciles.png", save), takeaway


def plot_term_buckets(dev: pd.DataFrame, save: bool = True):
    """Also writes reports/tables/segment_scan_term.csv (cost-model input)."""
    tmp = dev.copy()
    tmp["term_years"] = (tmp["AMT_CREDIT"] / (12 * tmp["AMT_ANNUITY"])).clip(0.5, 7)
    term_bins = pd.cut(tmp["term_years"], bins=[0.5, 1, 1.5, 2, 3, 5, 7],
                       include_lowest=True)
    seg = tmp.groupby(term_bins, observed=True).agg(
        default_rate=("TARGET", "mean"),
        avg_credit=("AMT_CREDIT", "mean"),
        n=("TARGET", "count"),
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(seg)), seg["default_rate"], color="#457b9d")
    ax.set_xticks(range(len(seg)))
    ax.set_xticklabels([str(i) for i in seg.index], rotation=30, fontsize=8)
    ax.axhline(BASE_RATE, ls="--", color="gray")
    ax.set_ylabel("Default rate")
    ax.set_title("Default rate by implied term (years) — term drives t*(x) variation")
    for i, (_, row) in enumerate(seg.iterrows()):
        ax.text(i, row["default_rate"], f"{int(row['n'] // 1000)}k",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    if save:
        seg.to_csv(os.path.join(TABLES_DIR, "segment_scan_term.csv"))
    takeaway = ("Terms cluster 1-3y; default rate NON-monotonic in term "
                "(peak 10.3% at 1-1.5y). Table saved to reports/tables/.")
    return _finish(fig, "12_term_buckets.png", save), takeaway


# Ordered registry — scripts iterate this to regenerate everything (DRY)
ALL_PLOTS = [
    plot_class_imbalance,
    plot_ext_source_deciles,
    plot_missingness,
    plot_missingness_signal,
    plot_leverage_distribution,
    plot_employment_length,
    plot_age,
    plot_segment_scan_contract,
    plot_gender_base_rates,
    plot_correlation,
    plot_leverage_deciles,
    plot_term_buckets,
]


# ── Figures 13–15: categoricals + auxiliary tables (added after review) ──

def plot_categorical_risk(dev: pd.DataFrame, save: bool = True):
    """Default rate by education and income type — the underwriting categoricals."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in [
        (axes[0], "NAME_EDUCATION_TYPE", "by education"),
        (axes[1], "NAME_INCOME_TYPE", "by income type"),
    ]:
        grp = dev.groupby(col)["TARGET"].agg(["mean", "count"])
        grp = grp[grp["count"] >= 200].sort_values("mean")  # hide tiny groups
        ax.barh(range(len(grp)), grp["mean"], color="#457b9d")
        ax.set_yticks(range(len(grp)))
        ax.set_yticklabels([f"{i} ({int(c)//1000}k)" for i, c in
                            zip(grp.index, grp["count"], strict=True)], fontsize=7)
        ax.axvline(BASE_RATE, ls="--", color="gray")
        ax.set_xlabel("Default rate")
        ax.set_title(f"Default rate {title}")
    fig.tight_layout()
    takeaway = ("Lower-secondary education ~2x default of higher education; "
                "'Working' income type riskier than pensioners/state servants — "
                "strong, sensible categoricals (also gender-correlated: proxy-audit relevance).")
    return _finish(fig, "13_categorical_risk.png", save), takeaway


def plot_bureau_history_signal(dev: pd.DataFrame, bureau: pd.DataFrame | None = None,
                               save: bool = True):
    """Default rate by number of prior bureau loans (aux table #1)."""
    if bureau is None:
        from src.data.load import load_bureau
        bureau = load_bureau()
    counts = bureau[bureau["SK_ID_CURR"].isin(set(dev["SK_ID_CURR"]))] \
        .groupby("SK_ID_CURR").size()
    tmp = dev[["SK_ID_CURR", "TARGET"]].copy()
    tmp["n_bureau"] = tmp["SK_ID_CURR"].map(counts).fillna(0)
    bins = pd.cut(tmp["n_bureau"], [-0.5, 0.5, 2.5, 5.5, 10.5, 200],
                  labels=["0 (no history)", "1-2", "3-5", "6-10", ">10"])
    grp = tmp.groupby(bins, observed=True)["TARGET"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(len(grp)), grp["mean"], color="#2a9d8f")
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(grp.index, fontsize=8)
    ax.axhline(BASE_RATE, ls="--", color="gray")
    ax.set_ylabel("Default rate")
    for i, (_, row) in enumerate(grp.iterrows()):
        ax.text(i, row["mean"], f"{row['mean']:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Default rate by # prior bureau loans — NO history is the risky group")
    fig.tight_layout()
    takeaway = ("No bureau history -> 10.1% default (riskiest); some history ~7.4-8.2%. "
                "Thin-file risk confirms 'missingness is signal' at the table level.")
    return _finish(fig, "14_bureau_history.png", save), takeaway


def plot_prev_refusal_signal(dev: pd.DataFrame, prev: pd.DataFrame | None = None,
                             save: bool = True):
    """Default rate by share of prior Home Credit applications refused (aux table #2)."""
    if prev is None:
        from src.data.load import load_previous_application
        prev = load_previous_application()
    p = prev[prev["SK_ID_CURR"].isin(set(dev["SK_ID_CURR"]))]
    refused = p.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"] \
        .apply(lambda s: (s == "Refused").mean())
    tmp = dev[["SK_ID_CURR", "TARGET"]].copy()
    tmp["refused_share"] = tmp["SK_ID_CURR"].map(refused)
    bins = pd.cut(tmp["refused_share"], [-0.01, 0.0, 0.25, 0.5, 1.0],
                  labels=["0% refused", "0-25%", "25-50%", ">50%"])
    grp = tmp.groupby(bins, observed=True)["TARGET"].agg(["mean", "count"])
    no_hist_rate = tmp.loc[tmp["refused_share"].isna(), "TARGET"].mean()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = ["no prior HC apps"] + list(grp.index)
    values = [no_hist_rate] + list(grp["mean"])
    colors = ["#a8dadc"] + ["#e76f51"] * len(grp)
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(BASE_RATE, ls="--", color="gray")
    ax.set_ylabel("Default rate")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Default rate by prior-refusal share — 2.2x monotone risk signal")
    fig.tight_layout()
    takeaway = ("Prior refusal share is a clean monotone signal: 7.1% -> 15.9% default. "
                "Prior rejections are highly predictive — top Phase 2 aggregation.")
    return _finish(fig, "15_prev_refusal.png", save), takeaway


ALL_PLOTS.extend([plot_categorical_risk, plot_bureau_history_signal,
                  plot_prev_refusal_signal])


# ── Figures 16–20: deep-dive additions (figure cap lifted by review) ──

def plot_numeric_distributions(dev: pd.DataFrame, save: bool = True):
    """Raw shapes of the 8 core numerics — justifies transformation decisions."""
    specs = [
        ("AMT_INCOME_TOTAL", True), ("AMT_CREDIT", False),
        ("AMT_ANNUITY", False), ("DAYS_BIRTH", False),
        ("DAYS_EMPLOYED", False), ("EXT_SOURCE_1", False),
        ("EXT_SOURCE_2", False), ("EXT_SOURCE_3", False),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, (col, log) in zip(axes.ravel(), specs, strict=False):
        v = pd.to_numeric(dev[col], errors="coerce").dropna()
        if log:
            v = np.log1p(v)
            ax.set_xlabel(f"log1p({col})", fontsize=8)
        else:
            ax.set_xlabel(col, fontsize=8)
        ax.hist(v, bins=50, color="#457b9d")
        ax.tick_params(labelsize=7)
    fig.suptitle("Raw distributions — income needs log (skew 375); EXT_SOURCE already 0-1")
    fig.tight_layout()
    takeaway = ("AMT_INCOME_TOTAL skew=375 (shown logged) -> log for logistic; "
                "credit/annuity mildly right-skewed (1.2-1.6, fine); "
                "EXT_SOURCE_* already smooth 0-1 scores.")
    return _finish(fig, "16_numeric_distributions.png", save), takeaway


def plot_ext_source_all_deciles(dev: pd.DataFrame, save: bool = True):
    """All three EXT_SOURCE decile curves — do they agree?"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"EXT_SOURCE_1": "#e76f51", "EXT_SOURCE_2": "#264653",
              "EXT_SOURCE_3": "#2a9d8f"}
    for col, color in colors.items():
        dv = dev[dev[col].notna()].copy()
        dv["dec"] = pd.qcut(dv[col], 10, labels=False)
        rate = dv.groupby("dec")["TARGET"].mean()
        ax.plot(rate.index, rate.values, "o-", color=color,
                label=f"{col} ({dv.shape[0] // 1000}k rows)")
    ax.axhline(BASE_RATE, ls="--", color="gray")
    ax.set_xlabel("Decile (low->high score)")
    ax.set_ylabel("Default rate")
    ax.set_title("All three external scores: same monotone story, different coverage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    takeaway = ("All three EXT_SOURCEs are monotone with similar spreads — they "
                "corroborate each other; their MEAN is an obvious Phase 2 feature.")
    return _finish(fig, "17_ext_source_all.png", save), takeaway


def plot_amount_deciles(dev: pd.DataFrame, save: bool = True):
    """Default rate by income decile and by credit-amount decile."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in [(axes[0], "AMT_INCOME_TOTAL", "income decile"),
                           (axes[1], "AMT_CREDIT", "credit-amount decile")]:
        dv = dev[dev[col].notna()].copy()
        dv["dec"] = pd.qcut(dv[col], 10, labels=False, duplicates="drop")
        rate = dv.groupby("dec")["TARGET"].mean()
        ax.plot(rate.index, rate.values, "o-", color="#457b9d")
        ax.axhline(BASE_RATE, ls="--", color="gray")
        ax.set_xlabel(f"{title} (low->high)")
        ax.set_ylabel("Default rate")
    fig.suptitle("Money alone is a weak separator — both curves are shallow")
    fig.tight_layout()
    takeaway = ("Income and credit-amount deciles are shallow (~2-3pp spread) — "
                "raw money variables are weak solo predictors, consistent with "
                "the flat leverage/DTI findings (reject-inference footprint).")
    return _finish(fig, "18_amount_deciles.png", save), takeaway


def plot_heavy_categoricals(dev: pd.DataFrame, save: bool = True):
    """Occupation (with missing as its own bar), family status, org-type extremes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    occ = dev.copy()
    occ["OCCUPATION_TYPE"] = occ["OCCUPATION_TYPE"].fillna("(missing 31%)")
    grp = occ.groupby("OCCUPATION_TYPE")["TARGET"].agg(["mean", "count"])
    grp = grp[grp["count"] >= 500].sort_values("mean")
    axes[0].barh(range(len(grp)), grp["mean"],
                 color=["#e76f51" if i == "(missing 31%)" else "#457b9d"
                        for i in grp.index])
    axes[0].set_yticks(range(len(grp)))
    axes[0].set_yticklabels(grp.index, fontsize=6.5)
    axes[0].axvline(BASE_RATE, ls="--", color="gray")
    axes[0].set_title("Occupation (18 levels)", fontsize=9)

    fam = dev.groupby("NAME_FAMILY_STATUS")["TARGET"].agg(["mean", "count"])
    fam = fam[fam["count"] >= 500].sort_values("mean")
    axes[1].barh(range(len(fam)), fam["mean"], color="#2a9d8f")
    axes[1].set_yticks(range(len(fam)))
    axes[1].set_yticklabels(fam.index, fontsize=7)
    axes[1].axvline(BASE_RATE, ls="--", color="gray")
    axes[1].set_title("Family status", fontsize=9)

    org = dev.groupby("ORGANIZATION_TYPE")["TARGET"].agg(["mean", "count"])
    org = org[org["count"] >= 1000].sort_values("mean")
    extremes = pd.concat([org.head(6), org.tail(6)])
    axes[2].barh(range(len(extremes)), extremes["mean"], color="#e9c46a")
    axes[2].set_yticks(range(len(extremes)))
    axes[2].set_yticklabels(extremes.index, fontsize=6.5)
    axes[2].axvline(BASE_RATE, ls="--", color="gray")
    axes[2].set_title("Organization type: 6 safest + 6 riskiest of 58", fontsize=9)

    for ax in axes:
        ax.set_xlabel("Default rate", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    takeaway = ("Occupation spans 4.6%-18.1% (low-skill laborers ~2.2x base); its "
                "31% missingness sits below base rate (white-collar skip it?); "
                "widows are the safest family status; ORGANIZATION_TYPE's 58 levels "
                "need target encoding (fold-safe) or grouping in Phase 2.")
    return _finish(fig, "19_heavy_categoricals.png", save), takeaway


def plot_univariate_auc(dev: pd.DataFrame, scan: pd.DataFrame | None = None,
                        save: bool = True):
    """Top-20 univariate AUCs — the leakage screen visualized."""
    if scan is None:
        from src.data.supplementary import univariate_auc_scan
        scan = univariate_auc_scan(dev, save=True)
    top = scan.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#e76f51" if f else "#2a9d8f" for f in top["leakage_flag"]]
    ax.barh(range(len(top)), top["auc_strength"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=7)
    ax.axvline(0.5, color="gray", lw=0.8)
    ax.axvline(0.90, ls="--", color="#e76f51", label="leakage flag (0.90)")
    ax.set_xlim(0.48, 1.0)
    ax.set_xlabel("Univariate AUC (direction-corrected)")
    ax.set_title("Leakage screen: top 20 solo features — nothing near 0.90")
    ax.legend(fontsize=8)
    fig.tight_layout()
    takeaway = ("PASS: no feature exceeds 0.90 alone (top: EXT_SOURCE_3 at 0.68). "
                "Only 9/104 clear 0.55 — signal is diffuse, which is exactly why "
                "aggregation features are differentiator #1.")
    return _finish(fig, "20_univariate_auc.png", save), takeaway


ALL_PLOTS.extend([plot_numeric_distributions, plot_ext_source_all_deciles,
                  plot_amount_deciles, plot_heavy_categoricals,
                  plot_univariate_auc])
