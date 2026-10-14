"""Figures of the results section, from results/tables and the selected return series."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from style import (PDATA, BLUE, DATA, FREQ_NAME, FREQS, GRAY, INK, LIGHT, NAVY, ORANGE, RULE, SAND, SKY, TEAL,
                   hgrid, save, setup)

T = lambda name: pd.read_csv(DATA / "results/tables" / f"{name}.csv")
BASE = ("ols", "ridge", "lasso", "ipca", "sipca", "xgb", "rf", "ebm")
TOPICS = ["lda", "lda_all", "minilm", "finbert", "nolbert", "nolbert20"]
TOPIC_NAME = {"lda": "LDA", "lda_all": "LDA, all articles", "minilm": "MiniLM", "finbert": "FinBERT",
              "nolbert": "NoLBERT", "nolbert20": "NoLBERT-20"}
DIVERGE = LinearSegmentedColormap.from_list("div", [ORANGE, "#F3D3C0", "#FFFFFF", "#BFDDE0", TEAL])


def kind(features):
    if features.endswith("_placebo"):
        return "placebo"
    if features.startswith("grand_"):
        return "grand"
    p = features.partition("_")[0]
    if p in ("narr", "text", "all") and features.endswith("_s63"):
        return p + "_s"
    return p


def topic_of(features):
    k = kind(features)
    if k in ("narr", "firm", "text", "all", "narr_s", "text_s", "all_s"):
        f = features.partition("_")[2]
        return f[:-4] if f.endswith("_s63") else f
    return None


# ------------------------------------------------------------------ the shock window
def fig_shock_window():
    sw = pd.read_csv(DATA / "results/tuning/shock_window.csv")
    fc = T("forecast")
    fc = fc[fc.model.isin(BASE) & (fc.split == "test")].copy()
    fc["kind"] = fc.features.map(kind)
    fc["topic"] = fc.features.map(topic_of)
    fig = plt.figure(figsize=(6.9, 2.75))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.15, 0.42, 1, 1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    v = sw.groupby(["freq", "window"]).ic_mean.mean().unstack()
    wins = [1, 5, 21, 63]
    cols = [LIGHT, SKY, BLUE, TEAL]
    for i, f in enumerate(FREQS):
        for k, w in enumerate(wins):
            ax.bar(i + (k - 1.5) * 0.2, 100 * v.loc[f, w], width=0.19, color=cols[k], label=f"{w} d" if i == 0 else None)
    ax.axhline(0, color=GRAY, lw=0.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([FREQ_NAME[f] for f in FREQS], fontsize=6.6)
    ax.set_ylabel("validation rank IC (×100)")
    ax.set_title("(a) Validation, 18 linear fits", fontsize=7.4)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=6, handlelength=0.8, columnspacing=0.8)
    hgrid(ax)
    for j, f in enumerate(FREQS):
        ax = fig.add_subplot(gs[j + 2])
        g = fc[fc.freq == f]
        five = g[g.kind.isin(["narr", "text", "all"])].groupby("topic").ic_mean.mean()
        sixty = g[g.kind.isin(["narr_s", "text_s", "all_s"])].groupby("topic").ic_mean.mean()
        y = np.arange(len(TOPICS))
        for i, t in enumerate(TOPICS):
            ax.annotate("", xy=(100 * sixty[t], i), xytext=(100 * five[t], i),
                        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.7, mutation_scale=7))
        ax.scatter(100 * five[TOPICS], y, s=16, color=SKY, zorder=3, label="5-day shocks")
        ax.scatter(100 * sixty[TOPICS], y, s=16, color=TEAL, zorder=3, label="63-day shocks")
        ax.axvline(0, color=GRAY, lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([TOPIC_NAME[t] for t in TOPICS] if j == 0 else [], fontsize=6.4)
        ax.invert_yaxis()
        ax.set_title(f"({'bcd'[j]}) {FREQ_NAME[f]}, test", fontsize=7.4)
        ax.set_xlabel("rank IC (×100)", fontsize=6.4)
        hgrid(ax, "x")
        if j == 1:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6)
    save(fig, "fig_shock_window")


# ------------------------------------------------------------------ the landscape
KINDS = [("price", "Price"), ("narr", "Narrative, 5 d"), ("narr_s", "Narrative, 63 d"), ("firm", "Firm news"),
         ("text", "Text, 5 d"), ("text_s", "Text, 63 d"), ("all", "Price + text, 5 d"), ("all_s", "Price + text, 63 d"),
         ("tone", "Tone (FinBERT, LM)"), ("emb", "Headline embeddings"), ("px", "Price + news"), ("placebo", "Placebo")]


def fig_landscape():
    L = T("landscape")
    L = L[(L.by == "kind")]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1), width_ratios=[1, 1])
    for ax, split, title in ((axes[0], "validation", "(a) Validation 2020–21"), (axes[1], "test", "(b) Test 2022–24")):
        g = L[L.split == split].set_index(["group", "freq"])
        tvals = np.array([[g.loc[(k, f), "ic_t"] for f in FREQS] for k, _ in KINDS])
        ic = np.array([[g.loc[(k, f), "ic_mean"] for f in FREQS] for k, _ in KINDS])
        ax.imshow(tvals, cmap=DIVERGE, norm=TwoSlopeNorm(0, -3.5, 3.5), aspect="auto")
        for i in range(len(KINDS)):
            for j in range(3):
                strong = abs(tvals[i, j]) > 2.0
                ax.text(j, i, f"{100 * ic[i, j]:+.1f}  ({tvals[i, j]:.1f})", ha="center", va="center",
                        fontsize=6.2, color="white" if abs(tvals[i, j]) > 2.6 else INK,
                        fontweight="semibold" if strong else "normal")
        ax.set_xticks(range(3)); ax.set_xticklabels([FREQ_NAME[f] for f in FREQS])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(KINDS)))
        ax.set_yticklabels([n for _, n in KINDS] if split == "validation" else [])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=7.6, pad=16)
    fig.subplots_adjust(wspace=0.04)
    save(fig, "fig_landscape")


# ------------------------------------------------------------------ cumulative returns
def fig_cumret():
    R = pd.read_csv(PDATA / "selected_returns.csv", parse_dates=["date", "hold_start"])
    sel = T("selection")
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.45), sharey=False)
    lines = [("grand_s63", "ens", "ls_buf", "Combination, price + text, 63 d", NAVY, 1.5, "-"),
             ("grand_text_s63", "ens", "ls_buf", "Combination, text, 63 d", TEAL, 1.3, "-"),
             ("price", "ens", "ls_buf", "Price ensemble", SKY, 1.1, "-"),
             (None, None, "ls_buf", "Single validated pick", ORANGE, 1.0, (0, (3, 1.5))),
             ("benchmark", None, "univ_vw", "Value-weighted market, excess", GRAY, 0.9, (0, (1, 1.2)))]
    for ax, f in zip(axes, FREQS):
        pick = sel[(sel.freq == f) & (sel.family == "any")].iloc[0]
        for feat, mod, rule, label, c, lw, ls in lines:
            if feat is None:
                feat, mod = pick.features, pick.model
            q = R[(R.freq == f) & (R.split == "test") & (R.features == feat) & (R.strategy == rule)]
            if mod:
                q = q[q.model == mod]
            q = q.sort_values("date")
            wealth = np.cumprod(1 + q["excess"].to_numpy())
            x = np.r_[q["hold_start"].iloc[0], q["date"].iloc[1:].to_numpy(), q["date"].iloc[-1] + pd.Timedelta(days=30)][:len(wealth) + 1]
            ax.plot(x, np.r_[1, wealth], color=c, lw=lw, ls=ls, label=label)
        ax.axhline(1, color=RULE, lw=0.6)
        ax.set_title(FREQ_NAME[f] + ", buffered long-short, net", fontsize=7.4)
        ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
        hgrid(ax)
    axes[0].set_ylabel("growth of $1 (excess return)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=6.1)
    fig.subplots_adjust(wspace=0.25)
    save(fig, "fig_cumret")


# ------------------------------------------------------------------ models
MODELS = [("ols", "OLS"), ("ridge", "Ridge"), ("lasso", "Lasso"), ("ipca", "IPCA"), ("sipca", "Sparse IPCA"),
          ("xgb", "XGBoost"), ("rf", "Random forest"), ("ebm", "EBM"), ("ens", "Ensemble of eight"),
          ("ols_rk", "OLS, ranked"), ("ridge_rk", "Ridge, ranked"), ("lasso_rk", "Lasso, ranked"),
          ("ipca_rk", "IPCA, ranked"), ("sipca_rk", "Sparse IPCA, ranked"), ("xgb_rk", "XGBoost, ranked")]


def fig_models():
    L = T("landscape")
    L = L[(L.by == "model")]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9), sharey=True)
    y = np.arange(len(MODELS))
    for ax, f in zip(axes, FREQS):
        t = L[(L.freq == f) & (L.split == "test")].set_index("group")
        v = L[(L.freq == f) & (L.split == "validation")].set_index("group")
        ic = np.array([t.loc[m, "ic_mean"] for m, _ in MODELS])
        vi = np.array([v.loc[m, "ic_mean"] for m, _ in MODELS])
        cols = [TEAL if m.endswith("_rk") else (NAVY if m == "ens" else BLUE) for m, _ in MODELS]
        ax.barh(y, 100 * ic, color=cols, height=0.62, label="test")
        ax.scatter(100 * vi, y, s=13, facecolors="white", edgecolors=ORANGE, linewidths=0.9, zorder=3, label="validation")
        ax.axvline(0, color=GRAY, lw=0.5)
        ax.set_title(FREQ_NAME[f], fontsize=7.6)
        ax.set_xlabel("average rank IC ×100", fontsize=6.4)
        hgrid(ax, "x")
    axes[0].set_yticks(y); axes[0].set_yticklabels([n for _, n in MODELS], fontsize=6.4); axes[0].invert_yaxis()
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.2)
    fig.subplots_adjust(wspace=0.1)
    save(fig, "fig_models")


# ------------------------------------------------------------------ portfolio rules
RULES = [("ls_ew", "Decile long-short"), ("ls_buf", "Buffered long-short"), ("ls_vw", "Decile, cap-weighted"),
         ("ls_buf_vw", "Buffered, cap-weighted"), ("rank", "Rank-weighted"), ("mvo", "Mean-variance"),
         ("long_ew", "Long top decile"), ("long_vw", "Long top decile, cap-w."), ("tilt_vw", "Cap-weighted tilt")]


def fig_rules_results():
    st = T("strategies")
    st = st[(st.split == "test") & (st.features != "benchmark") & ~st.features.str.endswith("_placebo")]
    st = st[st.model.isin(BASE)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.6), width_ratios=[1.35, 1], sharey=True)
    y = np.arange(len(RULES))
    marks = {"M": ("o", NAVY), "2W": ("s", TEAL), "W": ("D", ORANGE)}
    for f in FREQS:
        g = st[st.freq == f].groupby("strategy")
        med = g.sharpe_net.median(); q1 = g.sharpe_net.quantile(0.25); q3 = g.sharpe_net.quantile(0.75)
        tu = g.turnover.median()
        off = {"M": -0.22, "2W": 0, "W": 0.22}[f]
        for i, (r, _) in enumerate(RULES):
            col = "information_ratio" if r.startswith(("long", "tilt")) else None
            a1.plot([q1[r], q3[r]], [i + off] * 2, color=marks[f][1], lw=0.8, alpha=0.6)
        a1.scatter([med[r] for r, _ in RULES], y + off, marker=marks[f][0], s=14, color=marks[f][1], label=FREQ_NAME[f], zorder=3)
        a2.scatter([100 * tu[r] for r, _ in RULES], y + off, marker=marks[f][0], s=14, color=marks[f][1], zorder=3)
    a1.axvline(0, color=GRAY, lw=0.5)
    a1.set_yticks(y); a1.set_yticklabels([n for _, n in RULES], fontsize=6.6); a1.invert_yaxis()
    a1.set_xlabel("net Sharpe ratio, test (median and interquartile range)", fontsize=6.4)
    a2.set_xlabel("turnover per period, median (%)", fontsize=6.4)
    a1.legend(loc="lower left", fontsize=6)
    hgrid(a1, "x"); hgrid(a2, "x")
    a1.set_title("(a) Net Sharpe ratio over the 200 base experiments", fontsize=7.4)
    a2.set_title("(b) Turnover", fontsize=7.4)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig_rules_results")


# ------------------------------------------------------------------ halves
def fig_halves():
    h = T("halves")
    h = h[h.by == "kind"].set_index(["group", "freq"])
    show = [("price", "Price", GRAY, "-"), ("all_s", "Price + text, 63 d", NAVY, "-"),
            ("narr_s", "Narrative, 63 d", TEAL, "-"), ("narr", "Narrative, 5 d", ORANGE, (0, (3, 1.5)))]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2), sharey=True)
    for ax, f in zip(axes, FREQS):
        for k, name, c, ls in show:
            a, b = h.loc[(k, f), "ic_2022_23"], h.loc[(k, f), "ic_2024"]
            ax.plot([0, 1], [100 * a, 100 * b], color=c, lw=1.3, ls=ls, marker="o", ms=3.5, label=name)
        ax.axhline(0, color=GRAY, lw=0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["2022–23", "2024"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(FREQ_NAME[f], fontsize=7.6)
        hgrid(ax)
    axes[0].set_ylabel("average rank IC ×100")
    axes[2].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.2)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_halves")


# ------------------------------------------------------------------ the search
def fig_search():
    m = T("multiple_testing")
    rules = [r for r, _ in RULES if r in set(m.strategy)]
    names = dict(RULES)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.75), sharey=True)
    y = np.arange(len(rules))
    for ax, f in zip(axes, FREQS):
        for pool, off, c in (("main", -0.17, BLUE), ("all", 0.17, TEAL)):
            g = m[(m.freq == f) & (m.pool == pool)].set_index("strategy")
            best = np.array([g.loc[r, "best_sharpe_net"] for r in rules])
            exp = np.array([g.loc[r, "expected_best_sharpe_if_no_skill"] for r in rules])
            spa = np.array([g.loc[r, "spa_p"] for r in rules])
            for yy, e, b_ in zip(y + off, exp, best):
                ax.plot([e, b_], [yy, yy], color=c if b_ > e else LIGHT, lw=0.9, zorder=1)
            ax.scatter(exp, y + off, marker="|", s=60, color=GRAY, zorder=2)
            ax.scatter(best, y + off, s=[26 if p < 0.05 else 12 for p in spa], color=c,
                       edgecolors=[INK if p < 0.05 else c for p in spa], linewidths=0.8, zorder=3,
                       label={"main": "main design (375)", "all": "everything (617)"}[pool])
        ax.set_title(FREQ_NAME[f], fontsize=7.6)
        ax.set_xlabel("best net Sharpe ratio", fontsize=6.2)
        hgrid(ax, "x")
    axes[0].set_yticks(y); axes[0].set_yticklabels([names[r] for r in rules], fontsize=6.4); axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=BLUE, label="best, main design (375)"),
               Line2D([], [], ls="", marker="o", ms=4, color=TEAL, label="best, every experiment (617)"),
               Line2D([], [], ls="", marker="|", ms=7, color=GRAY, label="best expected without skill"),
               Line2D([], [], ls="", marker="o", ms=5.5, color="white", markeredgecolor=INK, label="SPA p < 0.05")]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=6.0,
                   columnspacing=1.0, handletextpad=0.3)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_search")


# ------------------------------------------------------------------ seeds
def fig_seed():
    tn = pd.read_csv(DATA / "results/data_audit/tuning_noise.csv")
    label = {"price": "price", "all_lda": "price + LDA", "all_finbert": "price + FinBERT", "text_finbert": "FinBERT text"}
    g = list(tn.groupby(["freq", "features", "model"]))
    order = sorted(g, key=lambda kv: (FREQS.index(kv[0][0]), kv[0][1], kv[0][2]))
    fig, ax = plt.subplots(figsize=(6.9, 2.1))
    for i, ((f, feat, mod), d) in enumerate(order):
        v = d.test_sharpe_net.to_numpy()
        ax.plot([v.min(), v.max()], [i, i], color=LIGHT, lw=3.2, solid_capstyle="round")
        ax.scatter(v, np.full(v.size, i), s=9, color={"M": NAVY, "2W": TEAL, "W": ORANGE}[f], zorder=3)
        ax.scatter(d[d.seed == 0].test_sharpe_net, [i], marker="|", s=80, color=INK, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{FREQ_NAME[f]}, {label.get(feat, feat)}, {mod.upper() if mod == 'ebm' else ('XGBoost' if mod == 'xgb' else mod)}"
                        for (f, feat, mod), _ in order], fontsize=6.3)
    ax.invert_yaxis(); ax.axvline(0, color=GRAY, lw=0.5)
    ax.set_xlabel("test net Sharpe ratio across ten seeds (| = seed of the main run)", fontsize=6.4)
    hgrid(ax, "x")
    save(fig, "fig_seed")


# ------------------------------------------------------------------ improvements
if __name__ == "__main__":
    setup()
    todo = sys.argv[1:] or [n for n in dir() if n.startswith("fig_")]
    for name in todo:
        globals()[name]()
        print("made", name)
