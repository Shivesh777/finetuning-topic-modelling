"""Figures of the data and methods sections."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from style import (BLUE, DATA, GRAY, INK, LIGHT, NAVY, ORANGE, RULE, SAND, SKY, SOFTBLUE, SOFTORANGE,
                   SOFTTEAL, TEAL, FREQ_NAME, hgrid, save, setup)

DECK = json.loads((DATA / "deck_data.json").read_text())


# ------------------------------------------------------------------ the sample
def fig_sample():
    news = pd.read_csv(DATA / "results/data_audit/news_coverage.csv")
    cov = pd.read_csv(DATA / "results/data_audit/firm_coverage.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(6.9, 2.15))
    years = news["year"].to_numpy()
    split = {**{y: ("training", SKY) for y in range(2012, 2020)}, 2020: ("validation", SAND), 2021: ("validation", SAND),
             **{y: ("test", NAVY) for y in range(2022, 2025)}}
    for y, a in zip(years, news["articles"]):
        lab, c = split[y]
        ax.bar(y, a / 1e3, width=0.7, color=c, edgecolor="none", label=lab)
        ax.text(y, a / 1e3 + 4, f"{a / 1e3:.0f}k", ha="center", va="bottom", fontsize=6, color=GRAY)
    h, l = ax.get_legend_handles_labels()
    keep = {lab: hh for hh, lab in zip(h, l)}
    ax.set_ylabel("Articles (thousands)")
    ax.set_ylim(0, 255)
    ax.set_xticks(range(2012, 2025))
    ax.set_xticklabels([str(y) if y % 2 == 0 else "" for y in range(2012, 2025)])
    hgrid(ax)
    axr = ax.twinx()
    line, = axr.plot(cov["date"].dt.year + (cov["date"].dt.month - 1) / 12 + 0.5 / 12,
                     100 * cov["news_in_252_days"], color=TEAL, lw=1.3)
    axr.set_ylim(0, 100)
    axr.set_ylabel("Stocks with firm news (%)", color=TEAL)
    axr.tick_params(axis="y", colors=TEAL)
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(TEAL)
    axr.spines["top"].set_visible(False)
    ax.legend([keep["training"], keep["validation"], keep["test"], line],
              ["training", "validation", "test", "stocks with an article in the past year"],
              loc="upper left", ncol=4, fontsize=6.4, handlelength=1.2, columnspacing=1.0)
    ax.set_xlim(2011.4, 2024.6)
    save(fig, "fig_sample")


# ------------------------------------------------------------------ what the news was about
def fig_finetune():
    log = json.loads((DATA / "results/encoders/nolbert_ft/train_log.json").read_text())
    sim = json.loads((DATA / "nolbert_similarity.json").read_text())
    h = pd.DataFrame(log["history"])
    fig = plt.figure(figsize=(6.9, 2.05))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.05, 0.12, 1.15, 1.05], left=0.055, right=0.995, top=0.87,
                          bottom=0.25, wspace=0.42)
    for k, (key, title) in enumerate([("base", "(a) Base NoLBERT"), ("ft", "(b) Adapted")]):
        ax = fig.add_subplot(gs[0, k])
        C = np.array(sim[key]["batch0_cos"])
        im = ax.imshow(C, cmap="Blues", vmin=-0.1, vmax=0.9)
        hit = C.argmax(1)
        wrong = hit != np.arange(len(C))
        ax.scatter(hit[wrong], np.flatnonzero(wrong), s=5, color=ORANGE, linewidths=0, zorder=3)
        ax.set_xticks([0, 31]); ax.set_xticklabels(["1", "32"])
        ax.set_yticks([0, 31]); ax.set_yticklabels(["1", "32"] if k == 0 else ["", ""])
        ax.tick_params(length=0, labelsize=6, pad=1.5)
        ax.set_xlabel("lead $j$", fontsize=6.6, labelpad=1)
        if k == 0:
            ax.set_ylabel("headline $i$", fontsize=6.6, labelpad=1)
        ax.set_title(f"{title}\n{int((~wrong).sum())} of 32 matched", fontsize=7.0)
        for s_ in ax.spines.values():
            s_.set_visible(True); s_.set_color(RULE)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.4, 0.8])
    cb.ax.set_yticklabels(["0", "0.4", "0.8"])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.8, length=2, pad=1)
    cb.ax.set_title("cos", fontsize=6.0, pad=3)
    ax = fig.add_subplot(gs[0, 4])
    ax.plot(h["step"], h["loss"], color=SKY, lw=0.6)
    ax.plot(h["step"], h["loss"].rolling(9, center=True, min_periods=1).mean(), color=NAVY, lw=1.3)
    ax.axhline(np.log(32), color=GRAY, ls=(0, (3, 2)), lw=0.7)
    ax.text(6200, np.log(32) + 0.2, "uniform guess", ha="right", fontsize=5.8, color=GRAY)
    ax.set_xticks([0, 3000, 6000]); ax.set_xticklabels(["0", "3k", "6k"])
    ax.set_xlabel("optimiser step", fontsize=6.6, labelpad=1)
    ax.set_ylabel("InfoNCE loss", fontsize=6.6, labelpad=1)
    ax.set_title("(c) Training loss", fontsize=7.0)
    ax.set_ylim(0, 6.2)
    hgrid(ax)
    ax = fig.add_subplot(gs[0, 5])
    groups = [("held out,\n2015–20", log["val_top1_before"], log["val_top1_after"]),
              ("test period,\n2022–24", sim["base"]["top1"], sim["ft"]["top1"])]
    for g, (lab, b0, b1) in enumerate(groups):
        for j, (v, c) in enumerate([(b0, LIGHT), (b1, TEAL)]):
            x = g + (j - 0.5) * 0.36
            ax.bar(x, 100 * v, width=0.34, color=c, label=["base", "adapted"][j] if g == 0 else None)
            ax.text(x, 100 * v + 2.5, f"{100 * v:.0f}", ha="center", fontsize=6.2, color=INK)
    ax.axhline(100 / 32, color=GRAY, ls=(0, (3, 2)), lw=0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups], fontsize=6.0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("matched (%)", fontsize=6.6, labelpad=1)
    ax.set_title("(d) Top-1 matching", fontsize=7.0)
    ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1, columnspacing=0.8)
    hgrid(ax)
    save(fig, "fig_finetune")


# ------------------------------------------------------------------ topic quality
def fig_topic_quality():
    t = pd.read_csv(DATA / "results/tables/topic_models.csv").set_index("name")
    order = ["lda", "lda_all", "minilm", "finbert", "nolbert", "nolbert20"]
    label = {"lda": "LDA", "lda_all": "LDA, all", "minilm": "MiniLM", "finbert": "FinBERT",
             "nolbert": "NoLBERT", "nolbert20": "NoLBERT-20"}
    t = t.loc[order]
    metrics = [("Coherence (NPMI)", ["npmi_training", "npmi_later"], ["to 2019", "after 2019"]),
               ("Stability across seeds", ["stability_share_kept"], ["articles kept"]),
               ("Attention-shock reliability", ["shock_reliability_training_150_per_day", "shock_reliability_later_150_per_day"],
                ["to 2019", "after 2019"]),
               ("Evenness (% of K)", ["evenness"], ["effective topics / K"])]
    t["evenness"] = 100 * t["effective_topics"].astype(float) / t["topics"].astype(float)
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.15), sharey=True)
    y = np.arange(len(order))
    for ax, (title, cols, legend) in zip(axes, metrics):
        w = 0.36 if len(cols) == 2 else 0.6
        for k, (c, lg) in enumerate(zip(cols, legend)):
            v = t[c].astype(float)
            if c == "shock_reliability_training_150_per_day":
                v = v.fillna(t["shock_reliability_training_100_per_day"])
            if c == "shock_reliability_later_150_per_day":
                v = v.fillna(t["shock_reliability_later_100_per_day"])
            off = (k - (len(cols) - 1) / 2) * w
            ax.barh(y + off, v, height=w * 0.92, color=[NAVY, SKY][k] if len(cols) == 2 else TEAL, label=lg)
        ax.set_title(title, fontsize=7.4)
        if c == "evenness":
            for yy, vv in zip(y, t["evenness"]):
                ax.text(vv + 2, yy, f"{vv:.0f}", va="center", fontsize=5.9, color=GRAY)
            ax.set_xlim(0, 110)
        ax.invert_yaxis() if ax is axes[0] else None
        hgrid(ax, "x")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        if len(cols) == 2:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=5.9, handlelength=0.9)
        if c == "stability_share_kept":
            ax.text(0.02, order.index("finbert"), "fixed classes", va="center", fontsize=6, color=GRAY)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label[o] for o in order])
    for lab in axes[0].get_yticklabels():
        if "NoLBERT" in lab.get_text():
            lab.set_color(TEAL); lab.set_fontweight("semibold")
    fig.subplots_adjust(wspace=0.22)
    save(fig, "fig_topic_quality")


# ------------------------------------------------------------------ EBM shapes
def lda_words(k, n=3):
    t = pd.read_csv(DATA / "topics/lda/topics.csv").set_index("topic")
    return ", ".join(str(t.loc[k, "top_terms_by_probability"]).split(", ")[:n])


def term_label(term):
    import re
    m = re.match(r"^(narr|firm)_lda(?:_news)?_t(\d+)$", term)
    if m:
        kind = "Exposure" if m.group(1) == "narr" else "Firm news share"
        return f"{kind}, LDA topic {int(m.group(2))}\n“{lda_words(int(m.group(2)))}”"
    return {"rev_21": "return, last 21 days", "mom_12_1": "momentum 12-1", "ivol": "idiosyncratic vol.",
            "beta": "market beta", "size": "market value", "vol_21": "volatility, 21 days",
            "rev_5": "return, last 5 days", "max_21": "max daily return"}.get(term, term)


def fig_ebm():
    fig = plt.figure(figsize=(6.9, 3.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.62, wspace=0.42)
    sh = pd.read_csv(DATA / "results/explain/W__all_lda/shapes.csv")
    terms = [t for t in sh["term"].unique() if "&" not in t][:4]
    for k, term in enumerate(terms):
        ax = fig.add_subplot(gs[0, k])
        g = sh[sh["term"] == term]
        xs = np.r_[g["from"].to_numpy(), g["to"].to_numpy()[-1]]
        ys = 1e4 * np.r_[g["score"].to_numpy(), g["score"].to_numpy()[-1]]
        ax.step(xs, ys, where="post", color=TEAL if term.startswith(("narr", "firm")) else NAVY, lw=1.2)
        ax.axhline(0, color=GRAY, lw=0.5, ls=(0, (3, 2)))
        ax.set_title(term_label(term), fontsize=6.3, fontweight="normal")
        ax.set_xlim(-1, 1); ax.tick_params(labelsize=6)
        ax.set_xlabel("feature rank", fontsize=6.2)
        if k == 0:
            ax.set_ylabel("basis points / week", fontsize=6.5)
    fig.text(0.07, 0.965, "(a) Weekly EBM on price and LDA text: four main-effect shape functions",
             fontsize=7.6, fontweight="semibold")
    # (b) importance shares by block
    ax = fig.add_subplot(gs[1, :2])
    blocks = ["price", "narr", "firm", "mixed pair"]
    names = {"price": "price", "narr": "narrative", "firm": "firm news", "mixed pair": "pairwise terms"}
    cols = {"price": NAVY, "narr": TEAL, "firm": SKY, "mixed pair": ORANGE}
    for r, f in enumerate(("M", "2W", "W")):
        b = pd.read_csv(DATA / f"results/explain/{f}__all_lda/blocks.csv").set_index("block")["share"]
        left = 0
        for bl in blocks:
            v = float(b.get(bl, 0))
            ax.barh(r, v, left=left, color=cols[bl], height=0.62, label=names[bl] if r == 0 else None)
            if v > 0.08:
                ax.text(left + v / 2, r, f"{100 * v:.0f}%", ha="center", va="center", fontsize=6, color="white")
            left += v
    ax.set_yticks(range(3)); ax.set_yticklabels([FREQ_NAME[f] for f in ("M", "2W", "W")])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=5.9, handlelength=0.9)
    ax.set_title("(b) Share of importance by block", fontsize=7.6)
    # (c) ablation
    ax = fig.add_subplot(gs[1, 2:])
    full, nopair = [], []
    for f in ("M", "2W", "W"):
        a = pd.read_csv(DATA / f"results/explain/{f}__all_lda/ablation.csv").set_index("deleted")
        full.append(a.loc["nothing", "sharpe_net"]); nopair.append(a.loc["block: mixed pair", "sharpe_net"])
    y = np.arange(3)
    for i in range(3):
        ax.annotate("", xy=(nopair[i], i), xytext=(full[i], i), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.8))
    ax.scatter(full, y, color=ORANGE, s=22, zorder=3, label="full model")
    ax.scatter(nopair, y, color=TEAL, s=22, zorder=3, label="pairwise terms deleted")
    ax.axvline(0, color=GRAY, lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    for i, f in enumerate(("M", "2W", "W")):
        ax.text(max(full[i], nopair[i]) + 0.04, i, FREQ_NAME[f], va="center", fontsize=6.3, color=GRAY)
    ax.set_xlim(min(full) - 0.1, max(nopair) + 0.35)
    ax.set_xlabel("test net Sharpe ratio, decile long-short", fontsize=6.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=5.9)
    ax.set_title("(c) Deleting the pairs, nothing re-estimated", fontsize=7.6)
    hgrid(ax, "x")
    save(fig, "fig_ebm_shapes")


if __name__ == "__main__":
    setup()
    for f in (fig_sample, fig_finetune, fig_topic_quality, fig_ebm):
        f()
        print("made", f.__name__)
