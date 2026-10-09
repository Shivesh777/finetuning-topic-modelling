#!/usr/bin/env python3
"""Build deck/js/data.js from the team repo at the pinned commit.

Every number the deck draws comes from this file's output. Each block names
the file it was read from, so a slide's source line can cite it.

    python tools/build_data.py            # writes deck/js/data.js
    python tools/build_data.py --check    # also prints the checks against FINDINGS.md

Reads, read-only:
  ..          the repo this folder sits in; the files it reads must match commit b97a27a
  ../data     predictions and panels outside git (backup slide B3 only); FTM_DATA points elsewhere
Writes deck/js/data.js (committed) and deck/js/portfolio.js (stock-level ranks, gitignored).
Parquet files are read with duckdb, because pyarrow fails on these files here.
Stock-level data (CRSP) never reaches the deck: only ranks of anonymous dots, and
portfolio-level returns (results/series) for the best performers of slide 5.5.
"""
import argparse
import json
import re
import math
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
APP = HERE.parent                                               # presentation/
REPO = Path(os.environ.get("FTM_REPO", APP.parent))             # the repo this folder sits in
LOOSE = Path(os.environ.get("FTM_DATA", REPO / "data"))         # predictions and panels, outside git
PIN = "b97a27a"
REPO_URL = "github.com/Shivesh777/finetuning-topic-modelling"
OUT = APP / "deck" / "js" / "data.js"
READS = ["FINDINGS.md", "results/tables", "results/explain", "results/encoders", "results/series", "data/topics"]


def same_as_pin():
    """True when every repo file the deck reads is as it was at the pinned commit."""
    try:
        return subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", PIN, "--"] + READS).returncode == 0
    except Exception:
        return None


def rnd(x, k=4):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, k)


def parquet(path):
    return duckdb.sql(f"select * from read_parquet('{path}')").df()


# ---------------------------------------------------------------- live news map (title, closing)
def news_map():
    """20 NoLBERT topics: monthly attention shares (slide 1.1) and the bead grid of the title
    and closing slides. A topic gets a bead in a month when its share of that month's news is
    above its median share of the previous 12 months, so each row uses only its own month and
    earlier ones. The grid starts in Sep 2020, the first month after every fitted part was
    frozen: topics and vocabulary were fitted on text up to 2019, and the encoder was adapted
    on articles dated before 1 Sep 2020."""
    src = "data/topics/nolbert20/daily_sums.parquet"
    df = parquet(REPO / src)
    df["date"] = pd.to_datetime(df["date"])
    tc = sorted(c for c in df.columns if c.startswith("t"))      # t00 ... t19, the topics.csv order
    m = df.set_index("date")[tc].resample("MS").sum()
    share = m.div(m.sum(axis=1), axis=0)
    hot = (share > share.shift(1).rolling(12).median()).loc["2020-09-01":]
    topics = pd.read_csv(REPO / "data/topics/nolbert20/topics.csv")
    terms = [", ".join(str(t).split(", ")[:3]) for t in topics["top_terms"]]
    monthly = share.loc["2013-01-01":]
    return {
        "source": f"{src}; data/topics/nolbert20/topics.csv",
        "terms": terms,
        "months": [d.strftime("%Y-%m") for d in monthly.index],
        "shares": [[rnd(v, 4) for v in monthly[c].values] for c in tc],
        "beads": {"rule": "share of the month's news above the topic's median share of the previous 12 months",
                  "months": [d.strftime("%Y-%m") for d in hot.index],
                  "rows": [[int(v) for v in r] for r in hot.values]},
    }


# ---------------------------------------------------------------- section 3
def training():
    src = "results/encoders/nolbert_ft/train_log.json"
    d = json.loads((REPO / src).read_text())
    cfg = d["config"]
    return {
        "source": src, "steps": d["steps"], "pairs_seen": d["pairs_seen"], "batch": cfg["batch"],
        "train_end": cfg["train_end"], "temperature": cfg["temperature"], "max_len": cfg["max_len"],
        "budget_min": cfg["budget_min"], "before": rnd(d["val_top1_before"], 4), "after": rnd(d["val_top1_after"], 4),
        "log": [[h["step"], rnd(h["loss"], 3), rnd(h["acc"], 3)] for h in d["history"]],
    }


def stability():
    out = []
    for m, name in [("nolbert", "NoLBERT"), ("minilm", "MiniLM"), ("lda", "LDA")]:
        q = json.loads((REPO / f"data/topics/{m}/quality.json").read_text())
        out.append({"model": name, "ari": rnd(q["stability_adjusted_rand"], 2),
                    "kept": rnd(q["stability_share_kept"], 2), "largest": rnd(q["largest_topic_share"], 3)})
    return {"source": "data/topics/{nolbert,minilm,lda}/quality.json", "rows": out}


def pooling():
    """Backup B11: the pooling probe of the adapted encoder (NPMI of 60 clusters on our vocabulary;
    AMI = agreement of the partition with MiniLM's)."""
    src = "results/encoders/nolbert_ft/nolbert_probe.json"
    p = json.loads((REPO / src).read_text())
    names = [("nolbert_last_raw", "Last layer, mean, no whitening"),
             ("nolbert_top4_raw", "Top four layers, mean"),
             ("nolbert_top4_abtt", "Top four layers, mean, all-but-the-top whitening"),
             ("nolbert_top4_sif_abtt", "Top four layers, SIF weighting, whitening"),
             ("minilm_last_raw", "MiniLM, last layer (reference)"),
             ("minilm_last_abtt", "MiniLM, last layer, whitening")]
    rows = [{"key": k, "label": lab, "npmi": rnd(p["results"][k]["npmi_mean"], 3), "ami": rnd(p["results"][k]["ami_vs_minilm"], 2)}
            for k, lab in names]
    c = p["config"]
    return {"source": src, "docs": int(c["n_docs"]), "chunks": int(c["n_chunks"]), "k": int(c["k"]), "rows": rows}


def arch():
    """Slide 2.2: the encoder's shape, from the adapted checkpoint's config (unchanged by the adaptation)."""
    src = "results/encoders/nolbert_ft/config.json"
    c = json.loads((REPO / src).read_text())
    return {"layers": int(c["num_hidden_layers"]), "heads": int(c["num_attention_heads"]), "hidden": int(c["hidden_size"]),
            "vocab_size": int(c["vocab_size"]), "arch_source": src}


# ---------------------------------------------------------------- shared by sections 4 and 5
FREQ = [("M", "M__all_nolbert", "monthly", "NoLBERT"),
        ("2W", "2W__all_finbert", "biweekly", "FinBERT"),
        ("W", "W__all_lda", "weekly", "LDA")]
FREQS = ["M", "2W", "W"]
TOPIC_MODELS = ["lda", "lda_all", "finbert", "minilm", "nolbert", "nolbert20"]
TM_NAME = {"lda": "LDA", "lda_all": "LDA, every article", "minilm": "MiniLM", "finbert": "FinBERT",
           "nolbert": "NoLBERT", "nolbert20": "NoLBERT-20"}
MAIN_SETS = {"price"} | {f"{k}_{t}" for k in ["narr", "firm", "text", "all"] for t in TOPIC_MODELS}
MAIN26 = MAIN_SETS | {"narr_lda_placebo"}           # the 26 feature sets of the main run
BASE = ["ols", "ridge", "lasso", "ipca", "sipca", "xgb", "rf", "ebm"]
MODEL_NAME = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "ipca": "IPCA", "sipca": "Sparse IPCA",
              "xgb": "XGBoost", "xgb_rk": "XGBoost, ranked target", "rf": "Random forest", "ebm": "EBM",
              "ens": "Ensemble of the eight"}
_TABLES = {}


def table(name):
    """results/tables/<name>.csv, read once."""
    if name not in _TABLES:
        _TABLES[name] = pd.read_csv(REPO / "results/tables" / f"{name}.csv")
    return _TABLES[name]


def set_name(fs):
    """A feature set's name on the slides."""
    fixed = {"price": "Price", "px_all": "Price + every news block", "px_news": "Price + tone and embeddings",
             "px_tone": "Price + tone", "narr_lda_placebo": "Placebo exposures"}
    if fs in fixed:
        return fixed[fs]
    kind, tm, w = re.fullmatch(r"(narr|firm|text|all)_([a-z0-9_]+?)(?:_s(\d+))?", fs).groups()
    name = {"narr": "{} exposures", "firm": "{} firm news", "text": "{} text", "all": "Price + {} text"}[kind].format(TM_NAME[tm])
    return name + (f", {w}-day shocks" if w else "")


def has_position(strategies):
    """(freq, features, model) of the experiments whose decile long-short trades in the test period:
    a flat forecast holds nothing (section 5 of FINDINGS.md leaves those out)."""
    q = strategies[(strategies.split == "test") & (strategies.strategy == "ls_ew") & (strategies.turnover > 0)]
    return set(zip(q.freq, q.features, q.model))


# ---------------------------------------------------------------- section 4: how it works
def pipeline():
    """Slide 4.1: the counts along the pipeline."""
    tm = table("topic_models").set_index("name")
    f, tun = table("forecast"), table("tuning")
    return {"source": "results/tables/{topic_models,forecast,tuning}.csv; FINDINGS.md:3-11, 499-501, 909",
            "articles": FACTS["periods"]["articles"], "labelled": int(tm.loc["lda", "articles_labelled"]),
            "fit": int(tm.loc["lda", "articles_fit"]), "firm": int(tm.loc["lda", "firm_articles"]),
            "fit_until": str(tm.loc["lda", "fit_until"]),
            "periods": {q: pipeline_periods(q, i) for i, q in enumerate(FREQS)},
            "features_all_lda": int(tun[(tun.features == "all_lda") & (tun.model == "ebm")].n_features.iloc[0]),
            "experiments": int(len(f[f.split == "test"])), "rules": 8, "models": 15}


# three example topics per reader: an oil topic (slide 4.3 follows one), a distinctive one, a templated one
READERS = [("lda", "LDA", [9, 44, 21]), ("minilm", "MiniLM", [43, 37, 0]),
           ("nolbert", "NoLBERT", [28, 0, 1]), ("finbert", "FinBERT", [6, 1, 4])]


def readers():
    """Slide 4.2: the four topic readers, their size and three of their topics each."""
    out = []
    for m, name, picks in READERS:
        t = pd.read_csv(REPO / f"data/topics/{m}/topics.csv")
        q = json.loads((REPO / f"data/topics/{m}/quality.json").read_text())
        rows = t.reset_index(drop=True) if m == "finbert" else t.set_index("topic")   # FinBERT: rows in class order
        ex = []
        for k in picks:
            r = rows.iloc[k] if m == "finbert" else rows.loc[k]
            ex.append({"k": int(k), "label": str(r["topic"]) if m == "finbert" else None,
                       "words": ", ".join(str(r["top_terms"]).split(", ")[:3])})
        out.append({"model": m, "name": name, "topics": int(q["topics"]), "effective": rnd(q["effective_topics"], 1),
                    "largest": rnd(q["largest_topic_share"], 3), "examples": ex})
    q20 = json.loads((REPO / "data/topics/nolbert20/quality.json").read_text())
    return {"source": "data/topics/<model>/{topics.csv,quality.json} (first three top words, one rule for every model)",
            "rows": out, "nolbert20": {"topics": int(q20["topics"]), "effective": rnd(q20["effective_topics"], 1)}}


# NYSE holidays from Jul 2021 to Apr 2022 (the days around slide 4.3's window that are not trading days)
NYSE_CLOSED = ["2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24", "2022-01-17", "2022-02-21", "2022-04-15"]


def attention(model="nolbert20", topic=10, start="2021-12-01", end="2022-04-29"):
    """Slide 4.3: one topic's daily attention and its shocks, as ftm/features/narrative.py builds them:
    calendar-day sums of article weights moved to the next trading day, days with fewer than 20
    articles missing, shock = attention minus its mean over the previous w trading days."""
    src = f"data/topics/{model}/daily_sums.parquet"
    df = parquet(REPO / src)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").loc["2021-07-01":end]
    days = pd.bdate_range("2021-07-01", end)
    days = days[~days.isin(pd.to_datetime(NYSE_CLOSED))]
    pos = np.searchsorted(days.values, df.index.values, side="left")
    ok = pos < len(days)
    tot = df[ok].groupby(pos[ok]).sum().reindex(range(len(days)), fill_value=0.0)
    tc = [c for c in df.columns if c.startswith("t")]
    theta = tot[tc].div(tot[tc].sum(axis=1).where(tot["n_articles"] >= 20), axis=0)
    theta.index = days
    s = theta[f"t{topic:02d}"]

    def trailing(w):
        return s.rolling(w, min_periods=min(w, max(2, w - 2))).mean().shift(1)
    keep = s.index >= pd.Timestamp(start)
    words = pd.read_csv(REPO / f"data/topics/{model}/topics.csv").set_index("topic").loc[topic, "top_terms"]
    peak = s[keep].idxmax()
    return {"source": f"{src}; topics fitted on text to 2019", "model": TM_NAME[model], "topic": topic,
            "words": ", ".join(str(words).split(", ")[:3]),
            "days": [d.strftime("%Y-%m-%d") for d in s.index[keep]],
            "theta": [rnd(v, 5) for v in s[keep]], "trail5": [rnd(v, 5) for v in trailing(5)[keep]],
            "shock": {str(w): [rnd(v, 5) for v in (s - trailing(w))[keep]] for w in (1, 5, 63)},
            "mean_2021h2": rnd(s.loc["2021-07-01":"2021-12-31"].mean(), 4),
            "peak": {"day": peak.strftime("%Y-%m-%d"), "theta": rnd(s[peak], 4)}}


EBM_TERMS = ["rev_21", "ivol", "narr_finbert_t06", "size"]


def ebm_story():
    """Slide 4.6: four shape functions of the biweekly EBM on price and FinBERT text (the EBM with the
    best validation IC there, opened by scripts/09_explain.py), and where the EBM stands among the models."""
    shapes = pd.read_csv(REPO / "results/explain/2W__all_finbert/shapes.csv")
    labels = pd.read_csv(REPO / "data/topics/finbert/topics.csv")["topic"]
    price = {"rev_21": "return over the last 21 days", "ivol": "idiosyncratic volatility", "size": "market value"}
    shp = []
    for term in EBM_TERMS:
        g = shapes[shapes.term == term]
        m = re.match(r"^narr_finbert_t(\d+)$", term)
        name = f"exposure to “{labels.iloc[int(m.group(1))]}”" if m else price[term]
        shp.append({"term": term, "name": name, "bins": [[rnd(r["from"], 4), rnd(r["to"], 4), rnd(r["score"] * 1e4, 4)] for _, r in g.iterrows()]})
    f = table("forecast")
    g = f[f.features.isin(MAIN26) & f.model.isin(BASE)]
    val = {q: g[(g.freq == q) & (g.split == "validation")].groupby("model").ic_mean.mean() for q in FREQS}
    tun = table("tuning")
    e = tun[(tun.model == "ebm") & tun.features.isin(MAIN26)]
    return {"source": "results/explain/2W__all_finbert/shapes.csv; results/tables/{forecast,tuning}.csv",
            "shapes_bp": shp, "model": "biweekly, price and FinBERT text", "rounds": FACTS["ebm"]["rounds_2w"],
            "val_ic": [rnd(val[q]["ebm"], 6) for q in FREQS],
            "val_best": [bool(val[q].idxmax() == "ebm") for q in FREQS],
            "fits": int(len(tun[tun.model == "ebm"])), "main_fits": int(len(e)),
            "pairs_chosen": int((e.setting == "interactions=10").sum())}


LABEL = {"price": "Price (main effects)", "narr": "Narrative exposures", "firm": "Firm news", "mixed pair": "Pairwise terms"}


ORDER = ["price", "narr", "firm", "mixed pair"]


def glassbox():
    models = []
    for code, folder, name, tm in FREQ:
        base = REPO / "results/explain" / folder
        blocks = pd.read_csv(base / "blocks.csv").set_index("block")
        abl = pd.read_csv(base / "ablation.csv")
        full = abl.loc[abl["deleted"] == "nothing"].iloc[0]
        rows = []
        for b in ORDER:
            a = abl.loc[abl["deleted"] == f"block: {b}"].iloc[0]
            rows.append({"block": b, "label": LABEL[b], "terms": int(blocks.loc[b, "terms"]),
                         "share": rnd(blocks.loc[b, "share"], 4),
                         "d_ic": rnd(full["ic_mean"] - a["ic_mean"], 5),
                         "sharpe_without": rnd(a["sharpe_net"], 3), "r2_without": rnd(a["r2_oos_pct"], 4)})
        imp = pd.read_csv(base / "importance.csv")
        models.append({"freq": code, "name": name, "topics": tm, "ic": rnd(full["ic_mean"], 5),
                       "sharpe": rnd(full["sharpe_net"], 3), "r2": rnd(full["r2_oos_pct"], 4),
                       "main_effects": int((imp["block"] != "mixed pair").sum()), "blocks": rows})
    shapes = pd.read_csv(REPO / "results/explain/W__all_lda/shapes.csv")
    lda = pd.read_csv(REPO / "data/topics/lda/topics.csv").set_index("topic")["top_terms"]   # the deck's one top-word rule

    def words(term):
        m = re.match(r"^(?:narr_lda_t|firm_lda_news_t)(\d+)$", term)
        return ", ".join(str(lda.loc[int(m.group(1))]).split(", ")[:3]) if m else None
    shp = [{"term": t, "words": words(t), "bins": [[rnd(r["from"], 3), rnd(r["to"], 3), rnd(r["score"] * 1e4, 4)] for _, r in g.iterrows()]}
           for t, g in shapes.groupby("term", sort=False)]
    return {"source": "results/explain/<freq>__all_<topics>/{blocks,ablation,importance,shapes}.csv",
            "models": models, "shapes_weekly_bp": shp}



def turnover():
    s = table("strategies")
    q = s[(s["freq"] == "W") & (s["split"] == "test") & (s["features"] == "all_lda") & (s["model"] == "ebm")]
    t = q.set_index("strategy")["turnover"] if "turnover" in q.columns else None
    return {"source": "results/tables/strategies.csv (W, test, all_lda, ebm)",
            "weeks": int(q.set_index("strategy")["periods"]["ls_ew"]),
            "ls_ew": rnd(t["ls_ew"], 3) if t is not None else None,
            "ls_buf": rnd(t["ls_buf"], 3) if t is not None else None}


# ---------------------------------------------------------------- section 5: results
def frequency():
    """Slide 5.1: price features alone, and the median decile long-short with and without the buffer,
    over the base experiments of the main run that take a position (FINDINGS.md sections 4.1 and 6)."""
    f, s = table("forecast"), table("strategies")
    pos = has_position(s)
    out = []
    for i, q in enumerate(FREQS):
        p = f[(f.freq == q) & (f.features == "price") & f.model.isin(BASE)]
        row = {"freq": q, "periods": pipeline_periods(q, i),
               "price_ic": {sp: rnd(p[p.split == sp].ic_mean.mean(), 6) for sp in ["validation", "test"]},
               "price_ic_models": [rnd(v, 6) for v in p[p.split == "test"].sort_values("ic_mean").ic_mean]}
        for rule in ["ls_ew", "ls_buf"]:
            g = s[(s.freq == q) & (s.split == "test") & (s.strategy == rule) & s.features.isin(MAIN26) & s.model.isin(BASE)]
            g = g[[(q, a, b) in pos for a, b in zip(g.features, g.model)]]
            row[rule] = {"n": int(len(g)), "gross": rnd(g.sharpe_gross.median(), 6), "net": rnd(g.sharpe_net.median(), 6),
                         "cost": rnd(g.cost_annual.median(), 6), "turnover": rnd(g.turnover.median(), 4)}
        out.append(row)
    return {"source": "results/tables/{forecast,strategies}.csv (main run, 8 base models, test period)", "rows": out}


def pipeline_periods(q, i):
    f = table("forecast")
    return [FACTS["periods"]["train"][i]] + [int(f[(f.freq == q) & (f.split == sp)].periods.max()) for sp in ["validation", "test"]]


def topic_compare():
    """Slide 5.2: text-only experiments by topic model (FINDINGS.md section 4.3), then the tuned
    shock windows paired with the 5-day ones (section 10.3)."""
    f = table("forecast")
    g = f[f.model.isin(BASE)]
    rows = []
    for t in TOPIC_MODELS:
        h = g[g.features.isin([f"narr_{t}", f"firm_{t}", f"text_{t}"])]
        rows.append({"model": t, "name": TM_NAME[t],
                     "test": [rnd(h[(h.freq == q) & (h.split == "test")].ic_mean.mean(), 6) for q in FREQS],
                     "validation": [rnd(h[(h.freq == q) & (h.split == "validation")].ic_mean.mean(), 6) for q in FREQS]})
    shock = []
    imp = table("improvements")
    for q, w in [("M", 63), ("W", 1)]:
        per = []
        for t in TOPIC_MODELS:
            sets = [f"{k}_{t}" for k in ["narr", "text", "all"]]
            a = g[(g.freq == q) & (g.split == "test") & g.features.isin(sets)].set_index(["features", "model"]).ic_mean
            b = g[(g.freq == q) & (g.split == "test") & g.features.isin([x + f"_s{w}" for x in sets])].copy()
            b["features"] = b.features.str.replace(f"_s{w}", "", regex=False)
            j = pd.concat([a, b.set_index(["features", "model"]).ic_mean], axis=1, keys=["base", "new"]).dropna()
            per.append({"model": t, "name": TM_NAME[t], "pairs": int(len(j)), "base": rnd(j.base.mean(), 6), "new": rnd(j.new.mean(), 6)})
        e = imp[(imp.freq == q) & (imp.split == "test") & (imp.comparison == "shock_window") & (imp.group == "every")].iloc[0]
        shock.append({"freq": q, "window": w, "topic_models": per, "pairs": int(e.pairs),
                      "ic": [rnd(e.ic_base, 6), rnd(e.ic_new, 6)],
                      "sharpe_median": [rnd(e.sharpe_base_median, 6), rnd(e.sharpe_new_median, 6)],
                      "improved_share": rnd(e.sharpe_improved_share, 4),
                      "up": int(sum(p["new"] > p["base"] for p in per))})
    return {"source": "results/tables/{forecast,improvements}.csv (8 base models)", "rows": rows, "shock": shock}


FORECASTERS = ["ols", "ridge", "lasso", "ipca", "sipca", "xgb", "xgb_rk", "rf", "ebm", "ens"]


def forecasters():
    """Slide 5.3: average IC of each model over the 26 feature sets of the main run (FINDINGS.md
    section 7); the lasso over the sets where its test forecast is not flat."""
    f = table("forecast")
    g = f[f.features.isin(MAIN26)]
    rows, other = [], []
    for m in FORECASTERS + ["ols_rk", "ridge_rk", "lasso_rk", "ipca_rk", "sipca_rk"]:
        r = {"model": m, "name": MODEL_NAME.get(m, m), "validation": [], "test": [], "sets": []}
        for q in FREQS:
            h = g[(g.freq == q) & (g.model == m)]
            if m.startswith("lasso"):
                t = h[h.split == "test"]
                keep = set(t[t.flat_periods < t.periods].features)
                h = h[h.features.isin(keep)]
            r["sets"].append(int(h[h.split == "test"].features.nunique()))
            for sp in ["validation", "test"]:
                r[sp].append(rnd(h[h.split == sp].ic_mean.mean(), 6))
        (rows if m in FORECASTERS else other).append(r)
    lo = min(v for r in other for v in r["test"])
    hi = max(v for r in other for v in r["test"])
    return {"source": "results/tables/forecast.csv (26 feature sets of the main run)", "rows": rows,
            "other_ranked": {"models": len(other), "test_lo": lo, "test_hi": hi}}


RULE_ORDER = [("ls_ew", "Decile long-short"), ("ls_buf", "Buffered long-short"), ("ls_vw", "Long-short, value weights"),
              ("rank", "Rank-weighted long-short"), ("mvo", "Mean-variance long-short"), ("long_ew", "Long-only top decile"),
              ("long_vw", "Long-only, value weights"), ("tilt_vw", "Cap-weighted tilt")]


def rules():
    """Slide 5.4: median test net Sharpe of every rule over the same experiments (the base experiments
    of the main run that take a position), and the value-weighted universe."""
    s = table("strategies")
    pos = has_position(s)
    rows = []
    for rule, label in RULE_ORDER:
        r = {"rule": rule, "name": label, "net": [], "gross": [], "turnover": [], "cost": [], "n": []}
        for q in FREQS:
            g = s[(s.freq == q) & (s.split == "test") & (s.strategy == rule) & s.features.isin(MAIN26) & s.model.isin(BASE)]
            g = g[[(q, a, b) in pos for a, b in zip(g.features, g.model)]]
            r["n"].append(int(len(g)))
            r["net"].append(rnd(g.sharpe_net.median(), 6))
            r["gross"].append(rnd(g.sharpe_gross.median(), 6))
            r["turnover"].append(rnd(g.turnover.median(), 4))
            r["cost"].append(rnd(g.cost_annual.median(), 6))
        rows.append(r)
    b = s[(s.features == "benchmark") & (s.split == "test")].set_index(["freq", "strategy"])
    imp = table("improvements")
    buf = [imp[(imp.freq == q) & (imp.split == "test") & (imp.comparison == "buffer") & (imp.group == "every")].iloc[0] for q in FREQS]
    return {"source": "results/tables/{strategies,improvements}.csv (main run, 8 base models, test period)", "rows": rows,
            "index_vw": [rnd(b.loc[(q, "univ_vw"), "sharpe_net"], 6) for q in FREQS],
            "buffer_gain": [rnd(e.return_diff_annual, 6) for e in buf]}


# the selection pools of slide 5.5, in the order the work added them (results/tables/selection.csv)
POOLS = [("any", "Main run", "every feature set,|5-day shocks"),
         ("shock", "Tuned shock window", "63 days monthly,|1 day weekly"),
         ("news", "Tone of the news", "tone and embedding|sets, some with price")]


def best():
    """Slide 5.5: what validation picks in each pool, and its test numbers; the best of each column
    (highest buffered net Sharpe) with its stats and its cumulative return next to the index."""
    sel, s, f = table("selection"), table("strategies"), table("forecast")
    cells = []
    for q in FREQS:
        col = []
        for pool, label, how in POOLS:
            r = sel[(sel.freq == q) & (sel.family == pool)]
            if r.empty:
                col.append(None)
                continue
            r = r.iloc[0]
            st = s[(s.freq == q) & (s.split == "test") & (s.features == r.features) & (s.model == r.model)].set_index("strategy")
            col.append({"pool": pool, "features": r.features, "model": r.model, "set": set_name(r.features),
                        "model_name": MODEL_NAME[r.model], "val_ic": rnd(r.validation_ic_mean, 6), "ic": rnd(r.test_ic_mean, 6),
                        "net": rnd(st.loc["ls_ew", "sharpe_net"], 6), "buf": rnd(st.loc["ls_buf", "sharpe_net"], 6),
                        "ret": rnd(st.loc["ls_buf", "mean_net"], 6), "vol": rnd(st.loc["ls_buf", "vol"], 6),
                        "dd": rnd(st.loc["ls_buf", "max_drawdown"], 6), "turnover": rnd(st.loc["ls_buf", "turnover"], 4)})
        win = max(range(len(col)), key=lambda i: col[i]["buf"] if col[i] else -9)
        cells.append({"freq": q, "cells": col, "best": win})
    curves = []
    for c in cells:
        q, w = c["freq"], c["cells"][c["best"]]
        p = duckdb.sql(f"select date, excess from read_parquet('{REPO / 'results/series' / f'{q}_returns.parquet'}') where split = 'test' "
                       f"and features = '{w['features']}' and model = '{w['model']}' and strategy = 'ls_buf' order by date").df()
        b = duckdb.sql(f"select date, excess from read_parquet('{REPO / 'results/series' / f'{q}_returns.parquet'}') where split = 'test' "
                       f"and strategy = 'univ_vw' order by date").df()
        j = p.merge(b, on="date", suffixes=("", "_index"))
        curves.append({"freq": q, "dates": [d.strftime("%Y-%m-%d") for d in j.date],
                       "pick": [rnd(v, 6) for v in np.cumprod(1 + j.excess) - 1],
                       "index": [rnd(v, 6) for v in np.cumprod(1 + j.excess_index) - 1],
                       "corr": rnd(np.corrcoef(j.excess, j.excess_index)[0, 1], 2)})
    b = s[(s.features == "benchmark") & (s.split == "test")].set_index(["freq", "strategy"])
    px = f[(f.freq == "W") & (f.split == "test") & (f.features == "price") & (f.model == "ebm")].iloc[0]
    pxs = s[(s.freq == "W") & (s.split == "test") & (s.features == "price") & (s.model == "ebm")].set_index("strategy")
    return {"source": "results/tables/{selection,strategies,forecast}.csv; results/series/<freq>_returns.parquet (portfolio returns)",
            "pools": [{"pool": p, "label": l, "how": h} for p, l, h in POOLS], "columns": cells, "curves": curves,
            "index_vw": [{"sharpe": rnd(b.loc[(q, "univ_vw"), "sharpe_net"], 6), "ret": rnd(b.loc[(q, "univ_vw"), "mean_net"], 6),
                          "vol": rnd(b.loc[(q, "univ_vw"), "vol"], 6), "dd": rnd(b.loc[(q, "univ_vw"), "max_drawdown"], 6)} for q in FREQS],
            "weekly_price_ebm": {"ic": rnd(px.ic_mean, 6), "net": rnd(pxs.loc["ls_ew", "sharpe_net"], 6), "buf": rnd(pxs.loc["ls_buf", "sharpe_net"], 6)}}


# ---------------------------------------------------------------- backup: validation against test, the search
def validation_drop():
    f = pd.read_csv(REPO / "results/tables/forecast.csv")
    out = []
    for code, _, name, _ in FREQ:
        g = f[(f.freq == code) & f.features.isin(MAIN_SETS) & f.model.isin(BASE)]
        p = g.pivot_table(index=["features", "model"], columns="split", values="ic_mean").dropna()
        top = set(p.sort_values("validation", ascending=False).head(10).index)
        rho = float(p["validation"].corr(p["test"], method="spearman"))
        fav = p.loc[list(top), "test"].mean()
        out.append({"freq": code, "name": name, "n": int(len(p)), "rho": rnd(rho, 2), "fav": rnd(fav, 3),
                    "pts": [[rnd(v, 4), rnd(t, 4), int(k in top), int(k[1] == "ebm")]
                            for k, (v, t) in zip(p.index, p[["validation", "test"]].values)]})
    return {"source": "results/tables/forecast.csv (8 base models x 25 real main-run feature sets; flat forecasts dropped)",
            "panels": out}


RULES = [("ls_ew", "Long-short"), ("ls_buf", "Buffered"), ("ls_vw", "Long-short VW"), ("rank", "Rank-weighted"),
         ("mvo", "Mean-variance"), ("long_ew", "Long-only"), ("long_vw", "Long-only VW")]


def search():
    mt = pd.read_csv(REPO / "results/tables/multiple_testing.csv")
    rows = []
    for code, _, name, _ in FREQ:
        for rule, label in RULES:
            r = mt[(mt.pool == "main") & (mt.freq == code) & (mt.strategy == rule)].iloc[0]
            rows.append({"freq": code, "rule": label, "best": rnd(r.best_sharpe_net, 2),
                         "luck": rnd(r.expected_best_sharpe_if_no_skill, 2), "p": rnd(r.best_sharpe_p, 3)})
    a = mt[(mt.pool == "all") & (mt.freq == "M") & (mt.strategy == "ls_vw")].iloc[0]
    return {"source": "results/tables/multiple_testing.csv (pool main: 7 rules x 3 frequencies; pool all: M ls_vw)",
            "rows": rows, "short": int(sum(r["best"] < r["luck"] for r in rows)),
            "all_mvw": {"best": rnd(a.best_sharpe_net, 2), "luck": rnd(a.expected_best_sharpe_if_no_skill, 2),
                        "p": rnd(a.best_sharpe_p, 3), "fdr": int(a.significant_10pct_fdr)}}


# ---------------------------------------------------------------- the paths behind 5.2 and 5.3
MODELS15 = BASE + [m + "_rk" for m in ["ols", "ridge", "lasso", "ipca", "sipca", "xgb"]] + ["ens"]
PER_YEAR = {"M": 12, "2W": 26, "W": 52}


def returns(freq):
    return parquet(REPO / f"results/series/{freq}_returns.parquet")


def race():
    """Slide 5.2: every weekly decile long-short of the main pool (25 feature sets x 15 models
    = 375, the pool of multiple_testing.csv), cumulative net return over the test weeks, in %.
    Weeks an experiment did not trade count as zero."""
    r = returns("W")
    r = r[(r.split == "test") & (r.strategy == "ls_ew") & r.features.isin(MAIN_SETS) & r.model.isin(MODELS15)]
    p = r.pivot_table(index="date", columns=["features", "model"], values="net").sort_index()
    cum = p.fillna(0).cumsum() * 100
    s = pd.read_csv(REPO / "results/tables/strategies.csv")
    s = s[(s.freq == "W") & (s.split == "test") & (s.strategy == "ls_ew")].set_index(["features", "model"])["sharpe_net"]
    keys = list(cum.columns)
    sharpe = [rnd(s.get(k), 3) for k in keys]
    best = int(np.argmax([v if v is not None else -9 for v in sharpe]))
    return {"source": "results/series/W_returns.parquet (test, ls_ew, 25 feature sets x 15 models); results/tables/strategies.csv",
            "dates": [str(pd.Timestamp(d).date()) for d in cum.index],
            "paths": [[rnd(v, 1) for v in cum[k].values] for k in keys],
            "sharpe": sharpe, "pick": keys.index(("all_lda", "ebm")), "best": best,
            "median": [rnd(v, 1) for v in cum.median(axis=1).values], "n": len(keys)}


def buffer_paths():
    """Slide 5.3: at each frequency, the average over the 200 base experiments of the net return
    of the plain decile long-short and of the buffered one, cumulated over the test period, in %."""
    out = []
    for code, _, name, _ in FREQ:
        r = returns(code)
        r = r[(r.split == "test") & r.strategy.isin(["ls_ew", "ls_buf"]) & r.features.isin(MAIN_SETS) & r.model.isin(BASE)]
        g = r.groupby(["strategy", "date"])["net"].mean().unstack(0).sort_index()
        cum = g.cumsum() * 100
        n = r[r.strategy == "ls_ew"][["features", "model"]].drop_duplicates().shape[0]
        out.append({"freq": code, "name": name, "n": int(n),
                    "dates": [str(pd.Timestamp(d).date()) for d in cum.index],
                    "plain": [rnd(v, 2) for v in cum["ls_ew"].values],
                    "buffered": [rnd(v, 2) for v in cum["ls_buf"].values],
                    "gain_year": rnd((g["ls_buf"] - g["ls_ew"]).mean() * PER_YEAR[code] * 100, 2)})
    return {"source": "results/series/{M,2W,W}_returns.parquet (test, ls_ew and ls_buf, 25 feature sets x 8 base models)",
            "panels": out}


# ---------------------------------------------------------------- the three checks of 5.1
def placebo_series():
    """Slide 5.1: the weekly rank IC of the narrative LDA exposures and of the same exposures
    misdated by 126 trading days, averaged over the 8 base models (lasso's forecast is flat, so 7
    count), test weeks; the means and the t of their difference from placebo.csv."""
    f = parquet(REPO / "results/series/W_forecast.parquet")
    f = f[(f.split == "test") & f.features.isin(["narr_lda", "narr_lda_placebo"]) & f.model.isin(BASE)]
    g = f.groupby(["features", "date"])["ic"].mean().unstack(0).sort_index()
    r = pd.read_csv(REPO / "results/tables/placebo.csv")
    r = r[(r.freq == "W") & (r.split == "test")].iloc[0]
    return {"source": "results/series/W_forecast.parquet (test, narr_lda vs narr_lda_placebo, base models); results/tables/placebo.csv",
            "dates": [str(pd.Timestamp(d).date()) for d in g.index],
            "real": [rnd(v, 4) for v in g["narr_lda"].values], "placebo": [rnd(v, 4) for v in g["narr_lda_placebo"].values],
            "mean_real": rnd(r.ic, 4), "mean_placebo": rnd(r.ic_placebo, 4), "t": rnd(r.difference_t, 2), "crit": rnd(r.t_critical, 2),
            "days": 126, "models": int(r.models)}


def alpha_rain():
    """Slide 5.1: the three-factor alpha t statistic of every equal-weighted decile long-short of
    the base experiments (25 feature sets x 8 models x 3 frequencies), test period. A portfolio
    whose forecast never ranked anything has no alpha."""
    s = pd.read_csv(REPO / "results/tables/strategies.csv")
    q = s[(s.split == "test") & (s.strategy == "ls_ew") & s.features.isin(MAIN_SETS) & s.model.isin(BASE)]
    a = q.dropna(subset=["alpha_ff3_t"])
    return {"source": "results/tables/strategies.csv (test, ls_ew, 25 feature sets x 8 base models)",
            "portfolios": int(len(q)), "never_traded": int(q["alpha_ff3_t"].isna().sum()), "n": int(len(a)),
            "crit": {c: rnd(a[a.freq == c]["t_critical"].iloc[0], 2) for c in ["M", "2W", "W"]},
            "above": int((a["alpha_ff3_t"] > a["t_critical"]).sum()), "below": int((a["alpha_ff3_t"] < -a["t_critical"]).sum()),
            "pts": [[rnd(t, 2), c] for t, c in zip(a["alpha_ff3_t"].values, a["freq"].values)]}


def text_vs_price():
    """Slide 5.1: price features alone against price and text, the six topic models x 8 base
    models x 3 frequencies; pairs without a t (a flat forecast on one side) are not counted."""
    t = pd.read_csv(REPO / "results/tables/text_vs_price.csv")
    w = t[t.features.isin({f"all_{m}" for m in TOPIC_MODELS}) & t.model.isin(BASE)].dropna(subset=["ic_diff_t"])
    return {"source": "results/tables/text_vs_price.csv (all_<topic model> vs price, 8 base models, 3 frequencies)",
            "pairs": int(len(w)), "better": int((w["ic_diff_t"] > w["t_critical"]).sum()),
            "weekly_lower": int(((w.freq == "W") & (w["ic_diff_t"] < 0)).sum()), "weekly": int((w.freq == "W").sum()),
            "pts": [[rnd(v, 2), rnd(c, 2), f] for v, c, f in zip(w["ic_diff_t"].values, w["t_critical"].values, w["freq"].values)]}


# ---------------------------------------------------------------- portfolio dots (tier 2)
def portfolio(weeks=16):
    """Weekly decile membership of the weekly LDA EBM in the test period, from its stored
    forecasts. Only ranks reach the deck: no identifiers, no returns."""
    try:
        pred = np.load(LOOSE / "predictions/W/all_lda__ebm.test.npy")
        idx = parquet(LOOSE / "panels/W/index.parquet")
    except Exception as e:  # noqa: BLE001 - the deck falls back to the rule diagram
        return {"available": False, "reason": str(e)[:200]}
    test = idx[idx["split"] == "test"].reset_index(drop=True)
    if len(test) != len(pred):
        return {"available": False, "reason": f"rows {len(test)} vs forecasts {len(pred)}"}
    test["f"] = pred
    dates = sorted(test["date"].unique())[:weeks]
    test = test[test["date"].isin(dates)]
    # anonymous, stable across weeks: the same stock keeps its number, nothing identifies it
    anon = {p: i for i, p in enumerate(sorted(test["permno"].unique()))}
    out, held_ew, held_buf, turn_ew, turn_buf = [], {}, {}, [], []
    for d in dates:
        g = test[test["date"] == d]
        pct = g["f"].rank(pct=True).values
        ids = [anon[p] for p in g["permno"].values]
        out.append({"date": str(pd.Timestamp(d).date()), "id": ids, "pct": [rnd(v, 3) for v in pct]})
        # turnover check: half the sum of absolute weight changes, legs of +1 and -1
        ew = {i: (1 if q > 0.9 else -1 if q <= 0.1 else 0) for i, q in zip(ids, pct)}
        buf = {}
        for i, q in zip(ids, pct):
            prev = held_buf.get(i, 0)
            buf[i] = 1 if (q > 0.9 or (prev == 1 and q > 0.8)) else -1 if (q <= 0.1 or (prev == -1 and q <= 0.2)) else 0
        for cur, prev_w, acc in [(ew, held_ew, turn_ew), (buf, held_buf, turn_buf)]:
            if prev_w:
                w1, w0 = weights(cur), weights(prev_w)
                acc.append(0.5 * sum(abs(w1.get(k, 0) - w0.get(k, 0)) for k in set(w1) | set(w0)))
        held_ew, held_buf = ew, buf
    return {"available": True, "source": "data/predictions/W/all_lda__ebm.test.npy; data/panels/W/index.parquet (outside git)",
            "weeks": out, "check_turnover": [rnd(np.mean(turn_ew), 3), rnd(np.mean(turn_buf), 3)]}


def stock_view(weeks=16):
    """Slide 4.1: one stock of the first test week, the one with the highest weekly LDA EBM
    forecast: its rank on each of the 12 plotted features (the stored panel blocks are already
    cross-sectional ranks scaled to [-1, 1]), what each shape function adds for it, its forecast,
    and the remainder (the intercept, the other main effects, the pairwise terms). Anonymous, with
    the same number as in 4.3; derived from the panel, so it stays in portfolio.js."""
    try:
        pred = np.load(LOOSE / "predictions/W/all_lda__ebm.test.npy")
        idx = parquet(LOOSE / "panels/W/index.parquet")
        names = json.loads((LOOSE / "panels/W/names.json").read_text())["names"]
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": str(e)[:200]}
    test = idx[idx["split"] == "test"].reset_index()          # "index": the row in the panel blocks
    if len(test) != len(pred):
        return {"available": False, "reason": f"rows {len(test)} vs forecasts {len(pred)}"}
    test["f"] = pred
    dates = sorted(test["date"].unique())
    first = test[test["date"] == dates[0]]
    row = first.loc[first["f"].idxmax()]
    anon = {p: i for i, p in enumerate(sorted(test[test["date"].isin(dates[:weeks])]["permno"].unique()))}
    shapes = pd.read_csv(REPO / "results/explain/W__all_lda/shapes.csv")
    terms = []
    for term, g in shapes.groupby("term", sort=False):
        blk = next(b for b in ["price", "narr_lda", "firm_lda"] if term in names[b])
        x = float(np.load(LOOSE / f"panels/W/{blk}.npy", mmap_mode="r")[int(row["index"]), names[blk].index(term)])
        b = g[(g["from"] <= x) & (x <= g["to"])].iloc[0]
        terms.append({"term": term, "x": rnd(x, 3), "bp": rnd(b["score"] * 1e4, 3)})
    f_bp = float(row["f"]) * 1e4
    return {"available": True, "source": "data/panels/W/{price,narr_lda,firm_lda}.npy; data/predictions/W/all_lda__ebm.test.npy (outside git); results/explain/W__all_lda/shapes.csv",
            "id": int(anon[row["permno"]]), "date": str(pd.Timestamp(dates[0]).date()), "n": int(len(first)),
            "rank_pct": rnd(float((first["f"] <= row["f"]).mean()), 3), "forecast_bp": rnd(f_bp, 2),
            "rest_bp": rnd(f_bp - sum(t["bp"] for t in terms), 2), "terms": terms}


def weights(pos):
    """Equal weights within each leg: the long leg sums to +1 and the short leg to -1."""
    nl = sum(1 for v in pos.values() if v == 1) or 1
    ns = sum(1 for v in pos.values() if v == -1) or 1
    return {k: (1 / nl if v == 1 else -1 / ns) for k, v in pos.items() if v != 0}


# ---------------------------------------------------------------- sections 1 and 5 after the 9 October briefs
def vt_points():
    """Slide 5.1: every experiment's validation and test rank IC, keyed by feature set and model, with the ten
    highest validation ICs marked. The population and the selection are those of validation_drop()."""
    f = pd.read_csv(REPO / "results/tables/forecast.csv")
    out = []
    for code, _, name, _ in FREQ:
        g = f[(f.freq == code) & f.features.isin(MAIN_SETS) & f.model.isin(BASE)]
        p = g.pivot_table(index=["features", "model"], columns="split", values="ic_mean").dropna()
        top = set(p.sort_values("validation", ascending=False).head(10).index)
        fav = p.loc[list(top), "test"].mean()
        out.append({"freq": code, "name": name.capitalize(), "n": int(len(p)), "fav": rnd(fav, 3),
                    "pts": [{"key": f"{k[0]}__{k[1]}", "val": rnd(v, 4), "test": rnd(t, 4), "top": int(k in top)}
                            for k, (v, t) in zip(p.index, p[["validation", "test"]].values)]})
    return {"source": "results/tables/forecast.csv; population and selection of validation_drop(): 8 base models x the main feature sets, "
                      "experiments with both validation and test ic_mean; top ten by validation ic_mean", "panels": out}


def weekly_r2():
    """Slide 5.2: weekly out-of-sample R2 against the historical mean, price controls alone and with the news,
    coefficients fitted before the test weeks."""
    want = [("pilot_results.csv", "CL", "N_oil", "Crude · NoLBERT oil"), ("pilot_results.csv", "CL", "F_oil", "Crude · FinBERT oil"),
            ("pilot_results.csv", "GC", "F_gold", "Gold · FinBERT gold"), ("btc_results.csv", "BTC", "F_fed", "Bitcoin · FinBERT Fed")]
    rows = []
    for fn, root, narr, label in want:
        d = pd.read_csv(REPO / "commodities/phase2" / fn)
        q = d[(d.kind == "forecasting") & (d.role == "primary") & (d.variant == "main") & (d.stage == "test: dev+val fit applied")
              & (d.root == root) & (d.narrative == narr)]
        assert len(q) == 1, (fn, root, narr, len(q))
        r = q.iloc[0]
        rows.append({"label": label, "file": fn, "root": root, "narrative": narr, "n": int(r.n),
                     "controls": rnd(r.oos_r2_controls_vs_hist_mean, 6), "news": rnd(r.oos_r2_vs_hist_mean, 6)})
    return {"source": "commodities/phase2/pilot_results.csv and btc_results.csv; kind=forecasting, role=primary, variant=main, "
                      "stage='test: dev+val fit applied'; oos_r2_controls_vs_hist_mean, oos_r2_vs_hist_mean", "rows": rows, "weeks": rows[0]["n"]}


def cross_asset():
    """Slide 5.3: the same-month explanatory gain of the equity factors over the market alone, held-out, four assets."""
    o = pd.read_csv(REPO / "commodities/phase2/ipca_oos.csv")
    rows = []
    for asset, label in [("CL", "Crude"), ("HG", "Copper"), ("BTC", "Bitcoin"), ("GC", "Gold")]:
        q = o[(o.role == "primary") & (o.spec == "narr_finbert__sipca") & (o.scope == "asset-specific") & (o.asset == asset)]
        assert len(q) == 1, (asset, len(q))
        r = q.iloc[0]
        rows.append({"asset": asset, "label": label, "n_fit": int(r.n_fit), "n_test": int(r.n_test), "gain": rnd(r.gain, 6),
                     "lo": rnd(r.gain_lo, 6), "hi": rnd(r.gain_hi, 6), "mspe_ratio": rnd(r.mspe_ratio, 6),
                     "cw_p": rnd(r.clark_west_p_onesided, 6), "p_holm_4": rnd(r.p_holm_4, 6)})
    return {"source": "commodities/phase2/ipca_oos.csv; role=primary, spec=narr_finbert__sipca, scope=asset-specific, asset in CL HG BTC GC; "
                      "gain, gain_lo, gain_hi, mspe_ratio, p_holm_4 (Clark-West, Holm over 4 tests)", "spec": "narr_finbert__sipca", "rows": rows}


def text_models():
    """Slide 5.4: the crude-oil result under three text models with the window and the factor count held fixed (post hoc, 63-day)."""
    r = pd.read_csv(REPO / "commodities/phase2/ipca_robustness_results.csv")
    rows = []
    for spec, label in [("narr_finbert_s63__sipca", "FinBERT-20"), ("narr_nolbert20_s63__sipca", "NoLBERT-20"), ("narr_lda_all_s63__sipca", "LDA-60")]:
        q = r[(r.analysis == "A: 63-day (post hoc)") & (r.scope == "asset-specific") & (r.asset == "CL") & (r.spec == spec)]
        assert len(q) == 1, (spec, len(q))
        x = q.iloc[0]
        rows.append({"spec": spec, "label": label, "n_test": int(x.n_test), "gain": rnd(x.gain, 6), "lo": rnd(x.gain_lo, 6), "hi": rnd(x.gain_hi, 6),
                     "holm12_analytic": rnd(x.holm12_analytic, 6), "holm12_calibrated": rnd(x.holm12_calibrated, 6)})
    return {"source": "commodities/phase2/ipca_robustness_results.csv; analysis='A: 63-day (post hoc)', scope=asset-specific, asset=CL; "
                      "gain, gain_lo, gain_hi, holm12_analytic, holm12_calibrated", "rows": rows}


# ---------------------------------------------------------------- facts typed from FINDINGS.md and the papers
FACTS = {
    "headlines": {"source": "FINDINGS.md:41-56", "rows": [
        {"freq": "Monthly", "pick": "Price and NoLBERT text, sparse IPCA", "val_ic": 0.068, "test_ic": 0.006, "net": -0.05, "lo": -1.18, "hi": 1.62},
        {"freq": "Biweekly", "pick": "Price and FinBERT text, XGBoost", "val_ic": 0.058, "test_ic": 0.017, "net": 0.37, "lo": -1.18, "hi": 1.89},
        {"freq": "Weekly", "pick": "Price and LDA text, EBM", "val_ic": 0.048, "test_ic": 0.010, "net": -0.66, "lo": -2.17, "hi": 0.63}]},
    "protocol": {"source": "FINDINGS.md:1192", "rows": [
        {"freq": "Monthly", "net": -0.21, "lo": -1.91, "hi": 0.85},
        {"freq": "Biweekly", "net": -0.34, "lo": -1.61, "hi": 0.69},
        {"freq": "Weekly", "net": -0.16, "lo": -1.25, "hi": 1.19}]},
    "index_vw": {"source": "FINDINGS.md:834-836 (test, value-weighted)", "M": 0.52, "2W": 0.61, "W": 0.47},
    "text_vs_price": {"source": "FINDINGS.md:122-124", "better": 0, "pairs": 134},
    "placebo": {"source": "FINDINGS.md:118-121, 709-711", "days": 126},
    "alpha": {"source": "FINDINGS.md:842-845; results/tables/strategies.csv", "portfolios": 581, "above": 1, "below": 97},
    "buffer": {"source": "FINDINGS.md:61-67", "gain": [0.6, 1.5, 2.0], "t": [2.0, 3.5, 5.2], "crit": [2.39, 2.25, 2.11]},
    "power": {"source": "FINDINGS.md:97-103", "se": [0.75, 0.80], "detect": 2.4},
    "crit": {"source": "FINDINGS.md:23-27", "test": [2.39, 2.25, 2.11]},
    "shock": {"source": "FINDINGS.md:180-200, 1355-1386", "weekly": 0.61, "weekly_buf": 0.65, "monthly": 0.68, "monthly_buf": 0.57},
    "tilt": {"source": "FINDINGS.md:1390-1401", "lo": 0.46, "hi": 0.60},
    "nolbert": {"source": "NolBert.pdf p. 1-3", "params": "109M", "arch": "DeBERTa v3 base", "vocab": "30,000",
                "window": [1976, 1995], "val": 1996, "epochs": 15,
                "t_ahead": 5.56, "d_ahead": 2.78, "t_back": -4.60, "d_back": -2.53, "words": 20},
    "presidents": {"source": "NolBert.pdf p. 9, Table A3", "rows": [
        ["Nixon", 0.05], ["Ford", 0.03], ["Carter", 0.14], ["Reagan", 0.58], ["G. H. W. Bush", 0.51],
        ["Clinton", 0.54], ["G. W. Bush", 0.45], ["Obama", 0.08]], "dividers": [1977, 2001]},
    "splits": {"source": "ftm/config.py:31-33; FINDINGS.md:12-15, 447-450",
               "train": ["2013-01-01", "2019-12-31"], "valid": ["2020-01-01", "2021-12-31"], "test": ["2022-01-01", "2024-12-31"],
               "unused_from": 2017, "test_months": 21},
    "encoders": {"source": "FINDINGS.md:543-544, 1492-1496; ftm/topics/classify.py:7-9",
                 "minilm": "2021", "finbert_tweets": "2022-09", "finbert_published": "2022-12"},
    "adaptation": {"source": "train_log.json; FINDINGS.md:1058-1087; 230ZA app.py:69-74 (corpus from 2015-01-01); train_nolbert.log ('train pairs 228883, val pairs 2048')",
                   "corpus_from": "2015-01-01", "before": "2020-09-01", "pairs": 228883, "held_out": 2048},
    "next": {"source": "platform.claude.com/docs/en/models/haiku-4-5/overview, read 2026-10-08; FINDINGS.md:97-103",
             "model": "Claude Haiku 4.5", "snapshot": "claude-haiku-4-5-20251001", "reliable": "2025-02", "training": "2025-07",
             "test_from": "2025-08-01", "test_to": "2026-10-01", "retirement": "not before 2026-10-15",
             "ic_detectable": {"M": 0.06, "2W": 0.04, "W": 0.03}},
    "periods": {"source": "FINDINGS.md:499-501 (articles), 909 (training periods)", "articles": 1255289, "train": [69, 143, 290]},
    # slide 1.2: Kevin's replication of Bybee, Kelly and Su with the authors' LDA outputs (230ZA app.py:51-52, the deck of 230ZA)
    "replication": {"source": "230ZA 230P-Group/ZA/app/app.py:51-52 (sparse IPCA, out-of-sample Sharpe by K, paper vs ours)",
                    "k": [1, 2, 3, 4, 5, 6], "paper": [0.480, 1.000, 1.100, 1.260, 1.320, 1.310], "replication": [0.480, 0.982, 1.106, 1.264, 1.315, 1.302]},
    # slide 1.3: the earlier FNSPID portfolio and its audit (Project_Reference.pdf, sections 1.2 and 17)
    "mvo": {"source": "Project_Reference.pdf 1.2 ('Out of sample (September 2020 to January 2024, 41 months) ... spherical clusters with multi-horizon "
                      "exposures 1.03') and 17 ('Market position alone 0.30, zero-net narrative tilt 0.89; average net position in the equal-weighted "
                      "market 4.1 times equity')",
            "sharpe": 1.03, "market": 0.30, "tilt": 0.89, "exposure": 4.1, "from": "2020-09", "to": "2024-01", "months": 41},
    # slide 1.2: the 230ZA pipeline (Project_Reference.pdf 1.2; 230ZA app.py PIPELINE: K = 3)
    "za": {"source": "Project_Reference.pdf 1.2 (multi-horizon exposures, spherical clusters, K = 3 factors); 230ZA app.py PIPELINE",
           "topics": 60, "half_lives": [20, 60, 120], "factors": 3},
    "ebm": {"source": "FINDINGS.md:1444 (boosting rounds kept for the main effects, biweekly)", "rounds_2w": 54},
}


def build(check=False):
    facts = dict(FACTS)
    facts["nolbert"] = dict(FACTS["nolbert"], **arch())
    data = {"pinned": PIN, "same_as_pin": same_as_pin(), "repo": REPO_URL,
            "map": news_map(), "training": training(), "stability": stability(), "pooling": pooling(),
            "vt": vt_points(), "weekly": weekly_r2(), "xasset": cross_asset(), "textmodels": text_models(),
            "glassbox": glassbox(), "turnover": turnover(), "validation": validation_drop(),
            "search": search(), "race": race(), "buffer_paths": buffer_paths(),
            "placebo_series": placebo_series(), "alpha_rain": alpha_rain(), "tvp": text_vs_price(),
            "pipeline": pipeline(), "readers": readers(), "attention": attention(), "ebm": ebm_story(),
            "frequency": frequency(), "topics": topic_compare(), "forecasters": forecasters(), "rules": rules(), "best": best(),
            "portfolio": portfolio(), "stock": stock_view(), "facts": facts}
    if check:
        sv = data["stock"]
        print("stock", sv.get("available"), sv.get("reason", ""), "id", sv.get("id"), "forecast bp", sv.get("forecast_bp"), "rest bp", sv.get("rest_bp"),
              "sum of 12", rnd(sum(t["bp"] for t in sv.get("terms", [])), 3), "rank pct", sv.get("rank_pct"))
        pl = data["placebo_series"]
        print("placebo", len(pl["dates"]), "weeks; means", pl["mean_real"], pl["mean_placebo"], "t", pl["t"], "crit", pl["crit"],
              "; series means", rnd(np.mean(pl["real"]), 4), rnd(np.mean(pl["placebo"]), 4), "expect 0.0048 0.0076 (FINDINGS.md:118-121)")
        al = data["alpha_rain"]
        print("alpha", al["portfolios"], "portfolios,", al["never_traded"], "never traded,", al["n"], "with an alpha; above", al["above"],
              "below", al["below"], "expect 1 and 97 (FINDINGS.md:842-845); slide text says", FACTS["alpha"]["portfolios"])
        tv = data["tvp"]
        print("text vs price", tv["pairs"], "pairs, better", tv["better"], "expect 134 and 0; weekly lower", tv["weekly_lower"], "of", tv["weekly"], "expect 42 of 44")
        r = data["race"]
        print("race", r["n"], "paths x", len(r["dates"]), "weeks; pick ends", r["paths"][r["pick"]][-1], "% (Sharpe", r["sharpe"][r["pick"]],
              "); best Sharpe", r["sharpe"][r["best"]], "expect 0.97; median ends", r["median"][-1], "%")
        for p in data["buffer_paths"]["panels"]:
            print("buffer", p["freq"], p["n"], "experiments, expect 200;", len(p["dates"]), "periods; gain", p["gain_year"], "% a year, expect",
                  dict(zip(["M", "2W", "W"], FACTS["buffer"]["gain"]))[p["freq"]], "(FINDINGS.md:61-67)")
        fr = data["frequency"]["rows"]
        print("price IC, test", [r["price_ic"]["test"] for r in fr], "expect 0.008 0.011 0.024 (FINDINGS.md:579)")
        for rule in ["ls_ew", "ls_buf"]:
            print(rule, "median net", [r[rule]["net"] for r in fr], "gross", [r[rule]["gross"] for r in fr],
                  "cost", [r[rule]["cost"] for r in fr], "turnover", [r[rule]["turnover"] for r in fr], "n", [r[rule]["n"] for r in fr])
        print("  expect net -0.21 -0.36 -0.48 / -0.12 -0.24 -0.27; gross 0.03 0.09 0.28; cost 2.9 4.6 8.2% / 2.2 3.1 5.1%;"
              " turnover 119 89 79% / 93 60 49% (FINDINGS.md:914-920)")
        for r in data["topics"]["rows"]:
            print("text-only", r["name"], "test", r["test"], "validation", r["validation"])
        print("  expect test: LDA .005 .002 .003; NoLBERT -.003 .003 .006; NoLBERT-20 .006 .001 .004 (FINDINGS.md:615-620)")
        for sh in data["topics"]["shock"]:
            print("shock", sh["freq"], sh["window"], "IC", sh["ic"], "Sharpe", sh["sharpe_median"], "up", sh["up"], "of 6",
                  [(p["name"], p["base"], p["new"]) for p in sh["topic_models"]])
        print("  expect M 0.003 -> 0.014, -0.07 -> 0.13; W 0.006 -> 0.011, -0.24 -> 0.11; 5 of 6 (FINDINGS.md:1331-1336)")
        for r in data["forecasters"]["rows"]:
            print("model", r["name"], "val", r["validation"], "test", r["test"], "sets", r["sets"])
        print("  other ranked-target models", data["forecasters"]["other_ranked"])
        print("  expect EBM val .022 .023 .022 test .002 -.005 .005; XGBoost ranked test .016 .007 .013 (FINDINGS.md:962-969)")
        for r in data["rules"]["rows"]:
            print("rule", r["name"], "net", r["net"], "n", r["n"])
        print("  index", data["rules"]["index_vw"], "buffer gain", data["rules"]["buffer_gain"],
              "expect index .52 .61 .47; gain 0.6 1.5 2.0% (FINDINGS.md:822-836, 1183)")
        for c in data["best"]["columns"]:
            print("best", c["freq"], [(x["set"], x["model"], x["ic"], x["net"], x["buf"]) if x else None for x in c["cells"]], "best cell", c["best"])
        print("  curves", [(c["freq"], len(c["dates"]), c["pick"][-1], c["index"][-1], c["corr"]) for c in data["best"]["curves"]])
        print("  weekly price-only EBM", data["best"]["weekly_price_ebm"], "expect IC .030 net -0.15 buffered 0.01")
        e = data["ebm"]
        print("ebm val IC", e["val_ic"], "best of base", e["val_best"], "fits", e["fits"], "main", e["main_fits"], "pairs", e["pairs_chosen"])
        a = data["attention"]
        print("attention", a["model"], a["words"], len(a["days"]), "days", a["days"][0], "->", a["days"][-1], "peak", a["peak"], "mean 2021H2", a["mean_2021h2"])
        print("pipeline", {k: v for k, v in data["pipeline"].items() if k != "source"})
        v = data["validation"]["panels"]
        print("rank corr", [p["rho"] for p in v], "expect -0.22 -0.12 0.14 (FINDINGS.md:595-597)")
        print("favorites", [p["fav"] for p in v], "expect -0.001 -0.010 0.004 (FINDINGS.md:597-598)")
        s = data["search"]
        print("search short", s["short"], "of", len(s["rows"]), "expect 17 of 21; all M ls_vw", s["all_mvw"])
        t = data["training"]
        print("retrieval", t["before"], "->", t["after"], "steps", t["steps"], "cut", t["train_end"])
        print("turnover", data["turnover"], "expect 0.789 -> 0.461")
        print("stability", data["stability"]["rows"])
        print("pooling", [(r["label"], r["npmi"], r["ami"]) for r in data["pooling"]["rows"]], "expect 0.343/0.45 first, MiniLM 0.308/1.00 (FINDINGS.md:1066-1073)")
        print("arch", {k: data["facts"]["nolbert"][k] for k in ("layers", "heads", "hidden", "vocab_size")}, "expect 12 12 768 30000 (results/encoders/nolbert_ft/config.json)")
        print("loss log", len(t["log"]), "points; last", t["log"][-1], "expect [6200, 0.284, ...]; first", t["log"][0])
        print("vt", [(q["name"], q["n"], q["fav"], sum(x["top"] for x in q["pts"])) for q in data["vt"]["panels"]], "expect 185/-0.001, 187/-0.010, 187/+0.004, ten each")
        print("weekly", [(r["label"], r["controls"], r["news"], r["n"]) for r in data["weekly"]["rows"]], "expect CL -0.0742 -> -0.0575 / -0.0642; GC 0.0002 -> -0.0042; BTC -0.0135 -> -0.0151; 147 weeks")
        print("xasset", [(r["label"], r["gain"], r["lo"], r["hi"], r["mspe_ratio"], r["p_holm_4"]) for r in data["xasset"]["rows"]], "expect CL 0.309 (-0.022, 0.526) ratio 0.691 p 0.103")
        print("textmodels", [(r["label"], r["gain"], r["lo"], r["hi"], r["holm12_analytic"], r["holm12_calibrated"]) for r in data["textmodels"]["rows"]], "expect FinBERT 0.248 (0.040, 0.461) 0.054/0.043; NoLBERT -0.161; LDA 0.100")
        rp = FACTS["replication"]; print("replication max abs diff", max(abs(a - b) for a, b in zip(rp["paper"], rp["replication"])), "expect 0.018")
        bd = data["map"]["beads"]
        print("beads", len(bd["rows"]), "months", bd["months"][0], "->", bd["months"][-1],
              "density", rnd(sum(map(sum, bd["rows"])) / (20 * len(bd["rows"])), 3))
        p = data["portfolio"]
        print("portfolio", p.get("available"), p.get("reason", ""), len(p.get("weeks", [])))
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    d = build(a.check)
    if not d["same_as_pin"]:
        print(f"WARNING: files the deck reads differ from commit {PIN}: {' '.join(READS)}", file=sys.stderr)
    # stock-level ranks are derived from licensed CRSP data: they go to portfolio.js, which git
    # ignores, and data.js (committed) only says they are absent; slide 4.3 then shows its fallback
    stocks = d.pop("portfolio")
    one = d.pop("stock")
    d["portfolio"] = {"available": False, "reason": "the stock-level ranks are built locally (tools/build_data.py with the data folder) and never committed"}
    d["stock"] = {"available": False, "reason": "built locally with the data folder, see portfolio"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = "window.DATA=" + json.dumps(d, separators=(",", ":"), allow_nan=False) + ";\n"
    out.write_text(s)
    print(f"wrote {out} ({len(s) // 1024} KB)")
    if stocks.get("available"):
        t = "window.DATA.portfolio=" + json.dumps(stocks, separators=(",", ":"), allow_nan=False) + ";\n"
        t += "window.DATA.stock=" + json.dumps(one, separators=(",", ":"), allow_nan=False) + ";\n"
        out.with_name("portfolio.js").write_text(t)
        print(f"wrote {out.with_name('portfolio.js')} ({len(t) // 1024} KB, stock-level: not for git)")
