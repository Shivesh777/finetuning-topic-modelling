#!/usr/bin/env python3
"""Build deck/js/data.js from the team repo at the pinned commit.

Every number the deck draws comes from this file's output. Each block names
the file it was read from, so a slide's source line can cite it.

    python tools/build_data.py            # writes deck/js/data.js
    python tools/build_data.py --check    # also prints the checks against FINDINGS.md

Reads, read-only:
  ..          the repo this folder sits in; the files it reads must match commit b97a27a
  ../data     predictions and panels outside git (slide 4.3 only); FTM_DATA points elsewhere
Writes deck/js/data.js (committed) and deck/js/portfolio.js (stock-level ranks, gitignored).
Parquet files are read with duckdb, because pyarrow fails on these files here.
Stock-level data (CRSP) never reaches the deck: only ranks of anonymous dots.
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
READS = ["FINDINGS.md", "results/tables", "results/explain", "results/encoders", "data/topics"]


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


# ---------------------------------------------------------------- section 4
FREQ = [("M", "M__all_nolbert", "monthly", "NoLBERT"),
        ("2W", "2W__all_finbert", "biweekly", "FinBERT"),
        ("W", "W__all_lda", "weekly", "LDA")]
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
    s = pd.read_csv(REPO / "results/tables/strategies.csv")
    q = s[(s["freq"] == "W") & (s["split"] == "test") & (s["features"] == "all_lda") & (s["model"] == "ebm")]
    t = q.set_index("strategy")["turnover"] if "turnover" in q.columns else None
    return {"source": "results/tables/strategies.csv (W, test, all_lda, ebm)",
            "weeks": int(q.set_index("strategy")["periods"]["ls_ew"]),
            "ls_ew": rnd(t["ls_ew"], 3) if t is not None else None,
            "ls_buf": rnd(t["ls_buf"], 3) if t is not None else None}


# ---------------------------------------------------------------- section 5
TOPIC_MODELS = ["lda", "lda_all", "finbert", "minilm", "nolbert", "nolbert20"]
MAIN_SETS = {"price"} | {f"{k}_{t}" for k in ["narr", "firm", "text", "all"] for t in TOPIC_MODELS}
BASE = ["ols", "ridge", "lasso", "ipca", "sipca", "xgb", "rf", "ebm"]


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


def weights(pos):
    """Equal weights within each leg: the long leg sums to +1 and the short leg to -1."""
    nl = sum(1 for v in pos.values() if v == 1) or 1
    ns = sum(1 for v in pos.values() if v == -1) or 1
    return {k: (1 / nl if v == 1 else -1 / ns) for k, v in pos.items() if v != 0}


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
    "adaptation": {"source": "train_log.json; FINDINGS.md:1058-1087; 230ZA app.py:69-74",
                   "corpus_from": "2015-01-01", "before": "2020-09-01", "pairs": 228883},
}


def build(check=False):
    data = {"pinned": PIN, "same_as_pin": same_as_pin(), "repo": REPO_URL,
            "map": news_map(), "training": training(), "stability": stability(),
            "glassbox": glassbox(), "turnover": turnover(), "validation": validation_drop(),
            "search": search(), "portfolio": portfolio(), "facts": FACTS}
    if check:
        v = data["validation"]["panels"]
        print("rank corr", [p["rho"] for p in v], "expect -0.22 -0.12 0.14 (FINDINGS.md:595-597)")
        print("favorites", [p["fav"] for p in v], "expect -0.001 -0.010 0.004 (FINDINGS.md:597-598)")
        print("points", [p["n"] for p in v], "expect 185 187 187")
        s = data["search"]
        print("search short", s["short"], "of", len(s["rows"]), "expect 17 of 21; all M ls_vw", s["all_mvw"])
        t = data["training"]
        print("retrieval", t["before"], "->", t["after"], "steps", t["steps"], "cut", t["train_end"])
        for m in data["glassbox"]["models"]:
            print(m["freq"], "ic", m["ic"], {b["block"]: b["d_ic"] for b in m["blocks"]}, "main effects", m["main_effects"])
        print("turnover", data["turnover"], "expect 0.789 -> 0.461")
        print("stability", data["stability"]["rows"])
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
    d["portfolio"] = {"available": False, "reason": "the stock-level ranks are built locally (tools/build_data.py with the data folder) and never committed"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = "window.DATA=" + json.dumps(d, separators=(",", ":"), allow_nan=False) + ";\n"
    out.write_text(s)
    print(f"wrote {out} ({len(s) // 1024} KB)")
    if stocks.get("available"):
        t = "window.DATA.portfolio=" + json.dumps(stocks, separators=(",", ":"), allow_nan=False) + ";\n"
        out.with_name("portfolio.js").write_text(t)
        print(f"wrote {out.with_name('portfolio.js')} ({len(t) // 1024} KB, stock-level: not for git)")
