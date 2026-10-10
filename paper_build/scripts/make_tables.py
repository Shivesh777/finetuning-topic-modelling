"""LaTeX tables of the results section. The three best cells of every metric column are shaded
(\\topA darkest, \\topB, \\topC); 'better' is higher, except where a column says otherwise."""
import re

import numpy as np
import pandas as pd

from style import DATA, FREQ_NAME, FREQS, ROOT

OUT = ROOT / "tables"
T = lambda name: pd.read_csv(DATA / "results/tables" / f"{name}.csv")
BASE = ("ols", "ridge", "lasso", "ipca", "sipca", "xgb", "rf", "ebm")
MODEL_NAME = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "ipca": "IPCA", "sipca": "Sparse IPCA",
              "xgb": "XGBoost", "rf": "Random forest", "ebm": "EBM", "ens": "Ensemble",
              "ols_rk": "OLS, ranked", "ridge_rk": "Ridge, ranked", "lasso_rk": "Lasso, ranked",
              "ipca_rk": "IPCA, ranked", "sipca_rk": "Sparse IPCA, ranked", "xgb_rk": "XGBoost, ranked",
              "best10": "Best ten", "combo": "Combo"}
TOPIC = {"lda": "LDA", "lda_all": "LDA (all articles)", "minilm": "MiniLM", "finbert": "FinBERT",
         "nolbert": "NoLBERT", "nolbert20": "NoLBERT-20"}


def feature_name(f):
    if f == "price":
        return "Price"
    if f.startswith("grand_text_s"):
        return f"Text, all readers, {f.rsplit('_s', 1)[1]} d"
    if f.startswith("grand_s"):
        return f"Price + text, all readers, {f.rsplit('_s', 1)[1]} d"
    fixed = {"tone_fb": "FinBERT tone", "tone_lm": "LM tone", "emb_fb": "FinBERT embeddings",
             "news_fb": "Tone + embeddings", "px_tone": "Price + tone", "px_news": "Price + tone + emb.",
             "px_all": "Price + all news", "px_pool": "Ten experiments"}
    if f in fixed:
        return fixed[f]
    m = re.match(r"^(narr|firm|text|all)_(.+?)(?:_s(\d+))?$", f)
    if not m:
        return f
    kind = {"narr": "Exposures", "firm": "Firm news", "text": "Text", "all": "Price + text"}[m.group(1)]
    win = f", {m.group(3)} d" if m.group(3) else ""
    return f"{kind}, {TOPIC.get(m.group(2), m.group(2))}{win}"


def shade(values, better="high", k=3):
    """Rank of each value among the finite ones: 1, 2, 3 for the best three, 0 otherwise."""
    v = np.array(values, dtype=float)
    out = np.zeros(v.size, dtype=int)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return out
    key = -v if better == "high" else v
    order = np.argsort(np.where(ok, key, np.inf), kind="stable")
    for r, i in enumerate(order[:k]):
        if ok[i]:
            out[i] = r + 1
    return out


def cell(text, rank):
    return {0: text, 1: r"\topA{" + text + "}", 2: r"\topB{" + text + "}", 3: r"\topC{" + text + "}"}[rank]


def num(x, fmt="{:.2f}", plus=False):
    if x is None or not np.isfinite(x):
        return "--"
    s = (("{:+" + fmt[2:]) if plus else fmt).format(x)
    return s.replace("-", "$-$") if not plus else s.replace("-", "$-$").replace("+", "$+$")


def write(name, body):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", name)


def render(df, cols, fmts, better, group_rules=None):
    """Rows of a tabular: df has the display columns; cols: metric columns; better: 'high'/'low'/None."""
    ranks = {c: shade(df[c].to_numpy(), b) if b else np.zeros(len(df), int) for c, b in zip(cols, better)}
    lines = []
    for i, (_, r) in enumerate(df.iterrows()):
        cells = [cell(f(r[c]), ranks[c][i]) for c, f in zip(cols, fmts)]
        lines.append(cells)
    return lines


# ------------------------------------------------------------------ main results, one panel per frequency
FAMILY = [("any", "Overall"), ("price", "Price"), ("text", "Text"), ("all", "Price\\,+\\,text"),
          ("news", "Tone/emb."), ("shock", "63\\,d shocks"), ("any_all", "Every set"),
          ("any_variant", "Ranked targets"), ("ensemble", "Ensembles")] + \
         [(f"topic_{t}", n) for t, n in TOPIC.items()]
COMBOS = [("price", "ens", "Price ensemble"), ("grand_s5", "ens", "All readers, 5 d"),
          ("grand_s63", "ens", "All readers, 63 d"), ("grand_text_s5", "ens", "All readers, text, 5 d"),
          ("grand_text_s63", "ens", "All readers, text, 63 d"), ("px_pool", "best10", "Best ten"),
          ("px_tone", "combo", "Price ens. + tone")]


def main_results():
    sel, st, fc = T("selection"), T("strategies"), T("forecast")
    st = st[st.split == "test"].set_index(["freq", "features", "model", "strategy"])
    fc = fc[fc.split == "test"].set_index(["freq", "features", "model"])
    for f in FREQS:
        rows = []
        s = sel[sel.freq == f].set_index("family")
        picks = {}
        for fam, label in FAMILY:
            if fam not in s.index:
                continue
            key = (s.loc[fam, "features"], s.loc[fam, "model"])
            picks.setdefault(key, []).append(label)
        entries = [("Selected", " $\\cdot$ ".join(v), k[0], k[1]) for k, v in picks.items()]
        entries += [("Combined", label, feat, mod) for feat, mod, label in COMBOS]
        for block, label, feat, mod in entries:
            g = lambda rule, col: st.loc[(f, feat, mod, rule), col] if (f, feat, mod, rule) in st.index else np.nan
            rows.append({"block": block, "label": label, "set": feature_name(feat), "model": MODEL_NAME.get(mod, mod),
                         "ic": fc.loc[(f, feat, mod), "ic_mean"] * 100, "ict": fc.loc[(f, feat, mod), "ic_t"],
                         "sr": g("ls_ew", "sharpe_net"), "srb": g("ls_buf", "sharpe_net"),
                         "srv": g("ls_vw", "sharpe_net"), "srbv": g("ls_buf_vw", "sharpe_net"),
                         "ret": 100 * g("ls_buf", "mean_net"), "vol": 100 * g("ls_buf", "vol"),
                         "dd": 100 * g("ls_buf", "max_drawdown"), "to": 100 * g("ls_buf", "turnover"),
                         "at": g("ls_buf", "alpha_ff3_t")})
        df = pd.DataFrame(rows)
        cols = ["ic", "ict", "sr", "srb", "srv", "srbv", "ret", "vol", "dd", "to", "at"]
        fmts = [lambda x: num(x, "{:.1f}", True), lambda x: num(x, "{:.2f}"), *([lambda x: num(x, "{:.2f}")] * 4),
                lambda x: num(x, "{:.1f}"), lambda x: num(x, "{:.1f}"), lambda x: num(x, "{:.0f}"),
                lambda x: num(x, "{:.0f}"), lambda x: num(x, "{:.2f}")]
        better = ["high", "high", "high", "high", "high", "high", "high", "low", "high", "low", "high"]
        cells = render(df, cols, fmts, better)
        out = []
        prev = None
        for (_, r), c in zip(df.iterrows(), cells):
            if r.block != prev:
                if prev is not None:
                    out.append(r"\addlinespace[2pt]\midrule")
                out.append(r"\multicolumn{14}{l}{\textit{" + ("Experiments selected within each family" if r.block == "Selected"
                           else "Equal-weight combinations") + r"}}\\")
                prev = r.block
            lab = r.label
            out.append(" & ".join([lab, r.set, r.model] + c) + r" \\")
        write(f"tab_main_{f}", "\n".join(out) + "\n")


# ------------------------------------------------------------------ kinds of feature sets
KINDS = [("price", "Price"), ("narr", "Narrative exposures, 5 d"), ("narr_s", "Narrative exposures, 63 d"),
         ("firm", "Firm news"), ("text", "Text, 5 d"), ("text_s", "Text, 63 d"), ("all", "Price + text, 5 d"),
         ("all_s", "Price + text, 63 d"), ("tone", "Tone (FinBERT, LM)"), ("emb", "Headline embeddings"),
         ("news", "Tone + embeddings"), ("px", "Price + news"), ("placebo", "Placebo exposures")]


def kinds_table():
    L = T("landscape")
    L = L[(L.by == "kind") & (L.split == "test")].set_index(["group", "freq"])
    rows = []
    for k, name in KINDS:
        r = {"name": name, "n": int(L.loc[(k, "M"), "experiments"])}
        for f in FREQS:
            g = L.loc[(k, f)]
            r[f"ic{f}"], r[f"t{f}"] = 100 * g.ic_mean, g.ic_t
            r[f"pos{f}"] = g.ic_positive / g.experiments * 100
            r[f"sr{f}"] = g.sharpe_net_median
        rows.append(r)
    df = pd.DataFrame(rows)
    cols, fmts, better = [], [], []
    for f in FREQS:
        cols += [f"ic{f}", f"t{f}", f"pos{f}", f"sr{f}"]
        fmts += [lambda x: num(x, "{:.1f}", True), lambda x: num(x, "{:.1f}"), lambda x: num(x, "{:.0f}"), lambda x: num(x, "{:.2f}")]
        better += ["high"] * 4
    cells = render(df, cols, fmts, better)
    write("tab_kinds", "\n".join(" & ".join([r["name"], str(r["n"])] + c) + r" \\" for (_, r), c in zip(df.iterrows(), cells)) + "\n")


# ------------------------------------------------------------------ models
MODELS = ["ols", "ridge", "lasso", "ipca", "sipca", "xgb", "rf", "ebm", "ens",
          "ols_rk", "ridge_rk", "lasso_rk", "ipca_rk", "sipca_rk", "xgb_rk"]


def models_table():
    L = T("landscape")
    L = L[(L.by == "model")].set_index(["group", "freq", "split"])
    rows = []
    for m in MODELS:
        r = {"name": MODEL_NAME[m]}
        for f in FREQS:
            r[f"v{f}"] = 100 * L.loc[(m, f, "validation"), "ic_mean"]
            r[f"ic{f}"] = 100 * L.loc[(m, f, "test"), "ic_mean"]
            r[f"t{f}"] = L.loc[(m, f, "test"), "ic_t"]
            r[f"sr{f}"] = L.loc[(m, f, "test"), "sharpe_net_median"]
        rows.append(r)
    df = pd.DataFrame(rows)
    cols, fmts, better = [], [], []
    for f in FREQS:
        cols += [f"v{f}", f"ic{f}", f"t{f}", f"sr{f}"]
        fmts += [lambda x: num(x, "{:.1f}", True), lambda x: num(x, "{:.1f}", True), lambda x: num(x, "{:.1f}"), lambda x: num(x, "{:.2f}")]
        better += [None, "high", "high", "high"]
    cells = render(df, cols, fmts, better)
    out = []
    for i, ((_, r), c) in enumerate(zip(df.iterrows(), cells)):
        if MODELS[i] == "ens" or MODELS[i] == "ols_rk":
            out.append(r"\addlinespace[2pt]")
        out.append(" & ".join([r["name"]] + c) + r" \\")
    write("tab_models", "\n".join(out) + "\n")


# ------------------------------------------------------------------ readers and windows
def readers_table():
    fc = T("forecast")
    fc = fc[fc.model.isin(BASE)]
    st = T("strategies")
    st = st[(st.split == "test") & st.model.isin(BASE) & (st.strategy == "ls_ew")]
    rows = []
    for t, name in TOPIC.items():
        r = {"name": name}
        for f in FREQS:
            for w, sets in (("5", [f"narr_{t}", f"text_{t}", f"all_{t}"]), ("63", [f"narr_{t}_s63", f"text_{t}_s63", f"all_{t}_s63"])):
                g = fc[(fc.freq == f) & (fc.split == "test") & fc.features.isin(sets)]
                r[f"ic{w}{f}"] = 100 * g.ic_mean.mean()
                s = st[(st.freq == f) & st.features.isin(sets)]
                r[f"sr{w}{f}"] = s.sharpe_net.median()
        rows.append(r)
    df = pd.DataFrame(rows)
    cols, fmts, better = [], [], []
    for f in FREQS:
        cols += [f"ic5{f}", f"ic63{f}", f"sr5{f}", f"sr63{f}"]
        fmts += [lambda x: num(x, "{:.2f}", True)] * 2 + [lambda x: num(x, "{:.2f}")] * 2
        better += ["high"] * 4
    cells = render(df, cols, fmts, better)
    write("tab_readers", "\n".join(" & ".join([r["name"]] + c) + r" \\" for (_, r), c in zip(df.iterrows(), cells)) + "\n")


# ------------------------------------------------------------------ portfolio rules
RULES = [("ls_ew", "Decile long-short"), ("ls_buf", "Buffered long-short"), ("ls_vw", "Decile, cap-weighted"),
         ("ls_buf_vw", "Buffered, cap-weighted"), ("rank", "Rank-weighted"), ("mvo", "Mean-variance"),
         ("long_ew", "Long top decile$^\\dagger$"), ("long_vw", "Long top decile, cap-w.$^\\dagger$"),
         ("tilt_vw", "Cap-weighted tilt$^\\dagger$")]


def rules_table():
    st = T("strategies")
    st = st[(st.split == "test") & st.model.isin(BASE) & (st.features != "benchmark") & ~st.features.str.endswith("_placebo")]
    rows = []
    for r_, name in RULES:
        r = {"name": name}
        for f in FREQS:
            g = st[(st.freq == f) & (st.strategy == r_)]
            metric = g.information_ratio if r_.startswith(("long", "tilt")) else g.sharpe_net
            r[f"med{f}"] = metric.median()
            r[f"pos{f}"] = 100 * (metric > 0).mean()
            r[f"to{f}"] = 100 * g.turnover.median()
        rows.append(r)
    df = pd.DataFrame(rows)
    cols, fmts, better = [], [], []
    for f in FREQS:
        cols += [f"med{f}", f"pos{f}", f"to{f}"]
        fmts += [lambda x: num(x, "{:.2f}"), lambda x: num(x, "{:.0f}"), lambda x: num(x, "{:.0f}")]
        better += ["high", "high", "low"]
    cells = render(df, cols, fmts, better)
    write("tab_rules", "\n".join(" & ".join([r["name"]] + c) + r" \\" for (_, r), c in zip(df.iterrows(), cells)) + "\n")


# ------------------------------------------------------------------ improvements
def improvements_table():
    imp = T("improvements")
    imp = imp[(imp.split == "test") & (imp.group == "all")].set_index(["comparison", "freq"])
    comps = [("shock_window", "63-day against 5-day shocks"), ("rank_target", "Ranked against raw target"),
             ("ensemble", "Ensemble against its members"), ("buffer", "Buffered against decile")]
    rows = []
    for c, name in comps:
        for f in FREQS:
            r = imp.loc[(c, f)]
            rows.append({"name": name if f == "M" else "", "freq": FREQ_NAME[f], "pairs": int(r.pairs),
                         "icd": 100 * r.ic_diff if np.isfinite(r.ic_diff) else np.nan, "ict": r.ic_diff_t,
                         "rd": 100 * r.return_diff_annual, "rt": r.return_diff_t,
                         "sb": r.sharpe_base_median, "sn": r.sharpe_new_median, "sh": 100 * r.sharpe_improved_share,
                         "tb": 100 * r.turnover_base, "tn": 100 * r.turnover_new})
    df = pd.DataFrame(rows)
    cols = ["icd", "ict", "rd", "rt", "sb", "sn", "sh", "tb", "tn"]
    fmts = [lambda x: num(x, "{:.2f}", True), lambda x: num(x, "{:.1f}"), lambda x: num(x, "{:.1f}", True),
            lambda x: num(x, "{:.1f}"), lambda x: num(x, "{:.2f}"), lambda x: num(x, "{:.2f}"),
            lambda x: num(x, "{:.0f}"), lambda x: num(x, "{:.0f}"), lambda x: num(x, "{:.0f}")]
    better = ["high", "high", "high", "high", None, "high", "high", None, "low"]
    cells = render(df, cols, fmts, better)
    out = []
    for i, ((_, r), c) in enumerate(zip(df.iterrows(), cells)):
        if i and i % 3 == 0:
            out.append(r"\addlinespace[3pt]")
        out.append(" & ".join([r["name"], r["freq"], str(r["pairs"])] + c) + r" \\")
    write("tab_improvements", "\n".join(out) + "\n")


# ------------------------------------------------------------------ significance
def significance_table():
    t = T("sharpe_tests")
    want = [("grand_s63", "ls_ew", ["price_ens", "grand_s5", "grand_text_s63", "headline"]),
            ("grand_s63", "ls_buf", ["price_ens", "grand_s5", "headline"]),
            ("grand_text_s63", "ls_ew", ["price_ens", "grand_text_s5", "headline"]),
            ("grand_text_s63", "ls_buf", ["price_ens", "grand_text_s5", "headline"]),
            ("price_ens", "ls_ew", ["price", "headline"]),
            ("any", "ls_ew", ["price"]), ("any", "ls_buf", ["ls_ew"])]
    fam_name = {"grand_s63": "All readers, price + text, 63 d", "grand_text_s63": "All readers, text, 63 d",
                "price_ens": "Price ensemble", "any": "Overall pick"}
    vs_name = {"price_ens": "price ensemble", "grand_s5": "same, 5-day shocks", "grand_text_s5": "same, 5-day shocks",
               "grand_text_s63": "text alone", "headline": "overall pick", "price": "price pick",
               "ls_ew": "decile rule"}
    rows = []
    for f in FREQS:
        for fam, rule, vss in want:
            g = t[(t.freq == f) & (t.family == fam) & (t.rule == rule)]
            if g.empty:
                continue
            first = True
            for vs in vss:
                h = g[g.vs == vs]
                if h.empty:
                    continue
                h = h.iloc[0]
                rows.append({"freq": FREQ_NAME[f] if first and fam == vss and False else "", "fam": f"{fam_name[fam]}, {'buffered' if rule == 'ls_buf' else 'decile'}" if first else "",
                             "f": f, "sr": h.sharpe_net if first else np.nan, "t": h.t_mean if first else np.nan,
                             "p": h.p_mean if first else np.nan, "psr": h.psr if first else np.nan,
                             "dsr": h.dsr if first else np.nan, "vs": vs_name[vs], "d": h.diff_sharpe,
                             "lwt": h.lw_t, "lwp": h.lw_p, "bp": h.boot_p})
                first = False
    df = pd.DataFrame(rows)
    out = []
    for f in FREQS:
        d = df[df.f == f]
        cols = ["sr", "t", "p", "psr", "dsr", "d", "lwt", "lwp", "bp"]
        fmts = [lambda x: num(x, "{:.2f}"), lambda x: num(x, "{:.2f}"), lambda x: num(x, "{:.3f}"), lambda x: num(x, "{:.2f}"),
                lambda x: num(x, "{:.2f}"), lambda x: num(x, "{:.2f}", True), lambda x: num(x, "{:.2f}"),
                lambda x: num(x, "{:.3f}"), lambda x: num(x, "{:.3f}")]
        better = ["high", "high", "low", "high", "high", "high", "high", "low", "low"]
        d = d.copy()
        dsr_shown = d["dsr"].copy()
        d["dsr"] = d["dsr"].where(d["dsr"] >= 0.005)          # a DSR that rounds to 0.00 is not shaded
        cells = render(d, cols, fmts, better)
        for i, v in enumerate(dsr_shown):
            if np.isfinite(v) and not np.isfinite(d["dsr"].iloc[i]):
                cells[i][4] = fmts[4](v)
        for i in range(len(d)):                                   # continuation rows: no zero-test cells
            if not np.isfinite(d["sr"].iloc[i]):
                cells[i][:5] = [""] * 5
        rk = {c: shade(d[c].to_numpy(), "high") for c in ("t", "lwt")}
        for i, row in enumerate(cells):
            if np.isfinite(d["p"].iloc[i]):
                row[2] = cell(fmts[2](d["p"].iloc[i]), rk["t"][i])
            row[7] = cell(fmts[7](d["lwp"].iloc[i]), rk["lwt"][i])
            row[8] = cell(fmts[8](d["bp"].iloc[i]), rk["lwt"][i])
        out.append(r"\multicolumn{11}{l}{\textit{" + FREQ_NAME[f] + r"}}\\")
        for (_, r), c in zip(d.iterrows(), cells):
            out.append(" & ".join([r["fam"]] + c[:5] + [r["vs"]] + c[5:]) + r" \\")
        if f != "W":
            out.append(r"\addlinespace[3pt]")
    write("tab_significance", "\n".join(out) + "\n")


# ------------------------------------------------------------------ the search
def search_table():
    m = T("multiple_testing")
    rules = [r for r, _ in RULES if r in set(m.strategy)]
    name = dict(RULES)
    rows = []
    for r_ in rules:
        row = {"name": name[r_]}
        for f in FREQS:
            for pool in ("main", "all"):
                g = m[(m.freq == f) & (m.pool == pool) & (m.strategy == r_)].iloc[0]
                row[f"b{pool}{f}"] = g.best_sharpe_net
                row[f"e{pool}{f}"] = g.expected_best_sharpe_if_no_skill
                row[f"p{pool}{f}"] = g.spa_p
        rows.append(row)
    df = pd.DataFrame(rows)
    cols, fmts, better = [], [], []
    for f in FREQS:
        for pool in ("main", "all"):
            cols += [f"b{pool}{f}", f"e{pool}{f}", f"p{pool}{f}"]
            fmts += [lambda x: num(x, "{:.2f}"), lambda x: num(x, "{:.2f}"), lambda x: num(x, "{:.2f}")]
            better += ["high", None, "low"]
    cells = render(df, cols, fmts, better)
    write("tab_search", "\n".join(" & ".join([r["name"]] + c) + r" \\" for (_, r), c in zip(df.iterrows(), cells)) + "\n")


# ------------------------------------------------------------------ significance, compact (main text)
def significance_compact():
    t = T("sharpe_tests")
    rows = []
    for f in FREQS:
        for fam, fname, short in (("grand_s63", "Price + text", "grand_s5"), ("grand_text_s63", "Text only", "grand_text_s5")):
            for rule, rname in (("ls_ew", "decile"), ("ls_buf", "buffered")):
                g = t[(t.freq == f) & (t.family == fam) & (t.rule == rule)]
                if g.empty:
                    continue
                r0 = g.iloc[0]
                row = {"freq": FREQ_NAME[f], "name": f"{fname}, {rname}", "sr": r0.sharpe_net, "t": r0.t_mean,
                       "p": r0.p_mean, "dsr": r0.dsr}
                for key, vs in (("pe", "price_ens"), ("w5", short), ("hd", "headline")):
                    h = g[g.vs == vs]
                    row[key + "d"] = h.diff_sharpe.iloc[0] if len(h) else np.nan
                    row[key + "p"] = h.lw_p.iloc[0] if len(h) else np.nan
                    row[key + "t"] = h.lw_t.iloc[0] if len(h) else np.nan
                rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["sr", "t", "p", "dsr", "ped", "pep", "w5d", "w5p", "hdd", "hdp"]
    f2, f3 = (lambda x: num(x, "{:.2f}")), (lambda x: num(x, "{:.3f}"))
    fp = lambda x: num(x, "{:+.2f}"[0:0] + "{:.2f}", True)
    fmts = [f2, f2, f3, f2, fp, f3, fp, f3, fp, f3]
    cells = render(df, cols, fmts, ["high", "high", None, "high", "high", None, "high", None, "high", None])
    # p-values follow the shading of their t statistics
    rk = {"p": shade(df["t"].to_numpy(), "high"), "pep": shade(df["pet"].to_numpy(), "high"),
          "w5p": shade(df["w5t"].to_numpy(), "high"), "hdp": shade(df["hdt"].to_numpy(), "high")}
    for i, row in enumerate(cells):
        for j, c in enumerate(cols):
            if c in rk:
                row[j] = cell(fmts[j](df[c].iloc[i]), rk[c][i])
    out, prev = [], None
    for (_, r), c in zip(df.iterrows(), cells):
        if r.freq != prev:
            if prev is not None:
                out.append(r"\addlinespace[3pt]")
            lab = r.freq
            prev = r.freq
        else:
            lab = ""
        out.append(" & ".join([lab, r["name"]] + c) + r" \\")
    write("tab_sig_compact", "\n".join(out) + "\n")


if __name__ == "__main__":
    for f in (main_results, kinds_table, models_table, readers_table, rules_table, improvements_table,
              significance_table, search_table, significance_compact):
        f()
