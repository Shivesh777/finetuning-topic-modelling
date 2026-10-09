"""Analyses A and B of ipca_robustness_prereg.md. POST HOC robustness: the 21-month holdout has already been examined, so none of this is
independent confirmation. The original 5-day results are untouched and stay primary.

A. 63-day factors (narr_nolbert20_s63, narr_lda_all_s63, narr_finbert_s63; all K=3): same procedure as ipca_regressions.py / ipca_cw_calibration.py.
B. 2024 sensitivity: original primary and secondary, frozen coefficients unchanged, test months with formation date >= 2023-12-01 dropped (14 remain).

    python3 ipca_robustness.py -> ipca_robustness_results.csv, ipca_robustness_results.md
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
SHA = hashlib.sha256((HERE / "ipca_robustness_prereg.md").read_bytes()).hexdigest()
DRAWS, SEED = 5000, 0
A_ = ["CL", "GC", "HG", "BTC"]
LBL = {"CL": "WTI crude", "GC": "Gold", "HG": "Copper", "BTC": "Bitcoin", "mkt": "Equity market"}
panel = pd.read_csv(HERE / "ipca_monthly_panel.csv", parse_dates=["date"])
common = panel[A_].notna().all(axis=1)

SPECS63 = {"narr_nolbert20_s63__sipca": "NoLBERT-20, 63-day, K=3", "narr_lda_all_s63__sipca": "LDA-60, 63-day, K=3", "narr_finbert_s63__sipca": "FinBERT-20, 63-day, K=3"}


def factors(spec):
    f = pd.read_csv(DATA / "results" / "ipca" / "M" / spec / "factors.csv", parse_dates=["date"])
    assert (f.date.values == panel.date.values).all() and (f.split.values == panel.split.values).all()
    cols = [c for c in f.columns if c.startswith("f") and c[1:].isdigit()]
    assert not f[cols].isna().any().any()
    return f[cols].to_numpy(), cols


def holm(p):
    o = np.argsort(p)
    out = np.empty(len(p))
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        out[i] = min(1.0, run)
    return out


def design(asset, F, rows):
    m = panel.mkt.to_numpy()[rows]
    if asset == "mkt":
        return np.ones((len(rows), 1)), np.c_[np.ones(len(rows)), F[rows]]
    return np.c_[np.ones(len(rows)), m], np.c_[np.ones(len(rows)), m, F[rows]]


def proc(y_f, y_t, X0f, X0t, X2f, X2t):
    b0 = np.linalg.lstsq(X0f, y_f, rcond=None)[0]
    b2 = np.linalg.lstsq(X2f, y_f, rcond=None)[0]
    p0, p2 = X0t @ b0, X2t @ b2
    e0, e2 = y_t - p0, y_t - p2
    gain = 1 - (e2 ** 2).sum() / (e0 ** 2).sum()
    d = e0 ** 2 - (e2 ** 2 - (p0 - p2) ** 2)
    return gain, d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), e0, e2


def evaluate(asset, F, fit_rows, test_rows, seed=SEED):
    """Held-out gain, bootstrap interval, analytic and calibrated Clark-West, calibrated gain p."""
    rng = np.random.default_rng(seed)
    y = panel[asset].to_numpy()
    X0f, X2f = design(asset, F, fit_rows)
    X0t, X2t = design(asset, F, test_rows)
    y_f, y_t = y[fit_rows], y[test_rows]
    gain, t, e0, e2 = proc(y_f, y_t, X0f, X0t, X2f, X2t)
    boots = np.array([1 - (e2[i] ** 2).sum() / (e0[i] ** 2).sum() for i in (rng.integers(0, len(e0), len(e0)) for _ in range(DRAWS))])
    out = dict(n_fit=len(fit_rows), n_test=len(test_rows), gain=gain, gain_lo=float(np.quantile(boots, 0.025)), gain_hi=float(np.quantile(boots, 0.975)),
               cw_t=float(t), p_analytic=float(stats.t.sf(t, len(y_t) - 1)))
    if asset != "mkt":
        Xall = np.vstack([X0f, X0t])
        yall = np.r_[y_f, y_t]
        mu = Xall @ np.linalg.lstsq(Xall, yall, rcond=None)[0]
        res = yall - mu
        gn, tn = np.empty(DRAWS), np.empty(DRAWS)
        for b in range(DRAWS):
            ys = mu + res * rng.choice([-1.0, 1.0], len(res))
            gn[b], tn[b], _, _ = proc(ys[:len(y_f)], ys[len(y_f):], X0f, X0t, X2f, X2t)
        out.update(p_calibrated_cw=float((1 + (tn >= t).sum()) / (1 + DRAWS)), p_calibrated_gain=float((1 + (gn >= gain).sum()) / (1 + DRAWS)))
    return out


def rows_for(asset, scope, drop_2024=False):
    fit = panel.split.isin(["train", "validation"]).to_numpy()
    test = (panel.split == "test").to_numpy()
    if drop_2024:
        test &= (panel.date < "2023-12-01").to_numpy()
    if scope == "common-month":
        fit &= common.to_numpy()
        test &= common.to_numpy()
    elif asset != "mkt":
        fit &= panel[asset].notna().to_numpy()
        test &= panel[asset].notna().to_numpy()
    return np.flatnonzero(fit), np.flatnonzero(test)


def desc(asset, F):
    m = panel[asset].notna().to_numpy() if asset != "mkt" else np.ones(len(panel), bool)
    y = panel[asset].to_numpy()[m]
    X0, X2 = design(asset, F, np.flatnonzero(m))
    f0 = sm.OLS(y, X0).fit()
    f2 = sm.OLS(y, X2).fit(cov_type="HC3", use_t=True)
    q = F.shape[1]
    idx = list(range(X2.shape[1] - q, X2.shape[1]))
    R = np.zeros((q, X2.shape[1]))
    for i, k in enumerate(idx):
        R[i, k] = 1
    w = f2.wald_test(R, use_f=True, scalar=True)
    adj0 = 0.0 if asset == "mkt" else f0.rsquared_adj
    std_beta = [f2.params[k] * F[m][:, j].std(ddof=1) / y.std(ddof=1) for j, k in enumerate(idx)]
    return dict(n=int(m.sum()), d_adj_r2=f2.rsquared_adj - adj0, joint_p=float(w.pvalue), std_beta=std_beta)


rows = []
# ------------------------------------------------------------ original specs (5-day) read from the frozen results, plus A (63-day) computed here
orig = pd.read_csv(HERE / "ipca_oos.csv")
cal = pd.read_csv(HERE / "ipca_cw_calibration.csv")
for role, spec, label in [("primary", "narr_finbert__sipca", "FinBERT-20, 5-day, K=3 (original primary)"), ("secondary", "narr_nolbert20__sipca", "NoLBERT-20, 5-day, K=1 (original secondary)")]:
    for scope in ["asset-specific", "common-month"]:
        for a in A_ + ["mkt"]:
            o = orig[(orig.role == role) & (orig.scope == scope) & (orig.asset == a)].iloc[0]
            r = dict(analysis="original", spec=spec, label=label, scope=scope, asset=a, n_fit=int(o.n_fit), n_test=int(o.n_test), gain=o.gain, gain_lo=o.gain_lo, gain_hi=o.gain_hi, cw_t=o.clark_west_t, p_analytic=o.clark_west_p_onesided)
            if role == "primary" and a != "mkt":
                c = cal[(cal.scope == scope) & (cal.asset == a)].iloc[0]
                r.update(p_calibrated_cw=c.p_calibrated_cw, p_calibrated_gain=c.p_calibrated_gain)
            rows.append(r)
for spec, label in SPECS63.items():
    F, cols = factors(spec)
    for scope in ["asset-specific", "common-month"]:
        for a in A_ + ["mkt"]:
            fit, test = rows_for(a, scope)
            rows.append(dict(analysis="A: 63-day (post hoc)", spec=spec, label=label, scope=scope, asset=a, **evaluate(a, F, fit, test)))
# ------------------------------------------------------------ B: 2024 exclusion, original specs, frozen coefficients
for spec, label in [("narr_finbert__sipca", "FinBERT-20, 5-day, K=3 (original primary)"), ("narr_nolbert20__sipca", "NoLBERT-20, 5-day, K=1 (original secondary)")]:
    F, cols = factors(spec)
    for a in A_ + ["mkt"]:
        fit, test = rows_for(a, "asset-specific", drop_2024=True)
        rows.append(dict(analysis="B: 2024 excluded (descriptive)", spec=spec, label=label, scope="asset-specific", asset=a, **evaluate(a, F, fit, test)))
res = pd.DataFrame(rows)

# ------------------------------------------------------------ multiple-testing corrections
for (an, spec, scope), g in res[res.asset.isin(A_)].groupby(["analysis", "spec", "scope"]):
    res.loc[g.index, "holm4_analytic"] = holm(g.p_analytic.values)
    if g.p_calibrated_cw.notna().all():
        res.loc[g.index, "holm4_calibrated"] = holm(g.p_calibrated_cw.values)
for scope in ["asset-specific", "common-month"]:
    g = res[(res.analysis == "A: 63-day (post hoc)") & (res.scope == scope) & res.asset.isin(A_)]
    res.loc[g.index, "holm12_analytic"] = holm(g.p_analytic.values)
    res.loc[g.index, "holm12_calibrated"] = holm(g.p_calibrated_cw.values)
res.to_csv(HERE / "ipca_robustness_results.csv", index=False)

# ------------------------------------------------------------ descriptive fit and loadings
dres = []
for spec, label in [("narr_finbert__sipca", "FinBERT-20, 5-day, K=3"), ("narr_nolbert20__sipca", "NoLBERT-20, 5-day, K=1")] + list(SPECS63.items()):
    F, cols = factors(spec)
    for a in A_ + ["mkt"]:
        d = desc(a, F)
        dres.append(dict(spec=spec, label=label, asset=a, **{k: v for k, v in d.items() if k != "std_beta"}, std_beta=", ".join(f"{b:+.2f}" for b in d["std_beta"])))
dres = pd.DataFrame(dres)
dres.to_csv(HERE / "ipca_robustness_descriptive.csv", index=False)


def top_loadings(spec, model_dir):
    g = pd.read_csv(DATA / "results" / "ipca" / "M" / spec / "gamma.csv")
    cols = [c for c in g.columns if c.startswith("f") and c[1:].isdigit()]
    if model_dir == "finbert":
        import json
        names = json.load(open(DATA / "topics" / "embeddings" / "finbert-topic_d150.labels.json"))
        g["name"] = [names[int(f.split("_t")[-1])] for f in g.feature]
    else:
        tp = pd.read_csv(DATA / "topics" / model_dir / "topics.csv")
        terms = {int(str(t).replace("t", "")): str(s).split(", ")[:3] for t, s in zip(tp.topic, tp.top_terms)} if tp.topic.dtype == object else {int(t): str(s).split(", ")[:3] for t, s in zip(tp.topic, tp.top_terms)}
        g["name"] = [f"t{int(f.split('_t')[-1]):02d}: " + ", ".join(terms.get(int(f.split("_t")[-1]), [])) for f in g.feature]
    out = {}
    for c in cols:
        top = g.reindex(g[c].abs().sort_values(ascending=False).index).head(3)
        out[c] = "; ".join(f"{r['name']} ({r[c]:+.2f})" for _, r in top.iterrows())
    return out


f2 = lambda x, d=2: "" if pd.isna(x) else f"{x:.{d}f}"
f3 = lambda x: "" if pd.isna(x) else f"{x:.3f}"
L = [f"# Post hoc robustness of the IPCA cross-asset result (analyses A and B)\n",
     f"Specification: `ipca_robustness_prereg.md` (SHA-256 `{SHA}`). **Post hoc:** the 21 test months were already examined in the original analysis, so none of this is independent confirmation. "
     "The original 5-day FinBERT K=3 result remains primary; nothing here replaces it. All factor files cover the same 104 months, so eligible observations are identical across specifications "
     "(59 fit months for crude, gold and copper, 28 for Bitcoin and for common-month fits, 83 for the equity market, 21 test months; 14 in analysis B). Intervals are unadjusted 95% bootstrap intervals; Holm families are stated.\n",
     "## A. Held-out gain over the equity market: original (5-day) and 63-day specifications side by side\n",
     "Gain = 1 - SSE(factors + market) / SSE(market only) in the test months, fitted on training + validation months only. p: one-sided Clark-West, analytic and calibrated (simulated null). "
     "Holm (4): across the four assets within a specification. Holm (12): across all 12 asset-model tests of the three 63-day specifications (conservative post hoc family).\n"]
for scope in ["asset-specific", "common-month"]:
    L.append(f"\n**{scope} fit sample**\n")
    L.append("| specification | asset | n fit / test | gain [95% CI] | analytic p | Holm (4) | calibrated p | Holm (4) cal. | Holm (12) analytic | Holm (12) cal. |\n|---|---|---|---|---|---|---|---|---|---|")
    d = res[(res.scope == scope) & res.analysis.isin(["original", "A: 63-day (post hoc)"])]
    for spec in ["narr_finbert__sipca", "narr_nolbert20__sipca"] + list(SPECS63):
        for a in A_ + ["mkt"]:
            r = d[(d.spec == spec) & (d.asset == a)].iloc[0]
            h4 = f3(r.holm4_analytic) if a != "mkt" else ""
            hc = f3(r.holm4_calibrated) if (a != "mkt" and "holm4_calibrated" in r and not pd.isna(r.get("holm4_calibrated"))) else ""
            L.append(f"| {r.label}{' (post hoc)' if r.analysis.startswith('A') else ''} | {LBL[a]} | {int(r.n_fit)} / {int(r.n_test)} | {f2(r.gain)} [{f2(r.gain_lo)}, {f2(r.gain_hi)}] | {f3(r.p_analytic)} | {h4} | {f3(r.get('p_calibrated_cw'))} | {hc} | {f3(r.get('holm12_analytic'))} | {f3(r.get('holm12_calibrated'))} |")

L.append("\n## A (descriptive). Fit on all available months, and exposures of crude oil\n")
L.append("Increase in adjusted R2 over the market benchmark and the joint test that the factor coefficients are zero (HC3). Descriptive: these months include those the loadings were fitted on.\n")
L.append("| specification | asset | n | increment adj R2 | joint p | standardised betas of the factors |\n|---|---|---|---|---|---|")
for r in dres.itertuples():
    L.append(f"| {r.label} | {LBL[r.asset]} | {r.n} | {f2(r.d_adj_r2)} | {f3(r.joint_p)} | {r.std_beta} |")
L.append("\n**Largest loadings of each 63-day factor** (latent factors: signs and rotations are arbitrary)\n")
for spec, label in SPECS63.items():
    md = {"narr_nolbert20_s63__sipca": "nolbert20", "narr_lda_all_s63__sipca": "lda_all", "narr_finbert_s63__sipca": "finbert"}[spec]
    try:
        tl = top_loadings(spec, md)
        L.append(f"- {label}: " + " | ".join(f"{k}: {v}" for k, v in tl.items()))
    except Exception as e:
        L.append(f"- {label}: (loadings not listed: {type(e).__name__})")

L.append("\n## B. 2024 sensitivity (descriptive): original specifications, frozen coefficients, 14 test months\n")
L.append("Test months with formation on or after 2023-12-01 are dropped (7 months); fit months and coefficients are unchanged. With 14 months every interval is wider and power lower; a null here is not evidence of no effect.\n")
L.append("| specification | asset | gain, 21 months | gain, 14 months [95% CI] | analytic p | Holm (4) | calibrated p | Holm (4) cal. |\n|---|---|---|---|---|---|---|---|")
for spec in ["narr_finbert__sipca", "narr_nolbert20__sipca"]:
    for a in A_ + ["mkt"]:
        b = res[(res.analysis.str.startswith("B")) & (res.spec == spec) & (res.asset == a)].iloc[0]
        o = res[(res.analysis == "original") & (res.spec == spec) & (res.scope == "asset-specific") & (res.asset == a)].iloc[0]
        hc = f3(b.get("holm4_calibrated")) if a != "mkt" else ""
        L.append(f"| {b.label} | {LBL[a]} | {f2(o.gain)} | {f2(b.gain)} [{f2(b.gain_lo)}, {f2(b.gain_hi)}] | {f3(b.p_analytic)} | {f3(b.holm4_analytic) if a != 'mkt' else ''} | {f3(b.get('p_calibrated_cw'))} | {hc} |")
(HERE / "ipca_robustness_results.md").write_text("\n".join(L))
print("done")
