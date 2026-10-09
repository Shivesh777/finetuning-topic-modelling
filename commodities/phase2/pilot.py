"""CL / GC pilot, exactly as fixed in pilot_prereg.md. Reads weekly_panel.parquet, runs each regression once.

    python3 pilot.py  ->  pilot_results.csv, pilot_results.md, pilot_coefficients.png

Seeds: bootstrap seed 0. No tuning of any kind: every number is a fixed-specification estimate.
"""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = Path(__file__).resolve().parent
PREREG_SHA = hashlib.sha256((HERE / "pilot_prereg.md").read_bytes()).hexdigest()
LAGS, BLOCK, DRAWS, SEED = 4, 4, 5000, 0

PRIMARY = {"CL": ["N_oil", "F_oil"], "GC": ["F_gold"]}
SECONDARY = {"GC": ["F_fed"]}
CROSS = {"CL": ["F_gold"], "GC": ["F_oil"]}             # cross-narrative controls
SENS_NARR = {"CL": ["N_oil_t55"]}
CTRL_C = ["mkt_std", "r_lag_std", "vol4_prev_std"]       # contemporaneous controls
CTRL_F = ["r_std", "vol4_cur_std"]                        # forecasting controls

panel = pd.read_parquet(HERE / "weekly_panel.parquet")


def prepare(root, kind, drop_flagged=False):
    p = panel[panel.root == root].sort_values("L").reset_index(drop=True).copy()
    p["y"] = p.r * 100
    p["em"] = p.r.expanding().mean().shift(1) * 100             # expanding historical mean, known before the week
    if kind == "contemporaneous":
        for n in {c[2:-4] for c in p if c.startswith("Z_") and c.endswith("_std")}:
            p[f"lead_{n}"] = p[f"Z_{n}_std"].shift(-1).where((p.L.shift(-1) - p.L).dt.days <= 10)
        out = p
    else:
        nxt = p.L.shift(-1)
        ok = (nxt - p.L).dt.days <= 10
        out = p.copy()
        out["y"] = p.y.shift(-1).where(ok)
        out["em"] = p.em.shift(-1).where(ok)
        out["window"] = p.window.shift(-1).where(ok)
        out["flag_t"] = p.degraded_in_week.shift(-1).where(ok)
        out["L_target"] = nxt.where(ok)
    if drop_flagged:
        bad = p.degraded_in_week.astype(bool)
        if kind == "forecasting":
            bad = bad | out.flag_t.fillna(False).astype(bool)
        out = out[~bad]
    return out


def ols(df, cols):
    d = df.dropna(subset=["y"] + cols)
    X = sm.add_constant(d[cols])
    return sm.OLS(d.y, X).fit(cov_type="HAC", cov_kwds={"maxlags": LAGS}), d


def fit_apply(train, test, cols):
    m, _ = ols(train, cols)
    d = test.dropna(subset=["y"] + cols)
    pred = m.predict(sm.add_constant(d[cols], has_constant="add"))
    return d, pred


def block_boot(e_full, e_base, rng):
    n = len(e_full)
    k = int(np.ceil(n / BLOCK))
    out = np.empty(DRAWS)
    for b in range(DRAWS):
        ix = (rng.integers(0, n, k)[:, None] + np.arange(BLOCK)).ravel()[:n] % n
        out[b] = 1 - (e_full[ix] ** 2).sum() / (e_base[ix] ** 2).sum()
    return np.quantile(out, [0.025, 0.975])


def analyse(root, narr, kind, role, rng, drop_flagged=False, tag=""):
    df = prepare(root, kind, drop_flagged)
    sig = f"Z_{narr}_std" if kind == "contemporaneous" else f"S_{narr}_std"
    ctrl = CTRL_C if kind == "contemporaneous" else CTRL_F
    win = "window"
    d = {w: df[df[win] == w] for w in ["dev", "val", "test"]}
    res = []
    base = dict(root=root, narrative=narr, kind=kind, role=role, variant=tag or "main")
    # own-window regressions (descriptive exposures, HAC)
    for w in ["dev", "val", "test"]:
        mf, dd = ols(d[w], [sig] + ctrl)
        mc, _ = ols(d[w], ctrl)
        ci = mf.conf_int().loc[sig]
        row = dict(base, stage=f"{w}: own-window OLS", n=len(dd), coef=mf.params[sig], se=mf.bse[sig], ci_lo=ci[0], ci_hi=ci[1],
                   p=mf.pvalues[sig], adj_r2=mf.rsquared_adj, adj_r2_controls=mc.rsquared_adj,
                   delta_adj_r2=mf.rsquared_adj - mc.rsquared_adj)
        res.append(row)
    # timing check
    if kind == "contemporaneous":
        dd = d["test"].dropna(subset=["y", f"lead_{narr}"] + CTRL_C)
        mt = sm.OLS(dd.y, sm.add_constant(dd[[f"lead_{narr}"] + CTRL_C])).fit(cov_type="HAC", cov_kwds={"maxlags": LAGS})
        ci = mt.conf_int().loc[f"lead_{narr}"]
        res.append(dict(base, stage="test: timing check (next week's shock)", n=int(mt.nobs), coef=mt.params[f"lead_{narr}"],
                        se=mt.bse[f"lead_{narr}"], ci_lo=ci[0], ci_hi=ci[1], p=mt.pvalues[f"lead_{narr}"]))
    # frozen-coefficient out-of-sample: validation (fit dev), test (fit dev + val)
    for stage, train, test in [("val: dev fit applied", d["dev"], d["val"]),
                               ("test: dev+val fit applied", pd.concat([d["dev"], d["val"]]), d["test"])]:
        dt, pf = fit_apply(train, test, [sig] + ctrl)
        _, pc = fit_apply(train, test, ctrl)
        dt = dt.loc[dt.index.intersection(pc.index)]
        ef, ec = (dt.y - pf.loc[dt.index]).values, (dt.y - pc.loc[dt.index]).values
        r = dict(base, stage=stage, n=len(dt), oos_r2_vs_controls=1 - (ef ** 2).sum() / (ec ** 2).sum())
        lo, hi = block_boot(ef, ec, rng)
        r.update(oos_r2_vs_controls_lo=lo, oos_r2_vs_controls_hi=hi)
        if kind == "forecasting":
            em = dt.em.values
            r["oos_r2_vs_hist_mean"] = 1 - (ef ** 2).sum() / ((dt.y.values - em) ** 2).sum()
            r["oos_r2_controls_vs_hist_mean"] = 1 - (ec ** 2).sum() / ((dt.y.values - em) ** 2).sum()
            hit = lambda e: float(np.mean(np.sign(dt.y.values - e) == np.sign(dt.y.values)))
            r["hit_rate_full"], r["hit_rate_controls"] = hit(ef), hit(ec)
            r["hit_rate_p_vs_half"] = stats.binomtest(int(round(r["hit_rate_full"] * len(dt))), len(dt), 0.5).pvalue
        res.append(r)
    return res


def holm(p):
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for root in ["CL", "GC"]:
        for kind in ["contemporaneous", "forecasting"]:
            for n in PRIMARY.get(root, []):
                rows += analyse(root, n, kind, "primary", rng)
            for n in SECONDARY.get(root, []):
                rows += analyse(root, n, kind, "secondary", rng)
            for n in CROSS.get(root, []):
                rows += analyse(root, n, kind, "cross-narrative control", rng)
            for n in SENS_NARR.get(root, []):
                rows += analyse(root, n, kind, "sensitivity: add t55", rng, tag="t55")
            for n in PRIMARY.get(root, []):
                rows += analyse(root, n, kind, "sensitivity: drop flagged weeks", rng, drop_flagged=True, tag="drop_flagged")
    res = pd.DataFrame(rows)
    prim = res[(res.role == "primary") & (res.stage == "test: own-window OLS")].copy()
    prim["p_holm_6"] = holm(prim.p.values)
    res = res.merge(prim[["root", "narrative", "kind", "role", "stage", "p_holm_6"]], how="left",
                    on=["root", "narrative", "kind", "role", "stage"])
    res["prereg_sha256"] = PREREG_SHA
    res.to_csv(HERE / "pilot_results.csv", index=False)
    write_md(res, prim)
    plot(res)
    print(f"prereg sha256 {PREREG_SHA}")
    print((HERE / "pilot_results.md").read_text())


def fmt(x, d=3):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def write_md(res, prim):
    L = [f"# CL / GC pilot results\n\nSpecification: `pilot_prereg.md` (SHA-256 `{PREREG_SHA}`). HAC standard errors, {LAGS} weekly lags. "
         f"Coefficients are percentage points of weekly return per 1 development-sample standard deviation of the shock. "
         f"Each regression was run once.\n"]
    for kind in ["contemporaneous", "forecasting"]:
        L.append(f"\n## {kind.capitalize()}: own-window regressions (descriptive exposures)\n")
        L.append("| asset | narrative | role | window | n | coef | 95% CI | p | adj R2 | adj R2 controls only |\n|---|---|---|---|---|---|---|---|---|---|")
        s = res[(res.kind == kind) & res.stage.str.contains("own-window") & res.variant.eq("main")]
        for r in s.itertuples():
            L.append(f"| {r.root} | {r.narrative} | {r.role} | {r.stage.split(':')[0]} | {r.n} | {fmt(r.coef)} | [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}] | {fmt(r.p)} | {fmt(r.adj_r2)} | {fmt(r.adj_r2_controls)} |")
        L.append(f"\n## {kind.capitalize()}: frozen-coefficient out-of-sample\n")
        extra = " | OOS R2 vs expanding mean | controls-only R2 vs mean | hit rate (full / controls) |" if kind == "forecasting" else " |"
        L.append("| asset | narrative | role | stage | n | OOS R2 vs controls-only (95% block-bootstrap CI) |" + extra)
        L.append("|---|---|---|---|---|---|" + ("---|---|---|" if kind == "forecasting" else ""))
        s = res[(res.kind == kind) & res.stage.str.contains("applied") & res.variant.eq("main")]
        for r in s.itertuples():
            tail = f" {fmt(r.oos_r2_vs_hist_mean)} | {fmt(r.oos_r2_controls_vs_hist_mean)} | {fmt(r.hit_rate_full, 2)} / {fmt(r.hit_rate_controls, 2)} |" if kind == "forecasting" else ""
            L.append(f"| {r.root} | {r.narrative} | {r.role} | {r.stage.split(':')[0]} | {r.n} | {fmt(r.oos_r2_vs_controls)} [{fmt(r.oos_r2_vs_controls_lo)}, {fmt(r.oos_r2_vs_controls_hi)}] |" + tail)
    L.append("\n## Timing check (contemporaneous regression with next week's shock, test window)\n")
    L.append("| asset | narrative | role | coef | 95% CI | p |\n|---|---|---|---|---|---|")
    for r in res[res.stage.str.contains("timing") & res.variant.eq("main")].itertuples():
        L.append(f"| {r.root} | {r.narrative} | {r.role} | {fmt(r.coef)} | [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}] | {fmt(r.p)} |")
    L.append("\n## Primary tests, test window, with Holm adjustment across the six\n")
    L.append("| asset | narrative | kind | coef | 95% CI | raw p | Holm p |\n|---|---|---|---|---|---|---|")
    for r in prim.itertuples():
        L.append(f"| {r.root} | {r.narrative} | {r.kind} | {fmt(r.coef)} | [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}] | {fmt(r.p)} | {fmt(r.p_holm_6)} |")
    L.append("\n## Sensitivities, test window\n")
    L.append("| asset | narrative | kind | variant | coef | p | OOS R2 vs controls-only |\n|---|---|---|---|---|---|---|")
    own = res[res.stage.eq("test: own-window OLS") & res.variant.ne("main")]
    oos = res[res.stage.str.startswith("test") & res.stage.str.contains("applied") & res.variant.ne("main")]
    for r in own.itertuples():
        o = oos[(oos.root == r.root) & (oos.narrative == r.narrative) & (oos.kind == r.kind) & (oos.variant == r.variant)]
        L.append(f"| {r.root} | {r.narrative} | {r.kind} | {r.variant} | {fmt(r.coef)} | {fmt(r.p)} | {fmt(o.oos_r2_vs_controls.iloc[0]) if len(o) else ''} |")
    (HERE / "pilot_results.md").write_text("\n".join(L))


def plot(res):
    s = res[res.stage.str.contains("own-window") & res.variant.eq("main") & res.role.isin(["primary", "secondary", "cross-narrative control"])]
    combos = s[["root", "narrative", "role"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(combos) + 2.5), sharey=True)
    for ax, kind in zip(axes, ["contemporaneous", "forecasting"]):
        for j, (root, n, role) in enumerate(combos):
            for k, w in enumerate(["dev", "val", "test"]):
                r = s[(s.root == root) & (s.narrative == n) & (s.kind == kind) & s.stage.str.startswith(w)]
                if len(r):
                    r = r.iloc[0]
                    ax.errorbar(r.coef, j + (k - 1) * 0.22, xerr=[[r.coef - r.ci_lo], [r.ci_hi - r.coef]], fmt="o", color=f"C{k}",
                                label=w if j == 0 else None, capsize=2)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(f"{kind}: coefficient (pp of weekly return per 1 dev-sd shock), 95% CI")
        ax.set_yticks(range(len(combos)))
        ax.set_yticklabels([f"{r} {n} ({ro.split()[0]})" for r, n, ro in combos])
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(HERE / "pilot_coefficients.png", dpi=150)


if __name__ == "__main__":
    main()
