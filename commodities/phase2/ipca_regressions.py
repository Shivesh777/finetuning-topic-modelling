"""IPCA factor exposures of CL, GC, HG, BTC and the equity market, exactly as fixed in ipca_prereg.md (including its logged deviations).

    python3 ipca_regressions.py  ->  ipca_results.csv, ipca_coefficients.csv, ipca_oos.csv, ipca_results.md, ipca_exposure_heatmap.png

Contemporaneous explanation only: no lagged factors, no forecasting regressions. Every regression is run once.
"""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
PREREG = hashlib.sha256((HERE / "ipca_prereg.md").read_bytes()).hexdigest()
SEED, DRAWS = 0, 5000
ROLES = {"primary": ("pri", "narr_finbert__sipca"), "secondary": ("sec", "narr_nolbert20__sipca")}
ASSETS = ["CL", "GC", "HG", "BTC"]
panel = pd.read_csv(HERE / "ipca_monthly_panel.csv", parse_dates=["date", "hold_start", "hold_end"])
panel["mi"] = panel.date.dt.year * 12 + panel.date.dt.month          # formation month index (calendar)
for a in ASSETS + ["mkt"]:
    panel[a] = panel[a].astype(float)
common = panel[ASSETS].notna().all(axis=1)


# ------------------------------------------------------------------ inference helpers
def ols(y, X):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="HC3", use_t=True)


def nw_calendar_cov(fit, mi):
    """Bartlett kernel with a one-calendar-month bandwidth: only observations in adjacent months are linked (weight 0.5)."""
    X = fit.model.exog
    e = fit.resid.values if hasattr(fit.resid, "values") else fit.resid
    n, k = X.shape
    s = X * e[:, None]
    S = s.T @ s
    order = np.argsort(mi)
    ms = np.asarray(mi)
    for a_ in range(n):
        for b_ in range(a_ + 1, n):
            if abs(ms[a_] - ms[b_]) == 1:
                S = S + 0.5 * (np.outer(s[a_], s[b_]) + np.outer(s[b_], s[a_]))
    bread = np.linalg.inv(X.T @ X)
    return bread @ S @ bread * n / (n - k)


def wald_f(beta, V, idx, df_resid):
    q = len(idx)
    r = beta[idx]
    stat = float(r @ np.linalg.solve(V[np.ix_(idx, idx)], r) / q)
    return stat, float(stats.f.sf(stat, q, df_resid))


def min_detectable_dr2(n, k_full, q, r2_base, alpha):
    """Smallest incremental R2 detectable with 80% power (noncentral F, homoskedastic normal approximation)."""
    df2 = n - k_full
    if df2 <= 0:
        return np.nan
    crit = stats.f.ppf(1 - alpha, q, df2)

    def power(d):
        f2 = d / max(1e-9, 1 - r2_base - d)
        return stats.ncf.sf(crit, q, df2, n * f2) - 0.8
    try:
        return float(optimize.brentq(power, 1e-6, 1 - r2_base - 1e-6))
    except ValueError:
        return np.nan


# ------------------------------------------------------------------ sample definitions
SAMPLES = {"all": panel.split.notna(), "train": panel.split == "train", "validation": panel.split == "validation", "test": panel.split == "test"}


def sample_mask(name, asset):
    base_name = name.replace("common_", "")
    m = SAMPLES[base_name].copy()
    if name.startswith("common_"):
        m &= common
    elif asset != "mkt":
        m &= panel[asset].notna()
    return m


# ------------------------------------------------------------------ main regressions
rows, coefs = [], []
for role, (pre, spec) in ROLES.items():
    fcols = [c for c in panel if c.startswith(pre + "_f")]
    q = len(fcols)
    for sample in ["all", "train", "validation", "test", "common_all", "common_train", "common_validation", "common_test"]:
        for asset in ASSETS + ["mkt"]:
            m = sample_mask(sample, asset)
            d = panel[m]
            n = len(d)
            if n < 8:
                continue
            y = d[asset]
            F = d[fcols]
            if asset == "mkt":
                m0 = sm.OLS(y, np.ones((n, 1))).fit()
                m1 = ols(y, F)
                m2 = m1
                r2_base, k_full = 0.0, q + 1
                adj0, adj1, adj2 = 0.0, m1.rsquared_adj, m1.rsquared_adj
                fidx = list(range(1, q + 1))
                names = ["const"] + fcols
            else:
                m0 = ols(y, d[["mkt"]])
                m1 = ols(y, F)
                m2 = ols(y, d[["mkt"] + fcols])
                r2_base, k_full = m0.rsquared, q + 2
                adj0, adj1, adj2 = m0.rsquared_adj, m1.rsquared_adj, m2.rsquared_adj
                fidx = list(range(2, q + 2))
                names = ["const", "mkt"] + fcols
            beta = np.asarray(m2.params)
            V_hc3 = np.asarray(m2.cov_params())
            V_nw = nw_calendar_cov(m2, d.mi.values)
            _, p_hc3 = wald_f(beta, V_hc3, fidx, m2.df_resid)
            _, p_nw = wald_f(beta, V_nw, fidx, m2.df_resid)
            rows.append(dict(role=role, spec=spec, asset=asset, sample=sample, n=n, adj_r2_M0=adj0, adj_r2_M1=adj1, adj_r2_M2=adj2,
                             delta_adj_r2=adj2 - adj0, r2_M2=m2.rsquared, joint_p_hc3=p_hc3, joint_p_nw_calendar=p_nw,
                             min_detectable_dr2_a05=min_detectable_dr2(n, k_full, q, r2_base, 0.05),
                             min_detectable_dr2_a0125=min_detectable_dr2(n, k_full, q, r2_base, 0.0125),
                             descriptive=sample != "test_frozen"))
            ci = m2.conf_int()
            ci_b = np.asarray(m2.params)[:, None] + np.array([-1, 1]) * stats.t.ppf(1 - 0.025 / 12, m2.df_resid) * np.asarray(m2.bse)[:, None]
            sd_r = float(y.std(ddof=1))
            for k, nm in enumerate(names):
                if nm in fcols:
                    coefs.append(dict(role=role, asset=asset, sample=sample, n=n, factor=nm, coef=beta[k], se_hc3=float(m2.bse.iloc[k] if hasattr(m2.bse, "iloc") else m2.bse[k]),
                                      ci_lo=float(ci.iloc[k, 0] if hasattr(ci, "iloc") else ci[k][0]), ci_hi=float(ci.iloc[k, 1] if hasattr(ci, "iloc") else ci[k][1]),
                                      ci_bonf12_lo=float(ci_b[k, 0]), ci_bonf12_hi=float(ci_b[k, 1]),
                                      std_beta=beta[k] * float(d[nm].std(ddof=1)) / sd_r, p=float(m2.pvalues.iloc[k] if hasattr(m2.pvalues, "iloc") else m2.pvalues[k])))
res = pd.DataFrame(rows)
cf = pd.DataFrame(coefs)

# ------------------------------------------------------------------ frozen out-of-sample (fit on train + validation only)
rng = np.random.default_rng(SEED)
oos = []
for role, (pre, spec) in ROLES.items():
    fcols = [c for c in panel if c.startswith(pre + "_f")]
    for scope in ["asset-specific", "common-month"]:
        for asset in ASSETS + ["mkt"]:
            fit_m = panel.split.isin(["train", "validation"])
            test_m = panel.split == "test"
            if scope == "common-month":
                fit_m &= common
                test_m &= common
            else:
                if asset != "mkt":
                    fit_m &= panel[asset].notna()
                    test_m &= panel[asset].notna()
            fit, test = panel[fit_m], panel[test_m]
            y_f, y_t = fit[asset], test[asset]
            if asset == "mkt":
                X0f, X0t = np.ones((len(fit), 1)), np.ones((len(test), 1))
                X2f, X2t = sm.add_constant(fit[fcols]).values, sm.add_constant(test[fcols], has_constant="add").values
            else:
                X0f, X0t = sm.add_constant(fit[["mkt"]]).values, sm.add_constant(test[["mkt"]], has_constant="add").values
                X2f, X2t = sm.add_constant(fit[["mkt"] + fcols]).values, sm.add_constant(test[["mkt"] + fcols], has_constant="add").values
            b0 = np.linalg.lstsq(X0f, y_f.values, rcond=None)[0]
            b2 = np.linalg.lstsq(X2f, y_f.values, rcond=None)[0]
            p0, p2 = X0t @ b0, X2t @ b2
            e0, e2 = y_t.values - p0, y_t.values - p2
            gain = 1 - (e2 ** 2).sum() / (e0 ** 2).sum()
            d_cw = e0 ** 2 - (e2 ** 2 - (p0 - p2) ** 2)
            t_cw = d_cw.mean() / (d_cw.std(ddof=1) / np.sqrt(len(d_cw)))
            boots = np.array([1 - (e2[i] ** 2).sum() / (e0[i] ** 2).sum() for i in (rng.integers(0, len(e0), len(e0)) for _ in range(DRAWS))])
            oos.append(dict(role=role, spec=spec, scope=scope, asset=asset, n_fit=len(fit), n_test=len(test), gain=gain,
                            gain_lo=float(np.quantile(boots, 0.025)), gain_hi=float(np.quantile(boots, 0.975)), clark_west_t=float(t_cw),
                            clark_west_p_onesided=float(stats.t.sf(t_cw, len(d_cw) - 1)),
                            mspe_ratio=float((e2 ** 2).mean() / (e0 ** 2).mean())))
oos = pd.DataFrame(oos)


def holm(p):
    o = np.argsort(p)
    out = np.empty(len(p))
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        out[i] = min(1.0, run)
    return out


for (role, scope), g in oos[oos.asset.isin(ASSETS)].groupby(["role", "scope"]):
    oos.loc[g.index, "p_holm_4"] = holm(g.clark_west_p_onesided.values)
tdesc = res[(res["sample"] == "test") & res.asset.isin(ASSETS)]
for role, g in tdesc.groupby("role"):
    res.loc[g.index, "joint_p_hc3_holm_4_descriptive"] = holm(g.joint_p_hc3.values)

# ------------------------------------------------------------------ checks and conventions
pn = panel.sort_values("date")
overlap = (pn.hold_end.shift(1) > pn.date).sum()          # previous holding window ends at or before the next formation date
overlap_after = int((pn.hold_end.shift(1) >= pn.hold_start).sum())
res.to_csv(HERE / "ipca_results.csv", index=False)
cf.to_csv(HERE / "ipca_coefficients.csv", index=False)
oos.to_csv(HERE / "ipca_oos.csv", index=False)

# ------------------------------------------------------------------ equity forecasting evidence (existing Phase 1 outputs, no new search)
p1 = pd.read_csv(HERE.parent / "phase1" / "robustness_summary.csv")
p1 = p1[(p1.freq == "M") & (p1.strategy == "buffer") & ~p1.placebo]
def eq_fc(spec):
    t = p1[(p1.experiment == spec) & (p1.split == "test")].iloc[0]
    v = p1[(p1.experiment == spec) & (p1.split == "validation")].iloc[0]
    rank = (p1[p1.split == "test"].sharpe_10bp < t.sharpe_10bp).mean()
    return t, v, rank

# ------------------------------------------------------------------ markdown
f2 = lambda x, d=3: "" if pd.isna(x) else f"{x:.{d}f}"
LBL = {"mkt": "Equity market", "CL": "WTI crude (CL)", "GC": "Gold (GC)", "HG": "Copper (HG)", "BTC": "Bitcoin (BTC)"}
L = [f"# IPCA factor exposures of crude oil, gold, copper, Bitcoin and the equity market\n",
     f"Specification: `ipca_prereg.md` (SHA-256 `{PREREG}`, including its logged deviations). Primary model `narr_finbert__sipca` (K = 3), secondary `narr_nolbert20__sipca` (K = 1). "
     "All results are **contemporaneous explanation**: the factors are estimated from same-month stock returns. Each regression was run once.\n",
     "## Conventions and checks\n",
     "- Factor month: return over (formation date, last trading day of the following calendar month]. Holding windows of consecutive factor months: "
     f"{int(overlap)} overlap with the next formation date (0 expected). Months with at least one Databento degraded day: {int((panel.CL_flag > 0).sum())} of {len(panel)}.",
     "- Futures returns are price returns on a single contract at a time; the contract is changed (volume rule) on a day's close and the roll gap is never a return, so no roll return is included. "
     "A futures price return needs no capital and is already an excess return; no collateral return is added. The equity market is Fama-French `mktrf` (excess of the risk-free rate) over the same dates.",
     "- Standard errors: HC3 with Student-t critical values (primary). The Newey-West sensitivity links only observations in adjacent calendar months. Intervals shown are 95%, unadjusted unless marked.",
     "- M0, M1, M2 use identical observations within each asset and sample. 'Common-month' repeats the analysis on the months in which CL, GC, HG and BTC all have returns (the Bitcoin months).\n"]

L.append("## 1. Held-out evaluation: coefficients fitted on training + validation months only, applied to the 21 test months\n")
L.append("Gain = 1 - SSE(M2)/SSE(M0) in the test months (for the equity market row, M0 is a constant). Interval: paired bootstrap of the 21 months (95%, unadjusted). "
         "Clark-West: one-sided test that M2 has lower squared error than M0, Holm-adjusted over CL, GC, HG, BTC. Contemporaneous factors are used, so this is explanation, not forecasting.\n")
for role in ROLES:
    for scope in ["asset-specific", "common-month"]:
        L.append(f"\n**{role} ({ROLES[role][1]}), {scope} fit sample**\n")
        L.append("| asset | fit months | gain over benchmark [95% CI] | MSPE ratio M2/M0 | Clark-West t | one-sided p | Holm p (4 assets) |\n|---|---|---|---|---|---|---|")
        for r in oos[(oos.role == role) & (oos.scope == scope)].itertuples():
            L.append(f"| {LBL[r.asset]} | {r.n_fit} | {f2(r.gain)} [{f2(r.gain_lo)}, {f2(r.gain_hi)}] | {f2(r.mspe_ratio)} | {f2(r.clark_west_t, 2)} | {f2(r.clark_west_p_onesided)} | {f2(r.p_holm_4) if r.asset in ASSETS else 'n/a'} |")

L.append("\n## 2. Descriptive fit by period (primary model; identical observations for M0, M1, M2)\n")
L.append("Adjusted R-squared of the benchmark (equity market), factors only, and benchmark + factors; joint test that the factor coefficients are zero. "
         "'all' includes months the loadings were fitted on and is **not** out-of-sample. Minimum detectable incremental R-squared (80% power, alpha 0.05 / 0.0125) shows what each sample could detect.\n")
for role in ROLES:
    L.append(f"\n**{role} ({ROLES[role][1]})**\n")
    L.append("| asset | sample | n | adj R2 M0 | M1 | M2 | increment (M2-M0) | joint p HC3 | joint p NW-calendar | detectable R2 increment |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in res[(res.role == role) & ~res["sample"].str.startswith("common_")].itertuples():
        L.append(f"| {LBL[r.asset]} | {r.sample} | {r.n} | {f2(r.adj_r2_M0)} | {f2(r.adj_r2_M1)} | {f2(r.adj_r2_M2)} | {f2(r.delta_adj_r2)} | {f2(r.joint_p_hc3)} | {f2(r.joint_p_nw_calendar)} | {f2(r.min_detectable_dr2_a05)} / {f2(r.min_detectable_dr2_a0125)} |")

L.append("\n## 3. Common-month comparison across assets (months with returns for all four futures; primary model)\n")
L.append("| asset | sample | n | adj R2 M0 | M1 | M2 | increment | joint p HC3 |\n|---|---|---|---|---|---|---|---|")
for r in res[(res.role == "primary") & res["sample"].str.startswith("common_")].itertuples():
    L.append(f"| {LBL[r.asset]} | {r.sample.replace('common_', '')} | {r.n} | {f2(r.adj_r2_M0)} | {f2(r.adj_r2_M1)} | {f2(r.adj_r2_M2)} | {f2(r.delta_adj_r2)} | {f2(r.joint_p_hc3)} |")

L.append("\n## 4. Factor coefficients (primary model, M2), descriptive\n")
L.append("Coefficient: monthly return of the asset per unit of factor return (both in decimals); standardised: coefficient x sd(factor) / sd(asset). "
         "Test-window rows are re-estimated on the 21 test months and are descriptive, not out-of-sample; they carry both the unadjusted 95% interval and a Bonferroni interval over the 12 primary coefficients.\n")
L.append("| asset | sample | n | factor | coef | 95% CI (unadjusted) | 95% CI (Bonferroni 12) | standardised | p |\n|---|---|---|---|---|---|---|---|---|")
for r in cf[(cf.role == "primary") & cf["sample"].isin(["all", "test"])].itertuples():
    L.append(f"| {LBL[r.asset]} | {r.sample} | {r.n} | {r.factor.replace('pri_', '')} | {f2(r.coef)} | [{f2(r.ci_lo)}, {f2(r.ci_hi)}] | [{f2(r.ci_bonf12_lo)}, {f2(r.ci_bonf12_hi)}] | {f2(r.std_beta)} | {f2(r.p)} |")
L.append("\nSecondary model (K = 1), same layout:\n")
L.append("| asset | sample | n | coef | 95% CI (unadjusted) | standardised | p |\n|---|---|---|---|---|---|---|")
for r in cf[(cf.role == "secondary") & cf["sample"].isin(["all", "test"])].itertuples():
    L.append(f"| {LBL[r.asset]} | {r.sample} | {r.n} | {f2(r.coef)} | [{f2(r.ci_lo)}, {f2(r.ci_hi)}] | {f2(r.std_beta)} | {f2(r.p)} |")

L.append("\n## 5. Comparison table: explanatory evidence versus forecasting evidence\n")
L.append("| asset | explanatory evidence (primary model, test months, held-out gain over the equity market) | descriptive fit, all months (increment in adj R2, joint p) | forecasting evidence (a different object, see note) |\n|---|---|---|---|")
fp, fs = eq_fc("narr_finbert__sipca"), eq_fc("narr_nolbert20__sipca")
cs = pd.read_csv(HERE / "cross_asset_primary.csv")
ts_fc = {"CL": "weekly topic shocks, oil: R2 vs expanding mean -5.8% / -6.4% (test); no forecasting gain",
         "GC": "weekly topic shock, gold: R2 vs mean -0.4%; hit rate 47%",
         "HG": "not tested", "BTC": "weekly topic shock, Fed: R2 vs mean -1.5%; hit rate 46%"}
for a in ASSETS + ["mkt"]:
    o = oos[(oos.role == "primary") & (oos.scope == "asset-specific") & (oos.asset == a)].iloc[0]
    d_ = res[(res.role == "primary") & (res["sample"] == "all") & (res.asset == a)].iloc[0]
    if a == "mkt":
        fc = (f"stock-level backtest of the same models (monthly, buffered, test): primary Sharpe {f2(fp[0].sharpe_0bp, 2)} gross / {f2(fp[0].sharpe_10bp, 2)} after 10 bp "
              f"(validation {f2(fp[1].sharpe_10bp, 2)}); secondary {f2(fs[0].sharpe_0bp, 2)} / {f2(fs[0].sharpe_10bp, 2)} (validation {f2(fs[1].sharpe_10bp, 2)})")
    else:
        fc = ts_fc[a]
    L.append(f"| {LBL[a]} | gain {f2(o.gain)} [{f2(o.gain_lo)}, {f2(o.gain_hi)}], one-sided p {f2(o.clark_west_p_onesided)}" + (f", Holm {f2(o.p_holm_4)}" if a in ASSETS else "") +
             f" | {f2(d_.delta_adj_r2)}, p {f2(d_.joint_p_hc3)} (n={d_.n}) | {fc} |")
L.append("\nNote on the forecasting column: for equities it is the out-of-sample stock-level backtest of the same two specifications (Phase 1, no new search). "
         "For CL, GC and BTC it is the earlier weekly topic-shock forecasting result, not an IPCA-factor forecast. IPCA factor forecasting of commodities was not tested.\n")
(HERE / "ipca_results.md").write_text("\n".join(L))

# ------------------------------------------------------------------ heatmap
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, smp in zip(axes, ["all", "test"]):
    g = cf[(cf.role == "primary") & (cf["sample"] == smp)]
    M = g.pivot(index="asset", columns="factor", values="std_beta").reindex(ASSETS)
    P = g.pivot(index="asset", columns="factor", values="p").reindex(ASSETS)
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([c.replace("pri_", "") for c in M.columns])
    ax.set_yticks(range(4))
    ax.set_yticklabels(ASSETS)
    for i in range(4):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:+.2f}" + ("*" if P.values[i, j] < 0.05 else ""), ha="center", va="center", fontsize=9)
    ax.set_title(f"standardised beta, {'all months (descriptive)' if smp == 'all' else '21 test months (descriptive)'}", fontsize=9)
fig.colorbar(im, ax=axes, shrink=0.8)
fig.text(0.01, 0.01, "* unadjusted p < 0.05; not corrected for 12 comparisons. Primary model narr_finbert__sipca.", fontsize=7)
fig.savefig(HERE / "ipca_exposure_heatmap.png", dpi=150, bbox_inches="tight")
print(f"sha {PREREG}\nwrote ipca_results.md; overlap check: {int(overlap)} windows overlap")
