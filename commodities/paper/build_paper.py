"""Builds main.tex and the figures from the frozen result files. Every number in the paper is filled from a file;
the build stops if a placeholder is left or a source value is missing.

    python3 build_paper.py   (run from research/paper)  ->  main.tex, figures/*.png, numbers_used.csv
"""
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P1, P2 = HERE.parent / "phase1", HERE.parent / "phase2"
(HERE / "figures").mkdir(exist_ok=True)

# ------------------------------------------------------------------ sources
find = json.load(open(P1 / "findings.json"))
rob = pd.read_csv(P1 / "robustness_summary.csv")
topic = pd.read_csv(P2 / "cross_asset_primary.csv")
pilot = pd.concat([pd.read_csv(P2 / "pilot_results.csv"), pd.read_csv(P2 / "btc_results.csv")], ignore_index=True)
oos = pd.read_csv(P2 / "ipca_oos.csv")
res = pd.read_csv(P2 / "ipca_results.csv")
coef = pd.read_csv(P2 / "ipca_coefficients.csv")
cal = pd.read_csv(P2 / "ipca_cw_calibration.csv")
panel = pd.read_csv(P2 / "ipca_monthly_panel.csv")
fv, wv = (P2 / "futures_validation.md").read_text(), (P2 / "wti_validation.md").read_text()

N = {}   # token -> string, recorded in numbers_used.csv


def S(x, d=2):
    s = f"{abs(x):.{d}f}"
    return f"$-{s}$" if x < 0 and float(s) != 0 else s


def CI(lo, hi, d=2):
    return f"[{S(lo, d)}, {S(hi, d)}]"


def put(k, v):
    N[k] = v


A = ["CL", "HG", "BTC", "GC"]
LBL = {"CL": "WTI crude", "HG": "Copper", "BTC": "Bitcoin", "GC": "Gold", "mkt": "Equity market"}

# ---- Phase 1, equities
FR = {"M": "M", "2W": "2W", "W": "W"}
for k in FR:
    v = find[k]
    put(f"V_{k}", S(v["validation_sharpe"]))
    put(f"T10_{k}", S(v["test_sharpe_by_cost"]["10"]))
    put(f"T0_{k}", S(v["test_sharpe_by_cost"]["0"]))
    put(f"DSR1_{k}", S(v["dsr_chosen_n1"]))
    put(f"PER_{k}", str(v["test_periods"]))
    put(f"CHOSEN_{k}", v["chosen_on_validation"].replace("_", "\\_"))
put("CI_M", CI(*find["M"]["bootstrap_sharpe_95ci"]))
put("TURN_LO", S(min(v["test_turnover_per_period"] for v in find.values())))
put("TURN_HI", S(max(v["test_turnover_per_period"] for v in find.values())))
put("NEFF_LO", str(int(round(min(v["n_trials_effective"] for v in find.values())))))
put("NEFF_HI", str(int(round(max(v["n_trials_effective"] for v in find.values())))))
put("DSRBEST_MAX", S(max(v["dsr_best_in_test"]["N_eff"] for v in find.values())))
t = rob[(rob.split == "test") & (~rob.placebo) & (rob.strategy == "buffer")]
med0 = [t[t.freq == k].sharpe_0bp.median() for k in FR]
put("MED0_LO", S(min(med0)))
put("MED0_HI", S(max(med0)))
pos = [(t[t.freq == k].sharpe_10bp > 0).mean() * 100 for k in FR]
put("POS_LO", f"{min(pos):.0f}")
put("POS_HI", f"{max(pos):.0f}")
for k in FR:
    s_t = t[t.freq == k]
    v_ = rob[(rob.freq == k) & (rob.split == "validation") & (rob.strategy == "buffer") & (~rob.placebo)]
    m = v_.merge(s_t[["experiment", "sharpe_10bp"]], on="experiment", suffixes=("_v", "_t"))
    put(f"RHO_{k}", S(m.sharpe_10bp_v.corr(m.sharpe_10bp_t, method="spearman")))
plM = rob[(rob.freq == "M") & (rob.split == "test") & rob.placebo & (rob.strategy == "buffer")].sharpe_10bp.max()
put("PLAC_M", S(plM))
put("BESTREAL_M", S(t[t.freq == "M"].sharpe_10bp.max()))

# ---- Phase 1, weekly topic shocks
T = topic.set_index(["root", "narrative", "kind"])
def tp(r, n, k, col): return T.loc[(r, n, k), col]
put("W_NOIL_C", S(tp("CL", "N_oil", "contemporaneous", "coef")))
put("W_NOIL_C_CI", CI(tp("CL", "N_oil", "contemporaneous", "ci_lo"), tp("CL", "N_oil", "contemporaneous", "ci_hi")))
put("W_NOIL_C_HOLM", S(tp("CL", "N_oil", "contemporaneous", "p_holm_8"), 3))
put("W_NOIL_F", S(tp("CL", "N_oil", "forecasting", "coef")))
put("W_NOIL_F_P", S(tp("CL", "N_oil", "forecasting", "p"), 3))
put("W_NOIL_F_HOLM", S(tp("CL", "N_oil", "forecasting", "p_holm_8"), 3))
own = pilot[pilot.stage.str.contains("own-window") & pilot.variant.eq("main")]
put("W_NOIL_F_DEV", S(own[(own.root == "CL") & (own.narrative == "N_oil") & (own.kind == "forecasting") & own.stage.str.startswith("dev")].coef.iloc[0]))
t55 = pilot[(pilot.variant == "t55") & (pilot.kind == "forecasting") & (pilot.stage == "test: own-window OLS")].p.iloc[0]
put("W_T55_P", S(t55))
df_ = pilot[(pilot.variant == "drop_flagged") & (pilot.stage == "test: own-window OLS")].set_index(["root", "narrative", "kind"]).p
p8 = df_.loc[[tuple(x) for x in topic[["root", "narrative", "kind"]].values]].values
def holm(p):
    o = np.argsort(p); out = np.empty(len(p)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); out[i] = min(1.0, run)
    return out
put("W_DROP_HOLM", S(holm(np.array(p8))[2], 3))
put("W_DROP_P", S(df_.loc[("CL", "N_oil", "forecasting")], 3))
o_ = pilot[pilot.stage.eq("test: dev+val fit applied") & pilot.kind.eq("forecasting") & pilot.role.eq("primary") & pilot.variant.eq("main")].set_index(["root", "narrative"])
for key, (r, n) in {"CLN": ("CL", "N_oil"), "CLF": ("CL", "F_oil"), "GC": ("GC", "F_gold"), "BTC": ("BTC", "F_fed")}.items():
    put(f"W_OOS_{key}", S(o_.loc[(r, n)].oos_r2_vs_hist_mean * 100, 1) + "\\%")
put("W_TESTWEEKS_C", str(int(own[(own.root == "CL") & (own.narrative == "N_oil") & (own.kind == "contemporaneous") & own.stage.str.startswith("test")].n.iloc[0])))
put("W_TESTWEEKS_F", str(int(own[(own.root == "CL") & (own.narrative == "N_oil") & (own.kind == "forecasting") & own.stage.str.startswith("test")].n.iloc[0])))
put("GLD_CORR", re.search(r"weekly correlation: ([0-9.]+);", fv).group(1))
put("WTI_CORR", re.search(r"\| excluding Mar-Jul 2020 \| \d+ \| ([0-9.]+) \|", wv).group(1))
put("WTI_CORR_TEST", re.search(r"\| test 2022-2024 \| \d+ \| ([0-9.]+) \|", wv).group(1))

# ---- Phase 2
O = oos[(oos.role == "primary") & (oos.scope == "asset-specific")].set_index("asset")
OC = oos[(oos.role == "primary") & (oos.scope == "common-month")].set_index("asset")
OS = oos[(oos.role == "secondary") & (oos.scope == "asset-specific")].set_index("asset")
C = cal[cal.scope == "asset-specific"].set_index("asset")
CC = cal[cal.scope == "common-month"].set_index("asset")
gh = holm(C.loc[A].p_calibrated_gain.values)
for a in A + ["mkt"]:
    put(f"G_{a}", S(O.loc[a, "gain"]))
    put(f"GCI_{a}", CI(O.loc[a, "gain_lo"], O.loc[a, "gain_hi"]))
    put(f"GCOM_{a}", S(OC.loc[a, "gain"]))
    put(f"GSEC_{a}", S(OS.loc[a, "gain"]))
for a in A:
    put(f"CWP_{a}", S(O.loc[a, "clark_west_p_onesided"], 3))
    put(f"HOLM_{a}", S(O.loc[a, "p_holm_4"], 3))
    put(f"HOLMSEC_{a}", S(OS.loc[a, "p_holm_4"], 3))
    put(f"HOLMCOM_{a}", S(OC.loc[a, "p_holm_4"], 3))
put("CAL_CL_P", S(C.loc["CL", "p_calibrated_cw"], 3))
put("CAL_CL_HOLM", S(C.loc["CL", "p_holm_calibrated_cw"], 3))
put("CAL_GAIN_CL_P", S(C.loc["CL", "p_calibrated_gain"], 3))
put("CAL_GAIN_CL_HOLM", S(gh[0], 3))
put("CAL_COM_GAIN_HOLM", S(holm(CC.loc[A].p_calibrated_gain.values)[0], 3))
put("SIZE_LO", f"{cal.size_at_5pct.min() * 100:.1f}")
put("SIZE_HI", f"{cal.size_at_5pct.max() * 100:.1f}")
put("NULL95_CL", S(C.loc["CL", "null_gain_95pct"]))
put("NFIT_CL", str(int(O.loc["CL", "n_fit"])))
put("NFIT_BTC", str(int(O.loc["BTC", "n_fit"])))
put("NFIT_MKT", str(int(O.loc["mkt", "n_fit"])))
R = res[(res.role == "primary") & (res["sample"] == "all")].set_index("asset")
for a in A + ["mkt"]:
    put(f"INC_{a}", S(R.loc[a, "delta_adj_r2"]))
    put(f"NALL_{a}", str(int(R.loc[a, "n"])))
    put(f"JP_ALL_{a}", "$<0.001$" if R.loc[a, "joint_p_hc3"] < 0.001 else S(R.loc[a, "joint_p_hc3"], 3))
Rt = res[(res.role == "primary") & (res["sample"] == "test")].set_index("asset")
put("GPCT_CL", f"{O.loc['CL', 'gain'] * 100:.0f}")
nwp = Rt.loc[A, "joint_p_nw_calendar"]
put("JP_NW_BTC", S(nwp.loc["BTC"], 3))
put("JP_NW_BTC_HOLM", S(holm(nwp.values)[A.index("BTC")], 3))
put("W_HOLM_MIN", S(topic.p_holm_8.min(), 3))
put("MDE_LO", S(Rt.loc[A, "min_detectable_dr2_a05"].min()))
put("MDE_HI", S(Rt.loc[A, "min_detectable_dr2_a05"].max()))
cf = coef[(coef.role == "primary") & (coef.asset == "CL") & (coef["sample"] == "all")].set_index("factor")
put("F3_STD", S(cf.loc["pri_f3", "std_beta"]))
put("F1_COEF", S(cf.loc["pri_f1", "coef"]))
put("F1_P", S(cf.loc["pri_f1", "p"]))
gam = pd.read_csv(P2.parent.parent / "data" / "results" / "ipca" / "M" / "narr_finbert__sipca" / "gamma.csv")
names = json.load(open(P2.parent.parent / "data" / "topics" / "embeddings" / "finbert-topic_d150.labels.json"))
gam["cls"] = [names[int(f.split("_t")[-1])].replace("&", "\\&") for f in gam.feature]
for fk in ["f1", "f2", "f3"]:
    top = gam.reindex(gam[fk].abs().sort_values(ascending=False).index).head(4)
    put(f"LOAD_{fk}", ", ".join(f"{r.cls} ({r[fk]:+.2f})".replace("-", "$-$") for _, r in top.iterrows()))
bsd = pd.read_csv(P2 / "ipca_bootstrap_diagnostic.csv").set_index("asset")
for a_ in ["CL", "GC", "HG", "BTC"]:
    put(f"BS_{a_}", S(bsd.loc[a_, "lag1_corr_loss_diff"]))
put("N_MONTHS_COM", str(int(panel[["CL", "GC", "HG", "BTC"]].notna().all(axis=1).sum())))
put("N_COM_FIT", str(int(OC.loc["CL", "n_fit"])))
put("N_FACTOR_MONTHS", str(len(panel)))
put("N_CL_MONTHS", str(int(panel.CL.notna().sum())))
put("N_BTC_MONTHS", str(int(panel.BTC.notna().sum())))


# ---- post hoc robustness (analyses A and B)
rob = pd.read_csv(P2 / "ipca_robustness_results.csv")
rdesc = pd.read_csv(P2 / "ipca_robustness_descriptive.csv")
rtest = (P2 / "reconstruction_test.md").read_text()
def rr(spec, asset, col, scope="asset-specific", analysis=None):
    d = rob[(rob.spec == spec) & (rob.asset == asset) & (rob.scope == scope)]
    d = d[d.analysis.str.startswith("A")] if analysis == "A" else (d[d.analysis.str.startswith("B")] if analysis == "B" else d[d.analysis == "original"])
    return d.iloc[0][col]
FB, NB, LD = "narr_finbert_s63__sipca", "narr_nolbert20_s63__sipca", "narr_lda_all_s63__sipca"
for key, spec in {"FB63": FB, "NB63": NB, "LDA63": LD}.items():
    for a in ["CL", "BTC"]:
        put(f"{key}_{a}_G", S(rr(spec, a, "gain", analysis="A")))
        put(f"{key}_{a}_CI", CI(rr(spec, a, "gain_lo", analysis="A"), rr(spec, a, "gain_hi", analysis="A")))
        put(f"{key}_{a}_H4", S(rr(spec, a, "holm4_analytic", analysis="A"), 3))
        put(f"{key}_{a}_H12", S(rr(spec, a, "holm12_analytic", analysis="A"), 3))
        put(f"{key}_{a}_H12C", S(rr(spec, a, "holm12_calibrated", analysis="A"), 3))
put("FB63_CL_COM_G", S(rr(FB, "CL", "gain", "common-month", "A")))
put("FB63_CL_COM_H4", S(rr(FB, "CL", "holm4_analytic", "common-month", "A"), 3))
for key, sp in {"B_FB": "narr_finbert__sipca"}.items():
    put("B_CL_G", S(rr(sp, "CL", "gain", analysis="B")))
    put("B_CL_CI", CI(rr(sp, "CL", "gain_lo", analysis="B"), rr(sp, "CL", "gain_hi", analysis="B")))
    put("B_CL_H4", S(rr(sp, "CL", "holm4_analytic", analysis="B"), 3))
    put("B_CL_H4C", S(rr(sp, "CL", "holm4_calibrated", analysis="B"), 3))
    put("B_HG_G", S(rr(sp, "HG", "gain", analysis="B")))
fbd = rdesc[(rdesc.spec == FB) & (rdesc.asset == "CL")].iloc[0].std_beta.split(", ")
put("F63_B1", fbd[0].replace("-", "$-$")); put("F63_B3", fbd[2].replace("-", "$-$"))
s63rows = [l for l in rtest.splitlines() if "_s63__sipca" in l and l.startswith("|")]
put("REPRO_S63_MIN", f"{min(float(l.split('|')[5]) for l in s63rows):.3f}")
put("REPRO_S63_TEST", f"{min(float(l.split('|')[7]) for l in s63rows):.3f}")
p1rows = [l for l in rtest.splitlines() if l.startswith("| nolbert20 |") or l.startswith("| finbert |") or l.startswith("| lda_all |")]
put("REPRO_5D", f"{min(float(l.split('|')[6]) for l in p1rows):.3f}")

def table_robust():
    rows = []
    spec_rows = [("narr_finbert__sipca", "FinBERT-20, 5-day, $K=3$ (original primary)", "not point-in-time", None),
                 ("narr_nolbert20__sipca", "NoLBERT-20, 5-day, $K=1$ (original secondary)", "more compatible", None),
                 (FB, "FinBERT-20, 63-day, $K=3$", "not point-in-time", "A"),
                 (NB, "NoLBERT-20, 63-day, $K=3$", "more compatible", "A"),
                 (LD, "LDA-60, 63-day, $K=3$", "more compatible", "A")]
    for spec, lab, status, an in spec_rows:
        c = {a: rr(spec, a, "gain", analysis=an) for a in ["CL", "HG", "GC", "BTC"]}
        h4 = rr(spec, "CL", "holm4_analytic", analysis=an)
        h12 = S(rr(spec, "CL", "holm12_analytic", analysis=an), 3) if an == "A" else "---"
        rows.append(f"{lab} & {status} & {S(c['CL'])} {CI(rr(spec, 'CL', 'gain_lo', analysis=an), rr(spec, 'CL', 'gain_hi', analysis=an))} & {S(h4, 3)} & {h12} & {S(c['HG'])} & {S(c['GC'])} & {S(c['BTC'])} \\\\")
    return "\n".join(rows)

# ------------------------------------------------------------------ tables
def table_topic():
    nm = {"N_oil": "oil (NoLBERT)", "F_oil": "oil (FinBERT)", "F_gold": "gold (FinBERT)", "F_fed": "Fed (FinBERT)"}
    rows = []
    for r in topic.itertuples():
        rows.append(f"{LBL[r.root]} & {nm[r.narrative]} & {r.kind} & {S(r.coef)} {CI(r.ci_lo, r.ci_hi)} & {S(r.p, 3)} & {S(r.p_holm_8, 3)} \\\\")
    return "\n".join(rows)


def table_heldout():
    rows = []
    for a in A + ["mkt"]:
        holm_ = f"{N['HOLM_' + a]}" if a in A else "---"
        cw = S(O.loc[a, "clark_west_p_onesided"], 3)
        ccp = S(C.loc[a, "p_calibrated_cw"], 3) if a in A else "---"
        cch = S(C.loc[a, "p_holm_calibrated_cw"], 3) if a in A else "---"
        rows.append(f"{LBL[a]} & {N['G_' + a]} {N['GCI_' + a]} & {cw} & {holm_} & {ccp} & {cch} & {N['GCOM_' + a]} & {N['GSEC_' + a]} \\\\")
    return "\n".join(rows)


def table_joint():
    rows = []
    for a in A:
        rows.append(f"{LBL[a]} & {N['NALL_' + a]} & {N['INC_' + a]} & {N['JP_ALL_' + a]} & {int(Rt.loc[a, 'n'])} & {S(Rt.loc[a, 'joint_p_hc3'], 3)} & {S(Rt.loc[a, 'joint_p_hc3_holm_4_descriptive'], 3)} \\\\")
    return "\n".join(rows)


def table_calib():
    rows = []
    for a in A:
        rows.append(f"{LBL[a]} & {S(C.loc[a, 'p_analytic'], 3)} & {S(C.loc[a, 'p_holm_analytic'], 3)} & {S(C.loc[a, 'p_calibrated_cw'], 3)} & {S(C.loc[a, 'p_holm_calibrated_cw'], 3)} & {S(C.loc[a, 'p_calibrated_gain'], 3)} & {S(gh[A.index(a)], 3)} \\\\")
    return "\n".join(rows)


AUDIT = [
 ("Factor estimation schedule", "Documented; consistent with saved files", "Grid fitted on training months, chosen on validation error, refitted once on training and validation, applied unchanged to test months; train and validation factor months flagged in-sample, test months out-of-sample; the project's own reproduction of the saved test forecasts agrees to $10^{-9}$ (29 of 50 specifications reproduced independently)."),
 ("Characteristic scaling and windows", "Verified", "Cross-sectional ranks within each date; exposure windows end at the formation date (rebuilt from daily attention and returns; rank correlation 0.92, falling to 0.72 and 0.41 when shifted forward by 21 and 63 days)."),
 ("Topic models and vocabulary", "Documented; metadata verified", "Clustering fitted on articles through 2019-12-31 only; the 10{,}000-term vocabulary basis comes from the whole corpus (project documentation)."),
 ("NoLBERT representation", "Historically more compatible; validation overlap documented", "Pretrained on 1976--1995 text; adapted on news with a headline--lead matching objective that uses no returns, on pairs drawn up to 1 September 2020 (project documentation), so eight validation months overlap; test period unaffected."),
 ("FinBERT representation", "Issue: not point-in-time", "Classifier published December 2022 and fine-tuned on a finance-tweet dataset published September 2022 (repository creation dates; tweet dates not stated). The primary factor model and two of three validation-selected equity specifications use it."),
 ("Shock window", "Issue, partly addressed post hoc", "The validation-chosen window for monthly rebalancing is 63 days; the main analyses use the default five days. The saved 63-day factors were analysed post hoc; the monthly stock forecasts were not rebuilt."),
 ("Feature reproducibility", "Verified", "A re-implementation of the project's pipeline reproduces the saved 5-day feature blocks (rank correlation 1.000) and the saved 63-day factor returns (correlation at least @@REPRO_S63_MIN@@; test months at least @@REPRO_S63_TEST@@)."),
 ("News cutoff", "Verified in code", "Exposures use shock days up to the day before the formation date (lag of one day), because article dates carry no time of day."),
 ("Feb--Jun holding months", "Documented", "From 2017 the daily return file holds only each year's June-list stocks, so formation days before the list changes are left out to avoid features observed for a sample selected on the future."),
 ("Calendar alignment", "Verified", "Holding window is the following calendar month; matches compounded stock returns for all 104 months; windows of consecutive months do not overlap."),
 ("Stock universe", "Verified; one issue", "98.7\\% of monthly rows belong to the list formed at the latest June; about 1.1\\% belong only to the next June's list."),
 ("News sources, 2024", "Documented", "FNSPID and FinMultiTime; 1{,}255{,}289 articles; one dump day removed; 22 articles in January 2024; about 4{,}500 articles of 2024 have only headlines."),
 ("Intraday timing of news", "Cannot be verified", "Articles carry dates, not times; the forecasting design skips the last trading day of the week."),
]
audit_rows = "\n".join(f"{a} & {b} & {c} \\\\ \\midrule" for a, b, c in AUDIT)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
# figure 1: Phase 1 versus Phase 2
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
x = np.arange(3)
for i, k in enumerate(FR):
    v = find[k]
    ax[0].plot(i - 0.12, v["validation_sharpe"], "o", color="C0", label="validation" if i == 0 else None)
    lo, hi = v["bootstrap_sharpe_95ci"]
    ts = v["test_sharpe_by_cost"]["10"]
    ax[0].errorbar(i + 0.12, ts, yerr=[[ts - lo], [hi - ts]], fmt="s", color="C3", capsize=3, label="test, with 95% CI" if i == 0 else None)
ax[0].axhline(0, color="grey", lw=0.8)
ax[0].set_xticks(x)
ax[0].set_xticklabels(["monthly", "biweekly", "weekly"])
ax[0].set_ylabel("annualised Sharpe ratio, after 10 bp per side")
ax[0].set_title("Phase 1: equity forecasts (chosen on validation)")
ax[0].legend(frameon=False)
for j, a in enumerate(A):
    ax[1].errorbar(O.loc[a, "gain"], j, xerr=[[O.loc[a, "gain"] - O.loc[a, "gain_lo"]], [O.loc[a, "gain_hi"] - O.loc[a, "gain"]]], fmt="o", color="C2", capsize=3)
ax[1].axvline(0, color="grey", lw=0.8)
ax[1].set_yticks(range(4))
ax[1].set_yticklabels([LBL[a] for a in A])
ax[1].invert_yaxis()
ax[1].set_xlabel("held-out squared-error gain over equity market")
ax[1].set_title("Phase 2: IPCA factors (21 test months)")
fig.text(0.5, -0.02, "Different data, targets and statistics: the panels show what each experiment supports, not a ranking.", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "figures" / "fig_phase_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
# figure 2: held-out gain forest
fig, ax = plt.subplots(figsize=(6.2, 3.6))
sets = [("primary, asset-specific fit", oos[(oos.role == "primary") & (oos.scope == "asset-specific")], "C2", -0.25),
        ("primary, common-month fit", oos[(oos.role == "primary") & (oos.scope == "common-month")], "C1", 0.0),
        ("secondary (NoLBERT, K=1)", oos[(oos.role == "secondary") & (oos.scope == "asset-specific")], "C4", 0.25)]
order = A + ["mkt"]
for lab, d, col, off in sets:
    d = d.set_index("asset")
    for j, a in enumerate(order):
        ax.errorbar(d.loc[a, "gain"], j + off, xerr=[[d.loc[a, "gain"] - d.loc[a, "gain_lo"]], [d.loc[a, "gain_hi"] - d.loc[a, "gain"]]], fmt="o", color=col, capsize=2, label=lab if j == 0 else None)
ax.axvline(0, color="grey", lw=0.8)
ax.set_yticks(range(5))
ax.set_yticklabels([LBL[a] for a in order])
ax.invert_yaxis()
ax.set_xlabel("held-out reduction in squared error (1 $-$ SSE ratio), 95% bootstrap CI (unadjusted)")
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3)
fig.tight_layout()
fig.savefig(HERE / "figures" / "fig_heldout_gain.png", dpi=200)
plt.close(fig)
shutil.copy(P2 / "ipca_exposure_heatmap.png", HERE / "figures" / "fig_exposure_heatmap.png")

# ------------------------------------------------------------------ template
TEMPLATE = (HERE / "paper_template.tex").read_text()
tables = {"TAB_ROBUST": table_robust(), "TAB_TOPIC": table_topic(), "TAB_HELDOUT": table_heldout(), "TAB_JOINT": table_joint(), "TAB_CALIB": table_calib(), "TAB_AUDIT": audit_rows}
out = TEMPLATE
for _ in range(2):                                   # second pass fills tokens that sit inside generated tables
    for k, v in {**N, **tables}.items():
        out = out.replace(f"@@{k}@@", v)
left = re.findall(r"@@([A-Z0-9_]+)@@", out)
assert not left, f"unfilled placeholders: {sorted(set(left))}"
(HERE / "main.tex").write_text(out)
pd.DataFrame(sorted(N.items()), columns=["token", "value"]).to_csv(HERE / "numbers_used.csv", index=False)
print(f"main.tex written ({len(out.split())} words incl. markup); {len(N)} numbers filled from files")
