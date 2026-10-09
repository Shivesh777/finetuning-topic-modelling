"""Bitcoin exploratory pilot, exactly as fixed in btc_prereg.md, using the functions of pilot.py unchanged.
Does not touch weekly_panel.parquet, pilot_results.csv or pilot_results.md.

    python3 pilot_btc.py -> weekly_panel_btc.parquet, btc_results.csv, btc_results.md, cross_asset_primary.csv
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import build_weekly_panel as bw
import pilot as p

HERE = Path(__file__).resolve().parent
PREREG = hashlib.sha256((HERE / "btc_prereg.md").read_bytes()).hexdigest()
M_FAMILY = 8                      # primary tests in the cross-asset family: CL x4, GC x2, BTC x2

# ---- panel (extra FinBERT classes registered here; the CL/GC panel does not use them)
bw.NARRATIVES.update({"F_macro": ("finbert", ["t14"]), "F_curr": ("finbert", ["t08"])})
btc = bw.build_root("BTC")
btc = btc[btc.L >= "2018-01-01"].copy()                    # development starts 2018-01
dev = btc[btc.window == "dev"]
for c in [c for c in btc if c.endswith("_std")]:
    base = c[:-4]
    btc[c] = btc[base] / dev[base].std()
btc.to_parquet(HERE / "weekly_panel_btc.parquet", index=False)
p.panel = btc

rng = np.random.default_rng(p.SEED)
rows = []
for kind in ["contemporaneous", "forecasting"]:
    rows += p.analyse("BTC", "F_fed", kind, "primary", rng)
    for n in ["F_macro", "F_curr"]:
        rows += p.analyse("BTC", n, kind, "secondary", rng)
    rows += p.analyse("BTC", "F_oil", kind, "cross-narrative control", rng)
    rows += p.analyse("BTC", "F_fed", kind, "sensitivity: drop flagged weeks", rng, drop_flagged=True, tag="drop_flagged")
res = pd.DataFrame(rows)
res["prereg_sha256"] = PREREG
res.to_csv(HERE / "btc_results.csv", index=False)

# ---- the eight-test primary family, with adjusted p and adjusted intervals
old = pd.read_csv(HERE / "pilot_results.csv")
fam = pd.concat([old, res], ignore_index=True)
fam = fam[(fam.role == "primary") & (fam.stage == "test: own-window OLS")].copy()
fam["p_holm_8"] = p.holm(fam.p.values)
z8 = stats.norm.ppf(1 - 0.025 / M_FAMILY)
fam["ci_bonf8_lo"], fam["ci_bonf8_hi"] = fam.coef - z8 * fam.se, fam.coef + z8 * fam.se
bt = fam[fam.root == "BTC"]
fam["p_holm_btc_only"] = np.nan
fam.loc[bt.index, "p_holm_btc_only"] = p.holm(bt.p.values)
fam.to_csv(HERE / "cross_asset_primary.csv", index=False)

f = p.fmt
L = [f"# Bitcoin pilot (exploratory)\n\nSpecification: `btc_prereg.md` (SHA-256 `{PREREG}`). Same construction as the CL/GC pilot. "
     f"Coefficients are percentage points of weekly return per 1 development-sample standard deviation of the shock; HAC errors, 4 lags. "
     f"Weeks: development {int((btc.window == 'dev').sum())}, validation {int((btc.window == 'val').sum())}, test {int((btc.window == 'test').sum())}.\n"]
for kind in ["contemporaneous", "forecasting"]:
    L.append(f"\n## {kind.capitalize()}: own-window regressions\n")
    L.append("| narrative | role | window | n | coef | 95% CI (unadjusted) | p | adj R2 | adj R2 controls only |\n|---|---|---|---|---|---|---|---|---|")
    for r in res[(res.kind == kind) & res.stage.str.contains("own-window") & res.variant.eq("main")].itertuples():
        L.append(f"| {r.narrative} | {r.role} | {r.stage.split(':')[0]} | {r.n} | {f(r.coef)} | [{f(r.ci_lo)}, {f(r.ci_hi)}] | {f(r.p)} | {f(r.adj_r2)} | {f(r.adj_r2_controls)} |")
    L.append(f"\n## {kind.capitalize()}: frozen-coefficient out-of-sample\n")
    L.append("| narrative | role | stage | n | OOS R2 vs controls-only [95% block-bootstrap CI] |" + (" OOS R2 vs expanding mean | hit rate (full / controls) |" if kind == "forecasting" else ""))
    L.append("|---|---|---|---|---|" + ("---|---|" if kind == "forecasting" else ""))
    for r in res[(res.kind == kind) & res.stage.str.contains("applied") & res.variant.eq("main")].itertuples():
        tail = f" {f(r.oos_r2_vs_hist_mean)} | {f(r.hit_rate_full, 2)} / {f(r.hit_rate_controls, 2)} |" if kind == "forecasting" else ""
        L.append(f"| {r.narrative} | {r.role} | {r.stage.split(':')[0]} | {r.n} | {f(r.oos_r2_vs_controls)} [{f(r.oos_r2_vs_controls_lo)}, {f(r.oos_r2_vs_controls_hi)}] |" + tail)
L.append("\n## Timing check (test window)\n\n| narrative | role | coef | 95% CI | p |\n|---|---|---|---|---|")
for r in res[res.stage.str.contains("timing") & res.variant.eq("main")].itertuples():
    L.append(f"| {r.narrative} | {r.role} | {f(r.coef)} | [{f(r.ci_lo)}, {f(r.ci_hi)}] | {f(r.p)} |")
L.append("\n## Sensitivity: drop weeks with Databento degraded days (test window)\n\n| kind | coef | p | OOS R2 vs controls-only |\n|---|---|---|---|")
for k in ["contemporaneous", "forecasting"]:
    a = res[(res.kind == k) & res.variant.eq("drop_flagged") & res.stage.eq("test: own-window OLS")].iloc[0]
    b = res[(res.kind == k) & res.variant.eq("drop_flagged") & res.stage.str.startswith("test") & res.stage.str.contains("applied")].iloc[0]
    L.append(f"| {k} | {f(a.coef)} | {f(a.p)} | {f(b.oos_r2_vs_controls)} |")
L.append("\n## Primary tests, test window\n\n| asset | narrative | kind | coef | 95% CI unadjusted | 95% CI Bonferroni (8 tests) | raw p | Holm p (BTC only, 2) | Holm p (all 8) |\n|---|---|---|---|---|---|---|---|---|")
for r in fam[fam.root == "BTC"].itertuples():
    L.append(f"| BTC | {r.narrative} | {r.kind} | {f(r.coef)} | [{f(r.ci_lo)}, {f(r.ci_hi)}] | [{f(r.ci_bonf8_lo)}, {f(r.ci_bonf8_hi)}] | {f(r.p)} | {f(r.p_holm_btc_only)} | {f(r.p_holm_8)} |")
(HERE / "btc_results.md").write_text("\n".join(L))
print(f"btc prereg sha256 {PREREG}")
print("\n".join(L))
