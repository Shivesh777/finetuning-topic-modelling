"""Calibration check of the Clark-West statistic used in ipca_regressions.py (a methodology check, not a new analysis).

The statistic, the data, the split, the fixed-scheme estimation and the models are exactly those of ipca_regressions.py (primary model).
What is new is only the reference distribution: under the null that the factors add nothing, returns are generated as
y* = fitted benchmark(market) + wild-bootstrap (Rademacher) residuals, keeping the observed market and factor values fixed. For each draw the whole
procedure is repeated (fit M0 and M2 on the pre-test months, apply to the 21 test months). This gives (i) the actual rejection rate of the analytic
one-sided Clark-West p-value at nominal levels and (ii) simulation-calibrated p-values for the observed statistics.

    python3 ipca_cw_calibration.py  ->  ipca_cw_calibration.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
panel = pd.read_csv(HERE / "ipca_monthly_panel.csv", parse_dates=["date"])
FC = [c for c in panel if c.startswith("pri_f")]
ASSETS = ["CL", "GC", "HG", "BTC"]
DRAWS, SEED = 5000, 0
common = panel[ASSETS].notna().all(axis=1)
rng = np.random.default_rng(SEED)


def procedure(y_fit, y_test, X0f, X0t, X2f, X2t):
    b0 = np.linalg.lstsq(X0f, y_fit, rcond=None)[0]
    b2 = np.linalg.lstsq(X2f, y_fit, rcond=None)[0]
    p0, p2 = X0t @ b0, X2t @ b2
    e0, e2 = y_test - p0, y_test - p2
    gain = 1 - (e2 ** 2).sum() / (e0 ** 2).sum()
    d = e0 ** 2 - (e2 ** 2 - (p0 - p2) ** 2)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return gain, t


def holm(p):
    o = np.argsort(p)
    out = np.empty(len(p))
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        out[i] = min(1.0, run)
    return out


rows = []
for scope in ["asset-specific", "common-month"]:
    for a in ASSETS:
        fit_m = panel.split.isin(["train", "validation"])
        test_m = panel.split == "test"
        if scope == "common-month":
            fit_m &= common
            test_m &= common
        else:
            fit_m &= panel[a].notna()
            test_m &= panel[a].notna()
        fit, test = panel[fit_m], panel[test_m]
        ones = lambda d: np.ones((len(d), 1))
        X0f, X0t = np.c_[ones(fit), fit.mkt], np.c_[ones(test), test.mkt]
        X2f, X2t = np.c_[ones(fit), fit.mkt, fit[FC]], np.c_[ones(test), test.mkt, test[FC]]
        y_f, y_t = fit[a].values, test[a].values
        g_obs, t_obs = procedure(y_f, y_t, X0f, X0t, X2f, X2t)
        # null: benchmark fitted on all months (fit + test), wild bootstrap residuals
        Xall0 = np.vstack([X0f, X0t])
        yall = np.r_[y_f, y_t]
        beta0 = np.linalg.lstsq(Xall0, yall, rcond=None)[0]
        mu0 = Xall0 @ beta0
        res = yall - mu0
        n_f = len(y_f)
        g_null, t_null = np.empty(DRAWS), np.empty(DRAWS)
        for b in range(DRAWS):
            ys = mu0 + res * rng.choice([-1.0, 1.0], len(res))
            g_null[b], t_null[b] = procedure(ys[:n_f], ys[n_f:], X0f, X0t, X2f, X2t)
        p_analytic_null = stats.t.sf(t_null, len(y_t) - 1)
        rows.append(dict(scope=scope, asset=a, n_fit=n_f, n_test=len(y_t), gain_obs=g_obs, cw_t_obs=t_obs,
                         p_analytic=stats.t.sf(t_obs, len(y_t) - 1),
                         p_calibrated_cw=float((1 + (t_null >= t_obs).sum()) / (1 + DRAWS)),
                         p_calibrated_gain=float((1 + (g_null >= g_obs).sum()) / (1 + DRAWS)),
                         null_gain_95pct=float(np.quantile(g_null, 0.95)), null_gain_median=float(np.median(g_null)),
                         size_at_5pct=float((p_analytic_null <= 0.05).mean()), size_at_2_5pct=float((p_analytic_null <= 0.025).mean()),
                         size_at_1pct=float((p_analytic_null <= 0.01).mean())))
out = pd.DataFrame(rows)
for scope, g in out.groupby("scope"):
    out.loc[g.index, "p_holm_calibrated_cw"] = holm(g.p_calibrated_cw.values)
    out.loc[g.index, "p_holm_analytic"] = holm(g.p_analytic.values)
out.to_csv(HERE / "ipca_cw_calibration.csv", index=False)
pd.set_option("display.width", 220)
print(out.round(3).to_string(index=False))
