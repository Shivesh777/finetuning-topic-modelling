"""Phase 1A: statistical robustness of the saved equity forecasts.

For every saved experiment (26 feature sets x 15 models = 390) at every frequency, on the
validation and test splits, this builds two equal-weighted long-short portfolios:

  decile   long the top 10% of forecasts, short the bottom 10%, rebuilt every period
  buffer   enter the top / bottom 10%, keep a name until it leaves the top / bottom 20%

and reports mean, volatility, Sharpe, maximum drawdown and turnover, with costs of 0, 5, 10
and 20 basis points per side. It then compares one model chosen on the validation split
with every other specification, the saved shuffled-topic placebos and a cross-sectional
permutation null, computes deflated Sharpe ratios, and bootstraps confidence intervals.

    python3 robustness.py

Reads ../../data; writes into this folder. All settings are in CONFIG and are saved to
config.json next to the outputs.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"

CONFIG = {
    "frequencies": {"M": 12, "2W": 26, "W": 52},          # periods per year, for annualising
    "decile": 0.10,                                         # entry band
    "buffer_exit": 0.20,                                    # buffer strategy exit band
    "costs_bps_per_side": [0, 5, 10, 20],
    "selection": {                                          # fixed before looking at the test split
        "strategy": "buffer",
        "cost_bps": 10,
        "split": "validation",
        "metric": "sharpe",
    },
    "permutation_draws": 1000,
    "bootstrap_draws": 10000,
    "bootstrap_block": {"M": 3, "2W": 4, "W": 8},          # about two months of returns per block
    "seed": 0,
}
COSTS = CONFIG["costs_bps_per_side"]


# ---------------------------------------------------------------- portfolios

def split_frame(freq, split):
    idx = pd.read_parquet(DATA / "panels" / freq / "index.parquet")
    s = idx[idx.split == split].reset_index(drop=True)
    groups = [(d, g.index.values) for d, g in s.groupby("date", sort=True)]
    return s, groups


def run_strategy(pred, frame, groups, kind):
    """Period returns (gross) and turnover sum|dw| of an equal-weighted long-short book."""
    permno, ret = frame.permno.values, frame.ret.values
    lo_in, hi_in = CONFIG["decile"], 1 - CONFIG["decile"]
    lo_out, hi_out = CONFIG["buffer_exit"], 1 - CONFIG["buffer_exit"]
    longs, shorts = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    old_w = {}
    out_ret, out_turn = [], []
    ds = np.array([g[0] for g in groups], dtype="datetime64[D]")
    gaps = np.diff(ds).astype(float)
    gap_limit = 1.6 * np.median(gaps)                       # a missing stretch of the panel, not a normal step
    for k, (_, rows) in enumerate(groups):
        reset = k > 0 and gaps[k - 1] > gap_limit
        if reset:                                           # the book is closed before the gap, rebuilt after it
            longs, shorts = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        p, ids, r = pred[rows], permno[rows], ret[rows]
        pct = (stats.rankdata(p, method="ordinal")) / len(p)
        if kind == "decile":
            is_long, is_short = pct > hi_in, pct <= lo_in
        else:
            is_long = (pct > hi_in) | (np.isin(ids, longs) & (pct > hi_out))
            is_short = (pct <= lo_in) | (np.isin(ids, shorts) & (pct <= lo_out))
        longs, shorts = ids[is_long], ids[is_short]
        out_ret.append(r[is_long].mean() - r[is_short].mean())
        new_w = dict(zip(longs, np.full(len(longs), 1 / len(longs))))
        new_w.update(zip(shorts, np.full(len(shorts), -1 / len(shorts))))
        keys = new_w.keys() | old_w.keys()
        t = sum(abs(new_w.get(q, 0) - old_w.get(q, 0)) for q in keys)
        if reset:
            t += sum(abs(v) for v in old_w.values())        # closing the old book
        out_turn.append(t)
        old_w = new_w
    return np.array(out_ret), np.array(out_turn)


def metrics(r, ann):
    sd = r.std(ddof=1)
    wealth = np.cumprod(1 + r)
    dd = 1 - wealth / np.maximum.accumulate(np.maximum(wealth, 1))
    return {
        "mean_ann": r.mean() * ann,
        "vol_ann": sd * np.sqrt(ann),
        "sharpe": r.mean() / sd * np.sqrt(ann) if sd > 0 else np.nan,
        "max_drawdown": dd.max(),
        "skew": stats.skew(r),
        "kurtosis": stats.kurtosis(r, fisher=False),
    }


# ---------------------------------------------------------------- statistics

def deflated_sharpe(sr, n_obs, skew, kurt, n_trials, var_sr):
    """Bailey and Lopez de Prado (2014). All Sharpe ratios are per period, not annualised."""
    g = 0.5772156649
    if n_trials <= 1:
        sr0 = 0.0
    else:
        sr0 = np.sqrt(var_sr) * ((1 - g) * stats.norm.ppf(1 - 1 / n_trials)
                                 + g * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(n_obs - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return stats.norm.cdf(z), sr0


def effective_trials(returns):
    """Participation ratio of the eigenvalues of the trials' return correlations."""
    c = np.corrcoef(returns)
    c = np.nan_to_num(c)
    lam = np.clip(np.linalg.eigvalsh(c), 0, None)
    return lam.sum() ** 2 / (lam ** 2).sum()


def block_bootstrap(series, block, draws, rng, stat):
    """Circular moving-block bootstrap of stat(*resampled series), same blocks for all series."""
    n = len(series[0])
    k = int(np.ceil(n / block))
    out = np.empty(draws)
    for b in range(draws):
        starts = rng.integers(0, n, k)
        ix = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        out[b] = stat(*[s[ix] for s in series])
    return out


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    rng = np.random.default_rng(CONFIG["seed"])
    (HERE / "config.json").write_text(json.dumps(CONFIG, indent=2))
    rows, series = [], {}

    for freq, ann in CONFIG["frequencies"].items():
        exps = sorted({p.name.split(".")[0] for p in (DATA / "predictions" / freq).glob("*.npy")})
        for split in ["validation", "test"]:
            frame, groups = split_frame(freq, split)
            for exp in exps:
                pred = np.load(DATA / "predictions" / freq / f"{exp}.{split}.npy").astype(float)
                fs, model = exp.split("__")
                for kind in ["decile", "buffer"]:
                    r, turn = run_strategy(pred, frame, groups, kind)
                    row = {"freq": freq, "split": split, "experiment": exp, "feature_set": fs,
                           "model": model, "strategy": kind, "periods": len(r),
                           "first_date": str(groups[0][0])[:10], "last_date": str(groups[-1][0])[:10],
                           "turnover_per_period": turn.mean(), "turnover_ann": turn.mean() * ann}
                    for c in COSTS:
                        m = metrics(r - turn * c / 1e4, ann)
                        if c == 0:
                            row.update(m)
                        row[f"sharpe_{c}bp"] = m["sharpe"]
                        row[f"mean_ann_{c}bp"] = m["mean_ann"]
                    rows.append(row)
                    series[(freq, split, exp, kind)] = (r, turn)
            print(f"{freq} {split}: {len(exps)} experiments done ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df["placebo"] = df.feature_set.str.contains("placebo")
    df["text_model"] = df.feature_set.str.split("_", n=1).str[1].fillna("none")
    df["block"] = df.feature_set.str.split("_").str[0]

    sel = CONFIG["selection"]
    col = f"{sel['metric']}_{sel['cost_bps']}bp"
    findings = {}
    for freq, ann in CONFIG["frequencies"].items():
        v = df[(df.freq == freq) & (df.split == sel["split"]) & (df.strategy == sel["strategy"]) & ~df.placebo]
        chosen = v.loc[v[col].idxmax(), "experiment"]
        test = df[(df.freq == freq) & (df.split == "test") & (df.strategy == sel["strategy"])]
        real, plac = test[~test.placebo], test[test.placebo]
        ch = test[test.experiment == chosen].iloc[0]

        # permutation null: give every stock another stock's whole forecast history (one fixed relabelling
        # per draw). This keeps each forecast's persistence, hence its turnover and costs, and the
        # cross-sectional spread, and removes only the link between forecasts and that stock's returns.
        frame, groups = split_frame(freq, "test")
        r_c, t_c = series[(freq, "test", chosen, sel["strategy"])]
        pred_c = np.load(DATA / "predictions" / freq / f"{chosen}.test.npy").astype(float)
        d_idx = pd.factorize(frame.date, sort=True)[0]
        p_idx, p_uni = pd.factorize(frame.permno)
        grid = np.full((d_idx.max() + 1, len(p_uni)), np.nan)
        grid[d_idx, p_idx] = pred_c
        date_med = np.nanmedian(grid, axis=1)
        null = np.empty(CONFIG["permutation_draws"])
        null_gross = np.empty(CONFIG["permutation_draws"])
        for b in range(len(null)):
            fake = grid[d_idx, rng.permutation(len(p_uni))[p_idx]]
            fake = np.where(np.isnan(fake), date_med[d_idx], fake)   # stock absent in the donor's date
            fr, ft = run_strategy(fake, frame, groups, sel["strategy"])
            null[b] = metrics(fr - ft * sel["cost_bps"] / 1e4, ann)["sharpe"]
            null_gross[b] = metrics(fr, ann)["sharpe"]

        # deflated Sharpe: for the chosen model (selected out of sample) and the best-in-test pick
        nets = np.array([series[(freq, "test", e, sel["strategy"])][0]
                         - series[(freq, "test", e, sel["strategy"])][1] * sel["cost_bps"] / 1e4
                         for e in real.experiment])
        sr_pp = nets.mean(1) / nets.std(1, ddof=1)
        var_sr = sr_pp.var(ddof=1)
        n_eff = effective_trials(nets)
        T = nets.shape[1]

        def dsr_for(exp, n_trials):
            x = series[(freq, "test", exp, sel["strategy"])]
            x = x[0] - x[1] * sel["cost_bps"] / 1e4
            sr = x.mean() / x.std(ddof=1)
            return deflated_sharpe(sr, len(x), stats.skew(x), stats.kurtosis(x, fisher=False), n_trials, var_sr)[0]

        best = real.loc[real[col].idxmax(), "experiment"]
        # bootstrap: chosen model's Sharpe, and its Sharpe minus the price-only model of the same type
        x = r_c - t_c * sel["cost_bps"] / 1e4
        pr = series[(freq, "test", f"price__{chosen.split('__')[1]}", sel["strategy"])]
        y = pr[0] - pr[1] * sel["cost_bps"] / 1e4
        shr = lambda a: a.mean() / a.std(ddof=1) * np.sqrt(ann)
        bs = block_bootstrap([x], CONFIG["bootstrap_block"][freq], CONFIG["bootstrap_draws"], rng, shr)
        bd = block_bootstrap([x, y], CONFIG["bootstrap_block"][freq], CONFIG["bootstrap_draws"], rng,
                             lambda a, b: shr(a) - shr(b))

        findings[freq] = {
            "chosen_on_validation": chosen,
            "validation_sharpe": float(v.loc[v.experiment == chosen, col].iloc[0]),
            "test_sharpe_by_cost": {c: float(ch[f"sharpe_{c}bp"]) for c in COSTS},
            "test_turnover_per_period": float(ch.turnover_per_period),
            "test_max_drawdown": float(ch.max_drawdown),
            "test_periods": int(T),
            "pct_rank_among_real_specs": float((real[col] < ch[col]).mean()),
            "share_real_specs_beating_placebo_max": float((real[col] > plac[col].max()).mean()),
            "placebo_sharpe_range": [float(plac[col].min()), float(plac[col].max())],
            "real_sharpe_median": float(real[col].median()),
            "permutation_p_value": float((1 + (null >= ch[col]).sum()) / (1 + len(null))),
            "permutation_null_95pct": float(np.quantile(null, 0.95)),
            "permutation_p_value_gross": float((1 + (null_gross >= ch["sharpe"]).sum()) / (1 + len(null))),
            "best_in_test": best,
            "best_in_test_sharpe": float(real[col].max()),
            "n_trials_raw": int(len(real)),
            "n_trials_effective": float(n_eff),
            "dsr_chosen_n1": float(dsr_for(chosen, 1)),
            "dsr_best_in_test": {"N_eff": float(dsr_for(best, n_eff)), "N_26": float(dsr_for(best, 26)),
                                 "N_raw": float(dsr_for(best, len(real)))},
            "bootstrap_sharpe_95ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "bootstrap_diff_vs_price_95ci": [float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975))],
            "price_model_same_type_sharpe": float(shr(y)),
            "_null": null.tolist(),
        }
        print(f"{freq}: chosen {chosen}, test Sharpe {ch[col]:.2f} ({time.time() - t0:.0f}s)")

    df.to_csv(HERE / "robustness_summary.csv", index=False)
    (HERE / "findings.json").write_text(json.dumps(
        {f: {k: v for k, v in d.items() if k != "_null"} for f, d in findings.items()}, indent=2))
    plot_placebo(df, findings, col)
    plot_costs(df, findings)
    print(f"done in {time.time() - t0:.0f}s")


def plot_placebo(df, findings, col):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, (freq, f) in zip(axes, findings.items()):
        t = df[(df.freq == freq) & (df.split == "test") & (df.strategy == CONFIG["selection"]["strategy"])]
        bins = np.linspace(min(t[col].min(), min(f["_null"])) - 0.1, max(t[col].max(), max(f["_null"])) + 0.1, 30)
        ax.hist(f["_null"], bins=bins, density=True, alpha=0.35, color="grey", label="relabelling null (chosen model)")
        ax.hist(t[~t.placebo][col], bins=bins, density=True, histtype="step", lw=2, color="C0",
                label="all real specifications")
        for i, v in enumerate(t[t.placebo][col]):
            ax.axvline(v, color="C3", lw=0.8, alpha=0.7, label="shuffled-topic placebos" if i == 0 else None)
        ch = t[t.experiment == f["chosen_on_validation"]][col].iloc[0]
        ax.axvline(ch, color="k", lw=2.5, label="chosen on validation")
        ax.set_title(f"{freq}: test Sharpe, buffer rule, 10 bp/side ({f['test_periods']} periods)")
        ax.set_xlabel("annualised Sharpe")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "placebo_comparison.png", dpi=150)


def plot_costs(df, findings):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (freq, f) in zip(axes, findings.items()):
        t = df[(df.freq == freq) & (df.split == "test") & (df.strategy == "buffer") & ~df.placebo]
        q = t[[f"sharpe_{c}bp" for c in COSTS]]
        ax.fill_between(COSTS, q.quantile(0.1), q.quantile(0.9), color="C0", alpha=0.15,
                        label="10th-90th pct of real specs")
        ax.plot(COSTS, q.median(), color="C0", label="median real spec")
        ch = t[t.experiment == f["chosen_on_validation"]].iloc[0]
        ax.plot(COSTS, [ch[f"sharpe_{c}bp"] for c in COSTS], "k-o", lw=2,
                label=f"chosen: {f['chosen_on_validation']}")
        b = t[t.experiment == f["best_in_test"]].iloc[0]
        ax.plot(COSTS, [b[f"sharpe_{c}bp"] for c in COSTS], "--", color="C3",
                label=f"best in test (hindsight): {f['best_in_test']}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{freq}: buffer rule, test split")
        ax.set_xlabel("cost, bp per side")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("annualised Sharpe")
    fig.tight_layout()
    fig.savefig(HERE / "performance_after_costs.png", dpi=150)


if __name__ == "__main__":
    main()
