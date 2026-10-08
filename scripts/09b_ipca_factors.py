"""Loadings (Gamma) and factor returns of the IPCA models, which the pipeline otherwise keeps
only as forecasts.

    python scripts/09b_ipca_factors.py [--freq M] [--models sipca ipca] [--features all_nolbert ...]

For each finished experiment of an IPCA model (default: sparse IPCA on every real feature set
of the monthly panel) the script refits the model that forecast the test period exactly as
scripts/06_run_models.py did: the setting that validation chose (results/tuning/), fitted on
the training and validation periods together. It checks that the refit reproduces the stored
test forecasts, then writes results/ipca/<freq>/<features>__<model>/:

    gamma.csv     Gamma: one row per feature (with its block), one column per factor; for the
                  sparse model, whether the group penalty kept the feature (a dropped feature
                  has a row of zeros)
    factors.csv   the factor return of every holding period: formation date, first day of the
                  holding period, split, f1 .. fK. Training and validation periods: the factors
                  estimated with Gamma (in sample). Test periods: the out-of-sample factors,
                  each period's cross-sectional regression of returns on Z_t Gamma with Gamma
                  fixed, the same formula as in the fit (a portfolio whose weights are known at
                  formation)
    summary.json  setting, periods, lambda (mean factor of the fit sample), the implied
                  coefficient of every feature (Gamma lambda, the forecast's slope),
                  normalisation and estimation schedule

Normalisation (both models): Gamma'Gamma = I; the factors orthogonal over the fit sample (the
sum of f_t f_t' is diagonal) and ordered by decreasing second moment, each with a positive
mean (Kelly, Pruitt and Su 2019). Plain IPCA is estimated under it. The sparse model's objective does not pin down a rotation of the
factors (its fit, ridge and group penalties are all unchanged by one), so its Gamma and factors
are put into the same form after the fit; that changes neither the forecasts (Gamma lambda) nor
which features are kept.

Returns are those the models are fitted on: next-period excess returns minus their mean across
the stocks of the period, in decimals per holding period. Nothing written is stock-level.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config, experiments
from ftm.models import ipca as ipca_module
from ftm.models.tuning import fit_for_test, get_model, target_of
from ftm.panel import Panel

IPCA_MODELS = ("sipca", "ipca", "sipca_rk", "ipca_rk")


def normalisation(gamma: np.ndarray, factors: np.ndarray):
    """(G, H) with Gamma G and factors H in the normalised form; Gamma G (factors H)' equals
    Gamma factors'. The transformation of ftm.models.ipca._normalise."""
    r = np.linalg.cholesky(gamma.T @ gamma).T
    u = np.linalg.svd(r @ (factors.T @ factors) @ r.T)[0]
    h = r.T @ u
    sign = np.where((factors @ h).mean(axis=0) < 0, -1.0, 1.0)
    return np.linalg.solve(r, u) * sign, h * sign


def out_of_sample(model, X: np.ndarray, y: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Factors of new periods with Gamma fixed, by the formula of the model's own fit."""
    S, q = ipca_module.period_moments(X, y, period)
    if isinstance(model, ipca_module.SparseIpca):
        counts = np.diff(np.flatnonzero(np.r_[True, period[1:] != period[:-1], True]))
        return model._factors(model.gamma_, S * counts[:, None, None], q * counts[:, None])
    return ipca_module._factors(model.gamma_, S, q, model.ridge)


def export(freq: str, features: str, model_name: str, out: Path) -> str:
    stem = f"{features}__{model_name}"
    meta = json.loads((config.RESULTS / "tuning" / freq / f"{stem}.json").read_text())
    panel = Panel.load(config.PANELS / freq, meta["blocks"])
    index = panel.index
    cls = get_model(model_name)
    period = index["period"].to_numpy()
    X = panel.X(meta["blocks"])
    y = target_of(cls, index["y_rel"].to_numpy(), period)
    train, val, test = (panel.rows(s) for s in ("train", "validation", "test"))
    model = fit_for_test(cls, meta["params"], X, y, train, val, groups=period)
    stored = np.load(experiments.prediction_path(freq, features, model_name, "test"))
    gap = float(np.abs(model.predict(X[test]) - stored).max())
    if gap > 1e-6 * max(1e-12, float(np.abs(stored).max())) + 1e-9:
        return f"skipped: the refit does not reproduce the stored test forecasts ({gap:.2e})"

    fit = train | val
    in_sample = model.factors_
    test_factors = out_of_sample(model, X[test], y[test], period[test])
    try:
        G, H = normalisation(model.gamma_, in_sample)
    except np.linalg.LinAlgError:                   # every feature dropped: nothing to rotate
        return "skipped: Gamma has no full column rank (the model keeps no feature)"
    gamma = model.gamma_ @ G
    k = gamma.shape[1]
    cols = [f"f{j + 1}" for j in range(k)]
    names = panel.feature_names(meta["blocks"])
    block_of = [b for b in meta["blocks"] for _ in panel.names[b]]
    loadings = pd.DataFrame(gamma, columns=cols).assign(feature=names, block=block_of)
    if hasattr(model, "active_"):
        loadings["kept"] = model.active_
    loadings = loadings[["feature", "block"] + (["kept"] if "kept" in loadings else []) + cols]

    dates = lambda rows: index.loc[rows, ["date", "hold_start", "split"]].drop_duplicates()
    factors = pd.concat([
        pd.concat([dates(fit).reset_index(drop=True),
                   pd.DataFrame(in_sample @ H, columns=cols)], axis=1).assign(estimate="in_sample"),
        pd.concat([dates(test).reset_index(drop=True),
                   pd.DataFrame(test_factors @ H, columns=cols)], axis=1).assign(estimate="out_of_sample"),
    ], ignore_index=True)

    lam = (in_sample @ H).mean(axis=0)
    # the normalisation must leave the forecast's slope (Gamma lambda) where it was
    drift = float(np.abs(gamma @ lam - model.coef_).max() / max(1e-12, np.abs(model.coef_).max()))
    if drift > 1e-6:
        return f"skipped: the normalisation moved Gamma lambda by {drift:.1e} (relative)"
    span = lambda rows: [str(index.loc[rows, "hold_start"].min().date()),
                         str(index.loc[rows, "hold_start"].max().date())]
    summary = {
        "freq": freq, "features": features, "model": model_name, "params": meta["params"],
        "factors": k, "features_total": len(names),
        "features_kept": int(loadings["kept"].sum()) if "kept" in loadings else len(names),
        "periods": {"train": int(index.loc[train, "period"].nunique()),
                    "validation": int(index.loc[val, "period"].nunique()),
                    "test": int(index.loc[test, "period"].nunique())},
        "holding_periods_start": {"train": span(train), "validation": span(val), "test": span(test)},
        "lambda": lam.tolist(),
        "implied_coefficients": dict(zip(names, (gamma @ lam).tolist())),
        "target": "next-period excess return minus its cross-sectional mean (y_rel), decimals",
        "features_scaling": "cross-sectional ranks scaled to [-1, 1] each period; no constant",
        "normalisation": "Gamma'Gamma = I; factors orthogonal over the fit sample (sum of f f' "
                         "diagonal), ordered by decreasing second moment, positive means "
                         "(Kelly, Pruitt and Su 2019)"
                         + ("; applied after the fit, which leaves forecasts and kept features "
                            "unchanged" if model_name.startswith("sipca") else ""),
        "schedule": "every setting of the grid fitted on the training periods and scored by mean "
                    "squared error on the validation periods; the chosen setting refitted once on "
                    "training plus validation periods and applied unchanged to the test periods "
                    "(no re-estimation inside the test period); forecast = z' Gamma lambda",
        "test_forecasts_reproduced_max_abs_error": gap,
    }
    folder = out / freq / stem
    folder.mkdir(parents=True, exist_ok=True)
    loadings.to_csv(folder / "gamma.csv", index=False, float_format="%.6g")
    factors.to_csv(folder / "factors.csv", index=False, float_format="%.6g")
    (folder / "summary.json").write_text(json.dumps(summary, indent=1))
    return (f"K = {k}, {summary['features_kept']} of {len(names)} features kept, "
            f"{len(factors)} periods")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=["M"])
    ap.add_argument("--models", nargs="+", default=["sipca"], choices=IPCA_MODELS)
    ap.add_argument("--features", nargs="*", help="feature sets (default: every real one)")
    a = ap.parse_args()
    out = config.RESULTS / "ipca"
    for freq in a.freq:
        for features, model in experiments.finished(freq):
            if model not in a.models or experiments.kind(features) == "placebo":
                continue
            if a.features and features not in a.features:
                continue
            print(f"{freq:>2} {features:<22} {model:<8} {export(freq, features, model, out)}",
                  flush=True)


if __name__ == "__main__":
    main()
