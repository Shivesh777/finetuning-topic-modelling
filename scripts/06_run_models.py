"""Tune, fit and forecast: every model on every feature set.

    python scripts/06_run_models.py [--freq M 2W W] [--features price all_lda ...]
                                    [--models ridge xgb ...] [--budget SECONDS]
                                    [--retry] [--force]

For each (frequency, feature set, model) the protocol is the same:

  1. tune    every hyperparameter setting is fitted on the TRAINING period and scored
             on the VALIDATION period; the forecasts of the chosen setting for the
             validation period are stored (they are what selects between experiments);
  2. final   the model is refitted on training plus validation with the chosen setting
             and forecasts the TEST period, once. (The EBM is not refitted: its validated
             model forecasts the test period; see ftm/models/ebm.py.)

Models named <model>_rk are fitted on the cross-sectional rank of the return instead of
the return (ftm.models.tuning.target_of); their R2 in the tuning files is against ranks.

Outputs
    data/predictions/<freq>/<features>__<model>.{validation,test}.npy   forecasts
    data/models/<freq>/<features>__ebm.joblib                           fitted EBMs
    results/tuning/<freq>/<features>__<model>.csv     validation score of every setting
    results/tuning/<freq>/<features>__<model>.json    chosen setting, sample sizes, timing
    results/tuning/<freq>/<features>__<model>.log     when the experiment started and finished
    results/importance/<freq>/<features>__<model>.csv coefficients or importances

Finished stages are skipped, so the script can be stopped (--budget) and run again;
--force runs them again, and so does emptying an experiment's .log file. An experiment
that was started and never finished (killed by a time limit) is skipped on later runs and
listed at the end; --retry tries those again.
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ftm import config, experiments
from ftm.models.tuning import fit_for_test, get_model, target_of, tune
from ftm.panel import Panel

IMPORTANCE_ROWS = 50_000            # rows on which mean absolute contributions are measured


def plain(value):
    """JSON form of the numpy scalars found in hyperparameters."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot store {type(value).__name__} in JSON")


def importance(model, names, X) -> pd.DataFrame:
    """Per feature (or EBM term): coefficient and mean absolute contribution, or gain share."""
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        return pd.DataFrame({"feature": names, "coef": coef,
                             "importance": np.abs(coef) * np.abs(X).mean(axis=0),
                             "kind": "mean_abs_contribution"})
    if hasattr(model, "importances"):                      # EBM: S(i) of the Glass Box paper
        terms = [" & ".join(names[i] for i in term) for term in model.term_features()]
        return pd.DataFrame({"feature": terms, "importance": model.importances(X),
                             "kind": "mean_abs_contribution"})
    if hasattr(model, "feature_importance"):
        return pd.DataFrame({"feature": names, "importance": model.feature_importance(),
                             "kind": "gain_share"})
    return pd.DataFrame({"feature": names, "importance": model.model_.feature_importances_,
                         "kind": "impurity_share"})


def run(freq: str, features: str, blocks: list[str], model_name: str, retry: bool,
        force: bool) -> str:
    """Run the stages of one experiment that are missing; returns what was done."""
    stem = f"{features}__{model_name}"
    pred = experiments.PREDICTIONS / freq
    tuning = config.RESULTS / "tuning" / freq
    paths = {split: experiments.prediction_path(freq, features, model_name, split)
             for split in experiments.EVALUATED}
    log = tuning / f"{stem}.log"                   # one line per stage: started, tuned, finished
    meta_path = tuning / f"{stem}.json"
    state = log.read_text() if log.exists() else ""
    if paths["test"].exists() and "finished" in state and not force:
        return "done"
    if state and "finished" not in state and not (retry or force):
        return "unfinished"
    for folder in (pred, tuning, config.RESULTS / "importance" / freq):
        folder.mkdir(parents=True, exist_ok=True)
    tuned = ("tuned" in state and paths["validation"].exists() and meta_path.exists()
             and not force)
    stamp = lambda stage: time.strftime(f"{stage} %Y-%m-%d %H:%M:%S\n")
    log.write_text(stamp("started") + (stamp("tuned earlier, resumed") if tuned else ""))

    panel = Panel.load(config.PANELS / freq, blocks)
    cls = get_model(model_name)
    period = panel.index["period"].to_numpy()
    X, y = panel.X(blocks), target_of(cls, panel.index["y_rel"].to_numpy(), period)
    panel.blocks = {}                  # X holds a copy: free the blocks (weekly px_all EBM on 3.9 GB)
    train, val, test = (panel.rows(s) for s in ("train", "validation", "test"))
    if not tuned:
        t0 = time.time()
        params, table, val_pred = tune(cls, X[train], y[train], X[val], y[val],
                                       config.TUNE_TOLERANCE, groups=period[train])
        table.to_csv(tuning / f"{stem}.csv", index=False)
        chosen = table[table["chosen"]].iloc[0]
        meta = {"freq": freq, "features": features, "model": model_name, "blocks": blocks,
                "n_features": int(X.shape[1]), "settings_tried": int(len(table)),
                "rows": {s: int(m.sum()) for s, m in (("train", train), ("validation", val),
                                                      ("test", test))},
                "params": params, "validation_r2": float(chosen["val_r2"]),
                "train_r2": float(chosen["train_r2"]), "seconds_tune": round(time.time() - t0, 1)}
        meta_path.write_text(json.dumps(meta, indent=1, default=plain))
        np.save(paths["validation"], val_pred.astype(np.float32))
        with log.open("a") as handle:
            handle.write(stamp("tuned"))
    meta = json.loads(meta_path.read_text())

    t0 = time.time()
    fit = train | val
    model = fit_for_test(cls, meta["params"], X, y, train, val, groups=period)
    sample = np.flatnonzero(fit)
    sample = sample[np.random.default_rng(config.SEED).choice(
        len(sample), min(len(sample), IMPORTANCE_ROWS), replace=False)]
    importance(model, panel.feature_names(blocks), X[np.sort(sample)]).to_csv(
        config.RESULTS / "importance" / freq / f"{stem}.csv", index=False)
    if model_name == "ebm":
        import joblib

        (config.DATA / "models" / freq).mkdir(parents=True, exist_ok=True)
        joblib.dump(model, config.DATA / "models" / freq / f"{stem}.joblib", compress=3)
    meta["seconds_final"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, indent=1, default=plain))
    np.save(paths["test"], model.predict(X[test]).astype(np.float32))
    with log.open("a") as handle:
        handle.write(stamp("finished"))
    return f"fitted (tune {meta['seconds_tune']:.0f} s, final {meta['seconds_final']:.0f} s, " \
           f"validation R2 {100 * meta['validation_r2']:.3f}%)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=list(config.FREQUENCIES))
    ap.add_argument("--features", nargs="*", help="feature sets (default: all the panel allows)")
    ap.add_argument("--models", nargs="*", default=list(experiments.MODEL_ORDER))
    ap.add_argument("--budget", type=float, help="start no new experiment after this many seconds")
    ap.add_argument("--retry", action="store_true", help="retry experiments that never finished")
    ap.add_argument("--force", action="store_true", help="run the selected experiments again")
    a = ap.parse_args()
    t0, unfinished, left = time.time(), [], 0
    tasks = []
    for model in [m for m in experiments.MODEL_ORDER if m in a.models]:      # cheapest first
        for freq in a.freq:
            names = json.loads((config.PANELS / freq / "names.json").read_text())["names"]
            for features, blocks in experiments.feature_sets(names).items():
                if a.features is None or features in a.features:
                    tasks.append((freq, features, blocks, model))
    for freq, features, blocks, model in tasks:
        if a.budget and time.time() - t0 > a.budget:
            left += not experiments.prediction_path(freq, features, model, "test").exists()
            continue
        outcome = run(freq, features, blocks, model, a.retry, a.force)
        if outcome == "unfinished":
            unfinished.append(f"{freq} {features} {model}")
        elif outcome != "done":
            print(f"{freq:>2} {features:<22} {model:<6} {outcome}", flush=True)
    print(f"{len(tasks)} experiments, {left} not started yet, {len(unfinished)} unfinished "
          f"({time.time() - t0:.0f} s)")
    if unfinished:
        print("started earlier and never finished (use --retry):", "; ".join(unfinished))


if __name__ == "__main__":
    main()
