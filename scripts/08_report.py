"""Tables with statistical inference, from the per-period series of scripts/07_evaluate.py.

    python scripts/08_report.py [--freq M 2W W]

Writes to results/tables/ (every number that FINDINGS.md quotes comes from these files or
from results/data_audit/). The tables are computed by ftm/evaluation/report.py, whose
docstrings define the columns; each is written as soon as it is computed, and a table
that fails is reported at the end without costing the others.

    forecast.csv         per experiment and split: out-of-sample R2, rank IC with a
                         Newey-West t statistic, Clark-West and Diebold-Mariano tests
                         against a zero forecast
    strategies.csv       per experiment, split and strategy: mean, volatility, Sharpe ratio
                         (gross and net of costs, with a standard error and a bootstrap
                         interval), drawdown, turnover, break-even cost, three-factor alpha;
                         for long-only portfolios also the result against the universe
    selection.csv        the experiment that the VALIDATION period selects (highest mean
                         rank IC) at each frequency, with its test result: among all
                         experiments ("any", the headline), among those of one family
                         (price, text, all) and among those of one topic model (topic_<name>)
    frequency.csv        the selected experiments under every strategy: monthly, biweekly
                         and weekly rebalancing side by side
    text_vs_price.csv    does text add to the price features? paired tests per model
    multiple_testing.csv how good is the best of all experiments, given that it is the best:
                         reality check, SPA test, false discovery rate (long-only
                         portfolios are measured against the universe)
    landscape.csv        groups of experiments (by feature set, kind of feature, topic model
                         and forecasting model): average rank IC, how many are positive, and
                         the t statistic and Sharpe ratio of the group's average
    contrasts.csv        group against group: topic model against topic model, nonlinear
                         against linear models, price and text against price alone
    halves.csv           the two halves of the test period against each other, per group
    placebo.csv          narrative exposures against exposures to misdated news: the test
                         of whether the content of the news matters
    improvements.csv     what the variants change: models fitted on ranked returns, the
                         ensemble of the eight models, the buffered long-short portfolio
    tuning.csv           what the validation period chose in each experiment: setting,
                         boosting rounds, R2, number of features with a weight, time
    topic_models.csv     quality of the topic models (from data/topics/*/quality.json)

"Splits" are validation, test, and the two halves of the test period, whose news comes
from different scrapes (test_2022_23, test_2024). t statistics are to be read against the
column t_critical, not against 1.96 (see ftm/evaluation/inference.py).
"""
from pathlib import Path
import argparse
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm import config
from ftm.evaluation import metrics, report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", nargs="+", default=list(config.FREQUENCIES))
    a = ap.parse_args()
    out = config.RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    ppy_of = {freq: metrics.PERIODS_PER_YEAR[freq] for freq in a.freq}
    series, returns, factors_of = report.load(config.RESULTS / "series", a.freq)
    made = {}                                   # tables that later tables are built from
    recipes = {
        "forecast": lambda: report.forecast_table(series, ppy_of),
        "strategies": lambda: report.strategy_table(returns, factors_of, ppy_of),
        "selection": lambda: report.selection_table(made["forecast"], made["strategies"]),
        "frequency": lambda: report.frequency_table(made["forecast"], made["strategies"],
                                                    made["selection"]),
        "text_vs_price": lambda: report.paired(series, returns, ppy_of),
        "multiple_testing": lambda: report.multiple_testing(returns, ppy_of),
        "landscape": lambda: report.landscape(series, returns, factors_of, ppy_of),
        "contrasts": lambda: report.contrasts(series),
        "halves": lambda: report.halves(series),
        "placebo": lambda: report.placebo_table(series),
        "improvements": lambda: report.improvements_table(series, returns, ppy_of),
        "tuning": lambda: report.tuning_table(config.RESULTS, a.freq),
        "topic_models": lambda: report.topic_table(config.TOPICS),
    }
    failed = []
    for name, recipe in recipes.items():
        path = out / f"{name}.csv"
        try:
            table = made[name] = recipe()
        except Exception:
            traceback.print_exc()
            failed.append(name)
            continue
        if len(table):
            table.to_csv(path, index=False, float_format="%.6g")
        else:
            path.unlink(missing_ok=True)        # no rows: do not keep the table of an earlier run
        print(f"{name}: {len(table)} rows", flush=True)
    if failed:
        raise SystemExit(f"tables that failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
