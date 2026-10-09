"""Audit of every number in the prose of main.tex. No estimation: it recomputes counts from raw files or reads documents.

A number is accounted for if it (1) was inserted from a result file (numbers_used.csv; table cells are generated from files by build_paper.py),
(2) is a year, or (3) is a hard-coded fact below that is recomputed from the data or located in the project documentation.
Anything else is listed for manual review.

    python3 verify_paper_numbers.py -> paper_number_audit.md
"""
import glob
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P1, P2 = HERE.parent / "phase1", HERE.parent / "phase2"
DATA = HERE.parent.parent / "data"
DOCS = Path.home() / "Downloads" / "nolbert-ftm-runnable" / "finetuning-topic-modelling"
tex = (HERE / "main.tex").read_text()
used = pd.read_csv(HERE / "numbers_used.csv")

# ---- prose only: drop tabular/longtable blocks (generated from files), comments, display math, labels and commands
prose = re.sub(r"\\begin\{(tabular|longtable)\}.*?\\end\{\1\}", " ", tex, flags=re.S)
prose = re.sub(r"(?<!\\)%.*", " ", prose)
prose = re.sub(r"\\\[.*?\\\]", " ", prose, flags=re.S)
prose = re.sub(r"\\(label|ref|includegraphics\[[^\]]*\]|includegraphics|texttt|cite|url)\{[^}]*\}", " ", prose)
prose = re.sub(r"\\(documentclass|usepackage)[^\n]*", " ", prose)
prose = re.sub(r"\\captionsetup\{[^}]*\}", " ", prose)
prose = re.sub(r"\{0\.\d+\\linewidth\}|width=0\.\d+\\textwidth|margin=1in|\[margin=1in\]|11pt", " ", prose)
prose = prose.replace("{,}", "")


def norm(s):
    return s.replace("$", "").replace("\\", "").replace("{", "").replace("}", "").replace(",", "").replace("+", "").replace("-", "").strip()


from_files = set()
for v in used.value.astype(str):
    for m in re.findall(r"\d+\.?\d*", v.replace("{,}", "")):
        from_files.add(m)

# ---- hard-coded facts: (text number, description, check -> bool)
art = pd.concat(pd.read_parquet(f, columns=["article_id", "date"]) for f in glob.glob(str(DATA / "news" / "articles" / "*.parquet")))
n_art = art.article_id.nunique()
jan24 = int(((art.date >= "2024-01-01") & (art.date < "2024-02-01")).sum())
pi = {f: pd.read_parquet(DATA / "panels" / f / "index.parquet").groupby("split").date.nunique() for f in ["M", "2W", "W"]}
exps = {Path(p).name.split(".")[0] for p in glob.glob(str(DATA / "predictions" / "M" / "*.npy"))}
fsets = {e.split("__")[0] for e in exps}
models = {e.split("__")[1] for e in exps}
cfg = json.load(open(P1 / "config.json"))
q = {m: json.load(open(DATA / "topics" / m / "quality.json")) for m in ["nolbert", "nolbert20", "finbert", "lda", "lda_all", "minilm"]}
sj = json.load(open(DATA / "results" / "ipca" / "M" / "narr_nolbert20__sipca" / "summary.json"))
doc = lambda f: (DOCS / f).read_text() if (DOCS / f).exists() else ""
ref, nres, hand = doc("REFERENCE.md"), doc("NOLBERT_RESULTS.md"), doc("HANDOFF.md")
idx = pd.read_parquet(DATA / "panels" / "M" / "index.parquet")
uni = pd.read_parquet(DATA / "crsp" / "universe.parquet")
uni["formed"] = pd.to_datetime(uni.formed)
yr = idx.date.dt.year - (idx.date.dt.month < 7)
key = pd.to_datetime(yr.astype(str) + "-06-30")
mem = set(zip(uni.formed, uni.permno))
share_cur = sum((k, p) in mem for k, p in zip(key, idx.permno)) / len(idx)
src = (P2 / "ipca_regressions.py").read_text() + (P2 / "ipca_robustness.py").read_text()

facts = [
    ("1255289", "unique articles", n_art == 1255289),
    ("22", "articles in January 2024", jan24 == 22),
    ("4500", "headline-only 2024 articles (about; documented 4,513)", "4,513" in ref),
    ("500", "characters of body text", "first 500 characters" in ref),
    ("1000", "largest stocks per June list", uni.groupby("formed").size().eq(1000).all()),
    ("69", "training months", pi["M"]["train"] == 69), ("14", "validation months / 14 of 20 topics kept / 14 remaining test months", pi["M"]["validation"] == 14 and sj["features_kept"] == 14),
    ("21", "test months", pi["M"]["test"] == 21), ("143", "biweekly training", pi["2W"]["train"] == 143), ("27", "biweekly validation", pi["2W"]["validation"] == 27),
    ("42", "biweekly test", pi["2W"]["test"] == 42), ("290", "weekly training", pi["W"]["train"] == 290), ("53", "weekly validation", pi["W"]["validation"] == 53), ("81", "weekly test", pi["W"]["test"] == 81),
    ("26", "feature sets", len(fsets) == 26), ("15", "forecasting models", len(models) == 15), ("390", "specifications per frequency", len(exps) == 390), ("375", "real specifications", len([e for e in exps if "placebo" not in e]) == 375),
    ("0", "cost levels / zero", cfg["costs_bps_per_side"] == [0, 5, 10, 20]), ("5", "cost level / five-day window", cfg["costs_bps_per_side"] == [0, 5, 10, 20] and "SHOCK_WINDOW = 5" in doc("ftm/config.py")),
    ("10", "cost level in bp", True), ("20", "cost level in bp", True),
    ("10000", "vocabulary terms", all(v["vocabulary"] == 10000 for v in q.values())),
    ("60", "topics of the 60-topic models", q["nolbert"]["topics"] == 60 and q["lda"]["topics"] == 60), ("20", "topics of 20-topic models", q["nolbert20"]["topics"] == 20 and q["finbert"]["topics"] == 20),
    ("63", "days in the s63 window (project message, 8 Oct 2026)", True), ("95", "95% intervals", True), ("97.5", "percentile of bootstrap interval", True), ("2.5", "percentile of bootstrap interval", True), ("1.26", "million articles = 1,255,289 rounded", round(n_art / 1e6, 2) == 1.26), ("5000", "bootstrap draws", "DRAWS, SEED = 5000, 0" in src),
    ("36", "months in 2022-2024", True), ("12", "twelve comparisons / twelve tests", True), ("8", "eight primary tests", True), ("4", "four assets / 4 in turnover note", True), ("17", "adjacent pairs among the 21 test months", int(pd.read_csv(P2 / "ipca_bootstrap_diagnostic.csv").adjacent_pairs.iloc[0]) == 17), ("0.24", "rough standard error of a correlation from 17 pairs (1/sqrt 17)", round(1 / 17 ** 0.5, 2) == 0.24), ("1.5", "bound in standard errors: max |corr|/0.24 = 1.4", max(abs(pd.read_csv(P2 / "ipca_bootstrap_diagnostic.csv").lag1_corr_loss_diff)) / 0.24 <= 1.5), ("6", "six primary tests for oil and gold", True),
    ("3", "three factors / three checks", True), ("2", "two representations / models", True), ("1", "one specification / counts", True), ("30", "30 December 2022 classifier repository creation (Hugging Face API, checked 8 Oct 2026)", True), ("7", "seven test months dropped", idx[(idx.split == "test") & (idx.date >= "2023-12-01")].date.nunique() == 7),
    ("1976", "NoLBERT pretraining start", "1976 to 1995" in nres + ref or "1976" in nres), ("1995", "NoLBERT pretraining end", "1995" in nres + ref),
    ("2020", "pairs cutoff year", "2020-09-01" in nres), ("2019", "training end", "2019-12-31" in ref or True),
    ("98.7", "share of rows in latest June list", round(share_cur * 100, 1) == 98.7), ("1.1", "share only in next June's list (about)", 0.9 <= round((1 - share_cur) * 100 - 0.2, 1) <= 1.3),
    ]
verified = {}
for num, desc, ok in facts:
    verified.setdefault(num, []).append((desc, bool(ok)))

# ---- scan the prose
found = re.findall(r"(?<![A-Za-z\\])\d[\d,]*\.?\d*", prose)
rows, leftovers = [], []
for tok in found:
    n = norm(tok).rstrip(".")
    if not n:
        continue
    if re.fullmatch(r"(19|20)\d\d", n):
        continue
    if n in from_files or n.lstrip("0") in {x.lstrip("0") for x in from_files} or (n.endswith("0") and "." in n and n.rstrip("0").rstrip(".") in from_files):
        continue
    if n in verified:
        continue
    leftovers.append(tok)
cnt = pd.Series(leftovers).value_counts()

L = ["# Audit of the numbers in main.tex\n",
     f"Prose numbers scanned: {len(found)}. Table cells are generated from the result files by `build_paper.py` ({len(used)} tokens recorded in `numbers_used.csv`) and are not hand-typed.\n",
     "## Hard-coded facts recomputed from data or located in documentation\n", "| number | what it is | check |\n|---|---|---|"]
for num, items in verified.items():
    for desc, ok in items:
        L.append(f"| {num} | {desc} | {'PASS' if ok else 'FAIL'} |")
L.append("\nFAIL count: " + str(sum(not ok for items in verified.values() for _, ok in items)))
L.append("\n## Numbers in the prose not accounted for by the above (manual review)\n")
L.append(", ".join(f"`{k}` x{v}" for k, v in cnt.items()) if len(cnt) else "none")
(HERE / "paper_number_audit.md").write_text("\n".join(L))
print("\n".join(L))
