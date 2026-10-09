"""Parse a FRED daily-series CSV with the standard library only; keep strictly date,number rows.
Run as: python3 -I clean_fred.py <downloaded.csv> <cleaned.csv>"""
import csv, re, sys
src, dst = sys.argv[1], sys.argv[2]
ok = bad = missing = 0
with open(src, newline="") as f, open(dst, "w", newline="") as g:
    r, w = csv.reader(f), csv.writer(g)
    next(r)
    w.writerow(["date", "price"])
    for row in r:
        if len(row) != 2 or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row[0]):
            bad += 1
            continue
        if row[1] in (".", ""):
            missing += 1
            continue
        try:
            w.writerow([row[0], float(row[1])])
            ok += 1
        except ValueError:
            bad += 1
print(f"kept {ok} rows, {missing} missing values, {bad} malformed rows")
