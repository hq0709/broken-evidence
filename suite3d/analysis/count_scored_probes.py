"""Total scored model outputs behind the paper: every row of every per-probe result file in the
volumetric repository (repository root and results_new/), plus the 2D per-model totals."""
import glob, os, csv, collections
R3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(os.path.dirname(R3), "results", "2d", "metrics_summary.csv")
files = sorted(glob.glob(f"{R3}/*.jsonl") + glob.glob(f"{R3}/results_new/*.jsonl"))
rows3 = 0; fam = collections.Counter()
import json
skipped = []
for f in files:
    n = sum(1 for l in open(f) if l.strip() and "prediction" in json.loads(l))   # model outputs only: corpora and pixel audits carry no prediction
    if n == 0: skipped.append(os.path.basename(f)); continue
    rows3 += n; fam[os.path.basename(f).split("_")[0]] += n
rows2 = sum(int(r["n_total"]) for r in csv.DictReader(open(R2)))
print(f"volumetric: {len(files)-len(skipped)} files with model outputs, {rows3:,} rows; skipped (no predictions): {skipped}"); print("  by prefix:", dict(fam.most_common()))
print(f"2D: {rows2:,} rows"); print(f"total scored: {rows2 + rows3:,}")
