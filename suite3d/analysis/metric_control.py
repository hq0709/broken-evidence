"""Metric-channel control: accuracy per model x background with volume-clustered bootstrap CIs.
Reads results_new/metric_<model>_<background>.jsonl written by spatialgen/run_metric_control.py."""
from __future__ import annotations
import json, sys, os, glob, collections
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [("qwen7b", "Qwen2.5-VL-7B"), ("internvl", "InternVL3-8B"), ("qwen3vl", "Qwen3-VL-8B"), ("qwen32b", "Qwen2.5-VL-32B")]
BG = ["grey", "ct", "real"]
def vol(q): return "_".join(q.split("_")[:2])
def boot(ok_by_vol, B=10000, seed=0):
    rng = np.random.default_rng(seed); vols = sorted(ok_by_vol); n = len(vols)
    tot = sum(len(v) for v in ok_by_vol.values())
    out = []
    for _ in range(B):
        pick = rng.integers(0, n, n); s = c = 0
        for i in pick:
            v = ok_by_vol[vols[i]]; s += sum(v); c += len(v)
        out.append(100 * s / c)
    return np.percentile(out, 2.5), np.percentile(out, 97.5)
rows = []
print(f"{'model':16} {'background':10} {'n':>4} {'acc':>6} {'95% CI':>15} {'modal':>14} {'gold-dist':>16}")
for tag, name in MODELS:
    for bg in BG:
        p = f"{R}/results_new/metric_{tag}_{bg}.jsonl"
        if not os.path.exists(p): print(f"{name:16} {bg:10} missing"); continue
        Rs = [json.loads(l) for l in open(p)]
        ok = collections.defaultdict(list)
        for r in Rs: ok[vol(r["qid"])].append(int(r["prediction"] == r["gold"]))
        acc = 100 * np.mean([x for v in ok.values() for x in v]); lo, hi = boot(ok)
        modal = collections.Counter(r["prediction"] for r in Rs).most_common(1)[0]
        gold = collections.Counter(r["gold"] for r in Rs)
        print(f"{name:16} {bg:10} {len(Rs):4d} {acc:6.1f} [{lo:5.1f}, {hi:5.1f}]   {modal[0]:>3} {100*modal[1]/len(Rs):5.1f}%   {dict(sorted(gold.items(), key=lambda kv: int(kv[0])))}")
        rows.append(dict(model=name, tag=tag, background=bg, n=len(Rs), acc=round(acc, 1), lo=round(lo, 1), hi=round(hi, 1), modal=modal[0], modal_share=round(100 * modal[1] / len(Rs), 1)))
if rows:
    import csv
    os.makedirs(f"{R}/figdata", exist_ok=True)
    with open(f"{R}/figdata/metric_control.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote figdata/metric_control.csv")
