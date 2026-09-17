"""PR / NEG / SDR accuracy on the growth-matched subset, per model, volume-clustered CIs;
plus agreement with the model's own answer on the unperturbed probe (mm_<tag>_sighted)."""
from __future__ import annotations
import json, os, glob, collections, csv
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [("qwen7b", "Qwen2.5-VL-7B"), ("internvl", "InternVL3-8B"), ("qwen3vl", "Qwen3-VL-8B"), ("qwen32b", "Qwen2.5-VL-32B")]
ARMS = ["pr", "neg", "sdr"]; FLIP = {"yes": "no", "no": "yes"}
def vol(q): return "_".join(q.split("_")[:2])
def boot(ok_by_vol, B=10000, seed=0):
    rng = np.random.default_rng(seed); vols = sorted(ok_by_vol); n = len(vols); out = []
    for _ in range(B):
        pick = rng.integers(0, n, n); s = c = 0
        for i in pick: v = ok_by_vol[vols[i]]; s += sum(v); c += len(v)
        out.append(100 * s / c)
    return np.percentile(out, 2.5), np.percentile(out, 97.5)
rows = []
print(f"{'model':16} {'arm':4} {'n':>5} {'acc':>6} {'95% CI':>15} {'modal':>12} {'agree-orig':>10}")
for tag, name in MODELS:
    base = {}
    p = f"{R}/mm_{tag}_sighted.jsonl"
    if os.path.exists(p):
        for l in open(p): r = json.loads(l); base[r["qid"]] = r["prediction"]
    for arm in ARMS:
        files = sorted(glob.glob(f"{R}/results_new/text_{arm}_Task*_{tag}.jsonl"))
        if len(files) < 4: print(f"{name:16} {arm:4} {len(files)}/4 organs done"); continue
        Rs = [json.loads(l) for f in files for l in open(f)]
        ok = collections.defaultdict(list)
        for r in Rs: ok[vol(r["qid"])].append(int(r["prediction"] == r["gold"]))
        acc = 100 * np.mean([x for v in ok.values() for x in v]); lo, hi = boot(ok)
        modal = collections.Counter(r["prediction"] for r in Rs).most_common(1)[0]
        # agreement with the unperturbed answer: for NEG the consistent answer is the flipped one
        agree = [ (r["prediction"] == (FLIP[base[r["qid"]]] if arm == "neg" else base[r["qid"]])) for r in Rs if r["qid"] in base]
        ag = 100 * np.mean(agree) if agree else float("nan")
        print(f"{name:16} {arm:4} {len(Rs):5d} {acc:6.1f} [{lo:5.1f}, {hi:5.1f}]   {modal[0]:>3} {100*modal[1]/len(Rs):5.1f}%   {ag:8.1f}")
        rows.append(dict(model=name, tag=tag, arm=arm, n=len(Rs), acc=round(acc, 1), lo=round(lo, 1), hi=round(hi, 1), modal=modal[0], modal_share=round(100 * modal[1] / len(Rs), 1), agree_orig=round(ag, 1)))
if rows:
    with open(f"{R}/figdata/text_perturbations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote figdata/text_perturbations.csv")
