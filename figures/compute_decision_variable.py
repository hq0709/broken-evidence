import os
"""Threshold-free analysis of the volumetric decision variable d = log p(yes) - log p(no)
on the growth-matched subset.  Three AUROCs per model, volume-clustered bootstrap CIs.
  gold   : d against the computed label over all probes
  pair   : within complete matched pairs (same volume, same rendering, only the growth number differs)
  text   : across volumes sharing (target, growth), i.e. an identical question sentence
Writes results/3d/decision_variable.csv.  Run from the repository root.
"""
import json, sys, glob, collections, csv, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D3 = os.path.join(REPO, "suite3d")
OUT = os.path.join(REPO, "figures", "out"); os.makedirs(OUT, exist_ok=True); sys.path.insert(0, D3)
from growth_matched import matched_subset, growth_of
from scipy.stats import rankdata
rng = np.random.default_rng(0); B = 2000
gold = {}; prov = {}; pair = {}
for l in open(f"{D3}/common_subset/qa/all.jsonl"):
    r = json.loads(l); gold[r["qid"]] = r["answer"]; prov[r["qid"]] = r["provenance"]; pair[r["qid"]] = r["pair_id"]
keep = list(matched_subset(gold, growth_of()))
vol_of = lambda q: "_".join(q.split("_")[:2])
def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, bool)
    if y.sum() == 0 or (~y).sum() == 0: return np.nan
    r = rankdata(s); return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
def pairwise(groups):
    """pooled pairwise AUROC over groups of (d, is_yes): every (yes, no) cross pair inside a group."""
    num = den = 0.0
    for g in groups:
        pos = [d for d, yy in g if yy]; neg = [d for d, yy in g if not yy]
        if not pos or not neg: continue
        for p in pos:
            for n in neg:
                num += 1.0 if p > n else (0.5 if p == n else 0.0); den += 1
    return num / den if den else np.nan
def stats(R, qs):
    d = {q: R[q]["logprobs"]["yes"] - R[q]["logprobs"]["no"] for q in qs}
    byvol = collections.defaultdict(list)
    for q in qs: byvol[vol_of(q)].append(q)
    vols = sorted(byvol)
    def one(vs):
        items = [q for v in vs for q in byvol[v]]
        g_auc = auroc([d[q] for q in items], [gold[q] == "yes" for q in items])
        bp = collections.defaultdict(list)
        for q in items: bp[pair[q]].append((d[q], gold[q] == "yes"))
        p_auc = pairwise([g for g in bp.values() if len(g) == 2])
        bt = collections.defaultdict(list)
        for q in items: bt[(prov[q]["target"], prov[q]["growth_mm"])].append(q)
        mixed = [q for g in bt.values() if len({gold[q] for q in g}) == 2 for q in g]   # authors' definition: pool items from mixed-label sentence groups
        t_auc = auroc([d[q] for q in mixed], [gold[q] == "yes" for q in mixed])
        return g_auc, p_auc, t_auc
    point = one(vols)
    boots = np.array([one(list(rng.choice(vols, size=len(vols), replace=True))) for _ in range(B)])
    lo = np.nanpercentile(boots, 2.5, axis=0); hi = np.nanpercentile(boots, 97.5, axis=0)
    acc = 100 * np.mean([R[q]["prediction"] == gold[q] for q in qs])
    npairs = sum(1 for g in collections.Counter(pair[q] for q in qs).values() if g == 2)
    bt = collections.defaultdict(list)
    for q in qs: bt[(prov[q]["target"], prov[q]["growth_mm"])].append(q)
    ngroups = sum(len(g) for g in bt.values() if len({gold[q] for q in g}) == 2)
    return point, lo, hi, acc, len(qs), npairs, ngroups
MODELS = [("smolvlm", "SmolVLM2-2.2B", "montage", 2.2), ("qwen3b", "Qwen2.5-VL-3B", "montage", 3), ("qwen7b", "Qwen2.5-VL-7B", "montage", 7),
          ("internvl", "InternVL3-8B", "montage", 8), ("qwen3vl", "Qwen3-VL-8B", "montage", 8), ("internvl14", "InternVL3-14B", "montage", 14),
          ("qwen32b", "Qwen2.5-VL-32B", "montage", 32), ("llavaov", "LLaVA-OneVision-7B", "montage", 7), ("idefics3", "Idefics3-8B", "montage", 8),
          ("pixtral", "Pixtral-12B", "montage", 12), ("m3d", "M3D-LaMed-Phi3-4B", "native", 4), ("m3dllama", "M3D-LaMed-Llama2-7B", "native", 7),
          ("med3dvlm", "Med3DVLM-7B", "native", 7)]
rows = []
for tag, name, inp, size in MODELS:
    R = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D3}/mm_{tag}_sighted.jsonl")}
    qs = [q for q in keep if q in R and "logprobs" in R[q]]
    (g, p, t), lo, hi, acc, n, npairs, ngroups = stats(R, qs)
    rows.append(dict(tag=tag, model=name, input=inp, params_b=size, arm="sighted", n=n, n_pairs=npairs, n_text_items=ngroups, acc=round(acc, 1),
                     auc_gold=g, auc_gold_lo=lo[0], auc_gold_hi=hi[0], auc_pair=p, auc_pair_lo=lo[1], auc_pair_hi=hi[1], auc_text=t, auc_text_lo=lo[2], auc_text_hi=hi[2]))
    print(f"{name:20} acc {acc:5.1f}  gold {g:.3f} [{lo[0]:.3f},{hi[0]:.3f}]  pair {p:.3f} [{lo[1]:.3f},{hi[1]:.3f}]  text {t:.3f} [{lo[2]:.3f},{hi[2]:.3f}]  n={n} pairs={npairs} groups={ngroups}", flush=True)
for cond in ["plain", "identified"]:
    R = {}
    for f in glob.glob(f"{D3}/results_new/*_qwen72b_{cond}.jsonl"):
        for l in open(f):
            r = json.loads(l); R[r["qid"]] = r
    qs = [q for q in keep if q in R]
    (g, p, t), lo, hi, acc, n, npairs, ngroups = stats(R, qs)
    rows.append(dict(tag="qwen72b", model="Qwen2.5-VL-72B", input="montage", params_b=72, arm=cond, n=n, n_pairs=npairs, n_text_items=ngroups, acc=round(acc, 1),
                     auc_gold=g, auc_gold_lo=lo[0], auc_gold_hi=hi[0], auc_pair=p, auc_pair_lo=lo[1], auc_pair_hi=hi[1], auc_text=t, auc_text_lo=lo[2], auc_text_hi=hi[2]))
    print(f"Qwen2.5-VL-72B/{cond:10} acc {acc:5.1f}  gold {g:.3f} [{lo[0]:.3f},{hi[0]:.3f}]  pair {p:.3f} [{lo[1]:.3f},{hi[1]:.3f}]  text {t:.3f} [{lo[2]:.3f},{hi[2]:.3f}]", flush=True)
with open(os.path.join(REPO, "results", "3d", "decision_variable.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote results/3d/decision_variable.csv")
