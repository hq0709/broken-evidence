"""Volumetric headline table with the 2D columns, growth-matched subset.
Columns: Original (mm_ sighted), PR / NEG / SDR (results_new/text_*), VGR = Acc(roi_only) - Acc(roi_masked)
pooled over the four organs (fill-labelled files preferred), Acc(roi_masked), LPA (lpa_ sighted, content-free calibrated),
SFR = calibrated weighted silent-failure rate on the anatomy-verified trap family (analyse_traps.py), CS = harmonic mean of
Cap = mean(Original, PR, NEG, SDR), Safe = 100 - SFR, Ground = (clip(VGR+50) + Acc_roi_masked)/2."""
from __future__ import annotations
import json, os, glob, sys, csv
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, R)
from growth_matched import growth_of, matched_subset
ref = {json.loads(l)["qid"]: json.loads(l)["answer"] for l in open(f"{R}/common_subset/qa/all.jsonl")}
keep = set(matched_subset(ref, growth_of()))
MODELS = [("qwen7b", "Qwen2.5-VL-7B"), ("internvl", "InternVL3-8B"), ("qwen3vl", "Qwen3-VL-8B"), ("qwen32b", "Qwen2.5-VL-32B"), ("m3d", "M3D-LaMed-Phi3-4B")]
from analyse_traps import rows_for as trap_rows, sfr_w as trap_sfr_w
def trap_sfr(tag):
    rows = trap_rows("traps", tag, "sighted")
    return trap_sfr_w(rows, "adj_fail") if rows else None
ORGANS = ["Task03_Liver", "Task06_Lung", "Task07_Pancreas", "Task10_Colon"]
def acc_file(p, subset=None):
    if not os.path.exists(p): return None, 0
    Rs = [json.loads(l) for l in open(p)]
    if subset is not None: Rs = [r for r in Rs if r["qid"] in subset]
    return (100 * np.mean([r["prediction"] == r["gold"] for r in Rs]) if Rs else None), len(Rs)
def arm_path(T, tag, arm):
    lab = f"{R}/roi_{T}_{tag}_{arm}_local.jsonl"; return lab if os.path.exists(lab) else f"{R}/roi_{T}_{tag}_{arm}.jsonl"
rows = []
print(f"{'model':18}{'Orig':>7}{'PR':>7}{'NEG':>7}{'SDR':>7}{'VGR':>7}{'ROI-m':>7}{'LPA':>7}{'SFR':>7}{'Cap':>7}{'Safe':>7}{'Grnd':>7}{'CS':>7}")
for tag, name in MODELS:
    orig, n = acc_file(f"{R}/mm_{tag}_sighted.jsonl", keep)
    pert = {}
    for arm in ["pr", "neg", "sdr"]:
        files = sorted(glob.glob(f"{R}/results_new/text_{arm}_Task*_{tag}.jsonl"))
        if len(files) == 4:
            Rs = [json.loads(l) for f in files for l in open(f)]; pert[arm] = 100 * np.mean([r["prediction"] == r["gold"] for r in Rs])
    only = masked = None; o_n = m_n = 0; o_s = m_s = 0
    for T in ORGANS:
        for arm in ["roi_only", "roi_masked"]:
            p = arm_path(T, tag if tag != "qwen7b" else "qwen7b", arm)
            if not os.path.exists(p) and tag == "qwen7b": p = arm_path(T, "qwen", arm)
            if not os.path.exists(p): continue
            Rs = [json.loads(l) for l in open(p)]; s = sum(r["prediction"] == r["gold"] for r in Rs)
            if arm == "roi_only": o_s += s; o_n += len(Rs)
            else: m_s += s; m_n += len(Rs)
    if o_n and m_n: only, masked = 100 * o_s / o_n, 100 * m_s / m_n
    vgr = (only - masked) if only is not None else None
    lpa = None
    if os.path.exists(f"{R}/lpa_{tag}_sighted.jsonl") and os.path.exists(f"{R}/lpacal_{tag}_sighted.jsonl"):
        cal = {}
        for l in open(f"{R}/lpacal_{tag}_sighted.jsonl"):
            c = json.loads(l); cal[(c["volume"], tuple(sorted(c["option_set"])))] = c["logprobs"]
        ok = []
        for l in open(f"{R}/lpa_{tag}_sighted.jsonl"):
            r = json.loads(l); base = cal.get(("_".join(r["qid"].split("_")[:2]), tuple(sorted(r["logprobs"]))))
            if base is None: continue
            adj = {c: r["logprobs"][c] - base.get(c, 0.0) for c in r["logprobs"]}; ok.append(max(adj, key=adj.get) == r["gold"])
        lpa = 100 * np.mean(ok) if ok else None                       # content-free calibrated
    sfr = trap_sfr(tag)
    cap = np.mean([orig] + [pert[a] for a in ["pr", "neg", "sdr"] if a in pert]) if orig is not None and len(pert) == 3 else None
    safe = 100 - sfr if sfr is not None else None
    ground = (np.clip(vgr + 50, 0, 100) + masked) / 2 if vgr is not None else None
    cs = 3 / (1 / cap + 1 / safe + 1 / ground) if all(v is not None and v > 0 for v in (cap, safe, ground)) else None
    f = lambda v: f"{v:7.1f}" if v is not None else f"{'--':>7}"
    print(f"{name:18}{f(orig)}{f(pert.get('pr'))}{f(pert.get('neg'))}{f(pert.get('sdr'))}{f(vgr)}{f(masked)}{f(lpa)}{f(sfr)}{f(cap)}{f(safe)}{f(ground)}{f(cs)}")
    rows.append(dict(model=name, tag=tag, original=orig, pr=pert.get("pr"), neg=pert.get("neg"), sdr=pert.get("sdr"), vgr=vgr, roi_masked=masked, lpa=lpa, sfr=sfr, cap=cap, safe=safe, ground=ground, cs=cs, n_matched=n))
with open(f"{R}/figdata/headline_3d.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows([{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
print("wrote figdata/headline_3d.csv")
