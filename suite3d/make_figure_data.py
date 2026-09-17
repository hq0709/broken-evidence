"""
Emit plot-ready CSVs for the journal figures.

The figures are drawn separately, so what this produces is the data and nothing
else: one CSV per figure, columns named as the caption uses them, no styling
decisions. Every value is recomputed from the jsonl files at run time rather
than transcribed, so a figure cannot disagree with the table it sits next to.

Figures whose data is not yet complete are skipped with a message naming what
is missing, rather than written from a partial run.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import os as _os
R = _os.environ.get("VOLUMETRIC_ROOT",
                    _os.path.dirname(_os.path.abspath(__file__)))
R = R if R.endswith("/") else R + "/"
OUT = R + "figdata/"
os.makedirs(OUT, exist_ok=True)

MONTAGE = ["smolvlm", "qwen3b", "qwen7b", "internvl", "qwen3vl", "internvl14",
           "qwen32b", "llavaov", "idefics3", "pixtral"]
NATIVE = ["m3d", "m3dllama", "med3dvlm"]
LABEL = {"smolvlm": "SmolVLM2-2.2B", "qwen3b": "Qwen2.5-VL-3B",
         "qwen7b": "Qwen2.5-VL-7B", "internvl": "InternVL3-8B",
         "qwen3vl": "Qwen3-VL-8B", "internvl14": "InternVL3-14B",
         "qwen32b": "Qwen2.5-VL-32B", "llavaov": "LLaVA-OneVision-7B",
         "idefics3": "Idefics3-8B", "pixtral": "Pixtral-12B",
         "m3d": "M3D-LaMed-Phi3-4B",
         "m3dllama": "M3D-LaMed-Llama2-7B", "med3dvlm": "Med3DVLM-7B"}
ORGANS = ["Task06_Lung", "Task10_Colon", "Task07_Pancreas", "Task03_Liver"]



def arm_path(T: str, model: str, arm: str) -> str:
    """Prefer the explicitly-fill-labelled file over the un-suffixed one.

    The re-measured arms are written as ..._roi_only_local.jsonl; the older
    un-suffixed files inherited whatever the argparse default was when their
    process started and cannot be attested. Mixing the two inside one four-arm
    comparison would reintroduce exactly the provenance problem the re-run
    exists to remove, so the labelled file wins whenever it exists.
    """
    labelled = f"{R}roi_{T}_{model}_{arm}_local.jsonl"
    return labelled if os.path.exists(labelled) else f"{R}roi_{T}_{model}_{arm}.jsonl"

def load(p):
    return ({json.loads(l)["qid"]: json.loads(l) for l in open(p)}
            if os.path.exists(p) else None)


def write(name, header, rows):
    with open(OUT + name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {name}  ({len(rows)} rows)")


def boot(D, keys, n=2000, seed=0):
    byv = defaultdict(list)
    for q in keys:
        byv["_".join(q.split("_")[:2])].append(q)
    vs = sorted(byv)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = [vs[rng.randrange(len(vs))] for _ in vs]
        qs = [q for v in s for q in byv[v]]
        out.append(100 * sum(D[q]["prediction"] == D[q]["gold"] for q in qs) / len(qs))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def fig_forest():
    """Figure 7: every model's confound-free accuracy with CI, against the 50 % null."""
    sys.path.insert(0, R)
    from growth_matched import matched_subset, growth_of
    gold = {}
    for l in open(R + "common_subset/qa/all.jsonl"):
        r = json.loads(l)
        gold[r["qid"]] = r["answer"]
    keep = matched_subset(gold, growth_of())
    rows = []
    for tag in MONTAGE + NATIVE:
        S, B = load(f"{R}mm_{tag}_sighted.jsonl"), load(f"{R}mm_{tag}_blind.jsonl")
        if not S or not B or len(S) != len(gold) or len(B) != len(gold):
            continue
        qs = [q for q in keep if q in S and q in B]
        a = 100 * sum(S[q]["prediction"] == S[q]["gold"] for q in qs) / len(qs)
        b = 100 * sum(B[q]["prediction"] == B[q]["gold"] for q in qs) / len(qs)
        lo, hi = boot(S, qs)
        modal = Counter(r["prediction"] for r in S.values()).most_common(1)[0]
        rows.append([LABEL[tag], "native" if tag in NATIVE else "montage",
                     f"{a:.1f}", f"{lo:.1f}", f"{hi:.1f}", f"{b:.1f}",
                     f"{a-b:+.1f}", f"{100*modal[1]/len(S):.1f}", modal[0]])
    write("fig7_forest_confound_free.csv",
          ["model", "input", "acc", "ci_lo", "ci_hi", "blind", "image_gain",
           "modal_share", "modal_token"], rows)


def fig_controls():
    """Figure 8: response-channel controls — perfect questions and yes-rate."""
    rows = []
    for tag in MONTAGE + NATIVE:
        p = f"{R}sanity_{tag}.jsonl"
        if not os.path.exists(p):
            continue
        rs = [json.loads(l) for l in open(p)]
        byq = defaultdict(list)
        for r in rs:
            byq[r.get("question") or r["qid"]].append(r)
        perfect = sum(all(x["prediction"] == x["gold"] for x in v)
                      for v in byq.values())
        rows.append([LABEL[tag], "native" if tag in NATIVE else "montage",
                     perfect, len(byq),
                     f"{100*sum(r['prediction']==r['gold'] for r in rs)/len(rs):.1f}",
                     f"{100*sum(r['prediction']=='yes' for r in rs)/len(rs):.1f}"])
    write("fig8_response_controls.csv",
          ["model", "input", "perfect_questions", "n_questions", "pass_rate",
           "yes_rate"], rows)


def fig_four_arm():
    """Figure 9: the ROI four-arm decomposition, per organ.

    Skipped for any organ whose arms are not all present at equal length --
    an intersection over unequal arms produced an n = 8 "result" once already.
    """
    rows, skipped = [], []
    for model in ("qwen32b", "internvl"):
        for T in ORGANS:
            arms = {c: load(arm_path(T, model, c))
                    for c in ("full", "roi_only", "roi_masked", "zero")}
            present = {c: d for c, d in arms.items() if d}
            if len(present) < 4 or len({len(d) for d in present.values()}) != 1:
                skipped.append(f"{T}/{model} ("
                               + ", ".join(f"{c}={len(d)}" for c, d in present.items())
                               + ")")
                continue
            keys = sorted(set.intersection(*[set(d) for d in present.values()]))
            for c in ("full", "roi_only", "roi_masked", "zero"):
                D = present[c]
                a = 100 * sum(D[q]["prediction"] == D[q]["gold"] for q in keys) / len(keys)
                lo, hi = boot(D, keys)
                modal = Counter(D[q]["prediction"] for q in keys).most_common(1)[0]
                rows.append([LABEL[model], T.split("_")[1], c, len(keys),
                             f"{a:.1f}", f"{lo:.1f}", f"{hi:.1f}",
                             f"{100*modal[1]/len(keys):.1f}"])
    if rows:
        write("fig9_roi_four_arm.csv",
              ["model", "organ", "arm", "n", "acc", "ci_lo", "ci_hi",
               "modal_share"], rows)
    for s in skipped:
        print(f"  skipped {s} — arms incomplete")


def fig_margin():
    """Figure 10: perturbation against flip rate, from the margin summary."""
    import json as _json
    src = R + "margin_summary.json"
    if not os.path.exists(src):
        print("  skipped fig10_margin — run analyse_margin.py first")
        return
    got = _json.load(open(src))
    rows = [[LABEL[t], "native" if t in NATIVE else "montage",
             f"{v['perturbation']:.3f}", f"{v['gap']:.3f}", f"{v['flip_rate']:.3f}"]
            for t, v in got.items() if t in LABEL]
    write("fig10_margin.csv",
          ["model", "input", "perturbation", "gap", "flip_rate"], rows)


if __name__ == "__main__":
    print("figure data ->", OUT)
    fig_forest()
    fig_controls()
    fig_four_arm()
    fig_margin()
