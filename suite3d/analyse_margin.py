"""
How far the volume moves the logits, and how often it moves a decision.

Why this exists as its own measurement
--------------------------------------
Chance-level accuracy is compatible with two states: nothing usable reaches the
decision, or something does and is out-voted. Accuracy cannot separate them
because both produce the same number. The separation, if there is one, is in
the gap between how much the image perturbs the scores and how often that
perturbation changes the answer.

Two quantities, both over the paired sighted/blind runs on identical probes:

  perturbation  mean |sighted - blind| of the option logprobs, in nats. How far
                the volume moves the scores at all.
  decision gap  mean |top1 - top2| in the sighted arm, in nats. How far the
                scores would have to move to change an answer.
  flip rate     share of probes whose argmax differs between arms. How often
                the perturbation actually wins.

A model with a large perturbation, a larger decision gap and a near-zero flip
rate is receiving the image and not acting on it. That is a different failure
from not receiving it, and it is the one the amplification experiment (§7.7)
targets.

An earlier draft cited numbers of this shape ("3D margins 1.375, perturbation
0.375, flip rate 1.5 %") against a 2D comparison, from an analysis that was
never written down and whose numbers no script in this repository reproduces.
The 3D side is recomputed here from the paired runs. The 2D side is not
recoverable -- the published 2D audit reports per-model metrics, not logits --
so no cross-modality comparison of these quantities is made.
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys
from collections import defaultdict

import os as _os
R = _os.environ.get("VOLUMETRIC_ROOT",
                    _os.path.dirname(_os.path.abspath(__file__)))
R = R if R.endswith("/") else R + "/"
ORDER = ["smolvlm", "qwen3b", "qwen7b", "internvl", "qwen3vl", "internvl14",
         "qwen32b", "llavaov", "idefics3", "pixtral",
         "m3d", "m3dllama", "med3dvlm"]
LABEL = {"smolvlm": "SmolVLM2-2.2B", "qwen3b": "Qwen2.5-VL-3B",
         "qwen7b": "Qwen2.5-VL-7B", "internvl": "InternVL3-8B",
         "qwen3vl": "Qwen3-VL-8B", "internvl14": "InternVL3-14B",
         "qwen32b": "Qwen2.5-VL-32B", "llavaov": "LLaVA-OneVision-7B",
         "idefics3": "Idefics3-8B", "pixtral": "Pixtral-12B",
         "m3d": "M3D-LaMed-Phi3-4B",
         "m3dllama": "M3D-LaMed-Llama2-7B", "med3dvlm": "Med3DVLM-7B"}
NATIVE = {"m3d", "m3dllama", "med3dvlm"}


def load(tag, arm):
    p = f"{R}mm_{tag}_{arm}.jsonl"
    if not os.path.exists(p):
        return None
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p)}


def boot(vals_by_vol, n=2000, seed=0):
    vols = sorted(vals_by_vol)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pick = [vols[rng.randrange(len(vols))] for _ in vols]
        xs = [x for v in pick for x in vals_by_vol[v]]
        out.append(st.mean(xs))
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def main() -> None:
    print(f"{'model':22}{'input':>9}{'perturbation':>14}{'decision gap':>14}"
          f"{'flip rate':>11}{'95% CI':>18}")
    rows = {}
    for tag in ORDER:
        S, B = load(tag, "sighted"), load(tag, "blind")
        if not S or not B:
            continue
        keys = [q for q in S if q in B and "logprobs" in S[q] and "logprobs" in B[q]]
        if not keys:
            continue
        pert, gap, flip = defaultdict(list), [], defaultdict(list)
        for q in keys:
            ls, lb = S[q]["logprobs"], B[q]["logprobs"]
            common = [k for k in ls if k in lb]
            if len(common) < 2:
                continue
            v = "_".join(q.split("_")[:2])
            pert[v].append(st.mean(abs(ls[k] - lb[k]) for k in common))
            top = sorted((ls[k] for k in common), reverse=True)
            gap.append(top[0] - top[1])
            flip[v].append(float(max(ls, key=ls.get) != max(lb, key=lb.get)))
        allp = [x for v in pert.values() for x in v]
        allf = [x for v in flip.values() for x in v]
        lo, hi = boot(flip)
        rows[tag] = (st.mean(allp), st.mean(gap), 100 * st.mean(allf))
        print(f"{LABEL[tag]:22}{'native' if tag in NATIVE else 'montage':>9}"
              f"{st.mean(allp):13.3f}{st.mean(gap):14.3f}"
              f"{100*st.mean(allf):10.1f}%  [{100*lo:.1f},{100*hi:.1f}]")

    print("\nperturbation = mean |sighted - blind| over option logprobs (nats)")
    print("decision gap = mean |top1 - top2| in the sighted arm (nats)")
    print("flip rate    = probes whose argmax differs between arms, "
          "bootstrapped over volumes")
    if rows:
        big = [t for t, (p, g, f) in rows.items() if p > 0.05 and f < 5.0]
        if big:
            print(f"\n{len(big)} of {len(rows)} models move their scores "
                  f"measurably while changing under 5 % of answers:")
            for t in big:
                p, g, f = rows[t]
                print(f"  {LABEL[t]:22} perturbs {p:.3f} nats against a "
                      f"{g:.3f}-nat gap, flips {f:.1f} %")
    json.dump({t: {"perturbation": p, "gap": g, "flip_rate": f}
               for t, (p, g, f) in rows.items()},
              open(f"{R}margin_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
