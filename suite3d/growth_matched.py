"""
Growth-magnitude confound: measure it, then re-score every model without it.

The problem
-----------
A matched pair asks the same question at two growth amounts straddling the true
surface gap, so the LARGER amount is always the "yes" member. Within a pair that
is exactly the point -- it forces chance to 50 % for any model that answers both
identically. Across the CORPUS, however, it leaves a marginal cue: yes-items
have median growth 26.0 mm against 15.3 mm for no-items, and a single threshold
on the number alone reaches 69.2 % (threshold chosen on the test set, so an
upper bound). "Chance is exactly 50 %" is therefore a statement about the label
distribution, not a bound on what a text-only model can score.

The fix, applied post hoc
-------------------------
Select a subset in which the marginal distribution of growth_mm is identical for
yes and no: bin growth, and within each bin keep equal numbers of each label.
On that subset the number carries no information about the answer by
construction, so any above-chance accuracy must come from somewhere else.

This needs no new inference -- it re-scores stored predictions -- so every model
can be re-reported on identical items.
"""
from __future__ import annotations

import json
import random
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


def growth_of() -> dict[str, float]:
    out = {}
    for l in open(f"{R}/common_subset/qa/all.jsonl"):
        r = json.loads(l)
        g = r.get("provenance", {}).get("growth_mm")
        if g is not None:
            out[r["qid"]] = float(g)
    return out


def threshold_ceiling(items: list[tuple[float, str]]) -> tuple[float, float]:
    best = (0.0, 0.0)
    for t in sorted({g for g, _ in items}):
        acc = sum((g > t) == (a == "yes") for g, a in items) / len(items)
        if acc > best[0]:
            best = (acc, t)
    return best


def matched_subset(gold: dict[str, str], growth: dict[str, float],
                   width: float = 2.0, seed: int = 0) -> set[str]:
    """Equal yes/no counts inside every growth bin of `width` mm."""
    bins: dict[int, dict[str, list[str]]] = defaultdict(lambda: {"yes": [], "no": []})
    for q, a in gold.items():
        if q in growth:
            bins[int(growth[q] // width)][a].append(q)
    rng = random.Random(seed)
    keep: set[str] = set()
    # Sort before shuffling. Without this the subset depends on the iteration
    # order of `gold`, which is the row order of whichever jsonl it was built
    # from -- so the same seed drew a different 1,368 items depending on an
    # arbitrary file's ordering, and the reported accuracies moved by a few
    # tenths with it. Sorting makes the subset a function of (gold, growth,
    # seed) and nothing else.
    for _, b in sorted(bins.items()):
        b["yes"].sort()
        b["no"].sort()
        rng.shuffle(b["yes"])
        rng.shuffle(b["no"])
        m = min(len(b["yes"]), len(b["no"]))
        keep.update(b["yes"][:m])
        keep.update(b["no"][:m])
    return keep


def boot(qs: list[str], ok: dict[str, bool], n: int = 2000, seed: int = 0):
    byv = defaultdict(list)
    for q in qs:
        byv["_".join(q.split("_")[:2])].append(ok[q])
    vols = sorted(byv)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = [x for _ in vols for x in byv[rng.choice(vols)]]
        out.append(100.0 * sum(s) / len(s))
    out.sort()
    return out[int(.025 * n)], out[int(.975 * n)]


def main() -> None:
    growth = growth_of()
    # gold comes from the corpus, not from a model's output file: the subset
    # must not depend on which model happened to be the reference
    ref = {}
    for l in open(f"{R}/common_subset/qa/all.jsonl"):
        r = json.loads(l)
        ref[r["qid"]] = r["answer"]

    full = [(growth[q], a) for q, a in ref.items() if q in growth]
    acc, thr = threshold_ceiling(full)
    print(f"full corpus n={len(full)}   growth-only threshold rule ceiling {100*acc:.1f}% "
          f"(threshold {thr:.1f} mm)")

    keep = matched_subset(ref, growth)
    sub = [(growth[q], ref[q]) for q in keep]
    acc2, thr2 = threshold_ceiling(sub)
    ys = sum(1 for q in keep if ref[q] == "yes")
    print(f"growth-matched subset n={len(keep)} ({ys} yes / {len(keep)-ys} no)   "
          f"same rule ceiling {100*acc2:.1f}% (threshold {thr2:.1f} mm)\n")

    print(f"{'model':18}{'full img':>11}{'matched img':>13}{'matched 95%CI':>17}"
          f"{'blind':>13}{'img gain':>10}")
    for name in ORDER:
        try:
            S = {json.loads(l)["qid"]: json.loads(l)
                 for l in open(f"{R}/mm_{name}_sighted.jsonl")}
            B = {json.loads(l)["qid"]: json.loads(l)
                 for l in open(f"{R}/mm_{name}_blind.jsonl")}
        except FileNotFoundError:
            print(f"{LABEL[name]:18}{'-- not run --':>24}")
            continue
        # an arm still being written must not enter the table: a partial file
        # yields a real-looking accuracy over whatever rows exist so far
        if len(S) != len(ref) or len(B) != len(ref):
            print(f"{LABEL[name]:18}{'-- running '+f'{min(len(S),len(B))}/{len(ref)}'+' --':>24}")
            continue
        common = [q for q in S if q in B]
        fa = 100.0 * sum(S[q]["prediction"] == S[q]["gold"]
                         for q in common) / len(common)
        qs = [q for q in common if q in keep]
        if not qs:
            continue
        okS = {q: S[q]["prediction"] == S[q]["gold"] for q in qs}
        okB = {q: B[q]["prediction"] == B[q]["gold"] for q in qs}
        ma = 100.0 * sum(okS.values()) / len(qs)
        mb = 100.0 * sum(okB.values()) / len(qs)
        lo, hi = boot(qs, okS)
        print(f"{LABEL[name]:18}{fa:10.1f}%{ma:12.1f}%  [{lo:5.1f},{hi:5.1f}]"
              f"{mb:12.1f}%{ma-mb:+9.1f}")

    print("\nOn the matched subset the growth amount carries no label information, so accuracy above 50% cannot come from that number.")


if __name__ == "__main__":
    main()
