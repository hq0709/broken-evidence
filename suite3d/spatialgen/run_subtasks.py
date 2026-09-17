"""
Sub-tasks: where does the failure live?

The composite probe asks a model to find a named lesion, find a named target,
judge the distance between them in millimetres, and compare that distance with a
growth amount. Failing the composite says nothing about which of those four
steps broke. Each is asked separately here, on the same volumes, the same
matched subset and the same `identified` rendering the composite uses, so the
comparison is within-item:

    localise    which organ holds the lesion outlined in red
                -- is the red outline perceived at all?
    name        which structure is outlined in cyan
                -- is the target identified?
    distance    how far apart the two outlined structures are, in mm
                -- is the metric relation perceived?

The fourth component, comparing a distance with a growth amount, is
`numeric-oracle` in run_identification_control, where models score 100%. So a
model that passes localise/name/distance and fails the composite has a
composition failure; one that fails distance has a metric-perception failure;
one that fails localise or name has a binding failure.

Two design points, and why:

* The distance options are 5 / 15 / 25 / 35 mm. The corpus caps the gap at
  40 mm (`counterfactual_qa.growth_pairs`, max_gap_mm=40), so a 60 mm option
  would never be correct; an option that is never right is a hint, not a
  distractor. The bucket edges (10, 20, 30) sit on the observed quartiles
  (10.4, 19.7, 29.4), so all four options are close to equiprobable and chance
  is 25%.
* `localise` pools all four organs and asks a four-way question. Asked one organ
  at a time the answer is constant within a file, and a model that always says
  "liver" would score 100% on the liver run.

Option order is shuffled per item from a seeded RNG, so position cannot carry
the answer, and the shuffled order is recorded.

Usage
-----
    python run_subtasks.py --subtask distance --model qwen32b --n 300 \
        --device cuda:0 --out results_new/id_common_qwen32b_subtask-distance.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_identification_control import (                        # noqa: E402
    cached_render, legend_for, score_choices,
)

SUBTASKS = ["localise", "name", "distance"]
ORGAN_WORD = {"Task03_Liver": "liver", "Task06_Lung": "lung",
              "Task07_Pancreas": "pancreas", "Task10_Colon": "colon"}
DISTANCE_OPTIONS = [5, 15, 25, 35]
RICHNESS_ARMS = ("slices3", "slices9", "axial25", "zoom")
CONDITION = "identified"          # the only condition that draws both outlines

# `localise` asks which organ holds the red outline, but the body region alone
# answers it: a chest CT is the lung task and an abdominal one is not, whether or
# not the model ever sees the outline. Running the same question on `plain`,
# which draws no outline at all and drops the legend sentence with it, separates
# "perceives the annotation" from "recognises the body region". Any gap between
# the two is what the annotation is worth; no gap means the sub-task measures
# anatomy recognition and cannot support a claim about binding.


def bucket(gap_mm: float) -> str:
    """Nearest option, which is what the question asks for."""
    return str(min(DISTANCE_OPTIONS, key=lambda o: abs(o - gap_mm)))


def question_for(subtask: str, pair: dict, rng: random.Random) -> tuple:
    target = pair["target"].replace("_", " ")
    if subtask == "localise":
        gold = ORGAN_WORD[pair["organ"]]
        choices = sorted(ORGAN_WORD.values())
        q = ("Which organ contains the lesion outlined in red? "
             "Answer with exactly one of: {opts}."
             if pair.get("annotated", True) else
             "Which organ is shown in these views? "
             "Answer with exactly one of: {opts}.")
    elif subtask == "name":
        gold = target
        # Distractors come from targets that occur in probes for THIS organ, not
        # from all 29 in the corpus. A sternum offered against an abdominal scan
        # is eliminable from body region alone, which would let the sub-task be
        # answered without ever resolving the cyan outline -- the thing it exists
        # to measure.
        pool = [t.replace("_", " ") for t in pair["organ_targets"]
                if t != pair["target"]]
        choices = [gold] + rng.sample(pool, min(3, len(pool)))
        q = ("Which structure is outlined in cyan? "
             "Answer with exactly one of: {opts}.")
    else:
        gold = bucket(pair["gap_mm"])
        choices = [str(o) for o in DISTANCE_OPTIONS]
        q = ("Approximately how many millimetres separate the red outlined "
             "lesion from the cyan outlined structure? "
             "Answer with exactly one of: {opts} mm.")
    choices = list(choices)
    rng.shuffle(choices)
    return q.format(opts=", ".join(choices)), gold, choices


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--subtask", choices=SUBTASKS, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300,
                    help="probes total, stratified equally across the organs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", default=CONDITION,
                    choices=["identified", "plain", "bestslice", "overlay",
                             "slices3", "slices9", "axial25", "zoom"],
                    help="rendering to ask the sub-task about. `plain` is the "
                         "no-annotation control for `localise`; the last four are "
                         "the input-richness arms, which carry the same annotation "
                         "as `identified` and vary only how much of the volume is "
                         "shown. Pre-render them first with runs/prerender.py "
                         "--conditions <arm>.")
    ap.add_argument("--render-cache",
                    default=str(Path(__file__).resolve().parent.parent
                                / "render_cache"))
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    from growth_matched import growth_of, matched_subset
    ref, rows = {}, {}
    for line in open(repo / "common_subset" / "qa" / "all.jsonl"):
        r = json.loads(line)
        ref[r["qid"]] = r["answer"]
        rows[r["qid"]] = r
    keep = matched_subset(ref, growth_of())

    # one item per (volume, lesion, target): the two members of a matched pair
    # share all three sub-task answers, so scoring both is duplicated work
    pairs: dict[str, dict] = {}
    for qid in sorted(keep):
        r = rows[qid]
        pid = r["pair_id"]
        if pid in pairs:
            continue
        vid = "_".join(qid.split("_")[:2])
        lk = next((p for p in qid.split("_") if p.startswith("lesion")), None)
        if lk is None:
            continue
        pairs[pid] = {"pair_id": pid, "qid": qid, "vid": vid, "lesion": lk,
                      "organ": r["organ"], "target": r["provenance"]["target"],
                      "gap_mm": float(r["provenance"]["gap_mm"])}
    all_targets = sorted({p["target"] for p in pairs.values()})
    organ_targets = defaultdict(set)
    for p in pairs.values():
        organ_targets[p["organ"]].add(p["target"])
    print("targets per organ: "
          + ", ".join(f"{o.split('_')[-1]} {len(t)}"
                      for o, t in sorted(organ_targets.items())), flush=True)

    by_organ = defaultdict(list)
    for p in pairs.values():
        by_organ[p["organ"]].append(p)
    rng = random.Random(args.seed)
    per = max(1, args.n // len(by_organ))
    picked = []
    for organ in sorted(by_organ):
        items = sorted(by_organ[organ], key=lambda p: p["pair_id"])
        rng.shuffle(items)
        picked += items[:per]
    picked.sort(key=lambda p: p["pair_id"])
    print(f"{len(picked)} probes, {per} per organ, subtask={args.subtask}",
          flush=True)

    from run_multimodel import MODEL_ID, MontageModel
    if args.model not in MODEL_ID:
        raise SystemExit(f"unknown model tag {args.model!r}")
    model = MontageModel(MODEL_ID[args.model], args.device)
    print("model ready", flush=True)

    # The richness arms draw outlines and a scale bar exactly as `identified`
    # does, so they take the identified legend -- this is the convention
    # run_input_richness.py already uses. Leaving the legend empty would describe
    # an annotation the model can see, which is its own confound.
    legend = legend_for("identified" if args.condition in RICHNESS_ARMS
                        else args.condition)
    written = missing = 0
    with open(args.out, "w") as fout:
        for p in picked:
            p["all_targets"] = all_targets
            p["organ_targets"] = sorted(organ_targets[p["organ"]])
            p["annotated"] = (args.condition in ("overlay", "identified")
                              or args.condition in RICHNESS_ARMS)
            qrng = random.Random(f"{args.seed}:{p['pair_id']}:{args.subtask}")
            question, gold, choices = question_for(args.subtask, p, qrng)
            try:
                image, geom = cached_render(
                    args.render_cache, p["organ"], p["vid"], p["lesion"],
                    p["target"], args.condition,
                    lambda: (_ for _ in ()).throw(
                        LookupError("render not in cache; run runs/prerender.py")))
            except LookupError as exc:
                missing += 1
                if missing <= 3:
                    print(f"  {p['pair_id']}: {exc}", file=sys.stderr)
                continue
            text = model.proc.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{legend} {question}"}]}],
                tokenize=False, add_generation_prompt=True)
            pred, lp = score_choices(model, text, choices, images=[image])
            fout.write(json.dumps({
                "qid": f"{p['pair_id']}_{args.subtask}", "organ": p["organ"],
                "condition": (f"subtask-{args.subtask}" if args.condition == CONDITION
                              else f"subtask-{args.subtask}-{args.condition}"),
                "prediction": pred, "gold": gold, "pair_id": None,
                "logprobs": lp, "choices": choices, "asked": question,
                "gap_mm": p["gap_mm"], "geometry": geom,
            }) + "\n")
            written += 1
            if written % 50 == 0:
                print(f"  {written} written", flush=True)

    got = [json.loads(l) for l in open(args.out)]
    acc = 100.0 * sum(r["prediction"] == r["gold"] for r in got) / max(len(got), 1)
    chance = 100.0 / len(got[0]["choices"]) if got else float("nan")
    print(f"wrote {args.out}: {written} scored, {missing} missing renders, "
          f"accuracy {acc:.1f}% against {chance:.1f}% chance")
    print(f"  gold distribution {dict(Counter(r['gold'] for r in got))}")
    print(f"  prediction distribution {dict(Counter(r['prediction'] for r in got))}")


if __name__ == "__main__":
    main()
