"""Text-side perturbations of the growth-contact probes, mirroring the 2D suite.

The 2D suite perturbs the question while the image stays intact: paraphrase
(gold preserved), negation (gold inverted) and specificity drop (a qualifier
removed, gold preserved).  The volumetric probe is a template, so the same three
operators apply to every item exactly and the gold stays computed:

  pr   one of three paraphrases, chosen per matched pair so both members share it
  neg  "would it stay clear of the X?"  -> gold inverted
  sdr  "in every direction" dropped     -> gold unchanged (isotropic growth is the default reading)

Writes cfqa_text/<arm>/<organ>.jsonl with the schema run_identification_control.py
consumes (qid, organ, question, answer, choices, pair_id, provenance), so the
runs use the published `plain` rendering and scoring path unchanged.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "common_subset" / "qa" / "all.jsonl"
OUT = REPO / "cfqa_text"
Q = re.compile(r"^If this lesion grew by (?P<g>[0-9.]+) mm in every direction, would it contact the (?P<t>.+?)\? Answer yes or no\.$")
PARAPHRASES = [
    "Suppose this lesion expanded by {g} mm outward in all directions. Would it reach the {t}? Answer yes or no.",
    "If the lesion enlarged uniformly by {g} mm, would it come into contact with the {t}? Answer yes or no.",
    "Would growing this lesion by {g} mm in every direction bring it into contact with the {t}? Answer yes or no.",
]
NEG = "If this lesion grew by {g} mm in every direction, would it stay clear of the {t}? Answer yes or no."
SDR = "If this lesion grew by {g} mm, would it contact the {t}? Answer yes or no."
FLIP = {"yes": "no", "no": "yes"}


def main() -> None:
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    rows = [r for r in rows if r.get("kind") == "growth_contact"]
    rng = random.Random(0)
    template_of_pair: dict = {}
    files: dict = {}
    counts = Counter()
    for r in rows:
        m = Q.match(r["question"])
        assert m, r["question"]
        g, t = m.group("g"), m.group("t")
        pid = r.get("pair_id") or r["qid"]
        if pid not in template_of_pair:
            template_of_pair[pid] = rng.randrange(len(PARAPHRASES))
        variants = {"pr": (PARAPHRASES[template_of_pair[pid]].format(g=g, t=t), r["answer"]),
                    "neg": (NEG.format(g=g, t=t), FLIP[r["answer"]]),
                    "sdr": (SDR.format(g=g, t=t), r["answer"])}
        for arm, (q, gold) in variants.items():
            out = dict(r); out["question"] = q; out["answer"] = gold; out["perturbation"] = arm
            out["original_answer"] = r["answer"]
            key = (arm, r["organ"])
            if key not in files:
                p = OUT / arm / f"{r['organ']}.jsonl"; p.parent.mkdir(parents=True, exist_ok=True); files[key] = open(p, "w")
            files[key].write(json.dumps(out) + "\n"); counts[arm] += 1
    for f in files.values():
        f.close()
    print({k: v for k, v in counts.items()}, "rows per arm;", len(rows), "source probes")
    ex = rows[0]; m = Q.match(ex["question"]); g, t = m.group("g"), m.group("t")
    print("example:", ex["question"]); print("  pr :", PARAPHRASES[template_of_pair[ex['pair_id']]].format(g=g, t=t))
    print("  neg:", NEG.format(g=g, t=t), "->", FLIP[ex["answer"]]); print("  sdr:", SDR.format(g=g, t=t))


if __name__ == "__main__":
    main()
