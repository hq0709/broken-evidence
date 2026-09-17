"""
Expand the counterfactual corpus into the five BrokenEvidence probe families.

Output rows carry the SAME schema run_multimodel.py already consumes (qid,
organ, question, choices, answer), so the families corpus is evaluated by the
existing runner with no changes to it. Family / severity / anchor metadata are
recovered at scoring time by joining on qid, which encodes them: a probe id is
"<anchor_qid>#<family>".

Options are scored as text strings, not letters. A five-option MCQ whose options
are rendered as "A/B/C/D/E" invites the model to answer with a letter whose
likelihood reflects position bias rather than content; and 3D medical VLMs
answer in referring-expression templates that embed the answer words. So the
letter mapping from families3d exists only to fix a canonical order, and the
prediction is the argmax over option text.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from families3d import build_probes, knowledge_probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="counterfactual corpus (jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--knowledge", type=int, default=200,
                    help="knowledge-only probes for the LPA axis")
    ap.add_argument("--anchors", type=int, default=None,
                    help="cap on anchor items; each yields 5 probes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.qa) if l.strip()]
    rows = [r for r in rows if r.get("kind") == "growth_contact"]
    rng = random.Random(args.seed)
    if args.anchors and len(rows) > args.anchors:
        # subsample whole matched PAIRS, so the anchor family keeps its exact
        # 50% chance level -- sampling items independently would not
        by_pair: dict = {}
        for r in rows:
            by_pair.setdefault(r.get("pair_id") or r["qid"], []).append(r)
        pairs = [v for v in by_pair.values() if len(v) == 2]
        rng.shuffle(pairs)
        rows = [r for p in pairs[: args.anchors // 2] for r in p]

    fam = Counter()
    n_out = 0
    with open(args.out, "w") as f:
        for r in rows:
            for p in build_probes(r, rng):
                letters = sorted(p.options)
                f.write(json.dumps({
                    "qid": p.qid,
                    "organ": r.get("organ") or p.qid.split("_")[0],
                    "question": p.question,
                    "choices": [p.options[l] for l in letters],
                    "answer": p.options[p.gold],
                    "family": p.family,
                    "severity": p.severity,
                    "anchor_qid": p.anchor_qid,
                    "pair_id": r.get("pair_id"),
                    "provenance": p.provenance}) + "\n")
                fam[p.family] += 1
                n_out += 1
        # Knowledge probes are answerable from clinical reasoning alone, but the
        # runner locates a volume from the qid, and LPA has to be measurable in
        # the SAME arm as everything else. So each is attached to a real scan
        # drawn from the anchor pool. The image is irrelevant to the answer by
        # construction -- that is precisely what the axis tests.
        hosts = sorted({(r["organ"], "_".join(r["qid"].split("_")[:2]))
                        for r in rows}) or [("liver", "liver_000")]
        for i in range(args.knowledge):
            p = knowledge_probe(rng, i)
            organ, vid = hosts[i % len(hosts)]
            letters = sorted(p.options)
            f.write(json.dumps({
                "qid": f"{vid}_knowledge_{i:04d}", "organ": organ,
                "question": p.question,
                "choices": [p.options[l] for l in letters],
                "answer": p.options[p.gold], "family": p.family,
                "severity": p.severity, "anchor_qid": None,
                "pair_id": None, "provenance": p.provenance}) + "\n")
            fam[p.family] += 1
            n_out += 1

    print(f"{len(rows)} anchors -> {n_out} probes")
    for k, v in sorted(fam.items()):
        print(f"  {k:10} {v:6d}")


if __name__ == "__main__":
    main()
