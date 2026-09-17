"""
Scoring harness for geometry-verified spatial QA.

Two families of measurement, deliberately separated:

  ACCURACY      -- per-category correctness.
  CONSISTENCY   -- whether the model's answers obey geometric axioms.

The second is a cheap diagnostic that needs no reference answer. A model that reads
volumetric evidence cannot say both "A is superior to B" and "B is superior to
A"; a model pattern-matching language priors does it constantly. Because
qa_gen.hard_negatives() emits each directional question together with its
inverse, antisymmetry is measurable directly -- no extra annotation, and it does
not care whether the model got the direction RIGHT, only whether it stayed
self-consistent. So a model can score 50% accuracy two ways: by being noisy
(inconsistent) or by holding a wrong-but-coherent frame (consistent). Accuracy
alone cannot tell those apart; this can.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CANON = {
    "left": "left", "l": "left", "leftward": "left",
    "right": "right", "r": "right", "rightward": "right",
    "superior": "superior", "above": "superior", "cranial": "superior",
    "cephalad": "superior", "upper": "superior",
    "inferior": "inferior", "below": "inferior", "caudal": "inferior",
    "lower": "inferior",
    "anterior": "anterior", "front": "anterior", "ventral": "anterior",
    "posterior": "posterior", "back": "posterior", "dorsal": "posterior",
    "bilateral": "bilateral", "midline": "bilateral", "central": "bilateral",
    "yes": "yes", "no": "no",
}

OPPOSITE = {
    "left": "right", "right": "left",
    "superior": "inferior", "inferior": "superior",
    "anterior": "posterior", "posterior": "anterior",
}


def normalise(text: str) -> str | None:
    """Map a free-text model answer onto the categorical vocabulary.

    Takes the FIRST recognised token rather than any match, so a hedged answer
    ("left, though it extends right") is scored on its actual commitment.
    """
    if text is None:
        return None
    for tok in re.findall(r"[a-z]+", text.lower()):
        if tok in CANON:
            return CANON[tok]
    return None


def parse_number(text: str) -> float | None:
    if text is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


def score_one(qa: dict, pred: str, num_tol_rel: float = 0.25) -> bool | None:
    """True/False, or None when the prediction is unparseable for this type."""
    gold = qa["answer"]
    if qa.get("category") == "extent":
        g, p = parse_number(gold), parse_number(pred)
        if g is None or p is None:
            return None
        # generous relative tolerance: radiologists round, and the point is
        # whether the model is in the right regime, not sub-millimetre accuracy
        return abs(p - g) <= max(num_tol_rel * abs(g), 2.0)
    g, p = normalise(gold), normalise(pred)
    if g is None or p is None:
        return None
    return g == p


def evaluate(qas: list[dict], preds: dict[str, str]) -> dict:
    per_cat = defaultdict(lambda: {"n": 0, "correct": 0, "unparsed": 0})
    for qa in qas:
        pred = preds.get(qa["qid"])
        cat = qa.get("category", "unknown")
        per_cat[cat]["n"] += 1
        if pred is None:
            per_cat[cat]["unparsed"] += 1
            continue
        r = score_one(qa, pred)
        if r is None:
            per_cat[cat]["unparsed"] += 1
        elif r:
            per_cat[cat]["correct"] += 1

    overall_n = sum(v["n"] for v in per_cat.values())
    overall_c = sum(v["correct"] for v in per_cat.values())

    return {
        "overall": {"n": overall_n, "correct": overall_c,
                    "accuracy": overall_c / overall_n if overall_n else 0.0},
        "per_category": {k: {**v, "accuracy": v["correct"] / v["n"] if v["n"] else 0.0}
                         for k, v in sorted(per_cat.items())},
    }


def consistency(qas: list[dict], preds: dict[str, str]) -> dict:
    """Antisymmetry violation rate over inverse question pairs.

    A pair (q, q_inv) asks the same geometric fact from both directions. The
    answers must be opposites. Equal answers are a hard geometric contradiction.
    Pairs where either side is unparseable are excluded, and reported.
    """
    by_qid = {qa["qid"]: qa for qa in qas}
    checked = violations = skipped = 0
    examples = []

    for qa in qas:
        qid = qa["qid"]
        if not qid.endswith("_inv"):
            continue
        src = qa.get("provenance", {}).get("source_qid")
        if src is None or src not in by_qid:
            continue
        p_inv, p_src = normalise(preds.get(qid, "")), normalise(preds.get(src, ""))
        if p_inv is None or p_src is None:
            skipped += 1
            continue
        if p_src not in OPPOSITE:
            skipped += 1
            continue
        checked += 1
        if OPPOSITE[p_src] != p_inv:
            violations += 1
            if len(examples) < 5:
                examples.append({"forward_qid": src, "forward_pred": p_src,
                                 "inverse_qid": qid, "inverse_pred": p_inv})

    return {"pairs_checked": checked, "violations": violations,
            "violation_rate": violations / checked if checked else None,
            "skipped_unparseable": skipped, "examples": examples}


def load_qas(path: Path) -> list[dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    out = []
    for f in files:
        with open(f) as fh:
            out.extend(json.loads(l) for l in fh if l.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="QA .jsonl file or directory")
    ap.add_argument("--pred", required=True,
                    help="predictions .jsonl with fields qid, prediction")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    qas = load_qas(Path(args.qa))
    preds = {}
    with open(args.pred) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                preds[r["qid"]] = r.get("prediction", "")

    acc = evaluate(qas, preds)
    cons = consistency(qas, preds)

    print(f"QA: {len(qas)}  predictions: {len(preds)}")
    print(f"\noverall accuracy: {acc['overall']['accuracy']:.1%} "
          f"({acc['overall']['correct']}/{acc['overall']['n']})")
    print("\nper category:")
    for cat, v in acc["per_category"].items():
        print(f"  {cat:16} {v['accuracy']:6.1%}  n={v['n']:5d}  "
              f"unparsed={v['unparsed']}")

    if cons["pairs_checked"]:
        print(f"\nantisymmetry: {cons['violation_rate']:.1%} violations "
              f"({cons['violations']}/{cons['pairs_checked']} pairs, "
              f"{cons['skipped_unparseable']} skipped)")
        for e in cons["examples"]:
            print(f"    contradiction: {e['forward_pred']} vs "
                  f"{e['inverse_pred']} ({e['forward_qid']})")
    else:
        print("\nantisymmetry: no evaluable inverse pairs")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"accuracy": acc, "consistency": cons}, f, indent=1)


def selftest() -> None:
    qas = [
        {"qid": "v_1", "category": "longitudinal", "answer": "superior",
         "provenance": {}},
        {"qid": "v_1_inv", "category": "longitudinal", "answer": "inferior",
         "provenance": {"source_qid": "v_1", "derived": "inverse_of"}},
        {"qid": "v_2", "category": "laterality", "answer": "left",
         "provenance": {}},
        {"qid": "v_2_inv", "category": "laterality", "answer": "right",
         "provenance": {"source_qid": "v_2", "derived": "inverse_of"}},
        {"qid": "v_3", "category": "extent", "answer": "40.0",
         "provenance": {}},
    ]
    # a perfectly consistent but HALF-WRONG model: it inverts the lateral frame
    # (calls left right) yet stays self-consistent. Accuracy must drop while the
    # antisymmetry violation rate stays 0 -- that separation is the whole point.
    preds = {"v_1": "superior", "v_1_inv": "inferior",
             "v_2": "right", "v_2_inv": "left",
             "v_3": "about 45 mm"}
    acc = evaluate(qas, preds)
    cons = consistency(qas, preds)
    assert acc["per_category"]["longitudinal"]["accuracy"] == 1.0
    assert acc["per_category"]["laterality"]["accuracy"] == 0.0
    assert acc["per_category"]["extent"]["accuracy"] == 1.0, "45 within 25% of 40"
    assert cons["pairs_checked"] == 2 and cons["violations"] == 0, cons

    # a self-contradicting model: same answer both directions
    preds2 = {"v_1": "superior", "v_1_inv": "superior",
              "v_2": "left", "v_2_inv": "left", "v_3": "40"}
    cons2 = consistency(qas, preds2)
    assert cons2["violations"] == 2, cons2
    acc2 = evaluate(qas, preds2)
    # note it still scores 50% on each directional category -- accuracy alone
    # cannot distinguish this from the coherent model above
    assert acc2["per_category"]["laterality"]["accuracy"] == 0.5, acc2

    assert normalise("The lesion is on the left side") == "left"
    assert normalise("banana") is None
    assert parse_number("approximately 12.5 mm") == 12.5

    print("selftest OK — accuracy and consistency separate a coherent-but-wrong "
          "model (0% acc, 0% violations) from a noisy one (50% acc, 100% violations)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
