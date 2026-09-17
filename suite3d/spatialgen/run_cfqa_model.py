"""
Evaluate a 3D medical VLM on counterfactual anatomical QA, sighted vs blind.

Why counterfactuals
-------------------
On factual anatomy a blind model can score by recalling that livers sit below
hearts, or by echoing what a radiology report would typically say. A
counterfactual has no such fallback -- "if this lesion grew 12 mm, would it reach the aorta?"
describes a state that never occurred, for a lesion whose position is specific
to this patient. Any blind accuracy above chance therefore has to come from
answer-format bias, not knowledge.

Two independent readouts:
  * accuracy against an exactly balanced 50% chance level;
  * PAIR CONSISTENCY -- each growth question is emitted with a matched partner
    that differs only in the growth amount, straddling the true gap. Answering
    both members the same way is geometrically impossible, and detecting it
    needs no reference to which answer is correct.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from m3d_infer import M3D, TARGET_SHAPE, preprocess_volume  # noqa: E402


def volume_id_of(qid: str) -> str:
    """CFQA ids look like '<volume>_<lesion|resect>_<rest>'."""
    parts = qid.split("_")
    for i, p in enumerate(parts):
        if p.startswith("lesion") or p == "resect":
            return "_".join(parts[:i])
    return "_".join(parts[:2])


def build_question(q: dict) -> str:
    ch = q.get("choices")
    if ch:
        return f"{q['question']} Answer with exactly one of: {', '.join(ch)}."
    return f"{q['question']} Reply with a number only."


def pair_consistency(rows: list[dict], qa: dict[str, dict]) -> dict:
    """Fraction of matched pairs answered identically -- a geometric impossibility."""
    by_pair = defaultdict(list)
    for r in rows:
        pid = qa[r["qid"]].get("pair_id")
        if pid:
            by_pair[pid].append(r["prediction"])
    complete = {k: v for k, v in by_pair.items() if len(v) == 2}
    bad = sum(1 for v in complete.values() if v[0] == v[1])
    return {"pairs": len(complete), "identical": bad,
            "violation_rate": bad / len(complete) if complete else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="cfqa/qa directory")
    ap.add_argument("--volumes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--blind", action="store_true")
    args = ap.parse_args()

    qa_dir = Path(args.qa)
    rows = [json.loads(l) for f in sorted(qa_dir.glob("*.jsonl"))
            for l in open(f) if l.strip()]
    choice_rows = [r for r in rows if r.get("choices")]
    by_vol = defaultdict(list)
    for r in choice_rows:
        by_vol[volume_id_of(r["qid"])].append(r)
    print(f"{len(choice_rows)} choice QA over {len(by_vol)} volumes "
          f"({'BLIND' if args.blind else 'sighted'})", flush=True)

    model = M3D(device=args.device)
    print("model ready", flush=True)

    vols_dir = Path(args.volumes)
    zero = np.zeros((1, *TARGET_SHAPE), dtype=np.float32)
    written = errors = 0
    with open(args.out, "w") as out:
        for vid, items in sorted(by_vol.items()):
            hits = [p for p in vols_dir.rglob(f"{vid}.nii*")
                    if not p.name.startswith("._")]
            if not hits:
                print(f"  {vid}: volume missing, skipping {len(items)}")
                continue
            vol = zero if args.blind else preprocess_volume(str(hits[0]))
            for r in items:
                try:
                    pred, sc = model.score_choices(
                        build_question(r), vol, r["choices"])
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  failed {r['qid']}: {e}", file=sys.stderr)
                    continue
                out.write(json.dumps({
                    "qid": r["qid"], "prediction": pred, "gold": r["answer"],
                    "kind": r["kind"],
                    "logprobs": {k: round(v, 4) for k, v in sc.items()},
                }) + "\n")
                written += 1
            out.flush()
            print(f"  {vid}: {len(items)} answered", flush=True)
    print(f"\nwrote {written} (errors: {errors})")


def summarise(pred_path: Path, qa_dir: Path) -> dict:
    qa = {json.loads(l)["qid"]: json.loads(l)
          for f in sorted(qa_dir.glob("*.jsonl")) for l in open(f) if l.strip()}
    rows = [json.loads(l) for l in open(pred_path) if l.strip()]
    ok = sum(r["prediction"] == r["gold"] for r in rows)
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r["prediction"] == r["gold"])
    # chance is exact here: growth pairs are balanced by construction
    chance = sum(1 / len(qa[r["qid"]]["choices"]) for r in rows) / max(len(rows), 1)
    return {
        "n": len(rows), "accuracy": ok / len(rows) if rows else 0.0,
        "chance": chance,
        "per_kind": {k: {"n": len(v), "acc": sum(v) / len(v)}
                     for k, v in sorted(by_kind.items())},
        "pred_dist": dict(Counter(r["prediction"] for r in rows)),
        "consistency": pair_consistency(rows, qa),
    }


def selftest() -> None:
    assert volume_id_of("lung_001_lesion3_aorta_g12") == "lung_001"
    assert volume_id_of("lung_010_resect_lung_upper_lobe_right") == "lung_010"

    qa = {"a": {"pair_id": "p1", "choices": ["no", "yes"]},
          "b": {"pair_id": "p1", "choices": ["no", "yes"]},
          "c": {"pair_id": "p2", "choices": ["no", "yes"]},
          "d": {"pair_id": "p2", "choices": ["no", "yes"]}}
    # p1 answered identically (impossible), p2 answered as a proper pair
    rows = [{"qid": "a", "prediction": "yes"}, {"qid": "b", "prediction": "yes"},
            {"qid": "c", "prediction": "no"}, {"qid": "d", "prediction": "yes"}]
    c = pair_consistency(rows, qa)
    assert c == {"pairs": 2, "identical": 1, "violation_rate": 0.5}, c

    q = build_question({"question": "Would it contact the aorta?",
                        "choices": ["no", "yes"]})
    assert "no, yes" in q, q

    print("selftest OK — volume ids parsed from both question families, "
          "pair-consistency detects the impossible identical-answer case")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
