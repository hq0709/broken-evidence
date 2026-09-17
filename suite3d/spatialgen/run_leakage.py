"""
Pretraining overlap: is the benchmark's imaging in the models' pretraining?

MSD is public, redistributed widely, and old enough to be in any web-scale
crawl. If a model recognises these volumes, "chance-level accuracy" and
"memorised the corpus" are different stories. The probe has to distinguish two things that a single question cannot:

    organ    which MSD task this volume belongs to. Answerable by looking -- a
             chest CT is the lung task. This is the POSITIVE CONTROL. Without
             it, a model that scores chance on the memorisation questions is
             indistinguishable from one whose image channel is simply dead, and
             this study has six models whose image channel IS effectively dead,
             so the control is not optional here.
    dataset  which public collection this scan comes from, against three
             plausible alternatives. Partly answerable from acquisition style.
    caseid   which case identifier this volume has, against three other real
             identifiers from the same task. Nothing in the pixels carries this.
             Above chance here is memorisation and nothing else.

Scored by likelihood over the option strings, like everything else in this
repository, and asked on the same montage the benchmark uses, so a positive
result would apply to the benchmark's own inputs rather than to some other
rendering of the same data.

Usage
-----
    python run_leakage.py --model qwen32b --probe caseid --volumes 60 \
        --device cuda:0 --out results_new/leak_qwen32b_caseid.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render import montage, orthogonal_views                      # noqa: E402
from scene_graph import load_ras                                  # noqa: E402

PROBES = ["organ", "dataset", "caseid"]
ORGAN_WORD = {"Task03_Liver": "liver", "Task06_Lung": "lung",
              "Task07_Pancreas": "pancreas", "Task10_Colon": "colon"}
DATASETS = ["the Medical Segmentation Decathlon", "the LIDC-IDRI collection",
            "the NIH DeepLesion collection", "the TCIA Pancreas-CT collection"]
QUESTION = {
    "organ": "Which organ's tumour segmentation task does this scan belong to? "
             "Answer with exactly one of: {opts}.",
    "dataset": "Which public dataset is this scan from? "
               "Answer with exactly one of: {opts}.",
    "caseid": "What is this volume's case identifier in its dataset? "
              "Answer with exactly one of: {opts}.",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", choices=PROBES, required=True)
    ap.add_argument("--msd-root", default="")
    ap.add_argument("--volumes", type=int, default=60,
                    help="volumes total, spread evenly over the four tasks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import os
    root = Path(args.msd_root or os.environ.get("MSD_ROOT", ""))
    repo = Path(__file__).resolve().parent.parent
    rng = random.Random(args.seed)

    # volumes the benchmark actually uses, so a hit would be a hit on the corpus
    per = max(1, args.volumes // len(ORGAN_WORD))
    picked, ids_by_organ = [], {}
    for organ in sorted(ORGAN_WORD):
        vids = sorted(p.stem.replace(".nii", "")
                      for p in (repo / f"cfqa_{organ}" / "qa").glob("*.jsonl"))
        ids_by_organ[organ] = vids
        sel = list(vids)
        rng.shuffle(sel)
        picked += [(organ, v) for v in sel[:per]]
    picked.sort()
    print(f"{len(picked)} volumes, probe={args.probe}", flush=True)

    from run_multimodel import MODEL_ID, MontageModel
    if args.model not in MODEL_ID:
        raise SystemExit(f"unknown model tag {args.model!r}")
    from PIL import Image
    model = MontageModel(MODEL_ID[args.model], args.device)
    print("model ready", flush=True)

    written = skipped = 0
    with open(args.out, "w") as fout:
        for organ, vid in picked:
            vp = root / organ / "imagesTr" / f"{vid}.nii.gz"
            if not vp.exists():
                skipped += 1
                continue
            vol, affine = load_ras(str(vp))
            img = Image.fromarray(
                montage(orthogonal_views(vol, np.abs(np.diag(affine)[:3])).available())
            ).convert("RGB")

            qrng = random.Random(f"{args.seed}:{vid}:{args.probe}")
            if args.probe == "organ":
                gold = ORGAN_WORD[organ]
                choices = sorted(ORGAN_WORD.values())
            elif args.probe == "dataset":
                gold = DATASETS[0]
                choices = list(DATASETS)
            else:
                gold = vid
                others = [v for v in ids_by_organ[organ] if v != vid]
                choices = [gold] + qrng.sample(others, min(3, len(others)))
            qrng.shuffle(choices)
            question = QUESTION[args.probe].format(opts=", ".join(choices))
            pred, lp = model.score(question, choices, img)
            fout.write(json.dumps({
                "qid": f"{vid}_{args.probe}", "organ": organ,
                "condition": f"leak-{args.probe}", "prediction": pred,
                "gold": gold, "pair_id": None, "logprobs": lp,
                "choices": choices, "asked": question,
            }) + "\n")
            written += 1

    rows = [json.loads(l) for l in open(args.out)]
    acc = 100.0 * sum(r["prediction"] == r["gold"] for r in rows) / max(len(rows), 1)
    chance = 100.0 / len(rows[0]["choices"]) if rows else float("nan")
    print(f"wrote {args.out}: {written} scored, {skipped} skipped, "
          f"accuracy {acc:.1f}% against {chance:.1f}% chance, "
          f"predictions {dict(Counter(r['prediction'] for r in rows))}")


if __name__ == "__main__":
    main()
