"""
Four-arm ROI evaluation, giving the 3D analogue of BrokenEvidence's Visual Grounding Ratio.

Arms
----
  full        untouched volume
  roi_only    keep the lesion, the target, and the corridor between them
  roi_masked  blank exactly that region, keep everything else
  zero        all zeros (the cruder control used earlier, kept for comparability)

VGR = Acc(roi_only) - Acc(roi_masked), in percentage points.

Both ROI arms keep a realistic volume, so a difference between them isolates the
evidence region rather than the presence of an image at all -- which is what the
all-zero arm conflates. Regions come from the same geometry that produced the
gold answer, so they are exact.

Only growth_contact probes are run: they name a single target structure, so the
evidence region is well defined. A "which structure would it reach first"
question depends on several targets at once and has no single ROI.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from lesion_binding import LESION_LABEL, find_lesions  # noqa: E402
from roi_control import apply_condition, evidence_region  # noqa: E402
from run_pipeline import label_map  # noqa: E402
from scene_graph import load_ras  # noqa: E402

CONDITIONS = ["full", "roi_only", "roi_masked", "zero"]


def lesion_key(qid: str) -> str | None:
    for part in qid.split("_"):
        if part.startswith("lesion"):
            return part
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--seg-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True,
                    help="m3d | qwen | qwen32b | internvl")
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fill", default="local",
                    help="masked-region fill: local | air | <float>. air is the volume 1st percentile, which on CT is background air and carves a detectable cavity; local (default) uses the median of the surrounding tissue shell.")
    ap.add_argument("--sample", type=int, default=None,
                    help="stratified subsample of matched PAIRS. VGR needs a "
                         "precise estimate, not the whole set; keeping pairs "
                         "intact preserves the exact 50%% chance level.")
    args = ap.parse_args()

    rows = [json.loads(l) for f in sorted(Path(args.qa).glob("*.jsonl"))
            for l in open(f) if l.strip()]
    rows = [r for r in rows if r.get("kind") == "growth_contact" and r.get("choices")]
    if args.limit:
        rows = rows[: args.limit]
    if args.sample:
        import random
        byp = defaultdict(list)
        for r in rows:
            byp[r.get("pair_id") or r["qid"]].append(r)
        pairs = [v for v in byp.values() if len(v) == 2]
        random.Random(0).shuffle(pairs)
        rows = [r for p in pairs[: max(1, args.sample // 2)] for r in p]

    by_vol = defaultdict(list)
    for r in rows:
        by_vol["_".join(r["qid"].split("_")[:2])].append(r)
    print(f"{len(rows)} probes over {len(by_vol)} volumes, "
          f"condition={args.condition}, model={args.model}", flush=True)

    task = Path(args.task_dir)
    organ = task.name
    lesion_label = LESION_LABEL[organ]
    lm = label_map()
    name2lab = {v: k for k, v in lm.items()}

    if args.model == "m3d":
        from m3d_infer import M3D, TARGET_SHAPE
        model = M3D(device=args.device, model_id=args.model)
    else:
        # reuse the montage wrapper the main benchmark uses, rather than a
        # second hardcoded Qwen loader: the ROI arms now cover models whose
        # loading path differs (InternVL3 needs the native image-text-to-text
        # class, not Qwen2_5_VLForConditionalGeneration)
        from PIL import Image

        from render import montage, orthogonal_views
        from run_multimodel import MODEL_ID, MontageModel
        if args.model not in MODEL_ID:
            raise SystemExit(f"unknown model tag {args.model!r}; known: "
                             + ", ".join(sorted(MODEL_ID)))
        model = MontageModel(MODEL_ID[args.model], args.device)
    print("model ready", flush=True)

    def prepare(volume, affine):
        """Volume -> model-ready input, computed ONCE per (lesion, target).

        Each (lesion, target) pair yields a matched question pair, so the
        resample to 32x256x256 (M3D) or the orthogonal-view render (Qwen) is
        the dominant cost and runs once per pair rather than per question.
        """
        if args.model == "m3d":
            from scipy.ndimage import zoom
            v = np.asarray(volume, dtype=np.float32)
            v = np.transpose(v, (2, 1, 0))
            f = [t / s for t, s in zip(TARGET_SHAPE, v.shape)]
            v = zoom(v, f, order=1)
            out = np.zeros(TARGET_SHAPE, np.float32)
            sl = tuple(slice(0, min(a, b)) for a, b in zip(TARGET_SHAPE, v.shape))
            out[sl] = v[sl]
            lo, hi = float(out.min()), float(out.max())
            out = (out - lo) / (hi - lo) if hi > lo else out * 0
            return out[None]
        r = orthogonal_views(volume.astype(np.int16),
                             np.abs(np.diag(affine)[:3]))
        return Image.fromarray(montage(r.available())).convert("RGB")

    def ask(question, choices, prepared):
        if args.model == "m3d":
            return model.score_choices(
                f"{question} Answer with exactly one of: {', '.join(choices)}.",
                prepared, choices)
        return model.score(question, choices, prepared)

    written = errors = 0
    with open(args.out, "w") as fout:
        for vid, items in sorted(by_vol.items()):
            segp = Path(args.seg_cache) / f"{vid}_seg.nii.gz"
            volp = task / "imagesTr" / f"{vid}.nii.gz"
            labp = task / "labelsTr" / f"{vid}.nii.gz"
            if not (segp.exists() and volp.exists() and labp.exists()):
                continue
            seg, affine = load_ras(str(segp))
            vol, _ = load_ras(str(volp))
            gt, _ = load_ras(str(labp))
            lesions = dict(find_lesions(gt == lesion_label, affine))
            if not lesions:
                continue

            cache = {}
            for r in items:
                lk = lesion_key(r["qid"])
                # explicit None check: `a or b` on numpy arrays raises, because
                # the truth value of a multi-element array is ambiguous
                lesion = lesions.get(lk)
                if lesion is None:
                    lesion = max(lesions.values(), key=lambda m: m.sum())
                tname = r["provenance"]["target"]
                if tname not in name2lab:
                    continue
                target = seg == name2lab[tname]
                if not target.any():
                    continue
                ck = (lk, tname)
                if ck not in cache:
                    roi = evidence_region(lesion, target, affine)
                    cache[ck] = prepare(apply_condition(vol, roi, args.condition,
                                                        fill=args.fill),
                                        affine)
                try:
                    pred, sc = ask(r["question"], r["choices"], cache[ck])
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  failed {r['qid']}: {e}", file=sys.stderr)
                    continue
                fout.write(json.dumps({
                    "qid": r["qid"], "prediction": pred, "gold": r["answer"],
                    "condition": args.condition, "target": tname,
                    # recorded per row: two arms of one VGR must agree on
                    # this field, and it cannot be inferred from the file.
                    "fill": args.fill,
                    "logprobs": {k: round(v, 4) for k, v in sc.items()}}) + "\n")
                written += 1
            fout.flush()
            print(f"  {vid}: {len(items)}", flush=True)

    print(f"\nwrote {written} (errors: {errors})")


if __name__ == "__main__":
    main()
