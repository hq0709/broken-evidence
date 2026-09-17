"""
Build a counterfactual anatomical QA set from MSD lesion datasets.

Pairs each volume's expert lesion mask with a TotalSegmentator anatomy map and
emits questions whose answers are computed by anatomy_sim. Segmentations are
reused from an existing seg_cache when present, since that is the expensive
step (~100 s per chest volume).

  python run_cfqa.py --task ../data/Task06_Lung --seg-cache ../out_lung/seg_cache \
                     --outdir ../cfqa --limit 8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from counterfactual_qa import check_balance, generate  # noqa: E402
from lesion_binding import LESION_LABEL, find_lesions  # noqa: E402
from run_pipeline import label_map, segment  # noqa: E402
from scene_graph import load_ras  # noqa: E402


def process(image: Path, label: Path, seg_dir: Path, outdir: Path,
            lesion_label: int, fast: bool, device: str,
            max_lesions: int = 3) -> dict:
    vid = image.name.split(".")[0]
    cached = seg_dir / f"{vid}_seg.nii.gz"
    seg_path = cached if cached.exists() else segment(
        image, outdir / "seg_cache", fast, device)

    anat_arr, affine = load_ras(str(seg_path))
    gt, gt_affine = load_ras(str(label))
    if gt.shape != anat_arr.shape or not np.allclose(gt_affine, affine, atol=1e-3):
        raise ValueError("label and anatomy grids disagree")

    lm = label_map()
    anatomy = {}
    for lab, name in lm.items():
        m = anat_arr == lab
        if m.any():
            anatomy[name] = m

    lesions = dict(find_lesions(gt == lesion_label, affine))
    if not lesions:
        return {"volume_id": vid, "n_lesions": 0, "n_qa": 0}

    # Cap lesions per volume. Liver cases are often multifocal and were yielding
    # ~93 QA from a single patient (pancreas: ~18), which both dominates runtime
    # -- one distance transform per lesion -- and inflates the apparent sample
    # size, since all those items share one patient. Keep the largest few.
    if len(lesions) > max_lesions:
        biggest = sorted(lesions, key=lambda k: -int(lesions[k].sum()))[:max_lesions]
        lesions = {k: lesions[k] for k in biggest}

    qs = generate(lesions, anatomy, affine, vid)
    (outdir / "qa").mkdir(parents=True, exist_ok=True)
    with open(outdir / "qa" / f"{vid}.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps(asdict(q)) + "\n")

    return {"volume_id": vid, "n_lesions": len(lesions), "n_qa": len(qs),
            "kinds": dict(Counter(q.kind for q in qs)),
            "balance": check_balance(qs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seg-cache", default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device", default="gpu:1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-lesions", type=int, default=3,
                    help="largest N lesions per volume; caps both runtime and "
                         "per-patient item counts")
    args = ap.parse_args()

    task = Path(args.task)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seg_dir = Path(args.seg_cache) if args.seg_cache else outdir / "seg_cache"

    lesion_label = LESION_LABEL.get(task.name)
    if lesion_label is None:
        sys.exit(f"unknown MSD task {task.name}")

    images = sorted(p for p in (task / "imagesTr").glob("*.nii*")
                    if not p.name.startswith("._"))
    if args.limit:
        images = images[: args.limit]
    print(f"{task.name}: {len(images)} volumes, lesion label {lesion_label}",
          flush=True)

    stats, failures = [], []
    for i, img in enumerate(images, 1):
        lab = task / "labelsTr" / img.name
        if not lab.exists():
            failures.append({"volume": img.name, "error": "no label"})
            continue
        try:
            s = process(img, lab, seg_dir, outdir, lesion_label,
                        args.fast, args.device, args.max_lesions)
            stats.append(s)
            print(f"[{i}/{len(images)}] {s['volume_id']}: {s['n_lesions']} lesions "
                  f"-> {s['n_qa']} QA {s.get('kinds', {})}", flush=True)
        except Exception as e:
            failures.append({"volume": img.name, "error": repr(e)})
            print(f"[{i}/{len(images)}] FAILED {img.name}: {e}", flush=True)

    total = sum(s["n_qa"] for s in stats)
    pairs = sum(s.get("balance", {}).get("complete_pairs", 0) for s in stats)
    kinds: Counter = Counter()
    for s in stats:
        kinds.update(s.get("kinds", {}))
    with open(outdir / "summary.json", "w") as f:
        json.dump({"stats": stats, "failures": failures}, f, indent=1)

    print(f"\n{len(stats)} ok / {len(failures)} failed | {total} QA, "
          f"{pairs} matched pairs")
    print(f"kinds: {dict(kinds)}")
    if failures:
        print(f"failure rate: {len(failures)}/{len(images)} "
              f"= {len(failures)/len(images):.1%}")


if __name__ == "__main__":
    main()
