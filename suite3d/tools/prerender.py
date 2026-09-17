"""
Fill the render cache on CPU, in parallel, before the model runs start.

Rendering a probe means decompressing an MSD volume, its lesion annotation and
its anatomy mask -- a quarter of a gigabyte of gzip per liver series, single
threaded -- and every model run was paying that again from scratch. The renders
do not depend on the model, so one pass over the volumes with as many workers as
the box has cores replaces 16 serial passes per organ, and the GPU jobs then
read PNGs.

This box has 224 cores and the GPUs are busy scoring, so this is free capacity.

    python runs/prerender.py --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "spatialgen"))
sys.path.insert(0, str(REPO))

CONDITIONS = ["plain", "bestslice", "overlay", "identified"]

# The input-richness arms draw the same annotation as `identified` but vary how
# much of the volume is shown. They render through run_input_richness.render_arm
# rather than run_identification_control.render, and both return (array, geom),
# so they cache identically. Pre-rendering them here is what lets run_subtasks.py
# ask a sub-task under a richness arm without loading volumes itself.
RICHNESS_ARMS = ["slices3", "slices9", "axial25", "zoom"]


def one_volume(job: tuple) -> tuple[int, int, str]:
    organ, vid, rows, msd_root, cache_dir, conditions = job
    from lesion_binding import LESION_LABEL, find_lesions
    from run_identification_control import cached_render, lesion_key, render
    from run_input_richness import render_arm
    from run_pipeline import label_map
    from scene_graph import load_ras

    task = Path(msd_root) / organ
    segp = REPO / f"cfqa_{organ}" / "seg_cache" / f"{vid}_seg.nii.gz"
    volp, labp = task / "imagesTr" / f"{vid}.nii.gz", task / "labelsTr" / f"{vid}.nii.gz"
    if not (segp.exists() and volp.exists() and labp.exists()):
        return 0, len(rows), f"{organ}/{vid}: inputs missing"

    pairs = set()
    name2lab = {v: k for k, v in label_map().items()}
    for r in rows:
        lk = lesion_key(r["qid"])
        tname = r.get("provenance", {}).get("target")
        if lk and tname in name2lab:
            pairs.add((lk, tname))

    # Nothing to do if every (pair, condition) entry is already on disk: skip
    # the volume read entirely, which is what makes a rerun of this script cheap.
    want = [(lk, t, c) for (lk, t) in sorted(pairs) for c in conditions]
    todo = [w for w in want
            if not (Path(cache_dir) / organ /
                    f"{vid}_{w[0]}_{w[1]}_{w[2]}.png").exists()]
    if not todo:
        return 0, 0, ""

    vol, affine = load_ras(str(volp))
    gt, _ = load_ras(str(labp))
    seg, _ = load_ras(str(segp))
    spacing = np.abs(np.diag(affine)[:3])
    lesions = dict(find_lesions(gt == LESION_LABEL[organ], affine))
    vol16 = vol.astype(np.int16)

    made = failed = 0
    for lk, tname, cond in todo:
        def make(lk=lk, tname=tname, cond=cond):
            if lk not in lesions:
                raise LookupError("lesion component absent")
            tmask = seg == name2lab[tname]
            if not tmask.any():
                raise LookupError("target absent from the mask")
            if cond in RICHNESS_ARMS:
                return render_arm(vol16, lesions[lk], tmask, spacing, cond)
            return render(vol16, lesions[lk], tmask, spacing, cond)
        try:
            cached_render(cache_dir, organ, vid, lk, tname, cond, make)
            made += 1
        except LookupError:
            failed += 1
    return made, failed, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--organs", nargs="*",
                    default=["Task03_Liver", "Task06_Lung", "Task07_Pancreas",
                             "Task10_Colon"])
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS,
                    help="any of " + ", ".join(CONDITIONS + RICHNESS_ARMS))
    ap.add_argument("--msd-root", default=os.environ.get("MSD_ROOT", ""))
    ap.add_argument("--cache", default=str(REPO / "render_cache"))
    ap.add_argument("--subset", choices=["matched", "all"], default="matched")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    keep = None
    if args.subset == "matched":
        from growth_matched import growth_of, matched_subset
        ref = {}
        for line in open(REPO / "common_subset" / "qa" / "all.jsonl"):
            r = json.loads(line)
            ref[r["qid"]] = r["answer"]
        keep = matched_subset(ref, growth_of())

    jobs = []
    for organ in args.organs:
        by_vol = defaultdict(list)
        for f in sorted((REPO / f"cfqa_{organ}" / "qa").glob("*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if keep is None or r["qid"] in keep:
                    by_vol["_".join(r["qid"].split("_")[:2])].append(r)
        jobs += [(organ, vid, rows, args.msd_root, args.cache, args.conditions)
                 for vid, rows in sorted(by_vol.items())]

    print(f"{len(jobs)} volumes x {len(args.conditions)} conditions, "
          f"{args.workers} workers", flush=True)
    made = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (m, f, msg) in enumerate(ex.map(one_volume, jobs), 1):
            made += m
            failed += f
            if msg:
                print(f"  {msg}", flush=True)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} volumes, {made} renders written, "
                      f"{failed} unrenderable", flush=True)
    n = sum(1 for _ in Path(args.cache).rglob("*.png"))
    print(f"\n{made} renders written, {failed} unrenderable; cache holds {n} PNGs")


if __name__ == "__main__":
    main()
