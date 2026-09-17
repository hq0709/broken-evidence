"""
End-to-end driver:  CT volume  ->  segmentation  ->  scene graph  ->  verifiable spatial QA

Usage
-----
  # one volume, quick look
  python run_pipeline.py --input scan.nii.gz --outdir out/ --fast

  # a directory of volumes, reusing cached segmentations
  python run_pipeline.py --input ct_dir/ --outdir out/ --limit 50

Design notes
------------
* Segmentation is cached per volume. It is the expensive step (~10-30 s/volume
  on an H100 with --fast) and is pure function of the input, so a rerun of the
  QA logic must never re-segment.
* Everything downstream of segmentation is CPU-only and cheap, which is what
  makes the criterion experiment affordable: iterate on QA design without
  touching the GPU.
* GPU 0 on this box is occupied by a vLLM server; --device lets you pin to 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scene_graph import (  # noqa: E402
    build_scene_graph, build_structures, check_antisymmetry,
    check_transitivity, load_ras,
)
from qa_gen import (  # noqa: E402
    filter_prior_answerable, generate, hard_negatives, validate_provenance,
)


def segment(volume: Path, cache_dir: Path, fast: bool, device: str) -> Path:
    """Run TotalSegmentator unless a cached multilabel mask already exists."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{volume.name.split('.')[0]}_seg.nii.gz"
    if out.exists():
        return out

    from totalsegmentator.python_api import totalsegmentator

    totalsegmentator(
        str(volume), str(out),
        fast=fast, ml=True, quiet=True, device=device,
    )
    return out


def label_map() -> dict[int, str]:
    """TotalSegmentator label id -> anatomical name."""
    from totalsegmentator.map_to_binary import class_map
    return dict(class_map["total"])


def process_one(
    volume: Path, outdir: Path, fast: bool, device: str, max_per_cat: int,
    do_adjacency: bool = True,
) -> dict:
    t0 = time.time()
    vid = volume.name.split(".")[0]

    seg_path = segment(volume, outdir / "seg_cache", fast, device)
    t_seg = time.time() - t0

    seg, affine = load_ras(str(seg_path))
    img, _ = load_ras(str(volume))

    structures = build_structures(seg, affine, label_map(), image=img)
    relations = build_scene_graph(structures, seg, affine,
                                  do_adjacency=do_adjacency)

    # our own graph is correct by construction; assert it, because a violation
    # here means a bug in the geometry, not an interesting model failure
    anti = check_antisymmetry(relations)
    trans = check_transitivity(relations)
    if anti or trans:
        print(f"  !! WARNING {vid}: own graph violates axioms "
              f"({len(anti)} anti, {len(trans)} trans) — geometry bug", flush=True)

    qas = generate(structures, relations, vid, max_per_category=max_per_cat)
    qas += hard_negatives(qas)

    # Drop questions a blind text model can answer from naming or canonical
    # anatomy. Measured on an unfiltered shard: a text-only LLM with no image
    # scored 89.1% on what this removes and 58.6% (== the trivial baseline) on
    # what it keeps. Items in the first group are not spatial supervision.
    qas, prior_reasons = filter_prior_answerable(qas)

    # Hard gate: every directional answer must be recomputable from its own
    # provenance. This is the property that distinguishes this data from
    # report-derived ground truth, so a shard that fails it is worthless and
    # must not be written.
    bad = validate_provenance(qas)
    if bad:
        raise ValueError(
            f"provenance does not reproduce {len(bad)}/{len(qas)} answers, "
            f"e.g. {bad[0]}")

    (outdir / "graphs").mkdir(parents=True, exist_ok=True)
    (outdir / "qa").mkdir(parents=True, exist_ok=True)
    with open(outdir / "graphs" / f"{vid}.json", "w") as f:
        json.dump({"structures": {k: asdict(v) for k, v in structures.items()},
                   "relations": [asdict(r) for r in relations]}, f)
    with open(outdir / "qa" / f"{vid}.jsonl", "w") as f:
        for qa in qas:
            f.write(json.dumps(asdict(qa)) + "\n")

    return {"volume_id": vid, "n_structures": len(structures),
            "n_relations": len(relations), "n_qa": len(qas),
            "prior_answerable_dropped": sum(prior_reasons.values()),
            "prior_reasons": prior_reasons,
            "seg_seconds": round(t_seg, 1),
            "total_seconds": round(time.time() - t0, 1),
            "axiom_violations": len(anti) + len(trans)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="NIfTI file or directory")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fast", action="store_true",
                    help="TotalSegmentator 3mm mode; much faster, coarser")
    ap.add_argument("--device", default="gpu:1",
                    help="gpu:1 by default — gpu:0 hosts a vLLM server here")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-per-category", type=int, default=40)
    ap.add_argument("--no-adjacency", action="store_true",
                    help="skip adjacency relations. They dominate runtime on "
                         "large volumes (chest CT is 79M voxels vs 14M for "
                         "abdominal), and the constancy test's central claim is "
                         "about DIRECTIONAL relations, which need only bounding "
                         "boxes.")
    args = ap.parse_args()

    inp, outdir = Path(args.input), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if inp.is_dir():
        # skip AppleDouble sidecars ('._foo.nii.gz') that macOS-created tars
        # carry -- they match *.nii* but are not gzip and fail every reader.
        vols = sorted([p for p in inp.rglob("*.nii*")
                       if "_seg" not in p.name and not p.name.startswith("._")])
    else:
        vols = [inp]
    if args.limit:
        vols = vols[: args.limit]
    if not vols:
        sys.exit(f"no NIfTI volumes found under {inp}")

    print(f"processing {len(vols)} volume(s) on {args.device}", flush=True)
    stats, failures = [], []
    for i, v in enumerate(vols, 1):
        try:
            s = process_one(v, outdir, args.fast, args.device,
                            args.max_per_category,
                            do_adjacency=not args.no_adjacency)
            stats.append(s)
            print(f"[{i}/{len(vols)}] {s['volume_id']}: "
                  f"{s['n_structures']} structures, {s['n_relations']} relations, "
                  f"{s['n_qa']} QA ({s['total_seconds']}s)", flush=True)
        except Exception as e:                      # keep going; report at end
            failures.append({"volume": str(v), "error": repr(e)})
            print(f"[{i}/{len(vols)}] FAILED {v.name}: {e}", flush=True)

    with open(outdir / "summary.json", "w") as f:
        json.dump({"stats": stats, "failures": failures}, f, indent=1)

    if stats:
        tot_qa = sum(s["n_qa"] for s in stats)
        mean_t = sum(s["total_seconds"] for s in stats) / len(stats)
        print(f"\ndone: {len(stats)} ok, {len(failures)} failed, "
              f"{tot_qa} QA total, {mean_t:.1f}s/volume mean")
        # segmentation failure rate is a real result to report in the paper,
        # since pathology-heavy scans are where TotalSegmentator degrades
        if failures:
            print(f"segmentation/pipeline failure rate: "
                  f"{len(failures)}/{len(vols)} = {len(failures)/len(vols):.1%}")


if __name__ == "__main__":
    main()
