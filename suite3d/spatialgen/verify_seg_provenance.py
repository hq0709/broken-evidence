"""
Does a regenerated segmentation cache reproduce the corpus it must explain?

Why this is not optional
------------------------
`cfqa_*/seg_cache/` is excluded from the repository, so every image condition of
the identification control has to draw its cyan target outline from masks
regenerated here, with whatever TotalSegmentator version is installed. The committed `provenance.gap_mm` was computed from the ORIGINAL
masks. If the regenerated mask of the aorta differs, the outline shown to the
model is not the structure the stored answer refers to -- silently, because
nothing in the run would fail.

`run_pipeline.segment` also has a mode switch (`--fast`, 3 mm) that the corpus
does not record. The two modes give different surface gaps, so the mode is not
a free choice either.

So this script recomputes, from the regenerated masks and the MSD lesion
annotation, exactly what `counterfactual_qa.growth_pairs` computed -- the
lesion-to-target surface gap -- and compares it with the stored value. It
reports two things:

  * agreement of the gap in millimetres, which is a property of the masks;
  * the DECISION-relevant quantity: on how many probes the reference rule
    (contact iff gap <= growth) changes its answer when recomputed. That is the
    number that matters, because a 0.4 mm disagreement 20 mm away from the
    growth amount changes nothing, while the same disagreement straddling it
    flips the label.

Run it per mode (`--seg-suffix _fast` vs `""`) to decide the mode empirically
rather than by assumption.

Usage
-----
    python verify_seg_provenance.py --repo . --msd-root /path/MSD \
        --organ Task03_Liver --seg-suffix _fast --volumes 8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anatomy_sim import surface_gap_mm            # noqa: E402
from lesion_binding import LESION_LABEL, find_lesions   # noqa: E402
from run_pipeline import label_map                # noqa: E402
from scene_graph import load_ras                  # noqa: E402


def lesion_key(qid: str) -> str | None:
    for part in qid.split("_"):
        if part.startswith("lesion"):
            return part
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--msd-root", required=True)
    ap.add_argument("--organ", required=True)
    ap.add_argument("--seg-suffix", default="")
    ap.add_argument("--volumes", type=int, default=8,
                    help="how many volumes to check (0 = all available)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    qa_dir = repo / f"cfqa_{args.organ}" / "qa"
    seg_dir = repo / f"cfqa_{args.organ}" / f"seg_cache{args.seg_suffix}"
    task = Path(args.msd_root) / args.organ
    name2lab = {v: k for k, v in label_map().items()}
    lesion_label = LESION_LABEL[args.organ]

    by_vol: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(qa_dir.glob("*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("provenance", {}).get("gap_mm") is None:
                continue
            by_vol["_".join(r["qid"].split("_")[:2])].append(r)

    avail = [v for v in sorted(by_vol)
             if (seg_dir / f"{v}_seg.nii.gz").exists()
             and (task / "imagesTr" / f"{v}.nii.gz").exists()]
    if not avail:
        raise SystemExit(f"no volume has both a mask in {seg_dir} and an image "
                         f"in {task / 'imagesTr'}")
    if args.volumes:
        avail = avail[: args.volumes]

    deltas: list[float] = []
    flips: list[dict] = []
    missing_target = n_probes = 0
    for vid in avail:
        gt, affine = load_ras(str(task / "labelsTr" / f"{vid}.nii.gz"))
        seg, _ = load_ras(str(seg_dir / f"{vid}_seg.nii.gz"))
        lesions = dict(find_lesions(gt == lesion_label, affine))
        cache: dict[tuple[str, str], float] = {}
        for r in by_vol[vid]:
            lk, p = lesion_key(r["qid"]), r["provenance"]
            tname = p["target"]
            if lk not in lesions or tname not in name2lab:
                missing_target += 1
                continue
            ck = (lk, tname)
            if ck not in cache:
                tmask = seg == name2lab[tname]
                if not tmask.any():
                    missing_target += 1
                    continue
                cache[ck] = surface_gap_mm(lesions[lk], tmask, affine)
            got, want = cache[ck], float(p["gap_mm"])
            n_probes += 1
            deltas.append(got - want)
            g = float(p["growth_mm"])
            if (got <= g) != (want <= g):
                flips.append({"qid": r["qid"], "stored_gap": want,
                              "recomputed_gap": round(got, 2), "growth": g,
                              "stored_answer": r["answer"]})
        print(f"  {vid}: {len(by_vol[vid])} probes, {len(cache)} distinct "
              f"(lesion,target) pairs", flush=True)

    if not deltas:
        raise SystemExit("nothing comparable: no probe had both a lesion "
                         "component and a target mask")
    d = np.abs(np.asarray(deltas))
    summary = {
        "organ": args.organ, "seg_suffix": args.seg_suffix,
        "volumes": len(avail), "probes_compared": n_probes,
        "targets_absent_from_mask": missing_target,
        "gap_abs_diff_mm": {"median": round(float(np.median(d)), 3),
                            "p90": round(float(np.percentile(d, 90)), 3),
                            "max": round(float(d.max()), 3),
                            "exact_le_0.01mm": int((d <= 0.01).sum())},
        "answer_flips": len(flips),
        "answer_flip_rate_pct": round(100.0 * len(flips) / n_probes, 2),
        "flip_examples": flips[:10],
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "flip_examples"},
                     indent=1))
    if flips:
        print(f"\n{len(flips)} of {n_probes} probes change answer under the "
              f"regenerated masks; first few:")
        for f in flips[:10]:
            print("  ", f)
    else:
        print(f"\nno probe changes answer: the regenerated masks reproduce "
              f"every stored label on {n_probes} probes over {len(avail)} volumes")
    if args.json_out:
        json.dump(summary, open(args.json_out, "w"), indent=1)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
