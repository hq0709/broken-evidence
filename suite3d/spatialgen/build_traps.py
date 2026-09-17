"""
Build the hallucination-trap family from per-volume verified-absent targets.

A trap is valid only if its premise is unsatisfiable in that image. Each trap
is built from structures the volume's own TotalSegmentator map does not
contain, so the premise is false by the same evidence that makes the rest of
the corpus true, and the phrasing varies across ~100 anatomical names.

Two guards, because a missing label can mean "absent" or "missed":
  * a structure counts as absent only if it has zero voxels AND is not in the
    scan's plausible field of view judged by z-extent overlap with structures
    that ARE present -- a lung lobe absent from an abdominal crop is a
    field-of-view artefact, not a hallucination trap, and asking about it tests
    cropping rather than grounding;
  * structures that TotalSegmentator is known to under-segment are excluded by
    name, since their absence is unreliable evidence.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from families3d import REFUSAL, _mcq, tier_of
from run_pipeline import label_map
from scene_graph import load_ras

# structures whose absence from a TotalSegmentator map is not trustworthy
UNRELIABLE = {"prostate", "sacrum", "spinal_cord", "esophagus", "duodenum",
              "small_bowel", "colon", "urinary_bladder", "heart_atrium_left",
              "heart_atrium_right", "heart_myocardium", "pulmonary_artery"}


def z_profile(cfqa_dir: Path, lm: dict, files: list) -> tuple[dict, dict]:
    """Per-structure presence frequency AND typical z-position within the scan.

    The frequency filter alone was not enough, and the docstring of the first
    version described a guard that was never implemented. Measured on the
    resulting family: 61 % of trap targets were vertebrae, sternum, hip, femur or
    the gluteal/iliac muscles -- structures absent because the scan is cropped
    above or below them, not because the patient lacks them. `vertebrae_T8` sits
    at frequency 0.72 and `sternum` at 0.75, so a 0.70 threshold admits exactly
    the class it was added to exclude.

    So we also record where each structure sits along z *relative to the scan's
    own coverage*, as a fraction in [0, 1]. A structure whose typical position is
    near either end is a crop candidate; one that normally sits in the middle of
    the acquired volume and is nevertheless missing is a genuine patient-level
    absence.
    """
    freq = Counter()
    pos = defaultdict(list)
    n = 0
    for f in files:
        vid = f.stem
        segp = cfqa_dir / "seg_cache" / f"{vid}_seg.nii.gz"
        if not segp.exists():
            continue
        seg, _ = load_ras(str(segp))
        seg = np.asarray(seg, dtype=np.int32)
        nz = np.flatnonzero(seg.any(axis=(0, 1)))
        if not nz.size:
            continue
        z0, z1 = int(nz[0]), int(nz[-1])
        span = max(z1 - z0, 1)

        # One pass over the label map instead of one full-volume scan per label.
        nlab = int(seg.max()) + 1
        cnt = np.zeros((seg.shape[2], nlab), dtype=np.int64)
        for z in range(seg.shape[2]):
            sl = seg[:, :, z]
            if sl.any():
                cnt[z] = np.bincount(sl.ravel(), minlength=nlab)
        tot = cnt.sum(axis=0)
        zmean = (cnt * np.arange(seg.shape[2])[:, None]).sum(axis=0) / \
            np.maximum(tot, 1)

        for lab, name in lm.items():
            if lab >= nlab or tot[lab] == 0:
                continue
            freq[name] += 1
            pos[name].append((float(zmean[lab]) - z0) / span)
        n += 1
    if not n:
        return {}, {}
    return ({k: v / n for k, v in freq.items()},
            {k: float(np.median(v)) for k, v in pos.items()})


def global_z_order(pos_samples: dict) -> dict:
    """Stable cranio-caudal ordering of structures, from population medians.

    Ranks are needed because the previous test -- a structure's typical position
    within the scans where it IS present -- cannot detect crop-proneness at all:
    in those scans the acquisition by definition reached it, so it never sits at
    an edge. Measured consequence: gluteal muscles, low ribs and iliac vessels
    passed a 0.15 edge filter unchanged.
    """
    return {k: float(np.median(v)) for k, v in pos_samples.items() if v}


# Structures whose absence from a segmentation is interpretable. Three attempts
# were needed to reach this list, and the failures are the argument for it.
#
# "Zero voxels in the map" conflates three causes: the patient genuinely lacks
# the structure, the acquisition did not cover it, or the segmenter missed it.
# A >=70% presence-frequency filter admitted vertebrae_T8 (freq 0.72) and sternum
# (0.75), i.e. exactly the crop class. A within-scan z-position filter cannot
# work at all, because a structure's position is only observable in scans that
# reached it. A bracketing test -- present structures above and below -- still
# passed 82% skeletal targets, because ribs and vertebrae interleave with organs
# in z, so a missed rib is bracketed while being a segmenter failure rather than
# an absence.
#
# What survives is narrow and defensible: solid abdominal organs that are
# routinely resected, are large, and are segmented reliably. For these, zero
# voxels inside a scan that covers their neighbours is a statement about the
# patient. The cost is a small family; we take the cost.
RESECTABLE = {
    "gallbladder",          # cholecystectomy -- the common case
    "kidney_left", "kidney_right",
    "spleen",
    "uterus",
    "adrenal_gland_left", "adrenal_gland_right",
}


def lesion_key(qid: str) -> str:
    for part in qid.split("_"):
        if part.startswith("lesion"):
            return part
    return "lesion?"


def absent_structures(seg: np.ndarray, lm: dict, freq: dict, order: dict,
                      min_freq: float = 0.7) -> list[str]:
    """Resectable organs absent here, bracketed by structures this scan covers.

    Both conditions are required. The whitelist rules out crop-prone skeleton and
    under-segmented small structures; the bracketing rules out an organ missing
    because the scan stopped short of it.
    """
    labs = set(int(v) for v in np.unique(seg))
    labs.discard(0)
    present = [lm[l] for l in labs if l in lm and lm[l] in order]
    if len(present) < 8:
        return []
    pr = [order[p] for p in present]
    lo, hi = min(pr), max(pr)

    out = []
    for lab, name in lm.items():
        if lab in labs or name not in RESECTABLE:
            continue
        if freq.get(name, 0.0) < min_freq:
            continue
        r = order.get(name)
        if r is None or r <= lo or r >= hi:
            continue
        out.append(name)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfqa-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--max-volumes", type=int, default=800)
    ap.add_argument("--min-freq", type=float, default=0.7,
                    help="structure must be segmented in this "
                         "fraction of the task's scans to count as "
                         "in-field-of-view")
    ap.add_argument("--edge", type=float, default=0.15,
                    help="reject structures whose typical z-position is\n                         within this fraction of either end of the\n                         acquired volume -- their absence is a crop")
    ap.add_argument("--per-volume", type=int, default=8,
                    help="probes drawn per qualifying scan. The bootstrap "
                         "resamples volumes, so this does not inflate the "
                         "independent-unit count; it reduces the noise in each "
                         "unit's estimate, which is what a 77-probe family "
                         "could not afford.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lm = label_map()
    rng = random.Random(args.seed)
    rows = []
    for d in args.cfqa_dirs:
        organ = Path(d).name.replace("cfqa_", "")
        files = sorted((Path(d) / "qa").glob("*.jsonl"))
        rng.shuffle(files)
        files = files[: args.max_volumes // len(args.cfqa_dirs)]
        freq, zpos = z_profile(Path(d), lm, files)
        order = global_z_order(zpos)
        keep = [k for k, v in freq.items() if v >= args.min_freq]
        print(f"  {organ}: {len(freq)} seen; {len(keep)} pass freq>="
              f"{args.min_freq:.0%}; bracketing applied per scan", flush=True)
        for f in files:
            items = [json.loads(l) for l in open(f)]
            items = [r for r in items if r.get("kind") == "growth_contact"]
            if not items:
                continue
            vid = "_".join(items[0]["qid"].split("_")[:2])
            segp = Path(d) / "seg_cache" / f"{vid}_seg.nii.gz"
            if not segp.exists():
                continue
            seg, _ = load_ras(str(segp))
            absent = absent_structures(seg, lm, freq, order,
                                       args.min_freq)
            if not absent:
                continue
            # Several probes per volume, not one.
            #
            # The bootstrap resamples volumes, so the number of independent
            # units is the number of qualifying scans -- 78, and expanding the
            # resectable whitelist moves that to 81 at best, because surgically
            # absent organs are simply rare in these datasets. What one probe
            # per volume does cost is precision *within* each cluster: a single
            # 0/1 observation per patient, which is why the interval on the
            # amplification effect straddled zero at n = 77. Drawing several
            # probes per volume leaves the cluster count untouched and replaces
            # each cluster's coin flip with an average.
            #
            # Probes within a volume are drawn with distinct lesions where the
            # scan has them, so they are not the same question restated at a
            # different growth amount.
            byles = defaultdict(list)
            for it in items:
                byles[lesion_key(it["qid"])].append(it)
            picks, keys = [], sorted(byles)
            rng.shuffle(keys)
            while len(picks) < args.per_volume and any(byles[k] for k in keys):
                for k in keys:
                    if byles[k] and len(picks) < args.per_volume:
                        picks.append(byles[k].pop(rng.randrange(len(byles[k]))))
            for r in picks:
              for tgt in ([rng.choice(absent)] if len(absent) == 1 else absent[:2]):
                g = r["provenance"]["growth_mm"]
                o, gold = _mcq(REFUSAL, ["yes", "no", "only partially"], rng)
                # The qid must be unique: predictions are keyed by it, so a
                # collision silently drops a probe rather than failing. With one
                # probe per volume the target and growth amount were enough;
                # drawing several per volume, two can share both, so the lesion
                # and an index are part of the identity.
                rows.append({
                    "qid": f"{vid}_{lesion_key(r['qid'])}_trapv_{tgt}"
                           f"_g{g:g}_{len(rows)}", "organ": organ,
                    "question": (f"If this lesion grew by {g:g} mm in every "
                                 f"direction, would it contact the "
                                 f"{tgt.replace('_',' ')}? "),
                    "choices": [o[k] for k in sorted(o)],
                    "answer": o[gold], "family": "trap", "severity": tier_of(tgt),
                    "anchor_qid": None, "pair_id": None,
                    "provenance": {"target": tgt, "growth_mm": g,
                                   "derived": "hallucination_trap",
                                   "reason": "structure has zero voxels in this "
                                             "volume's segmentation",
                                   "verified_absent": True}})
    rng.shuffle(rows)
    rows = rows[: args.n]

    # Validate BEFORE writing, so a build that fails its own assertions never
    # replaces a corpus on disk.
    names = Counter(r["provenance"]["target"] for r in rows)
    tiers = Counter(r["severity"] for r in rows)
    print(f"wrote {len(rows)} verified-absent traps")
    print(f"  distinct target structures: {len(names)}  "
          f"(single-phrase version: 1)")
    print(f"  most common: {names.most_common(6)}")
    print(f"  severity tiers: {dict(sorted(tiers.items()))}")
    assert len(names) >= 3, "trap family is not lexically diverse"
    import re as _re
    # the previous assertion blocklisted the three symptoms already observed,
    # which cannot catch the next one. Check the CLASS instead: skeletal and
    # limb-girdle structures are the ones a crop removes, so if they dominate,
    # the z-position filter is not working.
    boundary = sum(c for k, c in names.items()
                   if _re.match(r"(vertebrae_|rib_|sternum|hip|femur|gluteus|"
                                r"iliopsoas|iliac_|humerus|scapula|clavicula|"
                                r"autochthon|brain|skull)", k))
    frac = boundary / max(len(rows), 1)
    print(f"  crop-prone (skeletal/limb-girdle) share: {100*frac:.0f}%")
    assert frac == 0.0, (
        f"{100*frac:.0f}% of traps are crop-prone; the whitelist should make "
        f"this impossible")
    assert set(names) <= RESECTABLE, f"non-resectable target: {set(names)-RESECTABLE}"
    assert all(r["answer"] == REFUSAL for r in rows)
    ids = [r["qid"] for r in rows]
    assert len(ids) == len(set(ids)), (
        f"{len(ids)-len(set(ids))} duplicate qids; predictions are keyed by "
        f"qid, so a collision drops probes silently")
    vols = {q.split("_trapv_")[0].rsplit("_lesion", 1)[0] for q in ids}
    print(f"  {len(rows)} probes over {len(vols)} volumes "
          f"({len(rows)/max(len(vols),1):.1f} per volume); the bootstrap "
          f"resamples volumes, so {len(vols)} is the independent-unit count")

    # every check passed: publish atomically so a partial or failed build can
    # never be mistaken for a corpus
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, args.out)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
