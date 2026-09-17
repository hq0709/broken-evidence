"""
Input richness: are three slices too little, independent of annotation?

The identification control varies WHICH slices are shown and WHETHER the two
structures are marked. It does not vary HOW MUCH of the volume is shown, and the
main input is three planes out of a few hundred. Three slices may simply not
contain the geometry the question asks about; this experiment tests that
directly.

Annotation is held at the `identified` level in every arm, so the only thing
that varies is how much volume the model sees:

    slices3   three orthogonal planes through the joint-visibility slices.
              Identical to `identified`; present so the comparison is
              within-run rather than against another script's numbers.
    slices9   three planes per axis, spanning the lesion's extent along that
              axis, as a 3x3 grid. Same views, three times the sampling.
    axial25   a 5x5 grid of CONTIGUOUS axial slices centred on the lesion, so
              the lesion's whole cranio-caudal extent is on screen at the
              acquisition's own slice interval.
    zoom      the `identified` panels cropped to the bounding box of the two
              structures plus a margin. Same slices, same annotation, but the
              pixels are spent on the structures being asked about instead of
              on the whole body.

`zoom` is the arm that separates "too little volume" from "too few pixels on the
thing that matters": it shows strictly less anatomy than `slices3` while showing
the lesion and the target far larger.

Every arm resamples to square pixels and draws its 10 mm bar from the
post-resample spacing, exactly as the control does, so a millimetre judgement is
possible in all of them.

Usage
-----
    python run_input_richness.py --qa cfqa_Task03_Liver/qa \
        --task-dir $MSD_ROOT/Task03_Liver --seg-cache cfqa_Task03_Liver/seg_cache \
        --model qwen32b --arm axial25 --subset matched --device cuda:0 \
        --out results_new/id_Task03_Liver_qwen32b_richness-axial25.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from export_reader_study import _best_slice, outline              # noqa: E402
from lesion_binding import LESION_LABEL, find_lesions             # noqa: E402
from render import montage_rgb, to_display, window                # noqa: E402
from run_identification_control import (                          # noqa: E402
    LESION_RGB, TARGET_RGB, _isotropic, legend_for, lesion_key,
    matched_subset_keys,
)
from scene_graph import load_ras                                  # noqa: E402

ARMS = ["slices3", "slices9", "axial25", "zoom"]
AXES = [2, 1, 0]          # axial, coronal, sagittal -- montage's sorted order


def panel(grey3d, lesion, target, spacing, axis: int, index: int,
          scalebar: bool = True, crop=None) -> np.ndarray:
    """One annotated, isotropically resampled RGB panel."""
    sl = [slice(None)] * 3
    sl[axis] = int(np.clip(index, 0, grey3d.shape[axis] - 1))
    g = grey3d[tuple(sl)]
    le = outline(lesion[tuple(sl)])
    tg = outline(target[tuple(sl)])
    g, le, tg = (to_display(x, axis) for x in (g, le, tg))
    rem = [i for i in (0, 1, 2) if i != axis]
    g, le, tg, iso = _isotropic(g, le, tg,
                                float(spacing[rem[1]]), float(spacing[rem[0]]))
    if crop is not None:
        rows = np.flatnonzero((le | tg).any(axis=1))
        cols = np.flatnonzero((le | tg).any(axis=0))
        if rows.size and cols.size:
            from scipy.ndimage import zoom as _zoom

            full_h = g.shape[0]
            m = int(round(crop / iso))
            r0, r1 = max(0, rows[0] - m), min(g.shape[0], rows[-1] + m + 1)
            c0, c1 = max(0, cols[0] - m), min(g.shape[1], cols[-1] + m + 1)
            g, le, tg = g[r0:r1, c0:c1], le[r0:r1, c0:c1], tg[r0:r1, c0:c1]
            # Magnify the crop back to the height the uncropped panel had. Without
            # this the arm would vary two things at once -- what is in frame and
            # how many pixels the input has -- and a drop could be read as "less
            # context hurts" when it was "a tenth of the vision tokens". Now the
            # input budget is held roughly fixed and only the framing changes.
            f = full_h / max(g.shape[0], 1)
            if f > 1.01:
                g = _zoom(g, (f, f), order=1).astype(np.uint8)
                le, tg = ((_zoom(m_.astype(np.uint8) * 255, (f, f), order=1) > 127)
                          for m_ in (le, tg))
                le, tg = (m_[: g.shape[0], : g.shape[1]] for m_ in (le, tg))
                iso = iso / f          # the bar must shrink with the magnification
    rgb = np.dstack([g] * 3)
    rgb[le] = LESION_RGB
    rgb[tg] = TARGET_RGB
    if scalebar and rgb.shape[1] > 20 and rgb.shape[0] > 12:
        n = max(4, int(round(10.0 / iso)))
        h, w = rgb.shape[:2]
        rgb[h - 8:h - 5, 6:6 + min(n, w - 12)] = [255, 255, 255]
    return rgb


def grid(rows: list[list[np.ndarray]], pad: int = 8) -> np.ndarray:
    """Stack rows of panels. Rows are composited with the shared montage rule,
    then padded to a common width with the same 128-grey filler."""
    made = [montage_rgb(r, pad=pad) for r in rows]
    w = max(m.shape[1] for m in made)
    out = []
    for m in made:
        if m.shape[1] < w:
            m = np.hstack([m, np.full((m.shape[0], w - m.shape[1], 3), 128, np.uint8)])
        out.append(m)
        out.append(np.full((pad, w, 3), 128, np.uint8))
    return np.vstack(out[:-1])


def spread(mask: np.ndarray, axis: int, k: int, shape_n: int) -> list[int]:
    """`k` indices spanning the mask's extent along `axis`, inclusive."""
    idx = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
    if idx.size == 0:
        c = shape_n // 2
        return [c] * k
    lo, hi = int(idx[0]), int(idx[-1])
    if hi == lo:
        return [lo] * k
    return [int(round(lo + (hi - lo) * i / (k - 1))) for i in range(k)]


def render_arm(vol, lesion, target, spacing, arm: str) -> tuple[np.ndarray, dict]:
    grey = window(vol, "soft_tissue")
    if arm in ("slices3", "zoom"):
        idx = [_best_slice(lesion, a, target) for a in AXES]

        def seen(m):
            return any(m[tuple(slice(None) if i != a else k for i in range(3))].sum()
                       for a, k in zip(AXES, idx))
        if not seen(lesion):
            idx[0] = _best_slice(lesion, AXES[0])
        if not seen(target):
            idx[1] = _best_slice(target, AXES[1])
        crop = 15.0 if arm == "zoom" else None
        used = list(zip(AXES, idx))
        panels = [panel(grey, lesion, target, spacing, a, k, crop=crop)
                  for a, k in used]
        img = montage_rgb(panels)
    elif arm == "slices9":
        rows, used = [], []
        for a in AXES:
            ks = spread(lesion, a, 3, vol.shape[a])
            used += [(a, k) for k in ks]
            rows.append([panel(grey, lesion, target, spacing, a, k) for k in ks])
        img = grid(rows)
    elif arm == "axial25":
        a = 2
        zc = int(np.median(np.flatnonzero(lesion.any(axis=(0, 1))))) \
            if lesion.any() else vol.shape[a] // 2
        ks = [int(np.clip(zc - 12 + i, 0, vol.shape[a] - 1)) for i in range(25)]
        used = [(a, k) for k in ks]
        rows = [[panel(grey, lesion, target, spacing, a, k) for k in ks[r * 5:(r + 1) * 5]]
                for r in range(5)]
        img = grid(rows)
    else:
        raise SystemExit(f"unknown arm {arm!r}")

    # Distinct (axis, index) planes only: axial25 clips at the volume edge, so a
    # lesion near the top would otherwise have the same plane counted repeatedly
    # and report more voxels shown than the model was given.
    shown = {"lesion": 0, "target": 0}
    for a, k in sorted(set(used)):
        sl = [slice(None)] * 3
        sl[a] = int(np.clip(k, 0, vol.shape[a] - 1))
        shown["lesion"] += int(lesion[tuple(sl)].sum())
        shown["target"] += int(target[tuple(sl)].sum())
    return img, {"arm": arm, "n_panels": len(used),
                 "n_distinct_planes": len(set(used)),
                 "slices": [[int(a), int(k)] for a, k in used],
                 "lesion_voxels_shown": shown["lesion"],
                 "target_voxels_shown": shown["target"],
                 "pixels": int(img.shape[0]) * int(img.shape[1])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--qa", required=True)
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--seg-cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--subset", choices=["matched", "all"], default="matched")
    ap.add_argument("--sample", type=int, default=0,
                    help="subsample by pair, seeded; axial25 is 25 panels a probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    qa_path = Path(args.qa)
    files = sorted(qa_path.glob("*.jsonl")) if qa_path.is_dir() else [qa_path]
    items = [json.loads(l) for f in files for l in open(f) if l.strip()]
    if args.subset == "matched":
        keep = matched_subset_keys(qa_path)
        if keep is not None:
            items = [r for r in items if r["qid"] in keep]
    if args.sample:
        import random
        by_pair = defaultdict(list)
        for r in items:
            by_pair[r.get("pair_id") or r["qid"]].append(r)
        full = sorted(p for p, v in by_pair.items() if len(v) == 2)
        random.Random(args.seed).shuffle(full)
        picked = []
        for p in full:
            if len(picked) + 2 > args.sample:
                break
            picked += by_pair[p]
        items = sorted(picked, key=lambda r: r["qid"])
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no probes selected")
    print(f"{len(items)} probes, arm={args.arm}", flush=True)

    task = Path(args.task_dir)
    organ = task.name
    lesion_label = LESION_LABEL[organ]
    from run_multimodel import MODEL_ID, MontageModel
    if args.model not in MODEL_ID:
        raise SystemExit(f"unknown model tag {args.model!r}")
    from PIL import Image
    model = MontageModel(MODEL_ID[args.model], args.device)
    print("model ready", flush=True)

    from run_pipeline import label_map
    name2lab = {v: k for k, v in label_map().items()}
    legend = legend_for("identified")
    by_vol = defaultdict(list)
    for r in items:
        by_vol["_".join(r["qid"].split("_")[:2])].append(r)

    written = skipped = 0
    with open(args.out, "w") as fout:
        for vid, group in sorted(by_vol.items()):
            volp, labp = task / "imagesTr" / f"{vid}.nii.gz", task / "labelsTr" / f"{vid}.nii.gz"
            segp = Path(args.seg_cache) / f"{vid}_seg.nii.gz"
            if not (volp.exists() and labp.exists() and segp.exists()):
                skipped += len(group)
                continue
            vol, affine = load_ras(str(volp))
            gt, _ = load_ras(str(labp))
            seg, _ = load_ras(str(segp))
            spacing = np.abs(np.diag(affine)[:3])
            lesions = dict(find_lesions(gt == lesion_label, affine))
            v16 = vol.astype(np.int16)
            cache: dict[tuple[str, str], tuple] = {}
            for r in group:
                lk = lesion_key(r["qid"])
                tname = r.get("provenance", {}).get("target")
                if lk not in lesions or tname not in name2lab:
                    skipped += 1
                    continue
                ck = (lk, tname)
                if ck not in cache:
                    tmask = seg == name2lab[tname]
                    if not tmask.any():
                        skipped += 1
                        continue
                    img, geom = render_arm(v16, lesions[lk], tmask, spacing, args.arm)
                    cache[ck] = (Image.fromarray(img).convert("RGB"), geom)
                image, geom = cache[ck]
                choices = r.get("choices") or ["no", "yes"]
                pred, lp = model.score(f"{legend} {r['question']}", choices, image)
                fout.write(json.dumps({
                    "qid": r["qid"], "organ": organ,
                    "condition": f"richness-{args.arm}",
                    "prediction": pred, "gold": r["answer"],
                    "pair_id": r.get("pair_id"), "logprobs": lp, "geometry": geom,
                }) + "\n")
                written += 1
                if written % 100 == 0:
                    print(f"  {written} written", flush=True)

    rows = [json.loads(l) for l in open(args.out)]
    acc = 100.0 * sum(r["prediction"] == r["gold"] for r in rows) / max(len(rows), 1)
    px = np.median([r["geometry"]["pixels"] for r in rows]) if rows else 0
    print(f"wrote {args.out}: {written} scored, {skipped} skipped, accuracy {acc:.1f}%, "
          f"median input {px/1e6:.2f} MPx, predictions "
          f"{dict(Counter(r['prediction'] for r in rows))}")


if __name__ == "__main__":
    main()
