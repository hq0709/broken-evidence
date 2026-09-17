"""
Export a stratified probe sample for radiologist validation.

Purpose
-------
The pipeline's central claim is that ground truth can be COMPUTED rather than
annotated. That claim needs one thing human expertise alone can supply: evidence
that the computed answer is the clinically correct answer, and that the
generated question is one a radiologist would actually ask. A few hundred
expert judgements validate several thousand generated items -- which makes the
annotation-free argument stronger, not weaker.

Design decisions
----------------
* The lesion and the target structure are OUTLINED on the exported images. The
  question under test is whether the computed geometric relation matches expert
  judgement of the same anatomy, not whether a reader can find the lesion.
  Leaving them unmarked would measure detection, a different study.
* The reader never sees the growth amount's relation to the true gap, nor the
  computed answer: the form ships answers in a separate key file.
* Sampling is stratified over organ, severity tier and margin difficulty, so
  agreement can be reported per stratum rather than as a single number that
  hides where the method fails.
"""
from __future__ import annotations

import os
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from families3d import tier_of  # noqa: E402
from render import _rescale, montage_rgb, to_display, window  # noqa: E402
from run_pipeline import label_map  # noqa: E402
from scene_graph import load_ras  # noqa: E402


def outline(mask2d: np.ndarray) -> np.ndarray:
    """One-pixel boundary of a 2D mask."""
    from scipy.ndimage import binary_erosion

    return mask2d & ~binary_erosion(mask2d)


def closest_approach(lesion: np.ndarray, target: np.ndarray,
                     spacing: np.ndarray) -> tuple[int, int, int]:
    """Voxel in `target` closest to `lesion`, in millimetres.

    The probe asks about a 3D minimum surface distance. A plane through the lesion
    centroid need not contain the target at all, and for targets approaching cranio-caudally (sternum, T8-T10,
    lung lobes, heart) the judgement being asked was structurally impossible from
    that plane. The closest-approach point is the only location where the quantity
    in question is visible.
    """
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~lesion, sampling=spacing)
    masked = np.where(target, dist, np.inf)
    return np.unravel_index(int(np.argmin(masked)), masked.shape)


def _best_slice(m: np.ndarray, ax: int, other: np.ndarray | None = None) -> int:
    """Slice index along `ax` maximising joint visibility, over the FULL axis.

    Windowing around the closest-approach point failed for pairs separated by
    more than the window (a liver lesion and the heart, 28.5 mm apart, sit ~20
    slices apart in z), so the search is unrestricted.
    """
    counts = m.sum(axis=tuple(i for i in range(3) if i != ax))
    if other is not None:
        oc = other.sum(axis=tuple(i for i in range(3) if i != ax))
        joint = np.minimum(counts, oc)
        if joint.max() > 0:
            return int(np.argmax(joint))
    return int(np.argmax(counts))


def render_case(vol: np.ndarray, lesion: np.ndarray, target: np.ndarray,
                spacing: np.ndarray, preset: str = "soft_tissue") -> np.ndarray:
    """Three orthogonal panels, guaranteeing each structure appears in one.

    Two structures separated along all three axes cannot share a plane, so
    "both outlines in every panel" is not an achievable guarantee and asserting
    it fails on real cases (heart against a liver lesion, sternum against a
    cranio-caudal approach). The achievable and sufficient guarantee is that the
    reader sees each structure at least once across the three views.

    Each axis takes the slice maximising min(lesion, target) pixels; where the
    two never co-occur on that axis it falls back to whichever structure is
    still unseen. A 10 mm scale bar is drawn per panel from that panel's own
    in-plane spacing, since the question is metric.
    """
    idx = [_best_slice(lesion, a, target) for a in (2, 1, 0)]
    axes = [2, 1, 0]

    def counts(ax: int, k: int) -> tuple[int, int]:
        sl = [slice(None)] * 3
        sl[ax] = k
        return int(lesion[tuple(sl)].sum()), int(target[tuple(sl)].sum())

    # deterministic repair: if a structure is absent from all three chosen
    # slices, re-point the panel that contributes least to showing the other
    seen_l = any(counts(a, k)[0] for a, k in zip(axes, idx))
    seen_t = any(counts(a, k)[1] for a, k in zip(axes, idx))
    if not seen_l:
        idx[0] = _best_slice(lesion, axes[0])
    if not seen_t:
        idx[1] = _best_slice(target, axes[1])

    def panel(ax: int, k: int) -> np.ndarray:
        sl = [slice(None)] * 3
        sl[ax] = k
        g = window(vol, preset)[tuple(sl)]
        le = outline(lesion[tuple(sl)])
        tg = outline(target[tuple(sl)])
        # to_display, not `.T[::-1]`: the latter is the vertical flip only, so
        # the axial and coronal panels came out mirrored against the montage every
        # model was scored on -- a patient left/right swap, in the images used to
        # argue a human-versus-model comparison.
        g, le, tg = (to_display(a, ax) for a in (g, le, tg))
        # Resample to square pixels, as render.orthogonal_views does. Without this
        # the reader saw coronal and sagittal panels compressed ~7x along z while
        # being asked for a metric judgement, and the panels did not match what any
        # model was shown.
        rem = [i for i in (0, 1, 2) if i != ax]
        sr, sc = float(spacing[rem[1]]), float(spacing[rem[0]])
        g = _rescale(g, (sr, sc))
        le, tg = ((_rescale((m * 255).astype(np.uint8), (sr, sc)) > 127)
                  for m in (le, tg))
        le, tg = (m[: g.shape[0], : g.shape[1]] for m in (le, tg))
        rgb = np.dstack([g] * 3)
        rgb[le[: g.shape[0], : g.shape[1]]] = [255, 60, 60]
        rgb[tg[: g.shape[0], : g.shape[1]]] = [60, 200, 255]
        n = max(4, int(round(10.0 / min(sr, sc))))
        h, w = rgb.shape[:2]
        rgb[h - 8:h - 5, 6:6 + min(n, w - 12)] = [255, 255, 255]
        return rgb

    # Composite through the shared rule so this image and the identification
    # control's `identified` arm are the same picture; zero-padding against a
    # black gutter here while the control scaled up against a grey one meant the
    # reader and the model were not being shown the same thing.
    return montage_rgb([panel(a, k) for a, k in zip(axes, idx)])


def stratified(items: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Even coverage over organ x severity x margin difficulty.

    Two constraints beyond stratification, both of which matter for a
    blind reader study:

    1. **At most one probe per (lesion, target).** The corpus is built from
       matched pairs that differ only in growth amount and therefore always have
       opposite answers. Showing a reader both members leaks: having answered
       one, the other is determined without looking at the image.

    2. **Balanced answers.** The corpus is exactly 50/50 by construction, but a
       stratified draw does not inherit that -- the first export came out 73 no
       / 32 yes, so a reader answering "no" throughout would have scored 69.5 %
       and any agreement statistic would have been read against the wrong
       baseline. We balance explicitly and assert it before writing.
    """
    rng = random.Random(seed)

    def difficulty(r):
        p = r["provenance"]
        d = abs(p["growth_mm"] - p["gap_mm"])
        return "tight" if d <= 3 else ("moderate" if d <= 8 else "wide")

    def pair_key(r):
        p = r["provenance"]
        return (r["qid"].rsplit("_g", 1)[0], p["target"])

    buckets = defaultdict(list)
    for r in items:
        buckets[(r["organ"], tier_of(r["provenance"]["target"]),
                 difficulty(r), r["answer"])].append(r)

    # draw yes and no independently so the halves can be matched exactly
    half = n // 2
    picked = {"yes": [], "no": []}
    seen_pairs = set()
    for ans in ("yes", "no"):
        keys = sorted(k for k in buckets if k[3] == ans)
        pools = {}
        for k in keys:
            v = buckets[k][:]
            rng.shuffle(v)
            pools[k] = v
        # Round-robin across strata until the half is full, rather than one pass
        # of `half // len(keys)` per stratum. With n=104 there are ~29 strata per
        # answer, so that quotient floors to 0, the max(1, ...) turns it into a
        # single item per stratum, and the draw stopped at 58 of the 104 asked
        # for -- silently, since nothing downstream compared the two numbers.
        # The first pass still visits every stratum once, so coverage is
        # unchanged; the later passes only top the sample up.
        while len(picked[ans]) < half and any(pools.values()):
            progressed = False
            for k in keys:
                if len(picked[ans]) >= half:
                    break
                while pools[k]:
                    r = pools[k].pop()
                    pk = pair_key(r)
                    if pk in seen_pairs:
                        continue
                    seen_pairs.add(pk)
                    picked[ans].append(r)
                    progressed = True
                    break
            if not progressed:
                break

    m = min(len(picked["yes"]), len(picked["no"]), half)
    rng.shuffle(picked["yes"])
    rng.shuffle(picked["no"])
    out = picked["yes"][:m] + picked["no"][:m]
    rng.shuffle(out)

    assert len({pair_key(r) for r in out}) == len(out), \
        "a matched pair reached the reader form"
    assert sum(r["answer"] == "yes" for r in out) == \
        sum(r["answer"] == "no" for r in out), "reader form is not label-balanced"
    if len(out) < n:
        # Say so rather than return a short sample quietly: reader time is the
        # scarce resource here and the caller chose n for a reason.
        print(f"stratified: {len(out)} of {n} requested -- the constraint that "
              f"binds is one probe per (lesion, target) at balanced labels",
              file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfqa-dirs", nargs="+", required=True)
    ap.add_argument("--data-root", default="data/msd")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = []
    for d in args.cfqa_dirs:
        organ = Path(d).name.replace("cfqa_", "")
        for f in sorted((Path(d) / "qa").glob("*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if r.get("kind") != "growth_contact":
                    continue
                r["organ"] = organ
                r["volume_id"] = f.stem
                items.append(r)
    if not items:
        sys.exit("no growth_contact items found")

    sample = stratified(items, args.n, args.seed)
    outdir = Path(args.outdir)
    (outdir / "images").mkdir(parents=True, exist_ok=True)

    lm = label_map()
    name2lab = {v: k for k, v in lm.items()}
    from lesion_binding import LESION_LABEL, find_lesions
    from PIL import Image

    by_vol = defaultdict(list)
    for r in sample:
        by_vol[(r["organ"], r["volume_id"])].append(r)

    form, key = [], []
    n_ok = 0
    for (organ, vid), rs in sorted(by_vol.items()):
        # Resolve the mask from the repository root.
        root = Path(os.environ.get("VOLUMETRIC_ROOT",
                                   Path(__file__).resolve().parent.parent))
        segp = root / f"cfqa_{organ}" / "seg_cache" / f"{vid}_seg.nii.gz"
        if not segp.exists():
            for base in (root,):
                found = next(base.glob(f"out_*/seg_cache/{vid}_seg.nii.gz"), None) \
                    or next(base.glob(f"cfqa_*/seg_cache/{vid}_seg.nii.gz"), None)
                if found is not None:
                    segp = found
                    break
        labp = Path(args.data_root) / organ / "labelsTr" / f"{vid}.nii.gz"
        volp = Path(args.data_root) / organ / "imagesTr" / f"{vid}.nii.gz"
        if not (segp.exists() and labp.exists() and volp.exists()):
            continue

        seg, affine = load_ras(str(segp))
        gt, _ = load_ras(str(labp))
        vol, _ = load_ras(str(volp))
        lesions = dict(find_lesions(gt == LESION_LABEL[organ], affine))

        for r in rs:
            # `qid.split("_")[-2]` returned the last token of the TARGET name
            # ("right", "cava", "sternum") and matched no lesion key, so every
            # item silently fell through to the largest component -- the wrong
            # lesion on 104/104 exported cases. Parse the key, and fail loudly.
            m_lid = re.search(r"_(lesion\d+)_", r["qid"])
            if not m_lid:
                raise ValueError(f"cannot parse lesion id from qid {r['qid']}")
            lid = m_lid.group(1)
            if lid not in lesions:
                raise ValueError(f"{lid} not among lesions for {r['qid']} "
                                 f"(have {sorted(lesions)})")
            lesion = lesions[lid]
            tname = r["provenance"]["target"]
            if tname not in name2lab:
                continue
            target = seg == name2lab[tname]
            if not target.any():
                continue

            img = render_case(vol, lesion, target, np.abs(np.diag(affine)[:3]))
            # both structures must
            # actually be visible in what the reader is shown
            has_les = ((img[:, :, 0] > 180) & (img[:, :, 1] < 120)
                       & (img[:, :, 2] < 120)).sum()
            has_tgt = ((img[:, :, 0] < 120) & (img[:, :, 1] > 150)
                       & (img[:, :, 2] > 200)).sum()
            if not (has_les and has_tgt):
                raise AssertionError(
                    f"{r['qid']}: rendered panels missing "
                    f"{'lesion' if not has_les else 'target'} outline")
            fn = f"{r['qid'].replace('/', '_')}.png"
            Image.fromarray(img).save(outdir / "images" / fn)

            form.append({
                "probe_id": r["qid"], "image": f"images/{fn}", "organ": organ,
                "question": r["question"],
                "reader_answer__yes_no_unsure": "",
                "clinical_relevance__1to5": "",
                "severity_tier_shown": tier_of(tname),
                "severity_agree__yes_no": "",
                "comment": "",
            })
            key.append({"probe_id": r["qid"], "computed_answer": r["answer"],
                        "target": tname, "gap_mm": r["provenance"]["gap_mm"],
                        "growth_mm": r["provenance"]["growth_mm"],
                        "severity": tier_of(tname)})
            n_ok += 1

    with open(outdir / "reader_form.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(form[0].keys()))
        w.writeheader()
        w.writerows(form)
    # the key ships separately so the reader form can be handed over blind
    with open(outdir / "ANSWER_KEY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0].keys()))
        w.writeheader()
        w.writerows(key)

    print(f"exported {n_ok} probes to {outdir}")
    print("organ:", dict(Counter(r["organ"] for r in form)))
    print("severity:", dict(Counter(r["severity_tier_shown"] for r in form)))
    print("answer balance (key):",
          dict(Counter(k["computed_answer"] for k in key)))


def selftest() -> None:
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    o = outline(m)
    assert o.sum() > 0 and o.sum() < m.sum(), (o.sum(), m.sum())
    assert not o[10, 10], "interior must not be outlined"
    assert o[5, 5], "corner must be on the boundary"

    sp = np.array([1.0, 1.0, 4.0])          # anisotropic, as CT is

    # THE failure mode this selftest exists for: a target that does not
    # intersect the lesion's centroid plane. The previous synthetic case put
    # both structures on the same z, so it passed while 46 % of the real export
    # shipped with no target outline at all. Here the target sits entirely
    # cranial to the lesion, approaching along z.
    vol = np.full((30, 30, 12), -1000, dtype=np.int16)
    vol[10:20, 10:20, 2:11] = 60
    lesion = np.zeros((30, 30, 12), bool)
    lesion[12:16, 12:16, 2:4] = True        # centroid at z ~ 2.5
    target = np.zeros((30, 30, 12), bool)
    target[12:16, 12:16, 9:11] = True       # z ~ 9.5, no overlap in z

    zc = closest_approach(lesion, target, sp)
    assert target[zc], "closest-approach point must lie inside the target"
    assert 3 <= zc[2] <= 10, f"expected the point between the two, got {zc}"

    img = render_case(vol, lesion, target, sp)
    assert img.shape[2] == 3 and img.dtype == np.uint8, img.shape
    red = (img[:, :, 0] > 180) & (img[:, :, 1] < 120) & (img[:, :, 2] < 120)
    cyan = (img[:, :, 0] < 120) & (img[:, :, 1] > 150) & (img[:, :, 2] > 200)
    assert red.any(), "lesion outline missing"
    assert cyan.any(), "target outline missing on a cranio-caudal approach"

    # a centroid-plane render would have failed here; confirm it actually does,
    # so the assertion above is known to be discriminating rather than vacuous
    zl = int(round(np.argwhere(lesion)[:, 2].mean()))
    assert not target[:, :, zl].any(), \
        "synthetic case does not exercise the failure mode it was built for"

    # lesion-id parsing must not silently fall back to the largest component
    import re as _re
    qid = "liver_1_lesion4_lung_middle_lobe_right_g11.8"
    assert _re.search(r"_(lesion\d+)_", qid).group(1) == "lesion4"
    assert qid.split("_")[-2] == "right", \
        "a split on underscores reads the side, not the lesion id"

    items = [{"qid": f"liver_1_lesion1_aorta_g{g}", "answer": a,
              "provenance": {"target": "aorta", "gap_mm": 10.0, "growth_mm": g},
              "organ": "Task06_Lung"}
             for g, a in ((6.0, "no"), (18.0, "yes"), (7.0, "no"),
                          (22.0, "yes"))]
    for k, it in enumerate(items):
        it["qid"] = f"liver_{k}_lesion1_aorta_g{it['provenance']['growth_mm']}"
    s = stratified(items, 4)
    assert sum(r["answer"] == "yes" for r in s) == sum(r["answer"] == "no"
                                                       for r in s)

    print("selftest OK — outlining marks edges only; the closest-approach point "
          "lies inside the target and between the two structures; all three "
          "panels render with lesion (red) and target (cyan) visible on a "
          "cranio-caudal approach that a centroid-plane render would miss; "
          "lesion-id parsing is regression-guarded; stratification stays "
          "label-balanced")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
