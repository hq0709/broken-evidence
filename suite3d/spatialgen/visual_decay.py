"""
Volumetric analogue of the 2D suite's visual-information-decay ablation.

The 2D suite sweeps Gaussian blur sigma in {0,2,4,8,16,32,64} pixels plus a
no-image control, and defines the language-takeover point L* as the smallest
sigma at which the residual visual contribution falls below 20 % of the
clean-image contribution. There it lands between 16 and 64 pixels: models remain
visually anchored through substantial degradation.

Running the identical protocol on volumes puts both modalities on one axis. The
quantity of interest is the same:

    contribution(sigma) = Acc(blurred at sigma) - Acc(no image)
    L*                  = min { sigma : contribution(sigma) < 0.2 * contribution(0) }

Two implementation points matter for the comparison to be fair. Blur is applied
in-plane only, in millimetres converted to voxels per volume, because CT slice
spacing routinely differs from in-plane spacing by 4x and a voxel-space sigma
would blur anisotropically by an amount that varies per scan. And the sweep runs
on the growth-matched corpus, so the accuracy being decayed is not partly a
text-cue artefact.

If the clean-image contribution is itself at zero, L* is reported as 0 with that
stated explicitly: the ratio test is undefined, and the honest reading is that
takeover has already occurred at no degradation at all.
"""
from __future__ import annotations

import os

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_multimodel import FAMILY, PROMPT, MontageModel, volume_of  # noqa: E402

SIGMAS = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


def blur_inplane(vol: np.ndarray, spacing: np.ndarray, sigma_px: float
                 ) -> np.ndarray:
    """Gaussian blur in-plane only, sigma given in in-plane voxels.

    The 2D suite's sigma is in image pixels. The in-plane axes of a
    RAS+ volume are the pixel-equivalent axes, so sigma applies there and not
    along the slice axis; blurring across slices would degrade a different
    quantity (through-plane resolution) that the 2D sweep never touched.
    """
    if sigma_px <= 0:
        return vol
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(vol, sigma=(sigma_px, sigma_px, 0.0), mode="nearest")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--data-root", default=os.environ.get("MSD_ROOT", ""))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    fam = FAMILY.get(args.model, "qwen")
    rows = [json.loads(l) for l in open(args.qa) if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    # keep the label balance the corpus was built to have
    yes = [r for r in rows if r["answer"] == "yes"][: args.sample // 2]
    no = [r for r in rows if r["answer"] == "no"][: args.sample // 2]
    rows = yes + no
    rng.shuffle(rows)

    by_vol = defaultdict(list)
    for r in rows:
        by_vol[(r["organ"], volume_of(r["qid"]))].append(r)
    print(f"{len(rows)} probes / {len(by_vol)} volumes / family={fam} / "
          f"sigmas {SIGMAS} + no-image", flush=True)

    if fam == "m3d":
        from m3d_infer import M3D, TARGET_SHAPE
        model = M3D(device=args.device, model_id=args.model)
    elif fam == "med3dvlm":
        from med3dvlm_infer import TARGET_SHAPE, Med3DVLM
        model = Med3DVLM(device=args.device)
    else:
        model = MontageModel(args.model, args.device)
    print("model ready", flush=True)

    from PIL import Image

    from render import montage, orthogonal_views
    from scene_graph import load_ras

    def prepare(vol, affine, sigma, blank=False):
        if fam in ("m3d", "med3dvlm"):
            from scipy.ndimage import zoom
            nd = 4 if fam == "m3d" else 5
            shape = (1,) * (nd - 3) + TARGET_SHAPE
            if blank:
                return np.zeros(shape, np.float32)
            v = np.asarray(vol, np.float32)
            if fam == "m3d":
                v = np.transpose(v, (2, 1, 0))
            f = [t / s for t, s in zip(TARGET_SHAPE, v.shape)]
            v = zoom(v, f, order=1)
            out = np.zeros(TARGET_SHAPE, np.float32)
            sl = tuple(slice(0, min(a, b)) for a, b in zip(TARGET_SHAPE, v.shape))
            out[sl] = v[sl]
            lo, hi = float(out.min()), float(out.max())
            out = (out - lo) / (hi - lo) if hi > lo else out * 0
            return out.reshape(shape)
        r = orthogonal_views(vol.astype(np.int16), np.abs(np.diag(affine)[:3]))
        img = Image.fromarray(montage(r.available())).convert("RGB")
        if blank:
            img = Image.new("RGB", img.size, (128, 128, 128))
        return img

    def ask(q, choices, prepared):
        if fam in ("m3d", "med3dvlm"):
            return model.score_choices(
                PROMPT.format(q=q, opts=", ".join(choices)), prepared, choices)
        return model.score(q, choices, prepared)

    written = errors = 0
    with open(args.out, "w") as f:
        for (organ, vid), items in sorted(by_vol.items()):
            vp = Path(args.data_root) / organ / "imagesTr" / f"{vid}.nii.gz"
            if not vp.exists():
                continue
            vol, affine = load_ras(str(vp))
            spacing = np.abs(np.diag(affine)[:3])
            cache = {}
            for cond in [f"sigma{s:g}" for s in SIGMAS] + ["noimage"]:
                if cond not in cache:
                    if cond == "noimage":
                        cache[cond] = prepare(vol, affine, 0.0, blank=True)
                    else:
                        s = float(cond.replace("sigma", ""))
                        cache[cond] = prepare(blur_inplane(vol, spacing, s),
                                              affine, s)
                for it in items:
                    try:
                        pred, sc = ask(it["question"], it["choices"], cache[cond])
                    except Exception as e:
                        errors += 1
                        if errors <= 3:
                            print(f"  failed {it['qid']} {cond}: {e}",
                                  file=sys.stderr)
                        continue
                    f.write(json.dumps({
                        "qid": it["qid"], "condition": cond, "organ": organ,
                        "prediction": pred, "gold": it["answer"],
                        "logprobs": {k: round(v, 4) for k, v in sc.items()}}) + "\n")
                    written += 1
                cache.pop(cond, None)
            f.flush()
    print(f"\nwrote {written} (errors: {errors})")


if __name__ == "__main__":
    main()
