"""
Content-free calibration for the five-option probe families.

The problem this fixes
----------------------
The main tables use two options ("yes" / "no"), both one token, so raw
likelihood comparison is unbiased. The probe families use five options of very
different lengths and fluency ("yes" vs "cannot be determined from this image"),
and there mean-token-logprob is dominated by the prior plausibility of the
option STRING rather than by its correctness. Measured on M3D: the refusal
string is selected in 95 % of answerable growth questions and 100 % of traps,
while every knowledge probe goes to "only for cystic lesions". Read naively that
is a perfect safety score (SFR = 0) next to a total knowledge failure; in fact it
is one constant response wearing two hats.

The fix
-------
Standard contextual / PMI calibration: score each option by how much the actual
question raises its likelihood over a content-free prompt carrying the same
image and the same option list.

    score(c) = log P(c | question, image) - log P(c | "N/A", image)

The second term is exactly the option-string prior we need to remove. It depends
only on (volume, option-set, option) -- not on the individual probe -- so a
handful of forward passes per volume calibrates every probe on that volume.
That is why this runs as a cheap second pass over the SAME saved logprobs rather
than as a re-run of the main evaluation.

Calibration keeps the image, so it removes option-string bias WITHOUT removing
the image contribution -- which is a separate axis this paper measures and must
not be silently folded away.
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

CONTENT_FREE = "N/A"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--data-root", default=os.environ.get("MSD_ROOT", ""))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blind", action="store_true")
    args = ap.parse_args()

    fam = FAMILY.get(args.model, "qwen")
    rows = [json.loads(l) for l in open(args.qa) if l.strip()]

    # (organ, volume) -> the distinct option SETS appearing on that volume
    need = defaultdict(set)
    for r in rows:
        need[(r["organ"], volume_of(r["qid"]))].add(tuple(sorted(r["choices"])))
    print(f"{len(rows)} probes / {len(need)} volumes / family={fam} / "
          f"{'BLIND' if args.blind else 'sighted'} / "
          f"{sum(len(v) for v in need.values())} calibration prompts", flush=True)

    if fam == "m3d":
        from m3d_infer import M3D, TARGET_SHAPE, preprocess_volume
        model = M3D(device=args.device, model_id=args.model)
    elif fam == "med3dvlm":
        from med3dvlm_infer import TARGET_SHAPE, Med3DVLM, preprocess_volume
        model = Med3DVLM(device=args.device)
    else:
        model = MontageModel(args.model, args.device)
    print("model ready", flush=True)

    from PIL import Image

    from render import montage, orthogonal_views
    from scene_graph import load_ras

    written = errors = 0
    with open(args.out, "w") as f:
        for (organ, vid), sets in sorted(need.items()):
            vp = Path(args.data_root) / organ / "imagesTr" / f"{vid}.nii.gz"
            if not vp.exists():
                continue
            if fam in ("m3d", "med3dvlm"):
                nd = 4 if fam == "m3d" else 5
                shape = (1,) * (nd - 3) + TARGET_SHAPE
                prepared = (np.zeros(shape, np.float32) if args.blind
                            else preprocess_volume(str(vp)))
            else:
                vol, affine = load_ras(str(vp))
                r = orthogonal_views(vol, np.abs(np.diag(affine)[:3]))
                img = Image.fromarray(montage(r.available())).convert("RGB")
                if args.blind:
                    img = Image.new("RGB", img.size, (128, 128, 128))
                prepared = img

            for opts in sorted(sets):
                choices = list(opts)
                try:
                    if fam in ("m3d", "med3dvlm"):
                        _, sc = model.score_choices(
                            PROMPT.format(q=CONTENT_FREE,
                                          opts=", ".join(choices)),
                            prepared, choices)
                    else:
                        _, sc = model.score(CONTENT_FREE, choices, prepared)
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  failed {vid}: {e}", file=sys.stderr)
                    continue
                f.write(json.dumps({
                    "volume": vid, "organ": organ,
                    "option_set": list(opts),
                    "logprobs": {k: round(v, 4) for k, v in sc.items()}}) + "\n")
                written += 1
            f.flush()
    print(f"\nwrote {written} calibration rows (errors: {errors})")


if __name__ == "__main__":
    main()
