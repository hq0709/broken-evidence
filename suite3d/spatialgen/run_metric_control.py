"""Metric-channel control: the distance sub-task on synthetic geometry.

The sub-task decomposition finds every model at chance when asked how many
millimetres separate the red-outlined lesion from the cyan-outlined structure
(results_new/id_common_*_subtask-distance.jsonl).  A reader may object that
vision-language models cannot read a metric distance off ANY image, so the
result says nothing about medical volumes.  This runner asks the identical
question, with the identical legend, options, scale bar, colours and scoring
rule, on three backgrounds:

  real   the item's published `identified` montage (render_cache/*_slices3.png)
  ct     the same montage with its annotations erased and two SYNTHETIC outlines
         drawn over the real anatomy at a known surface gap
  grey   three uniform-grey panels with the same synthetic outlines

`grey` and `ct` carry the same gap distribution as the 300 real items, and the
gold is recomputed from the drawn masks with the same distance transform the
corpus uses.  If `grey` is answerable and `real` is not, the metric channel
exists and fails on anatomy; if `grey` is at chance too, the deficit is general
and the clinical task inherits it.  Either way the objection is measured.

    python spatialgen/run_metric_control.py --model qwen7b --background grey \
        --device cuda:0 --out results_new/metric_qwen7b_grey.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
from run_identification_control import (LESION_RGB, TARGET_RGB, legend_for,   # noqa: E402
                                        score_choices)
from run_subtasks import DISTANCE_OPTIONS, bucket, question_for            # noqa: E402

ITEMS = REPO / "results_new" / "id_common_qwen7b_subtask-distance.jsonl"
CACHE = REPO / "render_cache"
SYN = CACHE / "synthetic"
BACKGROUNDS = ["real", "ct", "grey"]
GUTTER = 128


# ----------------------------------------------------------------------------- montage helpers
def split_panels(img: np.ndarray) -> list[tuple[int, int]]:
    """Column ranges of the panels, found from the 128-grey gutters render.montage draws."""
    gutter = np.all(img == GUTTER, axis=(0, 2))
    cols, start = [], None
    for x, g in enumerate(gutter):
        if not g and start is None:
            start = x
        if g and start is not None:
            cols.append((start, x)); start = None
    if start is not None:
        cols.append((start, img.shape[1]))
    return cols


def bar_length(panel: np.ndarray) -> int:
    """Pixel length of the 10 mm bar render() draws at rows h-8:h-5 from column 6."""
    h = panel.shape[0]
    row = panel[h - 7, 6:, :]
    white = np.all(row == 255, axis=1)
    n = 0
    while n < len(white) and white[n]:
        n += 1
    return n


def erase_annotations(panel: np.ndarray) -> np.ndarray:
    """Replace outline and bar pixels by the local mean of un-annotated grey pixels."""
    from scipy.ndimage import uniform_filter

    p = panel.astype(np.float32)
    grey_like = (np.abs(p[..., 0] - p[..., 1]) < 2) & (np.abs(p[..., 1] - p[..., 2]) < 2)
    h = panel.shape[0]
    bar = np.zeros(grey_like.shape, bool); bar[h - 9:h - 4, :] = np.all(panel[h - 9:h - 4] == 255, axis=2)
    from scipy.ndimage import binary_dilation
    keep = ~binary_dilation(~grey_like | bar, iterations=2)      # annotation pixels plus a 2-px halo
    g = p[..., 0]
    num = uniform_filter(g * keep, 13); den = uniform_filter(keep.astype(np.float32), 13)
    fill = np.where(den > 0, num / np.maximum(den, 1e-6), g)
    out = np.where(keep, g, fill)
    return np.dstack([out] * 3).clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- synthetic geometry
def blob(h: int, w: int, cy: float, cx: float, r: float, ecc: float, theta: float) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    dy, dx = yy - cy, xx - cx
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    return (u / (r * ecc)) ** 2 + (v / (r / ecc)) ** 2 <= 1.0


def surface_gap_px(lesion: np.ndarray, target: np.ndarray, d=None) -> float:
    """Minimum lesion-to-target surface distance in px; `d` = EDT(~target) if already computed."""
    from scipy.ndimage import distance_transform_edt
    if d is None:
        d = distance_transform_edt(~target)
    return float(d[lesion].min()) if lesion.any() else float("nan")


def place_pair(h: int, w: int, gap_px: float, body: np.ndarray | None, rng: random.Random):
    """Two masks whose surface gap is gap_px (within 0.5 px), inside the panel / body."""
    margin = 10
    for _ in range(200):
        r_t = rng.uniform(18, 42); r_l = rng.uniform(6, 13)
        e_t, e_l = rng.uniform(0.8, 1.25), rng.uniform(0.8, 1.25)
        th_t, th_l = rng.uniform(0, np.pi), rng.uniform(0, np.pi)
        if body is not None:
            ys, xs = np.nonzero(body)
            if len(ys) < 100:
                return None
            k = rng.randrange(len(ys)); cy, cx = float(ys[k]), float(xs[k])
        else:
            cy, cx = rng.uniform(margin + 60, h - margin - 60), rng.uniform(margin + 60, w - margin - 60)
        target = blob(h, w, cy, cx, r_t, e_t, th_t)
        if target[:margin].any() or target[-margin:].any() or target[:, :margin].any() or target[:, -margin:].any():
            continue
        from scipy.ndimage import distance_transform_edt
        d_t = distance_transform_edt(~target)                # once per target, not once per bisection step
        ang = rng.uniform(0, 2 * np.pi); dy, dx = np.sin(ang), np.cos(ang)
        lo, hi = r_t * 0.5 + r_l * 0.5, r_t * 1.5 + r_l * 1.5 + gap_px + 4
        lesion = None
        for _ in range(28):                                   # bisection on centre distance
            d = 0.5 * (lo + hi)
            cand = blob(h, w, cy + dy * d, cx + dx * d, r_l, e_l, th_l)
            if not cand.any() or (cand & target).any():
                lo = d; continue
            g = surface_gap_px(cand, target, d_t)
            if abs(g - gap_px) <= 0.5:
                lesion = cand; break
            if g < gap_px: lo = d
            else: hi = d
        if lesion is None:
            continue
        if lesion[:margin].any() or lesion[-margin:].any() or lesion[:, :margin].any() or lesion[:, -margin:].any():
            continue
        if body is not None and (body[lesion].mean() < 0.9 or body[target].mean() < 0.9):
            continue
        return lesion, target, surface_gap_px(lesion, target, d_t)
    return None


def outline(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask)


def paint(panel: np.ndarray, lesion: np.ndarray, target: np.ndarray, iso: float) -> np.ndarray:
    rgb = panel.copy()
    rgb[outline(lesion)] = LESION_RGB
    rgb[outline(target)] = TARGET_RGB
    n = max(4, int(round(10.0 / iso))); h, w = rgb.shape[:2]
    rgb[h - 8:h - 5, 6:6 + min(n, w - 12)] = [255, 255, 255]
    return rgb


def synthesise(real: np.ndarray, background: str, gap_mm: float, seed: str):
    """Return (montage, per-panel measured gaps in mm, iso list)."""
    from PIL import Image
    rng = random.Random(seed)
    panels = []; gaps = []; isos = []
    for (a, b) in split_panels(real):
        p = real[:, a:b]
        n = bar_length(p); iso = 10.0 / n if n >= 4 else 0.78
        h, w = p.shape[:2]
        if background == "grey":
            base = np.full((h, w, 3), 110, np.uint8); body = None
        else:
            base = erase_annotations(p)
            from scipy.ndimage import binary_erosion
            body = binary_erosion(base[..., 0] > 40, iterations=12)
        got = place_pair(h, w, gap_mm / iso, body, rng)
        if got is None and background == "ct":
            got = place_pair(h, w, gap_mm / iso, None, rng)
        if got is None:
            return None
        lesion, target, gpx = got
        panels.append(paint(base, lesion, target, iso)); gaps.append(gpx * iso); isos.append(iso)
    tiles = []
    for p in panels:
        tiles.append(p); tiles.append(np.full((p.shape[0], 8, 3), GUTTER, np.uint8))
    return np.hstack(tiles[:-1]), gaps, isos


def load_items(limit: int | None):
    rows = [json.loads(l) for l in open(ITEMS)]
    out = []
    for r in rows:
        pid = r["qid"][: -len("_distance")]
        out.append({"pair_id": pid, "organ": r["organ"], "gap_mm": float(r["gap_mm"]),
                    "target": pid.split("lesion", 1)[1].split("_", 1)[1] if "_" in pid.split("lesion", 1)[1] else ""})
    return out[:limit] if limit else out


def montage_for(item: dict, background: str, seed: int):
    """PIL image + record; synthetic montages are cached as PNG so every model sees the same pixels."""
    from PIL import Image
    real = np.array(Image.open(CACHE / item["organ"] / f"{item['pair_id']}_slices3.png").convert("RGB"))
    if background == "real":
        return Image.fromarray(real), {"gap_mm": item["gap_mm"], "iso": None}
    path = SYN / background / f"{item['pair_id']}.png"; meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        return Image.open(path).convert("RGB"), json.load(open(meta))
    got = synthesise(real, background, item["gap_mm"], f"{seed}:{item['pair_id']}:{background}")
    if got is None:
        return None, None
    arr, gaps, isos = got
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path); rec = {"gap_mm": float(np.mean(gaps)), "gaps_mm": gaps, "iso": isos, "gap_mm_real": item["gap_mm"]}
    json.dump(rec, open(meta, "w"))
    return Image.fromarray(arr), rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--background", choices=BACKGROUNDS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--render-only", action="store_true", help="build the synthetic cache, load no model")
    args = ap.parse_args()

    items = load_items(args.limit or None)
    legend = legend_for("identified")
    model = None
    if not args.render_only:
        from run_multimodel import MODEL_ID, MontageModel
        model = MontageModel(MODEL_ID[args.model], args.device); print("model ready", flush=True)
    written = skipped = 0
    with open(args.out, "w") as fout:
        for it in items:
            image, rec = montage_for(it, args.background, args.seed)
            if image is None:
                skipped += 1; continue
            pair = {"target": it["target"], "gap_mm": rec["gap_mm"], "organ": it["organ"]}
            qrng = random.Random(f"{args.seed}:{it['pair_id']}:distance")     # same option order as run_subtasks
            question, gold, choices = question_for("distance", pair, qrng)
            if model is None:
                pred, lp = None, None
            else:
                text = model.proc.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{legend} {question}"}]}],
                    tokenize=False, add_generation_prompt=True)
                pred, lp = score_choices(model, text, choices, images=[image])
            fout.write(json.dumps({"qid": f"{it['pair_id']}_distance", "organ": it["organ"], "condition": f"metric-{args.background}",
                                   "background": args.background, "prediction": pred, "gold": gold, "logprobs": lp, "choices": choices,
                                   "asked": question, "gap_mm": rec["gap_mm"], "gap_mm_real": it["gap_mm"], "iso": rec["iso"]}) + "\n")
            written += 1
            if written % 50 == 0:
                print(f"  {written} written", flush=True)
    got = [json.loads(l) for l in open(args.out)]
    if model is not None:
        acc = 100.0 * sum(r["prediction"] == r["gold"] for r in got) / max(len(got), 1)
        from collections import Counter
        print(f"{args.model} {args.background}: n={len(got)} acc={acc:.1f}% (chance 25%) modal={Counter(r['prediction'] for r in got).most_common(1)} skipped={skipped}")
    else:
        print(f"rendered {written} ({skipped} skipped) for {args.background}")


if __name__ == "__main__":
    main()
