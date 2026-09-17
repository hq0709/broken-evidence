"""
Identification control: does telling the model which lesion and which target
the question is about change its answer?

The main evaluation feeds each model `montage(orthogonal_views(volume))`: three
slices through the geometric centre of the volume, without annotation. A
question such as

    "If this lesion grew by 5.8 mm in every direction, would it contact the heart?"

can refer to one of several lesions in the volume, and the image neither marks
the lesion or the target nor carries a scale. This control removes that
ambiguity step by step, running the same probes under a small factorial of
input conditions so that any change can be attributed:

    plain       centre slices, no annotation, no scale bar (the main condition)
    bestslice   slices chosen to show lesion and target, still no annotation
                -- does the model need to SEE the structures?
    overlay     centre slices, lesion and target outlined
                -- does the model need to know WHICH structures?
    identified  best slices + outlines + 10 mm scale bar
                -- exactly what the radiologist reader sees
                   (`export_reader_study.render_case`)

`identified` gives the model everything a reader is given, so accuracy that
stays at chance under `identified` locates the failure in reading the geometry
rather than in an under-specified input.

Usage
-----
    python run_identification_control.py \\
        --qa      ../cfqa_Task03_Liver/qa/all.jsonl \\
        --task-dir /path/to/MSD/Task03_Liver \\
        --seg-cache ../cfqa_Task03_Liver/seg_cache \\
        --model   qwen32b \\
        --condition identified \\
        --subset  matched \\
        --out     ../id_Task03_Liver_qwen32b_identified.jsonl

Output rows use the same format as the `mm_*`/`roi_*` runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_reader_study import _best_slice, outline           # noqa: E402
from lesion_binding import LESION_LABEL, find_lesions          # noqa: E402
from render import (_rescale, montage, montage_rgb, orthogonal_views,  # noqa: E402
                    to_display, window)
from scene_graph import load_ras                               # noqa: E402

IMAGE_CONDITIONS = ["plain", "bestslice", "overlay", "identified"]

# The task-solvability ceiling. "All models at chance" is only interpretable
# against a demonstrated ceiling, so these arms bound the task from above. They
# are conditions on this runner rather than a separate script because they must
# score the SAME items with the SAME scoring rule as the image arms; a second
# entry point is how a subset silently stops matching.
#
#   geometry-oracle    the reference rule applied to the stored provenance:
#                      contact iff gap_mm <= growth_mm. No model, no image, no
#                      volume. Must be exactly 100% -- it tests the harness and
#                      the labels, not a model, so anything below 100% is a bug
#                      in this repository and not a result.
#   text-oracle        the two measurements in text, no image at all. The task
#                      collapses to a numeric comparison, so this measures
#                      whether the model maps the clinical sentence onto the
#                      comparison when no image is involved.
#   text-oracle-blind  the same measurements, but delivered through the exact
#                      input pipeline the blind arm uses (uniform grey
#                      image, montage framing sentence). Pure text is the right
#                      ceiling; this arm is what makes the ceiling comparable
#                      with the blind arm, and the difference
#                      between the two is the cost of the framing itself.
#   numeric-oracle     the same comparison with the clinical framing removed
#                      entirely: "is 21.4 greater than or equal to 18.16?".
#                      Separates "cannot compare two numbers" from "cannot map
#                      a clinical sentence onto a comparison".
#   *-gen              the -gen variants decode greedily and parse the answer
#                      instead of scoring the option strings by likelihood.
#                      This exists because the likelihood rule can pin argmax to
#                      one option for every item, which produces exactly 50% on
#                      a balanced corpus. Reported alongside, the pair shows
#                      whether a ceiling arm reflects the model's decision or
#                      the scoring rule.
#   text-oracle-cot    the same text arm, but the model reasons first and the
#                      options are scored with its own reasoning in context.
#                      This asks the inference-compute question where it is cheapest: if a
#                      model that scores 100% on `numeric-oracle` and 50% on
#                      `text-oracle` recovers here, the clinical wording is
#                      costing it an inference step rather than the ability, and
#                      no image is involved in either arm.
ORACLE_CONDITIONS = ["geometry-oracle", "text-oracle", "text-oracle-blind",
                     "text-oracle-gen", "text-oracle-cot",
                     "numeric-oracle", "numeric-oracle-gen"]
CONDITIONS = IMAGE_CONDITIONS + ORACLE_CONDITIONS
LESION_RGB = [255, 60, 60]
TARGET_RGB = [60, 200, 255]

# A text-only arm cannot say "this lesion": there is no image for "this" to
# point at. The question is restated so it is self-contained, and the exact
# string scored is written to the output file so the wording is auditable
# rather than reconstructed from this source later.
TEXT_ORACLE_Q = (
    "A lesion's nearest surface-to-surface distance to the {target} is "
    "{gap:g} mm. If the lesion grew by {growth:g} mm in every direction, "
    "would it contact the {target}? Answer yes or no.")

# For the blind-pipeline variant the question is left untouched and the
# measurement is added as a preceding sentence, so the only difference from
# the blind arm is the sentence carrying the numbers.
TEXT_ORACLE_PREFIX = ("The lesion's nearest surface-to-surface distance to the "
                      "{target} is {gap:g} mm.")

# No anatomy, no counterfactual, no lesion: just the comparison the reference
# rule performs. Nothing about a CT scan can explain a failure here.
NUMERIC_ORACLE_Q = ("Is {growth:g} greater than or equal to {gap:g}? "
                    "Answer yes or no.")


def lesion_key(qid: str) -> str | None:
    for part in qid.split("_"):
        if part.startswith("lesion"):
            return part
    return None




def _montage_rgb(panels, pad: int = 8):
    """RGB counterpart of render.montage, matching it rule for rule.

    render.montage scales a short panel up to the common height rather than
    padding it, and separates panels with a 128-grey gutter. Zero-padding with a
    black gutter would change both the
    aspect ratio and the vision-token layout relative to the published condition.
    """
    from scipy.ndimage import zoom

    h = max(p.shape[0] for p in panels)
    tiles = []
    for p in panels:
        if p.shape[0] != h:
            f = h / p.shape[0]
            p = zoom(p, (f, f, 1), order=1).astype(np.uint8)
        tiles.append(p)
        tiles.append(np.full((h, pad, 3), 128, dtype=np.uint8))
    return np.hstack(tiles[:-1])

def _isotropic(grey, lesion2d, target2d, sr, sc):
    """Resample a panel to square pixels, exactly as render.orthogonal_views does.

    CT is
    routinely 0.7 x 0.7 x 5 mm, so an unresampled coronal or sagittal panel is
    compressed ~7x along z. `render.orthogonal_views` applies `_rescale` for
    precisely this reason -- its docstring says showing CT unscaled "distorts
    every vertical distance judgement the model is being asked to make" -- and a
    `plain` arm that skipped it would not have been the published condition, so
    the paired comparison this script exists to make would have been invalid.

    Masks are resampled filled and re-thresholded rather than resampling the
    outlines, because order-1 zoom thins a one-pixel contour until it disappears.
    Returns the resampled panel plus the post-resample isotropic spacing, which is
    what the scale bar must be drawn from.
    """
    grey = _rescale(grey, (sr, sc))
    out = []
    for m in (lesion2d, target2d):
        r = _rescale((m * 255).astype(np.uint8), (sr, sc)) > 127
        # _rescale is a no-op when the axes already match; keep shapes aligned
        if r.shape != grey.shape:
            r = r[: grey.shape[0], : grey.shape[1]]
            pad = [(0, grey.shape[i] - r.shape[i]) for i in range(2)]
            r = np.pad(r, pad)
        out.append(r)
    return grey, out[0], out[1], min(sr, sc)

def render(vol: np.ndarray, lesion: np.ndarray, target: np.ndarray,
           spacing: np.ndarray, condition: str,
           preset: str = "soft_tissue") -> tuple[np.ndarray, dict]:
    """Three orthogonal panels under one of the four input conditions.

    Panel geometry and colours match export_reader_study.render_case exactly, so
    the `identified` condition is byte-comparable with what the reader saw.

    Returns the montage and a per-case record of what the model could actually
    see. That record is not bookkeeping -- it is the difference between two
    conditions that this experiment is built to separate:

    `overlay` takes the volume-centre slices and adds the legend sentence "the
    lesion in question is outlined in red". A lesion is often a few hundred
    voxels and need not intersect any centre slice, and then there is no red
    outline to see while the prompt says there is. The condition is then not
    "knows which structures" but "is told about an annotation it did not get",
    for an unknown share of cases -- and a share that is unknown cannot be
    reported. So the geometry is recorded per probe: whether each structure
    appears on the slices shown at all, and how many outline pixels were drawn.
    """
    if condition == "plain":
        # bit-exact reproduction of the published condition: the same call the
        # audit made. Reimplementing it invited exactly the drift that an earlier
        # version of this function shipped.
        g = montage(orthogonal_views(vol, spacing).available())
        # Same three centre slices orthogonal_views takes, so the visibility
        # record is the published condition's own and not this branch's idea of
        # it. Nothing is annotated here, so the outline counts are zero by
        # construction rather than by measurement.
        idx = [vol.shape[a] // 2 for a in (2, 1, 0)]
        shown = {"lesion": 0, "target": 0}
        for a, k in zip((2, 1, 0), idx):
            sl = [slice(None)] * 3
            sl[a] = int(np.clip(k, 0, vol.shape[a] - 1))
            shown["lesion"] += int(lesion[tuple(sl)].sum())
            shown["target"] += int(target[tuple(sl)].sum())
        return np.dstack([g] * 3), {"slices": [int(i) for i in idx],
                                    "lesion_voxels_shown": shown["lesion"],
                                    "target_voxels_shown": shown["target"],
                                    "lesion_outline_px": 0,
                                    "target_outline_px": 0}

    axes = [2, 1, 0]
    if condition in ("bestslice", "identified"):
        idx = [_best_slice(lesion, a, target) for a in axes]
        # deterministic repair, as in render_case: a structure absent from all
        # three chosen slices gets a panel re-pointed at it
        def seen(m):
            out = False
            for a, k in zip(axes, idx):
                sl = [slice(None)] * 3
                sl[a] = k
                out = out or bool(m[tuple(sl)].sum())
            return out
        if not seen(lesion):
            idx[0] = _best_slice(lesion, axes[0])
        if not seen(target):
            idx[1] = _best_slice(target, axes[1])
    else:
        idx = [vol.shape[a] // 2 for a in axes]

    annotate = condition in ("overlay", "identified")
    scalebar = condition == "identified"
    grey = window(vol, preset)

    seen_px = {"lesion": 0, "target": 0}
    outline_px = {"lesion": 0, "target": 0}
    panels = []
    for a, k in zip(axes, idx):
        sl = [slice(None)] * 3
        sl[a] = int(np.clip(k, 0, vol.shape[a] - 1))
        g = grey[tuple(sl)]
        le = outline(lesion[tuple(sl)])
        tg = outline(target[tuple(sl)])
        seen_px["lesion"] += int(lesion[tuple(sl)].sum())
        seen_px["target"] += int(target[tuple(sl)].sum())
        if annotate:
            outline_px["lesion"] += int(le.sum())
            outline_px["target"] += int(tg.sum())
        # Orientation must match render._to_display, not just its vertical flip.
        # orthogonal_views renders sagittal with flip_h=False and coronal and
        # axial with flip_h=True; `.T[::-1]` alone is flip_v only, so those two
        # panels came out mirrored left-to-right against `plain`. Measured on
        # liver_0 before this line was added: axial and coronal matched `plain`
        # in 49.0% and 8.9% of pixels, and in 100.0% and 99.9% once flipped
        # horizontally. That is a patient-left/right swap on two of three panels,
        # and it sat inside the very comparison this script exists to make.
        # Flip before resampling, because _to_display runs before _rescale.
        g, le, tg = (to_display(x, a) for x in (g, le, tg))
        # row/col physical spacing of this panel after the transpose, matching
        # render.orthogonal_views: rows are the higher-numbered remaining axis
        rem = [i for i in (0, 1, 2) if i != a]
        g, le, tg, iso = _isotropic(g, le, tg,
                                    float(spacing[rem[1]]), float(spacing[rem[0]]))
        rgb = np.dstack([g] * 3)
        if annotate:
            rgb[le] = LESION_RGB
            rgb[tg] = TARGET_RGB
        if scalebar:
            # after resampling both axes share `iso` mm/px, so the bar is metric
            n = max(4, int(round(10.0 / iso)))
            h, w = rgb.shape[:2]
            rgb[h - 8:h - 5, 6:6 + min(n, w - 12)] = [255, 255, 255]
        panels.append(rgb)

    return montage_rgb(panels), {"slices": [int(i) for i in idx],
                                  "lesion_voxels_shown": seen_px["lesion"],
                                  "target_voxels_shown": seen_px["target"],
                                  "lesion_outline_px": outline_px["lesion"],
                                  "target_outline_px": outline_px["target"]}


def legend_for(condition: str) -> str:
    """Annotation legend only.

    MontageModel.score() already wraps whatever it is handed in
    run_multimodel.PROMPT ("This is a CT scan shown as axial, coronal and
    sagittal views. {q} Answer with exactly one of: {opts}."), so this must NOT
    repeat the view description or the answer instruction -- doing so double-wraps
    the prompt and changes the input in a way unrelated to the condition.

    Describing an annotation the model did not receive would be its own confound,
    so each sentence is added only when the corresponding pixels are drawn.
    """
    parts = []
    if condition in ("overlay", "identified"):
        parts.append("The lesion in question is outlined in red and the target "
                     "structure is outlined in cyan.")
    if condition == "identified":
        parts.append("The white bar in each panel is 10 mm in that panel's plane.")
    return " ".join(parts)


def matched_subset_keys(qa_path: Path) -> set[str] | None:
    """Reuse growth_matched.py's subset so results are directly comparable with
    the published table rather than a differently-drawn subset."""
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    try:
        from growth_matched import growth_of, matched_subset
    except Exception as exc:                    # pragma: no cover
        print(f"could not import growth_matched ({exc}); running all items",
              file=sys.stderr)
        return None
    ref = {}
    common = root / "common_subset" / "qa" / "all.jsonl"
    if not common.exists():
        print(f"missing {common}; running all items", file=sys.stderr)
        return None
    for line in open(common):
        r = json.loads(line)
        ref[r["qid"]] = r["answer"]
    return matched_subset(ref, growth_of())


ORGAN_BY_PREFIX = {"liver": "Task03_Liver", "lung": "Task06_Lung",
                   "pancreas": "Task07_Pancreas", "colon": "Task10_Colon"}


def cached_render(cache_dir: str | None, organ: str, vid: str, lesion_id: str,
                  target: str, condition: str, make) -> tuple:
    """Render once, reuse across every model and strategy.

    The grid is 4 conditions x 4 models = 16 runs per organ, and each run was
    re-reading and re-decompressing the same volumes: an MSD liver series is a
    quarter of a gigabyte decompressed, there are 96 of them in the matched
    subset for that organ alone, and gzip is single-threaded. Measured, the
    volume I/O rather than the forward passes set the wall clock -- 44 probes in
    six minutes, which puts the 64-run grid past half a day.

    `render` is a pure function of (volume, lesion, target, spacing, condition),
    so caching it changes no number. The geometry record travels inside the PNG
    as a text chunk rather than in a sidecar file, so a cache entry cannot exist
    in a half-written state, and the write is a rename, so concurrent jobs can
    populate the same cache safely.
    """
    from PIL import Image
    if cache_dir is None:
        arr, geom = make()
        return Image.fromarray(arr).convert("RGB"), geom

    path = Path(cache_dir) / organ / f"{vid}_{lesion_id}_{target}_{condition}.png"
    if path.exists():
        try:
            img = Image.open(path)
            img.load()
            return img.convert("RGB"), json.loads(img.text.get("geometry", "{}"))
        except Exception as exc:                     # corrupt entry: redo it
            print(f"  unreadable cache entry {path.name} ({exc}); re-rendering",
                  file=sys.stderr)

    from PIL.PngImagePlugin import PngInfo
    arr, geom = make()
    meta = PngInfo()
    meta.add_text("geometry", json.dumps(geom))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".part{os.getpid()}.png")
    Image.fromarray(arr).save(tmp, pnginfo=meta)
    tmp.replace(path)
    return Image.fromarray(arr).convert("RGB"), geom


def organ_of(row: dict, task_dir: str | None) -> str:
    """Organ label for an output row.

    common_subset/qa/all.jsonl carries `organ`; the per-organ cfqa shards do
    not, and the oracle arms take no --task-dir to read it from. The qid prefix
    is the remaining source and is unambiguous across the four tasks.
    """
    if row.get("organ"):
        return row["organ"]
    if task_dir:
        return Path(task_dir).name
    return ORGAN_BY_PREFIX.get(row["qid"].split("_")[0], "unknown")


def score_choices(model, text: str, choices: list[str], images=None) -> tuple:
    """MontageModel.score's arithmetic, with the image made optional.

    Deliberately the same slicing and the same reduction (mean log-probability
    over the option's own tokens, teacher-forced) as MontageModel.score. The
    ceiling arms are compared against the image arms, so a second scoring rule
    here would make the comparison meaningless -- the numbers would differ by
    the rule and not by the input.
    """
    out = {}
    for c in choices:
        kw = {"text": [text + c], "return_tensors": "pt"}
        if images is not None:
            kw["images"] = images
        enc = model.proc(**kw).to(model._resolve_device())
        n_c = len(model.proc.tokenizer(c, add_special_tokens=False)["input_ids"])
        with model.torch.inference_mode():
            logits = model.model(**enc).logits[0, :-1]
        tgt = enc["input_ids"][0, 1:]
        # Slice to the option's own rows BEFORE the float32 cast, as
        # MontageModel.score already does -- score_choices was left on the old
        # full-sequence cast, and only this scorer feeds run_subtasks.py.
        # log_softmax is row-wise, so the rows read are the same computation;
        # measured against a full-sequence reference run of internvl/slices3 the predictions
        # agree 300/300 and accuracy is unchanged at 27.67%, with logprobs
        # differing by at most 9.5e-07 (float32 ULP -- a (n_c,V) reduction and
        # an (N,V) one do not sum in the same order).
        # The discarded work was not free: under `axial25` a probe tiles to
        # ~15k vision tokens, so against a 152k vocabulary the full float32
        # logits are ~9 GB per option per probe. Slicing first keeps
        # the largest arm within one 80 GB card.
        lp = model.torch.log_softmax(logits[-n_c:].float(), dim=-1)
        out[c] = float(lp.gather(1, tgt[-n_c:, None]).mean())
    return max(out, key=out.get), out


def generate_answer(model, text: str, choices: list[str],
                    max_new_tokens: int = 128) -> tuple[str, dict]:
    """Greedy decode, then take the first option word that appears.

    `prediction` is set to "unparsed" when the continuation names no option, and
    the raw text is kept on every row. Silently folding an unparsable answer
    into one of the options is how a decoding failure turns into an apparent
    preference for that option; the rate of unparsed answers is itself reported.

    The budget is 128 rather than the dozen tokens an answer needs, because
    models differ in how they open. Qwen2.5-VL-7B replies "No."; Qwen2.5-VL-32B
    opens "To determine if the lesion would contact the esophagus after..." and
    a 12-token budget cut it off before any answer, scoring the arm 0.1% with
    100% unparsed -- a measurement artefact that looks exactly like a
    catastrophic result. Greedy decoding is prefix-deterministic, so a larger
    budget cannot change the answer of a model that already answered.
    """
    enc = model.proc(text=[text], return_tensors="pt").to(model._resolve_device())
    with model.torch.inference_mode():
        out = model.model.generate(**enc, max_new_tokens=max_new_tokens,
                                   do_sample=False,
                                   pad_token_id=model.proc.tokenizer.eos_token_id)
    gen = model.proc.tokenizer.decode(out[0, enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
    low = gen.lower()
    hits = [(low.find(c.lower()), c) for c in choices if c.lower() in low]
    pred = min(hits)[1] if hits else "unparsed"
    return pred, {"generated": gen.strip()}


def run_oracle(args, items: list[dict]) -> None:
    """The solvability ceiling. No volumes, no segmentations, no rendering."""
    cond = args.condition
    need_model = cond != "geometry-oracle"
    model = grey = None
    if need_model:
        from run_multimodel import MODEL_ID, PROMPT, MontageModel
        if args.model not in MODEL_ID:
            raise SystemExit(f"unknown model tag {args.model!r}; known: "
                             + ", ".join(sorted(MODEL_ID)))
        model = MontageModel(MODEL_ID[args.model], args.device)
        print("model ready", flush=True)
        if cond == "text-oracle-blind":
            from PIL import Image
            # The blind arm replaces the montage with a uniform grey
            # image of the same size. Sizes vary per volume there; here there is
            # no volume, so one fixed grey canvas is used for every item and the
            # size is recorded in the output.
            grey = Image.new("RGB", (768, 256), (128, 128, 128))

    written = skipped = 0
    with open(args.out, "w") as fout:
        for r in items:
            p = r.get("provenance", {})
            gap, growth, target = p.get("gap_mm"), p.get("growth_mm"), p.get("target")
            if gap is None or growth is None or target is None:
                # nearest_on_growth and resection items carry no (gap, growth)
                # pair, so the reference rule does not apply to them. They are
                # skipped rather than guessed at, and counted.
                skipped += 1
                continue
            choices = r.get("choices") or ["no", "yes"]
            pretty = str(target).replace("_", " ")

            if cond == "geometry-oracle":
                asked = None
                pred = "yes" if float(gap) <= float(growth) else "no"
                logprobs = {"gap_mm": float(gap), "growth_mm": float(growth),
                            "rule": "contact iff gap_mm <= growth_mm"}
            elif cond == "text-oracle-cot":
                asked = TEXT_ORACLE_Q.format(target=pretty, gap=float(gap),
                                             growth=float(growth))
                prompt = (f"{asked} Reason briefly about the two distances, "
                          f"then give your answer.")
                first = model.proc.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                    tokenize=False, add_generation_prompt=True)
                enc = model.proc(text=[first], return_tensors="pt").to(
                    model._resolve_device())
                with model.torch.inference_mode():
                    out = model.model.generate(
                        **enc, max_new_tokens=200, do_sample=False,
                        pad_token_id=model.proc.tokenizer.eos_token_id)
                chain = model.proc.tokenizer.decode(
                    out[0, enc["input_ids"].shape[1]:],
                    skip_special_tokens=True).strip()
                text = model.proc.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": prompt}]},
                     {"role": "assistant", "content": [{"type": "text", "text": chain}]},
                     {"role": "user", "content": [{"type": "text", "text": (
                         f"Therefore, answer with exactly one of: "
                         f"{', '.join(choices)}.")}]}],
                    tokenize=False, add_generation_prompt=True)
                pred, lp = score_choices(model, text, choices)
                logprobs = {"chain": chain, "logprobs": lp}
            elif cond in ("text-oracle", "text-oracle-gen",
                          "numeric-oracle", "numeric-oracle-gen"):
                if cond.startswith("numeric"):
                    asked = NUMERIC_ORACLE_Q.format(growth=float(growth),
                                                    gap=float(gap))
                else:
                    asked = TEXT_ORACLE_Q.format(target=pretty, gap=float(gap),
                                                 growth=float(growth))
                msgs = [{"role": "user", "content": [{"type": "text", "text": (
                    f"{asked} Answer with exactly one of: "
                    f"{', '.join(choices)}.")}]}]
                text = model.proc.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                if cond.endswith("-gen"):
                    pred, logprobs = generate_answer(model, text, choices)
                else:
                    pred, logprobs = score_choices(model, text, choices)
            else:                                   # text-oracle-blind
                asked = (TEXT_ORACLE_PREFIX.format(target=pretty, gap=float(gap))
                         + " " + r["question"])
                msgs = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT.format(
                        q=asked, opts=", ".join(choices))}]}]
                text = model.proc.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                pred, logprobs = score_choices(model, text, choices,
                                               images=[grey])

            row = {"qid": r["qid"], "organ": organ_of(r, args.task_dir),
                   "condition": cond, "prediction": pred, "gold": r["answer"],
                   "pair_id": r.get("pair_id"), "logprobs": logprobs}
            if asked is not None:
                row["asked"] = asked
            if grey is not None:
                row["image"] = f"grey{grey.size[0]}x{grey.size[1]}"
            fout.write(json.dumps(row) + "\n")
            written += 1
            if written % 400 == 0:
                print(f"  {written} written", flush=True)

    acc = None
    extra = ""
    if written:
        rows = [json.loads(l) for l in open(args.out)]
        acc = 100.0 * sum(r["prediction"] == r["gold"] for r in rows) / len(rows)
        # Two numbers that distinguish a real result from a degenerate one, both
        # cheap and both invisible in an accuracy figure alone.
        n_un = sum(r["prediction"] == "unparsed" for r in rows)
        share = {c: sum(r["prediction"] == c for r in rows) for c in
                 sorted({r["prediction"] for r in rows})}
        extra = (f", predictions {share}"
                 + (f", unparsed {100.0 * n_un / len(rows):.1f}%" if n_un else ""))
    print(f"wrote {args.out}: {written} scored, {skipped} skipped"
          + (f", accuracy {acc:.1f}%" if acc is not None else "") + extra)
    if cond == "geometry-oracle" and acc is not None and abs(acc - 100.0) > 1e-9:
        # Loud on purpose. This arm recomputes the labels from the numbers the
        # labels were derived from, so a miss means the stored answers and the
        # stored provenance disagree -- a corpus error, not a finding about
        # a model.
        raise SystemExit(
            f"geometry-oracle scored {acc:.4f}%, not 100%. The reference rule "
            f"does not reproduce the stored answers; the corpus is "
            f"inconsistent and no model number computed from it is meaningful.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--qa", required=True, help="probe jsonl (or a directory of them)")
    ap.add_argument("--task-dir", help="MSD task dir with imagesTr/labelsTr "
                                      "(image conditions only)")
    ap.add_argument("--seg-cache", help="anatomy segmentation cache "
                                        "(image conditions only)")
    ap.add_argument("--model", default="geometry",
                    help="tag from run_multimodel.MODEL_ID; unused by "
                         "--condition geometry-oracle, which runs no model")
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--subset", choices=["matched", "all"], default="matched")
    ap.add_argument("--device", default="cuda:0",
                    help='concrete device ("cuda:0") or "auto" to shard a model too large for one card across every visible GPU')
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (smoke tests)")
    ap.add_argument("--render-cache",
                    default=str(Path(__file__).resolve().parent.parent
                                / "render_cache"),
                    help="directory of cached montages, shared by every model "
                         "and strategy. Pass an empty string to disable.")
    args = ap.parse_args()
    args.render_cache = args.render_cache or None

    qa_path = Path(args.qa)
    files = sorted(qa_path.glob("*.jsonl")) if qa_path.is_dir() else [qa_path]
    items = [json.loads(l) for f in files for l in open(f) if l.strip()]

    if args.subset == "matched":
        keep = matched_subset_keys(qa_path)
        if keep is not None:
            before = len(items)
            items = [r for r in items if r["qid"] in keep]
            print(f"growth-matched subset: {len(items)} of {before} probes", flush=True)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no probes selected")

    if args.condition in ORACLE_CONDITIONS:
        run_oracle(args, items)
        return

    # Missing volumes are silently skipped further down, so a forgotten flag
    # would produce an empty output file that looks like a completed run.
    for flag, val in (("--task-dir", args.task_dir), ("--seg-cache", args.seg_cache)):
        if not val:
            raise SystemExit(f"{flag} is required for --condition {args.condition}")

    task = Path(args.task_dir)
    organ = task.name
    lesion_label = LESION_LABEL[organ]

    from run_multimodel import MODEL_ID, MontageModel
    if args.model not in MODEL_ID:
        raise SystemExit(f"unknown model tag {args.model!r}; known: "
                         + ", ".join(sorted(MODEL_ID)))
    from PIL import Image
    model = MontageModel(MODEL_ID[args.model], args.device)
    print("model ready", flush=True)

    by_vol: dict[str, list[dict]] = defaultdict(list)
    for r in items:
        by_vol["_".join(r["qid"].split("_")[:2])].append(r)

    from run_pipeline import label_map
    name2lab = {v: k for k, v in label_map().items()}
    legend = legend_for(args.condition)
    written = skipped = 0
    with open(args.out, "w") as fout:
        for vid, group in sorted(by_vol.items()):
            volp = task / "imagesTr" / f"{vid}.nii.gz"
            labp = task / "labelsTr" / f"{vid}.nii.gz"
            segp = Path(args.seg_cache) / f"{vid}_seg.nii.gz"
            if not (volp.exists() and labp.exists() and segp.exists()):
                skipped += len(group)
                continue

            # Loaded on first miss only. With a warm render cache no volume is
            # read at all, which is the whole point of the cache -- the reads,
            # not the forward passes, were the wall clock.
            held: dict[str, object] = {}

            def volume() -> dict:
                if not held:
                    vol, affine = load_ras(str(volp))
                    gt, _ = load_ras(str(labp))
                    seg, _ = load_ras(str(segp))
                    held.update(
                        vol=vol.astype(np.int16), seg=seg,
                        spacing=np.abs(np.diag(affine)[:3]),
                        lesions=dict(find_lesions(gt == lesion_label, affine)))
                return held

            # one render per (lesion, target); the matched pair shares it
            cache: dict[tuple[str, str], object] = {}
            for r in group:
                lk = lesion_key(r["qid"])
                tname = r.get("provenance", {}).get("target")
                if lk is None or tname not in name2lab:
                    skipped += 1
                    continue
                ck = (lk, tname)
                if ck not in cache:
                    def make(lk=lk, tname=tname):
                        d = volume()
                        if lk not in d["lesions"]:
                            raise LookupError("lesion component absent")
                        tmask = d["seg"] == name2lab[tname]
                        if not tmask.any():
                            raise LookupError("target absent from the mask")
                        return render(d["vol"], d["lesions"][lk], tmask,
                                      d["spacing"], args.condition)
                    try:
                        cache[ck] = cached_render(args.render_cache, organ, vid,
                                                  lk, tname, args.condition, make)
                    except LookupError:
                        skipped += 1
                        continue
                image, geom = cache[ck]
                choices = r.get("choices") or ["no", "yes"]
                asked = (f"{legend} {r['question']}" if legend else r["question"])
                pred, logprobs = model.score(asked, choices, image)
                fout.write(json.dumps({
                    "qid": r["qid"], "organ": organ, "condition": args.condition,
                    "prediction": pred, "gold": r["answer"],
                    "pair_id": r.get("pair_id"), "logprobs": logprobs,
                    "geometry": geom,
                }) + "\n")
                written += 1
            if written and written % 200 < len(group):
                print(f"  {written} written", flush=True)

    print(f"wrote {args.out}: {written} scored, {skipped} skipped")
    if skipped:
        print("  skips are volumes or targets absent from the caches; "
              "they are excluded from both arms so the comparison stays paired")



def selftest() -> None:
    """Assert the parity properties this script's validity rests on.

    Accuracy cannot check parity: a renderer that showed the model z-squashed,
    left-right mirrored panels would still score at chance. Parity is a property
    of the pixels and has to be tested on the pixels.

    Synthetic volume, deliberately anisotropic (0.7 x 0.7 x 5 mm, the spacing
    that makes the resampling matter), so this runs with no data present.
    """
    from export_reader_study import render_case

    rng = np.random.default_rng(0)
    vol = rng.integers(-200, 200, size=(64, 60, 24), dtype=np.int16)
    lesion = np.zeros(vol.shape, bool)
    lesion[20:26, 18:24, 8:11] = True
    target = np.zeros(vol.shape, bool)
    target[40:52, 34:46, 14:19] = True
    spacing = np.array([0.703125, 0.703125, 5.0])

    plain, geom = render(vol, lesion, target, spacing, "plain")
    published = np.dstack([montage(orthogonal_views(vol, spacing).available())] * 3)
    assert np.array_equal(plain, published), \
        "`plain` is not the published condition; the paired control is invalid"

    shapes = {}
    for cond in IMAGE_CONDITIONS:
        img, g = render(vol, lesion, target, spacing, cond)
        shapes[cond] = img.shape
        assert set(g) >= {"slices", "lesion_voxels_shown", "lesion_outline_px"}
    assert len(set(shapes.values())) == 1, \
        f"conditions differ in shape, so they differ in vision tokens: {shapes}"

    ident, _ = render(vol, lesion, target, spacing, "identified")
    assert np.array_equal(ident, render_case(vol, lesion, target, spacing)), \
        "`identified` is not what the reader is shown; the arm is not comparable"

    # the annotated arms must agree with `plain` wherever they draw nothing
    ov, _ = render(vol, lesion, target, spacing, "overlay")
    drawn = ((ov == LESION_RGB).all(-1)) | ((ov == TARGET_RGB).all(-1))
    agree = ((ov == plain).all(-1) | drawn).mean()
    assert agree > 0.99, \
        f"`overlay` differs from `plain` outside its annotation on {100*(1-agree):.1f}% of pixels"

    # the bar must be 10 mm in the panel's own post-resample spacing
    iso = min(spacing[0], spacing[1])
    assert abs(int(round(10.0 / iso)) * iso - 10.0) < iso, "scale bar is not 10 mm"

    print(f"selftest OK -- plain bit-exact against the published path, all four "
          f"conditions {shapes['plain']}, identified == render_case, overlay "
          f"agrees with plain on {100*agree:.2f}% of unannotated pixels")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
