"""
Inference-time compute: does more decoding recover the signal?

The 2D suite includes chain-of-thought and self-verification ablations; this
runs them in 3D. The question is narrow and worth stating precisely: the
volumetric evaluation uses a SINGLE
unbatched forward pass per option, scored by likelihood. That is the cheapest
possible decoding strategy, and "the model cannot do this" and "the model was
not given the chance to work" are different claims.

Arms, all on the same probes and the same images as the identification control,
so every number here is comparable with the identification control row for row:

    score      likelihood over the option strings. Identical to the control; present so
               the baseline is measured inside this run rather than borrowed.
    greedy     decode greedily, parse the answer out of the text.
    cot        decode a short chain of thought, then score the options with the
               reasoning in context. Scoring rather than parsing the final
               answer keeps the decision rule identical to the baseline, so a
               difference is attributable to the reasoning and not to a
               different way of reading the answer off.
    sc5, sc10  self-consistency: k sampled chains, majority vote over the parsed
               answers. k=10 is ten generations per probe and is the compute-
               hungry arm.
    verify     answer, then be asked to check that answer against the image,
               then answer again.

Everything about the input -- rendering, condition, subset, legend -- is
imported from run_identification_control rather than reimplemented, because an
arm that silently rendered its images differently would be measuring the
renderer.

Sampling is seeded per probe from the qid, so a rerun reproduces the same
chains and `sc10` is not a different experiment every time it runs.

Usage
-----
    python run_inference_compute.py --qa cfqa_Task03_Liver/qa \
        --task-dir $MSD_ROOT/Task03_Liver --seg-cache cfqa_Task03_Liver/seg_cache \
        --model qwen32b --condition identified --strategy sc10 \
        --subset matched --device cuda:0 \
        --out id_Task03_Liver_qwen32b_identified-sc10.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lesion_binding import LESION_LABEL, find_lesions          # noqa: E402
from run_identification_control import (                        # noqa: E402
    IMAGE_CONDITIONS, legend_for, lesion_key, matched_subset_keys, render,
    score_choices,
)
from scene_graph import load_ras                                # noqa: E402

STRATEGIES = ["score", "greedy", "cot", "sc5", "sc10", "verify"]

COT_INSTRUCTION = ("Reason briefly about the distances involved, then give your "
                   "answer.")
FINAL_INSTRUCTION = "Therefore, answer with exactly one of: {opts}."
VERIFY_INSTRUCTION = ("Check that answer against the image. Is it correct? "
                      "Explain briefly.")


def build_text(model, turns: list[tuple[str, str]], with_image: bool) -> str:
    """Chat template for a list of (role, text) turns, image on the first user
    turn only -- the model is looking at one montage for the whole exchange."""
    msgs = []
    for i, (role, text) in enumerate(turns):
        content = []
        if i == 0 and with_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        msgs.append({"role": role, "content": content})
    return model.proc.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)


def generate(model, text: str, img, max_new_tokens: int, seed: int | None = None,
             temperature: float = 0.7, top_p: float = 0.95) -> str:
    kw = {"text": [text], "return_tensors": "pt"}
    if img is not None:
        kw["images"] = [img]
    enc = model.proc(**kw).to(model._resolve_device())
    gen_kw = dict(max_new_tokens=max_new_tokens,
                  pad_token_id=model.proc.tokenizer.eos_token_id)
    if seed is None:
        gen_kw["do_sample"] = False
    else:
        model.torch.manual_seed(seed)
        gen_kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    with model.torch.inference_mode():
        out = model.model.generate(**enc, **gen_kw)
    return model.proc.tokenizer.decode(out[0, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip()


def parse(text: str, choices: list[str]) -> str:
    """First option word to appear. Never guesses: "unparsed" is a real outcome
    and is reported rather than folded into one of the options."""
    low = text.lower()
    hits = [(low.find(c.lower()), c) for c in choices if c.lower() in low]
    return min(hits)[1] if hits else "unparsed"


def seed_of(qid: str, k: int) -> int:
    """Deterministic per (probe, sample index)."""
    return zlib.crc32(f"{qid}#{k}".encode()) & 0x7FFFFFFF


def answer(model, strategy: str, question: str, choices: list[str], img,
           qid: str) -> tuple[str, dict]:
    opts = ", ".join(choices)
    base = f"This is a CT scan shown as axial, coronal and sagittal views. {question}"

    if strategy == "score":
        # exactly the baseline rule, via the shared helper
        text = build_text(model, [("user", f"{base} Answer with exactly one of: "
                                          f"{opts}.")], True)
        return score_choices(model, text, choices, images=[img])

    if strategy == "greedy":
        text = build_text(model, [("user", f"{base} Answer with exactly one of: "
                                          f"{opts}.")], True)
        gen = generate(model, text, img, 12)
        return parse(gen, choices), {"generated": gen}

    if strategy in ("cot", "sc5", "sc10", "verify"):
        prompt = f"{base} {COT_INSTRUCTION}"
        first = build_text(model, [("user", prompt)], True)

        if strategy == "cot":
            chain = generate(model, first, img, 256)
            text = build_text(model, [("user", prompt), ("assistant", chain),
                                      ("user", FINAL_INSTRUCTION.format(opts=opts))],
                              True)
            pred, lp = score_choices(model, text, choices, images=[img])
            return pred, {"chain": chain, "logprobs": lp}

        if strategy in ("sc5", "sc10"):
            k = 5 if strategy == "sc5" else 10
            votes, chains = [], []
            for i in range(k):
                chain = generate(model, first, img, 256, seed=seed_of(qid, i))
                # a sampled chain need not end in an option word, so the answer
                # is asked for explicitly and parsed from a short continuation
                text = build_text(model, [("user", prompt), ("assistant", chain),
                                          ("user", FINAL_INSTRUCTION.format(opts=opts))],
                                  True)
                votes.append(parse(generate(model, text, img, 8), choices))
                chains.append(chain)
            tally = Counter(v for v in votes if v != "unparsed")
            if not tally:
                pred = "unparsed"
            else:
                top = tally.most_common()
                pred = "tie" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]
            return pred, {"votes": votes, "tally": dict(tally),
                          "chains": chains[:2]}

        # verify: answer, be asked to check it, answer again
        gen = generate(model, first, img, 128)
        turns = [("user", prompt), ("assistant", gen),
                 ("user", VERIFY_INSTRUCTION)]
        critique = generate(model, build_text(model, turns, True), img, 160)
        turns += [("assistant", critique),
                  ("user", FINAL_INSTRUCTION.format(opts=opts))]
        pred, lp = score_choices(model, build_text(model, turns, True), choices,
                                 images=[img])
        return pred, {"first": gen, "critique": critique, "logprobs": lp}

    raise SystemExit(f"unknown strategy {strategy!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--qa", required=True)
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--seg-cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", choices=IMAGE_CONDITIONS, default="identified")
    ap.add_argument("--strategy", choices=STRATEGIES, required=True)
    ap.add_argument("--subset", choices=["matched", "all"], default="matched")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="subsample to about this many probes, seeded. Sampling "
                         "is BY PAIR, not by probe: the matched pair is the only "
                         "unit on which the composite question can be scored "
                         "without the gap-magnitude shortcut, so a probe-level "
                         "sample would spend ten generations each on items whose "
                         "partner is missing. k=10 self-consistency is 10 chains "
                         "per probe and does not fit the full subset here.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    qa_path = Path(args.qa)
    files = sorted(qa_path.glob("*.jsonl")) if qa_path.is_dir() else [qa_path]
    items = [json.loads(l) for f in files for l in open(f) if l.strip()]
    if args.subset == "matched":
        keep = matched_subset_keys(qa_path)
        if keep is not None:
            before = len(items)
            items = [r for r in items if r["qid"] in keep]
            print(f"growth-matched subset: {len(items)} of {before} probes",
                  flush=True)
    if args.sample:
        import random as _random
        by_pair: dict[str, list[dict]] = defaultdict(list)
        for r in items:
            by_pair[r.get("pair_id") or r["qid"]].append(r)
        full = sorted(p for p, v in by_pair.items() if len(v) == 2)
        rest = sorted(p for p, v in by_pair.items() if len(v) != 2)
        rng = _random.Random(args.seed)
        rng.shuffle(full)
        rng.shuffle(rest)
        picked: list[dict] = []
        # complete pairs first, whole: half a pair costs the same to run and
        # cannot answer the question the pair exists to ask
        for p in full:
            if len(picked) + 2 > args.sample:
                break
            picked += by_pair[p]
        for p in rest:
            if len(picked) >= args.sample:
                break
            picked += by_pair[p]
        items = sorted(picked, key=lambda r: r["qid"])
        n_pairs = sum(1 for p in full if by_pair[p][0] in items)
        print(f"sampled {len(items)} probes ({n_pairs} complete pairs), "
              f"seed={args.seed}", flush=True)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no probes selected")

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
    legend = legend_for(args.condition)
    by_vol: dict[str, list[dict]] = defaultdict(list)
    for r in items:
        by_vol["_".join(r["qid"].split("_")[:2])].append(r)

    written = skipped = 0
    with open(args.out, "w") as fout:
        for vid, group in sorted(by_vol.items()):
            volp = task / "imagesTr" / f"{vid}.nii.gz"
            labp = task / "labelsTr" / f"{vid}.nii.gz"
            segp = Path(args.seg_cache) / f"{vid}_seg.nii.gz"
            if not (volp.exists() and labp.exists() and segp.exists()):
                skipped += len(group)
                continue
            vol, affine = load_ras(str(volp))
            gt, _ = load_ras(str(labp))
            seg, _ = load_ras(str(segp))
            spacing = np.abs(np.diag(affine)[:3])
            lesions = dict(find_lesions(gt == lesion_label, affine))
            cache: dict[tuple[str, str], tuple] = {}
            for r in group:
                lk = lesion_key(r["qid"])
                tname = r.get("provenance", {}).get("target")
                if lk is None or lk not in lesions or tname not in name2lab:
                    skipped += 1
                    continue
                ck = (lk, tname)
                if ck not in cache:
                    tmask = seg == name2lab[tname]
                    if not tmask.any():
                        skipped += 1
                        continue
                    img, geom = render(vol.astype(np.int16), lesions[lk], tmask,
                                       spacing, args.condition)
                    cache[ck] = (Image.fromarray(img).convert("RGB"), geom)
                image, geom = cache[ck]
                choices = r.get("choices") or ["no", "yes"]
                asked = (f"{legend} {r['question']}" if legend else r["question"])
                pred, detail = answer(model, args.strategy, asked, choices,
                                      image, r["qid"])
                fout.write(json.dumps({
                    "qid": r["qid"], "organ": organ,
                    "condition": f"{args.condition}-{args.strategy}",
                    "prediction": pred, "gold": r["answer"],
                    "pair_id": r.get("pair_id"), "logprobs": detail,
                    "geometry": geom,
                }) + "\n")
                written += 1
                if written % 50 == 0:
                    print(f"  {written} written", flush=True)

    rows = [json.loads(l) for l in open(args.out)]
    acc = 100.0 * sum(r["prediction"] == r["gold"] for r in rows) / max(len(rows), 1)
    share = Counter(r["prediction"] for r in rows)
    print(f"wrote {args.out}: {written} scored, {skipped} skipped, "
          f"accuracy {acc:.1f}%, predictions {dict(share)}")


if __name__ == "__main__":
    main()
