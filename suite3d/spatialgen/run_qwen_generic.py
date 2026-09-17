"""
Qwen2.5-VL runner shared by every remaining cell of the design.

One script rather than several near-copies: the cells differ only in how the
volume becomes an image (a single axial slice for the 2D cell, orthogonal views
for the 3D cells) and in which QA file is read. Everything that must stay
constant across cells -- likelihood scoring, the matched blind arm, the prompt
skeleton -- is therefore literally the same code, which is what makes the
comparison between cells meaningful.

  --render slice    one axial slice at the index the QA was generated from
  --render montage  axial + coronal + sagittal side by side
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from render import montage, orthogonal_views, window  # noqa: E402
from scene_graph import load_ras  # noqa: E402


def volume_id_of(qid: str, mode: str) -> str:
    if mode == "slice":                     # '<vid>_z<idx>_<a>__<b>_<axis>'
        return qid.split("_z")[0]
    parts = qid.split("_")                  # counterfactual ids
    for i, p in enumerate(parts):
        if p.startswith("lesion") or p == "resect":
            return "_".join(parts[:i])
    return "_".join(parts[:2])


def render_slice(vol: np.ndarray, z: int, preset: str = "soft_tissue") -> np.ndarray:
    """One axial plane, oriented radiologically (patient right on image left)."""
    w = window(vol, preset)
    z = int(np.clip(z, 0, w.shape[2] - 1))
    plane = w[:, :, z].T                    # -> (y, x) so rows are A->P
    return np.ascontiguousarray(plane[::-1, ::-1])


def render_montage(vol: np.ndarray, spacing: np.ndarray,
                   preset: str = "soft_tissue") -> np.ndarray:
    return montage(orthogonal_views(vol, spacing, preset=preset).available())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--volumes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", choices=["slice", "montage"], required=True)
    ap.add_argument("--slices", default=None,
                    help="json mapping volume id -> slice index (slice mode)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    qa_dir = Path(args.qa)
    rows = [json.loads(l) for f in sorted(qa_dir.glob("*.jsonl"))
            for l in open(f) if l.strip()]
    rows = [r for r in rows if r.get("choices")]
    if args.limit:
        rows = rows[: args.limit]

    slice_idx = {}
    if args.slices:
        slice_idx = json.load(open(args.slices))

    by_vol = defaultdict(list)
    for r in rows:
        by_vol[volume_id_of(r["qid"], args.render)].append(r)
    print(f"{len(rows)} QA over {len(by_vol)} volumes, render={args.render} "
          f"({'BLIND' if args.blind else 'sighted'})", flush=True)

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    print("model ready", flush=True)

    view_desc = ("an axial CT slice" if args.render == "slice"
                 else "a CT scan shown as axial, coronal and sagittal views")

    def score(question: str, choices: list[str], img) -> tuple:
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text",
             "text": (f"This is {view_desc}. Orientation is radiological: the "
                      f"patient's right appears on the left of the image. "
                      f"{question} Answer with exactly one of: "
                      f"{', '.join(choices)}.")}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        out = {}
        for c in choices:
            enc = proc(text=[text + c], images=[img],
                       return_tensors="pt").to(args.device)
            n_c = len(proc.tokenizer(c, add_special_tokens=False)["input_ids"])
            with torch.inference_mode():
                logits = model(**enc).logits[0, :-1].float()
            tgt = enc["input_ids"][0, 1:]
            lp = torch.log_softmax(logits, dim=-1)
            out[c] = float(lp[-n_c:].gather(1, tgt[-n_c:, None]).mean())
        return max(out, key=out.get), out

    vols = Path(args.volumes)
    written = errors = missing = 0
    with open(args.out, "w") as f:
        for vid, items in sorted(by_vol.items()):
            hits = [p for p in vols.rglob(f"{vid}.nii*")
                    if not p.name.startswith("._")]
            if not hits:
                missing += len(items)
                continue
            vol, affine = load_ras(str(hits[0]))
            if args.render == "slice":
                z = int(slice_idx.get(vid, vol.shape[2] // 2))
                arr = render_slice(vol, z)
            else:
                arr = render_montage(vol, np.abs(np.diag(affine)[:3]))
            img = Image.fromarray(arr).convert("RGB")
            if args.blind:
                img = Image.new("RGB", img.size, (128, 128, 128))

            for r in items:
                try:
                    pred, sc = score(r["question"], r["choices"], img)
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  failed {r['qid']}: {e}", file=sys.stderr)
                    continue
                f.write(json.dumps({
                    "qid": r["qid"], "prediction": pred, "gold": r["answer"],
                    "category": r.get("category") or r.get("kind"),
                    "pair_id": r.get("pair_id"),
                    "logprobs": {k: round(v, 4) for k, v in sc.items()},
                }) + "\n")
                written += 1
            f.flush()
            print(f"  {vid}: {len(items)} answered", flush=True)

    print(f"\nwrote {written} (errors: {errors}, missing volumes: {missing})")


def selftest() -> None:
    assert volume_id_of("spleen_10_z44_liver__aorta_lateral", "slice") == "spleen_10"
    assert volume_id_of("lung_001_lesion1_aorta_g12", "montage") == "lung_001"
    assert volume_id_of("lung_010_resect_lung_upper_lobe_right",
                        "montage") == "lung_010"

    vol = np.full((40, 50, 20), -1000, dtype=np.int16)
    vol[30:38, 20:30, 10] = 900             # bright, high x  -> patient right
    s = render_slice(vol, 10)
    assert s.dtype == np.uint8 and s.ndim == 2, (s.dtype, s.shape)
    assert s.max() == 255, s.max()
    # radiological convention: patient right must land on the LEFT half
    h, w = s.shape
    left_half = s[:, : w // 2].max()
    right_half = s[:, w // 2:].max()
    assert left_half > right_half, (left_half, right_half,
                                    "patient right is not on image left")

    # out-of-range slice index must clamp, not raise
    assert render_slice(vol, 9999).shape == s.shape

    print("selftest OK — volume ids parsed for both id families; axial slice "
          "renders uint8 with patient right on the image left; index clamps")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
