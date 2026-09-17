"""
Run a vision-language model on geometry-derived spatial QA.

Ties together the pieces: render.py turns a RAS+ volume into the orthogonal
views that actually carry each question's information, the model answers, and
eval_spatial.py scores accuracy plus the antisymmetry consistency probe.

View selection is not a detail. A question about superior/inferior is
undecidable from a single axial slice, so asking it that way measures the
harness rather than the model. `--views category` therefore shows each question
only the planes render.views_for_category() says can answer it; `--views all`
shows all three, which is the fairer default when comparing against models that
expect a fixed input.

  python vlm_infer.py --qa ../out_spleen/qa --volumes ../data/Task09_Spleen/imagesTr \
                      --model Qwen/Qwen2.5-VL-7B-Instruct --out preds.jsonl --limit 200
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from render import montage, orthogonal_views, views_for_category  # noqa: E402
from scene_graph import load_ras  # noqa: E402


def volume_id_of(qid: str) -> str:
    """Recover the volume id from a QA id.

    qa_gen emits '<volume>_<5-digit counter>' and hard_negatives appends '_inv'.
    Stripping every trailing numeric part is WRONG: volume ids themselves end in
    digits ('spleen_10'), so a greedy strip turns 'spleen_10_00042' into
    'spleen', and then no volume is ever found. Remove one '_inv' and exactly
    one zero-padded counter, nothing more. Ids in other formats (lesion QA)
    pass through untouched.
    """
    s = qid[:-4] if qid.endswith("_inv") else qid
    return re.sub(r"_\d{4,}$", "", s)


def find_volume(volumes_dir: Path, vid: str) -> Path | None:
    for p in volumes_dir.rglob(f"{vid}.nii*"):
        if not p.name.startswith("._"):
            return p
    return None


def to_png_b64(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_for(qa: dict, volume: Path, preset: str, view_mode: str) -> np.ndarray:
    vol, affine = load_ras(str(volume))
    spacing = np.abs(np.diag(affine)[:3])
    r = orthogonal_views(vol, spacing, preset=preset)
    avail = r.available()
    if view_mode == "category":
        wanted = set(views_for_category(qa.get("category", "")))
        avail = {k: v for k, v in avail.items() if k in wanted} or avail
    return montage(avail)


def build_prompt(qa: dict, view_names: list[str]) -> str:
    choices = qa.get("choices")
    hint = (f" Answer with exactly one of: {', '.join(choices)}."
            if choices else " Answer concisely.")
    return (
        f"This is a CT scan shown as {', '.join(view_names)} view(s), left to right. "
        "Orientation is radiological: patient's right appears on the left of the "
        "image; superior is up.\n"
        f"Question: {qa['question']}{hint} Reply with the answer only."
    )


class LocalVLM:
    """In-process Qwen2.5-VL inference.

    Preferred over standing up a vLLM server here: this box is shared, GPU 0
    already hosts someone else's engine, and a heavyweight install risks
    disturbing their environment. Throughput is lower but the workload is a few
    hundred short completions, so it does not matter.
    """

    def __init__(self, model_id: str, device: str = "cuda:1",
                 max_new_tokens: int = 16):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device).eval()

    def ask(self, prompt: str, img) -> str:
        from PIL import Image
        import numpy as np

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img],
                                return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()


def ask_openai_compatible(endpoint: str, model: str, prompt: str, png_b64: str,
                          timeout: int = 120) -> str:
    import urllib.request

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
        ]}],
        "max_tokens": 16, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--volumes", required=True)
    ap.add_argument("--endpoint", default="http://localhost:8001")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stratify", action="store_true",
                    help="sample evenly ACROSS volumes instead of taking the "
                         "first N QA. Without it --limit 300 drew from only 2 "
                         "volumes, so the effective sample size was 2, not 300.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", default="soft_tissue")
    ap.add_argument("--views", choices=["all", "category"], default="all")
    ap.add_argument("--save-renders", default=None)
    ap.add_argument("--backend", choices=["local", "endpoint"], default="local")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--blind", action="store_true",
                    help="ask WITHOUT the image -- the paired null for "
                         "margin_response.py; any margin dependence measured "
                         "here is confound, not grounding")
    args = ap.parse_args()

    qa_path = Path(args.qa)
    files = sorted(qa_path.glob("*.jsonl")) if qa_path.is_dir() else [qa_path]
    qas = []
    for f in files:
        with open(f) as fh:
            qas.extend(json.loads(l) for l in fh if l.strip())
    if args.limit and args.stratify:
        import random
        from collections import defaultdict as _dd
        buckets = _dd(list)
        for qa in qas:
            buckets[volume_id_of(qa["qid"])].append(qa)
        rng = random.Random(args.seed)
        per = max(1, args.limit // max(len(buckets), 1))
        picked = []
        for vid in sorted(buckets):
            items = buckets[vid]
            rng.shuffle(items)
            picked.extend(items[:per])
        rng.shuffle(picked)
        qas = picked[: args.limit]
    elif args.limit:
        qas = qas[: args.limit]
    print(f"{len(qas)} QA over {len(files)} shard(s)")

    # group by volume so each is loaded and rendered once, not per question
    by_vol: dict[str, list[dict]] = defaultdict(list)
    for qa in qas:
        by_vol[volume_id_of(qa["qid"])].append(qa)
    print(f"{len(by_vol)} distinct volumes")

    local = None
    if args.backend == "local":
        print(f"loading {args.model} on {args.device} ...", flush=True)
        local = LocalVLM(args.model, device=args.device)
        print("model ready", flush=True)

    vols_dir = Path(args.volumes)
    written = missing = errors = 0
    with open(args.out, "w") as out:
        for vid, items in sorted(by_vol.items()):
            vpath = find_volume(vols_dir, vid)
            if vpath is None:
                missing += len(items)
                print(f"  {vid}: volume not found, skipping {len(items)} QA")
                continue

            cache: dict = {}
            for qa in items:
                key = args.views if args.views == "all" else qa.get("category", "")
                if key not in cache:
                    img = render_for(qa, vpath, args.window, args.views)
                    cache[key] = img if args.backend == "local" else to_png_b64(img)
                    if args.save_renders:
                        d = Path(args.save_renders)
                        d.mkdir(parents=True, exist_ok=True)
                        from render import save_png
                        save_png(img, str(d / f"{vid}_{key or 'all'}.png"))
                names = (["axial", "coronal", "sagittal"] if args.views == "all"
                         else list(views_for_category(qa.get("category", ""))))
                try:
                    prompt = build_prompt(qa, sorted(names))
                    if args.blind:
                        # identical prompt, no image: isolates how much of any
                        # margin dependence comes from language priors alone
                        prompt = prompt.split("Question:", 1)[-1]
                        prompt = "Question:" + prompt
                    if local is not None:
                        import numpy as np
                        img_in = (np.zeros((64, 64), dtype=np.uint8)
                                  if args.blind else cache[key])
                        pred = local.ask(prompt, img_in)
                    else:
                        pred = ask_openai_compatible(
                            args.endpoint, args.model, prompt, cache[key])
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  request failed: {e}", file=sys.stderr)
                    continue
                out.write(json.dumps({"qid": qa["qid"],
                                      "prediction": pred.strip()}) + "\n")
                written += 1
            print(f"  {vid}: {len(items)} QA answered", flush=True)

    print(f"\nwrote {written} predictions to {args.out}"
          f"  (missing volumes: {missing}, request errors: {errors})")
    print(f"score with:  python eval_spatial.py --qa {args.qa} --pred {args.out}")


def selftest() -> None:
    assert volume_id_of("spleen_10_00042") == "spleen_10"
    assert volume_id_of("spleen_10_00042_inv") == "spleen_10"
    assert volume_id_of("lung_001_lesion3_container") == "lung_001_lesion3_container"

    qa = {"question": "Is A superior to B?", "choices": ["inferior", "superior"],
          "category": "longitudinal"}
    p = build_prompt(qa, ["coronal", "sagittal"])
    assert "inferior, superior" in p and "coronal, sagittal" in p, p
    # the prompt must state orientation, or left/right answers are unscoreable
    assert "right appears on the left" in p

    # category view selection must never hand a question a plane that cannot
    # answer it
    assert "axial" not in views_for_category("longitudinal")

    img = np.zeros((16, 16), dtype=np.uint8)
    b = to_png_b64(img)
    assert isinstance(b, str) and len(b) > 20

    print("selftest OK — volume-id parsing, orientation-stating prompt, "
          "category-aware view selection, PNG encoding")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
