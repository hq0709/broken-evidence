"""
Paired sighted/blind evaluation across model families on one common subset.

Every model sees the SAME 2,262 probes (1,131 yes / 1,131 no, four organs), so
differences between rows are model differences, not sampling differences. Many
families are evaluated because two systems cannot support a claim about
volumetric VLMs as a class.

Scoring is likelihood over the option strings for every family, never text
parsing -- 3D medical VLMs answer in referring-expression templates whose
wording embeds the answer words.

Families and how they receive the volume:
  qwen      orthogonal-view montage (2.5-VL 3B / 7B / 32B)
  internvl  orthogonal-view montage
  huatuo    orthogonal-view montage (LLaVA-style medical 2D model)
  m3d       native volume, resampled to 1x32x256x256
  med3dvlm  native volume, resampled to the model's own input size
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from render import montage, orthogonal_views  # noqa: E402
from scene_graph import load_ras  # noqa: E402

FAMILY = {
    "Qwen/Qwen2.5-VL-3B-Instruct": "qwen",
    "Qwen/Qwen2.5-VL-7B-Instruct": "qwen",
    "Qwen/Qwen2.5-VL-32B-Instruct": "qwen",
    # Scale. 72.7B params = 146.8 GB of bf16 weights, so this needs
    # `--device auto` across two 80 GB cards and leaves ~12 GB for activations;
    # it cannot share either card with anything else. Two larger candidates
    # do not fit on two 80 GB cards and are deliberately absent:
    # InternVL3-78B is 156.8 GB (2.4 GB of headroom on 2x80 GB, which activations
    # exceed) and Llama-3.2-90B-Vision is 177.2 GB and gated.
    "Qwen/Qwen2.5-VL-72B-Instruct": "qwen",
    # the "-hf" repo, not OpenGVLab/InternVL3-8B. The original is model_type
    # `internvl_chat` whose AutoProcessor resolves to a bare Qwen2Tokenizer with
    # no `.tokenizer` and no image handling; the -hf conversion is model_type
    # `internvl`, natively supported, and returns a real InternVLProcessor.
    "OpenGVLab/InternVL3-8B-hf": "internvl",
    "OpenGVLab/InternVL3-14B-hf": "internvl",
    "Qwen/Qwen3-VL-8B-Instruct": "qwen3vl",
    # five further families, chosen for architectural distance from the above
    # and for native transformers support -- remote-code repositories cost two
    # days of debugging earlier in this study (InternVL3, HuatuoGPT)
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": "llavaov",
    "HuggingFaceM4/Idefics3-8B-Llama3": "idefics3",
    "mistral-community/pixtral-12b": "pixtral",
    "rhymes-ai/Aria": "aria",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct": "smolvlm",
    "FreedomIntelligence/HuatuoGPT-Vision-7B": "huatuo",
    "GoodBaiBai88/M3D-LaMed-Phi-3-4B": "m3d",
    # third native volumetric medical VLM: same M3D input pipeline, different
    # LLM backbone, so it isolates backbone from vision tower
    "GoodBaiBai88/M3D-LaMed-Llama-2-7B": "m3d",
    "MagicXin/Med3DVLM-Qwen-2.5-7B": "med3dvlm",
}

# tag -> HF id, the inverse of FAMILY for the tags used as filename stems.
# run_roi_arms.py used to carry its own three-entry copy of this; a second
# hardcoded model map is how InternVL3 ended up being fed through the Qwen
# preprocessing path for 226 probes.
MODEL_ID = {
    "smolvlm": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl": "OpenGVLab/InternVL3-8B-hf",
    "internvl14": "OpenGVLab/InternVL3-14B-hf",
    "llavaov": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "idefics3": "HuggingFaceM4/Idefics3-8B-Llama3",
    "pixtral": "mistral-community/pixtral-12b",
    "aria": "rhymes-ai/Aria",
}
assert all(v in FAMILY for v in MODEL_ID.values()), \
    "every tag must resolve to an id run_multimodel knows how to feed"

PROMPT = ("This is a CT scan shown as axial, coronal and sagittal views. "
          "{q} Answer with exactly one of: {opts}.")

# A model that receives the volume itself is not looking at three views, so
# native volumetric models get a prompt that describes the input they receive.
NATIVE_PROMPT = ("This is a CT volume. {q} Answer with exactly one of: {opts}.")


def volume_of(qid: str) -> str:
    return "_".join(qid.split("_")[:2])


def _patch_tied_weights_compat() -> None:
    """Let remote-code models written against transformers 4.x still load.

    transformers 5.x sets `all_tied_weights_keys` (a {target: source} dict) in
    PreTrainedModel.post_init(). Repositories whose custom classes predate that
    -- InternVL3 among them -- never set it, and weight loading dies with
    AttributeError before a single probe runs. Supplying a CLASS-level empty
    default is safe: any model whose post_init did run has an instance
    attribute that shadows it, so only the classes that would otherwise crash
    are affected.

    Empty means "no tied weights". If a model does tie its embeddings and the
    checkpoint omits the tied copy, that shows up as a missing-key warning at
    load time, which we check rather than assume.
    """
    from transformers.modeling_utils import PreTrainedModel
    if "all_tied_weights_keys" not in vars(PreTrainedModel):
        PreTrainedModel.all_tied_weights_keys = {}


class MontageModel:
    """Any HF vision-language model that consumes a single 2D image."""

    def __init__(self, model_id: str, device: str):
        """`device` is either a concrete device ("cuda:0") or "auto".

        "auto" hands device_map to accelerate, which shards the weights across
        every visible GPU -- required for the 32B and 25B-MoE tags, which do not
        fit one 80 GB card in bf16. When sharded there is no single model device,
        so input tensors go to the device holding the input embeddings; sending
        them to the literal string "auto" is not a device and raises.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.device = device
        self.sharded = (device == "auto")
        self.model_id = model_id
        _patch_tied_weights_compat()
        # Prefer the natively implemented processor, exactly as the model
        # loading below does, and reach for remote code only when there is no
        # native one. Passing trust_remote_code=True unconditionally makes
        # transformers fetch the repository's own processor even when it ships
        # a broken one: rhymes-ai/Aria declares a vision_processor.py it does
        # not contain, so every Aria run died at load with a missing-file error
        # while transformers had a working AriaProcessor all along.
        try:
            self.proc = AutoProcessor.from_pretrained(model_id)
        except Exception as native_err:
            try:
                self.proc = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True)
            except Exception:
                raise native_err
        # No try/except here on purpose. A Qwen2.5-VL checkpoint MUST load
        # through this class, so any failure is a real failure. A broad
        # except here would turn a CUDA OOM into a misleading "Unrecognized
        # configuration class" error further down.
        if "Qwen2.5-VL" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map=device,
                output_loading_info=True)
            self.model = self.model.eval()
            self._assert_weights_loaded(model_id, info)
            return
        # Prefer the natively supported path over the repository's remote code.
        # InternVL3 ships a custom class written against transformers 4.x that
        # dies during weight loading on 5.x; transformers 5.14 implements the
        # architecture itself, so the native loader is both correct and the one
        # that stays maintained.
        # Fall back ONLY for the reason the fallback exists -- the repository
        # is not natively supported. Any other failure (OOM, missing dependency,
        # corrupt download) must surface here rather than be re-raised later as
        # "Unrecognized configuration class".
        from transformers import AutoConfig, AutoModelForImageTextToText
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES as _ITT)
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if getattr(cfg, "model_type", None) in _ITT:
            self.model, info = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map=device,
                output_loading_info=True)
            self.model = self.model.eval()
            self._assert_weights_loaded(model_id, info)
            return
        print(f"  {getattr(cfg, 'model_type', '?')} has no native "
              f"image-text-to-text class; using remote code", file=sys.stderr)
        self.model, info = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True, output_loading_info=True)
        self.model = self.model.eval()
        self._assert_weights_loaded(model_id, info)

    def _resolve_device(self):
        """Device that model inputs must be placed on."""
        if not self.sharded:
            return self.device
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    @staticmethod
    def _assert_weights_loaded(model_id: str, info: dict) -> None:
        """Refuse a model whose pretrained weights did not map onto it.

        transformers reports a key mismatch as a warning and returns a model
        with randomly initialised parameters. That model runs, emits logits and
        produces a benchmark score -- of a random network. rhymes-ai/Aria ships
        a checkpoint keyed `language_model.model.layers.*` against a native
        implementation expecting `model.language_model.*`, so every
        language-model weight was unexpected and every expected one missing.
        `from_pretrained`
        already knows exactly which keys it could not fill, so the check is
        exact rather than inferential.
        """
        missing = [k for k in info.get("missing_keys", [])
                   if "language_model" in k or ".layers." in k]
        if not missing:
            return
        raise SystemExit(
            f"{model_id}: {len(missing)} language-model weights were newly "
            f"initialised because the checkpoint does not contain them "
            f"(e.g. {missing[0]}). The checkpoint does not map onto this "
            f"architecture; any score from it would be a random network's.")

    def score(self, question: str, choices: list[str], img) -> tuple:
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT.format(q=question,
                                                   opts=", ".join(choices))}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        out = {}
        for c in choices:
            enc = self.proc(text=[text + c], images=[img],
                            return_tensors="pt").to(self._resolve_device())
            n_c = len(self.proc.tokenizer(c, add_special_tokens=False)["input_ids"])
            with self.torch.inference_mode():
                logits = self.model(**enc).logits[0, :-1]
            tgt = enc["input_ids"][0, 1:]
            # Only the option's own token positions are ever read, so cast and
            # normalise just those rows. log_softmax is row-wise, so this is the
            # same number to the bit -- but the discarded work was not free: a
            # montage tiled to ~3,000 vision tokens against a 152k vocabulary
            # makes the full float32 logits 1.8 GB, allocated per option per
            # probe, and that allocation is what pushed three 8B jobs sharing a
            # card into CUDA OOM at the log_softmax line.
            lp = self.torch.log_softmax(logits[-n_c:].float(), dim=-1)
            out[c] = float(lp.gather(1, tgt[-n_c:, None]).mean())
        return max(out, key=out.get), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--data-root", default="data/msd")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # A model this code has not been told how to feed is a caller error, not a default.
    if args.model not in FAMILY:
        raise SystemExit(
            f"unknown model id {args.model!r}; add it to FAMILY with the "
            "preprocessing family it needs. Known ids:\n  "
            + "\n  ".join(sorted(FAMILY)))
    fam = FAMILY[args.model]
    rows = [json.loads(l) for l in open(args.qa) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    by_vol = defaultdict(list)
    for r in rows:
        by_vol[(r["organ"], volume_of(r["qid"]))].append(r)
    print(f"{len(rows)} probes / {len(by_vol)} volumes / family={fam} / "
          f"{'BLIND' if args.blind else 'sighted'}", flush=True)

    if fam == "m3d":
        from m3d_infer import M3D, TARGET_SHAPE, preprocess_volume
        model = M3D(device=args.device, model_id=args.model)
    elif fam == "med3dvlm":
        # different input geometry from M3D -- see med3dvlm_infer for why the
        # volume is NOT transposed here
        from med3dvlm_infer import TARGET_SHAPE, Med3DVLM, preprocess_volume
        model = Med3DVLM(device=args.device)
    else:
        model = MontageModel(args.model, args.device)
    print("model ready", flush=True)

    from PIL import Image
    written = errors = 0
    with open(args.out, "w") as f:
        for (organ, vid), items in sorted(by_vol.items()):
            vp = Path(args.data_root) / organ / "imagesTr" / f"{vid}.nii.gz"
            if not vp.exists():
                continue
            if fam in ("m3d", "med3dvlm"):
                # blind arm must match the sighted tensor's rank exactly: M3D
                # takes (C, D, H, W), Med3DVLM takes (B, C, H, W, T)
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

            for it in items:
                try:
                    if fam in ("m3d", "med3dvlm"):
                        pred, sc = model.score_choices(
                            NATIVE_PROMPT.format(q=it["question"],
                                                 opts=", ".join(it["choices"])),
                            prepared, it["choices"])
                    else:
                        pred, sc = model.score(it["question"], it["choices"],
                                               prepared)
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  failed {it['qid']}: {e}", file=sys.stderr)
                    continue
                f.write(json.dumps({
                    "qid": it["qid"], "organ": organ, "prediction": pred,
                    "gold": it["answer"], "pair_id": it.get("pair_id"),
                    "logprobs": {k: round(v, 4) for k, v in sc.items()}}) + "\n")
                written += 1
            f.flush()
    print(f"\nwrote {written} (errors: {errors})")


if __name__ == "__main__":
    main()
