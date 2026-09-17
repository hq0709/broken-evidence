"""
Run M3D-LaMed-Phi-3-4B on geometry-derived spatial QA.

Why this model
--------------
M3D-LaMed consumes the volume directly rather than 2D projections of it, and is
trained on medical data, so it tests the volumetric setting with a native
medical VLM.

Preprocessing (per the official repo, since the HF card is empty)
-----------------------------------------------------------------
  * resize the volume to 1 x 32 x 256 x 256
  * Min-Max normalise to [0, 1]
Prompt format is Phi-3 chat with 256 `<im_patch>` placeholders: the vision tower
emits 32/4 x 256/16 x 256/16 = 8x16x16 patches, and the spatial projector pools
by 2 per axis, giving 4x8x8 = 256 image tokens.

The blind arm passes an all-zero volume of the same shape, so the two arms
differ only in image CONTENT, not in architecture path or token count.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scene_graph import load_ras  # noqa: E402
from vlm_infer import find_volume, volume_id_of  # noqa: E402

DEFAULT_MODEL_ID = "GoodBaiBai88/M3D-LaMed-Phi-3-4B"
# kept for backward compatibility with scripts that import MODEL_ID directly
MODEL_ID = DEFAULT_MODEL_ID
TARGET_SHAPE = (32, 256, 256)
N_IMAGE_TOKENS = 256
IMAGE_TOKEN = "<im_patch>"


def preprocess_volume(path: str, target=TARGET_SHAPE) -> np.ndarray:
    """NIfTI -> (1, 32, 256, 256) float32 in [0, 1].

    The volume is loaded in RAS+ (x=right, y=anterior, z=superior) and
    transposed to (z, y, x) so that axis 0 indexes AXIAL slices, which is what a
    depth-32 stack means for a CT model. Feeding (x, y, z) unchanged would hand
    the model 32 sagittal slices instead -- silently, with no error.
    """
    from scipy.ndimage import zoom

    vol, _ = load_ras(path)
    vol = np.asarray(vol, dtype=np.float32)
    vol = np.transpose(vol, (2, 1, 0))          # (x,y,z) -> (z,y,x) = axial stack

    factors = [t / s for t, s in zip(target, vol.shape)]
    vol = zoom(vol, factors, order=1)
    # zoom can land one voxel off on odd sizes; pad or crop to be exact
    out = np.zeros(target, dtype=np.float32)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(target, vol.shape))
    out[sl] = vol[sl]

    lo, hi = float(out.min()), float(out.max())
    if hi > lo:
        out = (out - lo) / (hi - lo)
    else:
        out[:] = 0.0
    return out[None]                             # add channel -> (1, 32, 256, 256)


def _patch_monai_compat() -> None:
    """Let M3D's 2024 modeling code run against a current MONAI.

    M3D calls `PatchEmbeddingBlock(..., pos_embed="perceptron")`. MONAI renamed
    that argument to `proj_type` (>=1.4), so the call raises TypeError. Pinning
    MONAI 1.3 is not an option here: it calls `importer.find_module`, removed in
    Python 3.12, and this environment is 3.14.

    So translate the old kwarg instead. This changes no numerics -- it is the
    same argument under a new name.
    """
    import inspect

    from monai.networks.blocks import patchembedding as pe
    from monai.networks.nets import vit as vit_mod

    # Both the block and the ViT that wraps it took `pos_embed`; the rename
    # touched every caller, so shim each class that still lacks the old name.
    for module, cls_name in ((pe, "PatchEmbeddingBlock"), (vit_mod, "ViT")):
        cls = getattr(module, cls_name, None)
        if cls is None or getattr(cls, "_m3d_shim", False):
            continue
        params = inspect.signature(cls.__init__).parameters
        if "pos_embed" in params:
            continue                       # old MONAI: nothing to translate

        orig = cls.__init__

        def shimmed(self, *args, __orig=orig, **kwargs):
            if "pos_embed" in kwargs:
                kwargs["proj_type"] = kwargs.pop("pos_embed")
            return __orig(self, *args, **kwargs)

        cls.__init__ = shimmed
        cls._m3d_shim = True


# Arguments current transformers passes into forward() that a 2024-era custom
# model never declared. Each is a pure optimisation hint whose absence changes
# speed, not outputs -- `logits_to_keep` limits how many positions get a logit
# computed. Only these are dropped; anything else still raises, so a genuine
# incompatibility is not silently swallowed.
_DROPPABLE_FORWARD_KWARGS = {"logits_to_keep", "num_logits_to_keep"}


def _drop_unsupported_generate_kwargs(model) -> None:
    import inspect

    orig = model.forward
    accepted = set(inspect.signature(orig).parameters)
    if "kwargs" in accepted:
        return                                   # already permissive

    def forward(*args, **kwargs):
        unknown = set(kwargs) - accepted
        droppable = unknown & _DROPPABLE_FORWARD_KWARGS
        for k in droppable:
            kwargs.pop(k)
        return orig(*args, **kwargs)

    model.forward = forward


class M3D:
    def __init__(self, device: str = "cuda:1", max_new_tokens: int = 16,
                 model_id: str = DEFAULT_MODEL_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _patch_monai_compat()
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        # the id must come from the caller: this was hardcoded, so requesting
        # M3D-LaMed-Llama-2-7B silently loaded Phi-3 instead and produced an
        # output file byte-identical to the Phi-3 run, which would have been
        # reported as a third native volumetric system
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(
            model_id, model_max_length=768, padding_side="right",
            use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True).eval()
        _drop_unsupported_generate_kwargs(self.model)

    def score_choices(self, question: str, volume: np.ndarray,
                      choices: list[str]) -> tuple[str, dict[str, float]]:
        """Pick the choice with the highest length-normalised log-likelihood.

        Free-text parsing is unusable for this model. M3D-LaMed answers almost
        everything with a referring-expression template -- asked for France's
        capital it replies "The object in question is Paris", and asked whether
        A is left or right of B it replies "The object is rib left 5". That
        string contains the token "left" purely because it is part of a STRUCTURE
        NAME, so a regex scorer would read it as the answer and manufacture an
        accuracy driven by which organ happens to be named. Scoring by
        likelihood of each candidate answer is immune to the template.

        Normalising by token count keeps multi-token options ("posterior") from
        being penalised against single-token ones ("left").
        """
        import torch

        prompt = (f"<|user|>\n{IMAGE_TOKEN * N_IMAGE_TOKENS}{question}"
                  f"<|end|>\n<|assistant|>\n")
        p_ids = self.tok(prompt, return_tensors="pt")["input_ids"].to(self.device)
        img = torch.from_numpy(volume)[None].to(self.device,
                                                dtype=torch.bfloat16)

        scores: dict[str, float] = {}
        for c in choices:
            c_ids = self.tok(c, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(self.device)
            full = torch.cat([p_ids, c_ids], dim=1)
            with torch.inference_mode():
                out = self.model(images=img, input_ids=full)
            logits = out.logits[0, :-1].float()
            tgt = full[0, 1:]
            logp = torch.log_softmax(logits, dim=-1)
            n_c = c_ids.shape[1]
            tok_lp = logp[-n_c:].gather(1, tgt[-n_c:, None]).squeeze(1)
            scores[c] = float(tok_lp.mean())      # length-normalised
        best = max(scores, key=scores.get)
        return best, scores

    def ask(self, question: str, volume: np.ndarray) -> str:
        import torch

        prompt = (f"<|user|>\n{IMAGE_TOKEN * N_IMAGE_TOKENS}{question}"
                  f"<|end|>\n<|assistant|>\n")
        ids = self.tok(prompt, return_tensors="pt")["input_ids"].to(self.device)
        img = torch.from_numpy(volume)[None].to(
            device=self.device, dtype=torch.bfloat16)      # (1,1,32,256,256)
        with torch.inference_mode():
            out = self.model.generate(img, ids, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def build_question(qa: dict) -> str:
    ch = qa.get("choices")
    hint = (f" Answer with exactly one of: {', '.join(ch)}." if ch
            else " Answer concisely.")
    return f"{qa['question']}{hint} Reply with the answer only."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--volumes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stratify", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind", action="store_true",
                    help="feed an all-zero volume of identical shape")
    args = ap.parse_args()

    qa_path = Path(args.qa)
    files = sorted(qa_path.glob("*.jsonl")) if qa_path.is_dir() else [qa_path]
    qas = []
    for f in files:
        with open(f) as fh:
            qas.extend(json.loads(l) for l in fh if l.strip())

    if args.limit and args.stratify:
        import random
        buckets = defaultdict(list)
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

    by_vol = defaultdict(list)
    for qa in qas:
        by_vol[volume_id_of(qa["qid"])].append(qa)
    print(f"{len(qas)} QA over {len(by_vol)} volumes "
          f"({'BLIND' if args.blind else 'sighted'})", flush=True)

    print(f"loading {MODEL_ID} on {args.device} ...", flush=True)
    model = M3D(device=args.device)
    print("model ready", flush=True)

    vols_dir = Path(args.volumes)
    written = errors = 0
    with open(args.out, "w") as out:
        for vid, items in sorted(by_vol.items()):
            vpath = find_volume(vols_dir, vid)
            if vpath is None:
                print(f"  {vid}: volume not found, skipping {len(items)}")
                continue
            vol = (np.zeros((1, *TARGET_SHAPE), dtype=np.float32) if args.blind
                   else preprocess_volume(str(vpath)))
            for qa in items:
                try:
                    ch = qa.get("choices")
                    if ch and len(ch) >= 2:
                        # likelihood scoring, not text parsing: this model's
                        # replies embed structure names that contain the answer
                        # words, so a regex scorer reads "rib left 5" as "left"
                        pred, scores = model.score_choices(
                            build_question(qa), vol, ch)
                        rec = {"qid": qa["qid"], "prediction": pred,
                               "logprobs": {k: round(v, 4)
                                            for k, v in scores.items()}}
                    else:
                        rec = {"qid": qa["qid"],
                               "prediction": model.ask(build_question(qa), vol)}
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  failed: {e}", file=sys.stderr)
                    continue
                out.write(json.dumps(rec) + "\n")
                written += 1
            out.flush()
            print(f"  {vid}: {len(items)} answered", flush=True)

    print(f"\nwrote {written} predictions to {args.out} (errors: {errors})")


def selftest() -> None:
    import tempfile
    import nibabel as nib

    # an anisotropic volume with a known bright marker in the superior half
    arr = np.zeros((64, 80, 40), dtype=np.float32)
    arr[:, :, 30:] = 500.0
    arr[10:20, 10:20, 5:10] = -1000.0
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        nib.save(nib.Nifti1Image(arr, np.eye(4)), f.name)
        v = preprocess_volume(f.name)

    assert v.shape == (1, *TARGET_SHAPE), v.shape
    assert v.dtype == np.float32
    assert abs(v.min()) < 1e-6 and abs(v.max() - 1.0) < 1e-6, (v.min(), v.max())

    # axis 0 must index AXIAL slices: the bright band was at high z, so the
    # later entries along axis 0 must be brighter than the earlier ones
    early = v[0, :8].mean()
    late = v[0, -8:].mean()
    assert late > early, (early, late, "axis 0 is not the axial stack")

    # a constant volume must not produce NaN through the min-max division
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f2:
        nib.save(nib.Nifti1Image(np.full((20, 20, 20), 7.0, np.float32),
                                 np.eye(4)), f2.name)
        c = preprocess_volume(f2.name)
    assert np.isfinite(c).all() and c.max() == 0.0

    q = build_question({"question": "Is A superior to B?",
                        "choices": ["inferior", "superior"]})
    assert "inferior, superior" in q

    print(f"selftest OK — preprocess gives {v.shape} float32 in [0,1], "
          f"axis 0 verified as the axial stack (late {late:.3f} > early "
          f"{early:.3f}), constant volume handled without NaN")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        selftest()
    else:
        main()
