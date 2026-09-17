"""
Med3DVLM-Qwen-2.5-7B adapter: a second NATIVE volumetric medical VLM.

Why this model matters to the study
-----------------------------------
Every volumetric result so far rests on M3D-LaMed. One native 3D system cannot
support a claim about volumetric medical VLMs as a class. Med3DVLM uses a
different vision tower (dcformer, not vit3d), a different backbone (Qwen2.5, not
Phi-3), and a different input geometry -- so agreement between the two is
evidence about the regime rather than about one checkpoint.

Input convention -- read from the model's own source, not assumed
----------------------------------------------------------------
config.input_size = (256, 256, 128). The axis order is NOT the same as M3D's.
modeling.py line ~386 does:

    H, W, T = size
    rearrange(x, "b (h w t) c -> b c h w t", h=H, w=W, t=T)

so the tuple is (H, W, T) with T the slice axis: 128 axial slices of 256x256.
M3D's (32, 256, 256) is (D, H, W) with D first, which is why its adapter
transposes to put z first. Here a RAS+ array (x, y, z) already has the slice
axis last and must NOT be transposed. Guessing this wrong feeds coronal planes
to a model expecting axial ones, silently and without error -- the same class of
failure this project has repeatedly had to catch by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

MODEL_ID = "MagicXin/Med3DVLM-Qwen-2.5-7B"
TARGET_SHAPE = (256, 256, 128)      # (H, W, T) per the model's rearrange
N_IMAGE_TOKENS = 256                # config.proj_out_num
IMAGE_TOKEN = "<im_patch>"


def preprocess_volume(path: str, target=TARGET_SHAPE) -> np.ndarray:
    """NIfTI -> (1, 1, 256, 256, 128) float32 in [0, 1], slice axis LAST."""
    from scipy.ndimage import zoom

    from scene_graph import load_ras

    vol, _ = load_ras(path)
    vol = np.asarray(vol, dtype=np.float32)      # RAS+ (x, y, z); z is axial idx

    factors = [t / s for t, s in zip(target, vol.shape)]
    vol = zoom(vol, factors, order=1)
    out = np.zeros(target, dtype=np.float32)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(target, vol.shape))
    out[sl] = vol[sl]

    lo, hi = float(out.min()), float(out.max())
    out = (out - lo) / (hi - lo) if hi > lo else out * 0.0
    return out[None, None]                        # (B=1, C=1, H, W, T)


class Med3DVLM:
    def __init__(self, device: str = "cuda:0", max_new_tokens: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True).eval()
        from m3d_infer import _drop_unsupported_generate_kwargs
        _drop_unsupported_generate_kwargs(self.model)

    def _prompt(self, question: str) -> str:
        return (f"<|im_start|>user\n{IMAGE_TOKEN * N_IMAGE_TOKENS}{question}"
                f"<|im_end|>\n<|im_start|>assistant\n")

    def score_choices(self, question: str, volume: np.ndarray,
                      choices: list[str]) -> tuple[str, dict[str, float]]:
        """Length-normalised log-likelihood of each option string.

        Same protocol as the M3D adapter: never parse free text, because these
        models answer in templates whose wording embeds the answer words.
        """
        import torch

        p_ids = self.tok(self._prompt(question),
                         return_tensors="pt")["input_ids"].to(self.device)
        img = torch.from_numpy(volume).to(self.device, dtype=torch.bfloat16)

        scores: dict[str, float] = {}
        for c in choices:
            c_ids = self.tok(c, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(self.device)
            full = torch.cat([p_ids, c_ids], dim=1)
            with torch.inference_mode():
                out = self.model(images=img, input_ids=full)
            logits = out.logits[0, :-1].float()
            tgt = full[0, 1:]
            lp = torch.log_softmax(logits, dim=-1)
            n_c = c_ids.shape[1]
            scores[c] = float(lp[-n_c:].gather(1, tgt[-n_c:, None]).mean())
        return max(scores, key=scores.get), scores


def selftest() -> None:
    import tempfile

    import nibabel as nib

    # marker high in z: after preprocessing it must remain at high T, since the
    # slice axis is last for this model (unlike M3D)
    arr = np.zeros((80, 90, 40), dtype=np.float32)
    arr[:, :, 30:] = 800.0
    arr[10:20, 10:20, 2:6] = -1000.0
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        nib.save(nib.Nifti1Image(arr, np.eye(4)), f.name)
        v = preprocess_volume(f.name)

    assert v.shape == (1, 1, *TARGET_SHAPE), v.shape
    assert v.dtype == np.float32
    assert abs(v.min()) < 1e-6 and abs(v.max() - 1.0) < 1e-6, (v.min(), v.max())

    # the bright band sat at high z; with the slice axis LAST it must show up in
    # the later entries of axis -1
    early = v[0, 0, :, :, :20].mean()
    late = v[0, 0, :, :, -20:].mean()
    assert late > early, (early, late, "slice axis is not last")

    # and it must NOT look like M3D's convention (slice axis first)
    early0 = v[0, 0, :20].mean()
    late0 = v[0, 0, -20:].mean()
    assert abs(late0 - early0) < abs(late - early), \
        "contrast should live on the last axis, not the first"

    const = np.full((20, 20, 20), 5.0, np.float32)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f2:
        nib.save(nib.Nifti1Image(const, np.eye(4)), f2.name)
        c = preprocess_volume(f2.name)
    assert np.isfinite(c).all() and c.max() == 0.0

    print(f"selftest OK — preprocess gives {v.shape} float32 in [0,1] with the "
          f"slice axis LAST (late {late:.3f} > early {early:.3f}), matching this "
          f"model's (H,W,T) rearrange and NOT M3D's (D,H,W); constant volume "
          f"handled without NaN")


if __name__ == "__main__":
    selftest()
