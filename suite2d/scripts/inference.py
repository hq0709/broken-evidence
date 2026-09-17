"""Local inference for LLaVA-Med v1.5 (Mistral-7B), used by bench_run_llava_med.py.

Requires the official LLaVA-Med package in its own environment:

    pip install git+https://github.com/microsoft/LLaVA-Med.git

Decoding is greedy, matching the rest of the audit.
"""
from __future__ import annotations

import os

MODEL_PATHS = {"llava-med": os.environ.get("LLAVA_MED_PATH", "microsoft/llava-med-v1.5-mistral-7b")}


class MedVLMInference:
    def __init__(self, model_key: str = "llava-med", conv_mode: str = "mistral_instruct"):
        import torch
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        path = MODEL_PATHS[model_key]
        self.torch = torch
        self.conv_mode = conv_mode
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            path, None, get_model_name_from_path(path))
        self.model.eval()

    def ask(self, image_path: str | None, prompt: str, max_new_tokens: int = 128) -> str:
        from PIL import Image
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        conv = conv_templates[self.conv_mode].copy()
        text = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}" if image_path else prompt
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt").unsqueeze(0).to(self.model.device)
        kwargs = {}
        if image_path:
            image = Image.open(image_path).convert("RGB")
            kwargs["images"] = process_images([image], self.image_processor, self.model.config).to(
                self.model.device, dtype=self.torch.float16)
        with self.torch.inference_mode():
            out = self.model.generate(ids, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True, **kwargs)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
