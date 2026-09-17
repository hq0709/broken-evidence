"""API helpers shared by the 2D runners: environment loading and an OpenAI client wrapper.

Keys are read from the environment, or from a `.env` file at the repository root
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_BASE_URL`, ...).
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Populate os.environ from KEY=VALUE lines without overriding variables already set."""
    candidates = [Path(path)] if path else [Path.cwd() / ".env",
                                            Path(__file__).resolve().parents[2] / ".env",
                                            Path(__file__).resolve().parents[1] / ".env"]
    for p in candidates:
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        break


class GPT4oInference:
    """Thin wrapper exposing an OpenAI-compatible client as `.client`."""

    def __init__(self, model: str = "gpt-4o", base_url: str | None = None, api_key: str | None = None):
        _load_dotenv()
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                             base_url=base_url or os.environ.get("OPENAI_BASE_URL"))

    def ask(self, prompt: str, max_tokens: int = 256) -> str:
        r = self.client.chat.completions.create(model=self.model, temperature=0, max_tokens=max_tokens,
                                                messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""
