"""Wrapper around a locally hosted Llama model."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - fallback when dependency missing
    Llama = None  # type: ignore


class LocalLLM:
    """Thin wrapper over ``llama_cpp.Llama``."""

    def __init__(self, model_path: str, n_ctx: int = 2048):
        if Llama is None:
            raise RuntimeError("llama_cpp is not installed")
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.llm = Llama(model_path=str(path), n_ctx=n_ctx)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        result = self.llm(prompt, max_tokens=max_tokens, stop=["</s>"])
        return result["choices"][0]["text"].strip()


__all__ = ["LocalLLM"]
