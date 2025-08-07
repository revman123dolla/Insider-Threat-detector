"""Utilities for splitting transcripts into overlapping token chunks."""
from __future__ import annotations

from typing import List, Tuple


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return text.split()


def chunk_text(text: str, chunk_size: int = 700, overlap: float = 0.2) -> List[Tuple[int, int, str]]:
    """Split ``text`` into chunks of ``chunk_size`` tokens with ``overlap``.

    Returns a list of tuples ``(start_token, end_token, chunk_text)``.
    """
    tokens = _tokenize(text)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be between 0 and 1")
    step = max(1, int(chunk_size * (1 - overlap)))
    chunks: List[Tuple[int, int, str]] = []
    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break
        chunks.append((start, min(end, len(tokens)), " ".join(chunk_tokens)))
        if end >= len(tokens):
            break
    return chunks


__all__ = ["chunk_text"]
