"""Light-weight deterministic embedding implementation."""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


class Embedder:
    """Produce fixed-size embeddings using SHA256 hashing.

    This avoids heavyweight model dependencies while providing deterministic
    vectors suitable for tests and small deployments.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        data = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        vec = np.tile(data, int(np.ceil(self.dim / len(data))))[: self.dim]
        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        return vec

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self._hash(t) for t in texts]).astype("float32")


__all__ = ["Embedder"]
