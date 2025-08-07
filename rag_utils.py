"""Utilities for retrieval and prompt construction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import faiss

from embed import Embedder

INDEX_DIR = Path("vector_db")
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "meta.json"


class Retriever:
    """Perform vector similarity search over transcript chunks."""

    def __init__(self, index_file: Path = INDEX_FILE, meta_file: Path = META_FILE):
        self.index = faiss.read_index(str(index_file))
        self.meta: List[Dict[str, object]] = json.loads(meta_file.read_text(encoding="utf-8"))
        self.embedder = Embedder()

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        vec = self.embedder.embed([query])
        scores, ids = self.index.search(vec, k)
        results: List[Dict[str, object]] = []
        for score, idx in zip(scores[0], ids[0]):
            meta = dict(self.meta[idx])
            meta["score"] = float(score)
            results.append(meta)
        return results


def format_context(chunks: List[Dict[str, object]]) -> str:
    """Format retrieved chunks into a context block."""
    return "\n\n".join(
        f"[{c['episode_id']} chunk {c['chunk_id']}] {c['text']}" for c in chunks
    )


__all__ = ["Retriever", "format_context"]
