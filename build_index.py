"""Build a FAISS index from transcript embeddings."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from embed import Embedder
from ingest import ingest_transcripts

INDEX_DIR = Path("vector_db")
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "meta.json"


def build_index() -> tuple[Path, Path]:
    """Ingest transcripts, embed them and persist a FAISS index."""
    INDEX_DIR.mkdir(exist_ok=True)
    records = ingest_transcripts()
    texts = [r["text"] for r in records]
    embedder = Embedder()
    vectors = embedder.embed(texts)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))
    with META_FILE.open("w", encoding="utf-8") as fh:
        json.dump(records, fh)
    return INDEX_FILE, META_FILE


if __name__ == "__main__":
    build_index()
