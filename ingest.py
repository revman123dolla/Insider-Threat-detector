"""Ingestion and normalisation of transcript files."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List

from chunker import chunk_text

TIMESTAMP_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
AD_RE = re.compile(r"(?im)^ad:.*$")


def clean_text(text: str) -> str:
    """Normalise whitespace, drop timestamps and ad sections."""
    text = unicodedata.normalize("NFKC", text)
    text = TIMESTAMP_RE.sub("", text)
    text = AD_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ingest_transcripts(transcript_dir: str = "transcripts") -> List[Dict[str, object]]:
    """Load, clean and chunk all transcripts in ``transcript_dir``."""
    records: List[Dict[str, object]] = []
    for path in sorted(Path(transcript_dir).glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        episode_id = path.stem
        chunks = chunk_text(cleaned)
        for idx, (start, end, chunk) in enumerate(chunks):
            records.append(
                {
                    "episode_id": episode_id,
                    "chunk_id": idx,
                    "start_token": start,
                    "end_token": end,
                    "text": chunk,
                }
            )
    return records


__all__ = ["ingest_transcripts", "clean_text"]
