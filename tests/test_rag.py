import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

from build_index import build_index
from rag_utils import Retriever


def setup_module(module):
    index_file, _ = build_index()
    assert index_file.exists()


def test_retrieve_sleep_topic():
    retriever = Retriever()
    results = retriever.retrieve("sleep", k=1)
    assert results[0]["episode_id"] == "episode1"
    assert results[0]["score"] <= 1.0
