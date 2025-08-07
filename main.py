"""CLI entry point for querying the RAG system."""
from __future__ import annotations

import argparse

from answer import RAGPipeline
from local_llm import LocalLLM
from rag_utils import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Huberman Lab transcripts")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument(
        "--model", default="models/llama.gguf", help="Path to GGUF model"
    )
    args = parser.parse_args()
    pipeline = RAGPipeline(Retriever(), LocalLLM(args.model))
    result = pipeline.answer(args.question)
    print(result["answer"])
    for src in result["sources"]:
        print(
            f"- {src['episode_id']} chunk {src['chunk_id']} (score {src['score']:.3f})"
        )


if __name__ == "__main__":
    main()
