"""RAG orchestration combining retrieval and local LLM generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from local_llm import LocalLLM
from rag_utils import Retriever, format_context

SYSTEM_PROMPT = "You are a helpful assistant answering questions about the Huberman Lab podcast."


def _compose_prompt(question: str, context: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{context}\n\nUser: {question}\nAssistant:"


@dataclass
class RAGPipeline:
    retriever: Retriever
    llm: LocalLLM
    k: int = 5

    def answer(self, question: str) -> Dict[str, object]:
        chunks = self.retriever.retrieve(question, self.k)
        context = format_context(chunks)
        prompt = _compose_prompt(question, context)
        response = self.llm.generate(prompt)
        return {"answer": response, "sources": chunks}


__all__ = ["RAGPipeline"]
