"""Minimal Flask web application exposing the RAG pipeline."""
from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from answer import RAGPipeline
from local_llm import LocalLLM
from rag_utils import Retriever

app = Flask(__name__)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(Retriever(), LocalLLM("models/llama.gguf"))
    return _pipeline


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    result = get_pipeline().answer(question)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11434)
