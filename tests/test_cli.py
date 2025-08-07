import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import sys
from types import SimpleNamespace

import main


class DummyLLM:
    def __init__(self, model_path: str) -> None:
        pass

    def generate(self, prompt: str, max_tokens: int = 256) -> str:  # pragma: no cover - simple stub
        return "dummy"


def test_cli_invocation(monkeypatch, capsys):
    monkeypatch.setattr(main, "LocalLLM", DummyLLM)

    class DummyPipeline:
        def __init__(self, retriever, llm):
            pass

        def answer(self, question: str):
            return {"answer": "dummy", "sources": []}

    monkeypatch.setattr(main, "RAGPipeline", lambda r, l: DummyPipeline(r, l))
    monkeypatch.setattr(main, "Retriever", lambda: None)

    testargs = ["main.py", "--question", "hi?", "--model", "fake"]
    monkeypatch.setattr(sys, "argv", testargs)
    main.main()
    out = capsys.readouterr().out.strip()
    assert "dummy" in out
