import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from chunker import chunk_text


def test_chunk_overlap():
    text = " ".join(str(i) for i in range(200))
    chunks = chunk_text(text, chunk_size=50, overlap=0.2)
    assert chunks[0][1] - chunks[0][0] == 50
    assert chunks[1][0] == 40
