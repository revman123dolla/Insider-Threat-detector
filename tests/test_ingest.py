import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from ingest import clean_text


def test_clean_text_removes_timestamps_and_ads():
    raw = "00:01:02 Hello\nAd: buy now\nworld"
    cleaned = clean_text(raw)
    assert "00:01:02" not in cleaned
    assert "Ad:" not in cleaned
    assert cleaned == "Hello world"
