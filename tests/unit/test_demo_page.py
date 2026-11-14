"""The listening page scores written-form text with the shared two-sided WER rules."""

import io
import json
from types import SimpleNamespace

import pytest

from scripts.demo_page import transcribe


@pytest.mark.parametrize("reference,hypothesis", [
    ("CHÀO, BẠN!", "Chào bạn."),
    ("Chào bạn.", "CHÀO, BẠN!"),
    ("Giá 50 đồng.", "giá năm mươi đồng"),
])
def test_demo_scores_both_sides_without_changing_display_text(tmp_path, reference, hypothesis):
    (tmp_path / "clip.wav").write_bytes(b"audio supplied by the caller")

    def response(data, **fields):
        assert data == b"audio supplied by the caller"
        if fields["response_format"] == "verbose_json":
            return io.BytesIO(json.dumps({"text": hypothesis, "duration": 1.0}).encode())
        return io.BytesIO(f"1\n00:00:00,200 --> 00:00:01,000\n{hypothesis}\n\n".encode())

    case = {"file": "clip.wav", "reference": reference}
    transcribe(SimpleNamespace(transcribe=response), case, tmp_path)

    assert case["reference"] == reference
    assert case["hypothesis"] == hypothesis
    # No digit-to-spelled-number normalization: one substitution and one insertion over three words.
    assert case["wer"] == (round(2 / 3, 4) if "50" in reference else 0.0)
