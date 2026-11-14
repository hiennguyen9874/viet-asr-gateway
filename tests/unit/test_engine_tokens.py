"""Byte-fallback tokens (`<0xNN>`) are merged into the characters they spell, keeping the first byte's time."""

from app.alignment import EMISSION_SHIFT, words
from app.chunking import Span
from app.engine import ChunkResult, merge_byte_tokens


def spell(tokens) -> str:
    return "".join(tokens).replace("\u2581", " ").strip()


def test_tokens_without_bytes_are_unchanged():
    tokens, times = ("\u2581Các", "\u2581sản", ","), (0, 640, 1280)
    assert merge_byte_tokens(tokens, times) == (tokens, times)


def test_two_byte_character_inside_a_word():
    # Observed on vi_bytes.wav: "▁" "<0xC3>" "<0x9A>" "t" spells "Út" (U+00DA is not in the vocabulary).
    tokens = ("\u2581việc", "\u2581", "<0xC3>", "<0x9A>", "t", "\u2581trở")
    merged = merge_byte_tokens(tokens, (0, 640, 1280, 1920, 2560, 3200))
    assert merged == (("\u2581việc", "\u2581", "Ú", "t", "\u2581trở"), (0, 640, 1280, 2560, 3200))
    assert spell(merged[0]) == "việc Út trở"


def test_three_byte_character_at_the_end_and_separate_runs():
    tokens = ("<0xC3>", "<0x9A>", "\u2581Người", "\u2581", "<0xE1>", "<0xBA>", "<0xA2>")
    merged, times = merge_byte_tokens(tokens, (0, 1, 2, 3, 4, 5, 6))
    assert merged == ("Ú", "\u2581Người", "\u2581", "Ả")
    assert times == (0, 2, 3, 4)


def test_one_run_of_several_characters_is_one_token():
    merged, times = merge_byte_tokens(("\u2581", "<0xC3>", "<0x9A>", "<0xE1>", "<0xBA>", "<0xA2>"), (0, 1, 2, 3, 4, 5))
    assert merged == ("\u2581", "ÚẢ") and times == (0, 1)


def test_invalid_bytes_become_replacement_characters_like_the_recognizer_text():
    # Observed: a lone "<0xC3>" between words renders as U+FFFD in sherpa-onnx's text ("và � Áp").
    merged, times = merge_byte_tokens(("\u2581và", "\u2581", "<0xC3>", "\u2581Áp"), (0, 640, 1280, 1920))
    assert merged == ("\u2581và", "\u2581", "\ufffd", "\u2581Áp") and times == (0, 640, 1280, 1920)
    assert spell(merged) == "và \ufffd Áp"


def test_merged_tokens_align():
    tokens, times = merge_byte_tokens(("\u2581việc", "\u2581", "<0xC3>", "<0x9A>", "t", "."), (0, 640, 1280, 1920, 2560, 3200))
    result = ChunkResult("việc Út.", tokens, times)
    got = [(w.text, w.start - EMISSION_SHIFT) for w in words(Span(0, 16000), result)]
    assert got == [("việc", 0), ("Út.", 640)]
