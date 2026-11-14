"""Timestamps from token emission times (app/alignment.py, docs/api-contract.vi.md §4), CPU only.

Failure modes covered: words get another word's time; times are not offset by the chunk start or leak outside the
chunk; the last word of a chunk is not capped or runs past the chunk; times go backwards within or across chunks;
tokens that do not spell the text are accepted; empty-text tokens create words or shift char ranges; cues straddle
pauses or exceed their limits; cue/word texts do not rebuild the transcript.
"""

import random

import pytest

from app.alignment import (
    align,
    AlignmentError,
    LAST_WORD_SAMPLES,
    MAX_CUE_CHARS,
    MAX_CUE_SAMPLES,
    PAUSE_SAMPLES,
    Segment,
    segments,
    Word,
    words,
)
from app.chunking import Span
from app.engine import ChunkResult
from app.transcription import Joiner
from tests.fake_engine import fake_result

S = "\u2581"


def result(text, tokens, times):
    return ChunkResult(text, tuple(tokens), tuple(times))


# -- words ----------------------------------------------------------------------------------------------------------

def test_word_runs_from_its_first_token_to_the_next_word():
    r = result("các sản phẩm", [S + "các", S + "sản", S + "ph", "ẩm"], [1000, 4000, 7000, 8000])
    got = words(Span(16000, 16000 + 20000), r)
    assert got == [Word("các", 17000, 20000), Word("sản", 20000, 23000), Word("phẩm", 23000, 23000 + LAST_WORD_SAMPLES)]


def test_last_word_is_capped_and_clipped_to_the_chunk():
    assert words(Span(0, 16000), result("a", [S + "a"], [100])) == [Word("a", 100, 100 + LAST_WORD_SAMPLES)]
    near_end = 16000 - 10
    assert words(Span(5, 5 + 16000), result("a", [S + "a"], [near_end])) == [Word("a", 5 + near_end, 5 + 16000)]


def test_tokens_without_text_make_no_words():
    # An empty piece emits nothing; a lone "▁" is the first token of the next word, so it sets that word's start.
    r = result("các a", ["", S + "các", S, "a"], [0, 100, 300, 500])
    assert words(Span(0, 20000), r) == [Word("các", 100, 300), Word("a", 300, 300 + LAST_WORD_SAMPLES)]
    r = result("a", [S, "a"], [40, 200])
    assert words(Span(0, 20000), r) == [Word("a", 40, 40 + LAST_WORD_SAMPLES)]
    # A token without "▁" continues the previous word.
    r = result("các aa", [S + "các", S + "a", "a"], [100, 500, 900])
    assert words(Span(0, 20000), r) == [Word("các", 100, 500), Word("aa", 500, 500 + LAST_WORD_SAMPLES)]


def test_equal_times_inside_a_word_are_fine():
    assert words(Span(0, 20000), result("aa", [S + "a", "a"], [10, 10])) == [Word("aa", 10, 10 + LAST_WORD_SAMPLES)]


def test_empty_text_has_no_words():
    assert words(Span(0, 3200), result("", [], [])) == []
    assert words(Span(0, 3200), result("", [S], [2])) == []


@pytest.mark.parametrize("r", [
    result("các", [S + "sản"], [1]),  # tokens spell another text
    result("các sản", [S + "các"], [1]),
    result("cácsản", [S + "các", S + "sản"], [1, 2]),
    result("các  sản", [S + "các", S, S + "sản"], [1, 2, 3]),  # double space: words cannot rebuild the text
    result("a", [S + "a"], [3200]),  # time at the chunk end
    result("a", [S + "a"], [-1]),
    result("a b", [S + "a", S + "b"], [300, 100]),  # times go backwards
    result("a b", [S + "a", S + "b"], [100, 100]),  # a word without duration
    result("a", [S + "a"], [1, 2]),  # tokens and times differ in length
    result("a", [S + "a", S + "b"], [1]),
    result("ab c", [S + "a", "b" + S + "c"], [100, 200]),  # marker inside a token
    result("a b", [S + "a b"], [100]),  # literal whitespace is not a word marker
    result("a b", [S + "a", " ", "b"], [100, 200, 300]),
    result("a\tb", [S + "a\tb"], [100]),
    result("a\nb", [S + "a\nb"], [100]),
])
def test_inconsistent_results_raise(r):
    with pytest.raises(AlignmentError):
        words(Span(0, 3200), r)


def test_random_fake_results_round_trip():
    rnd = random.Random(1)
    letters = "aăâbcdđeêghiklmnoôơpqrstuưvxyàáảãạ0123456789-"
    for _ in range(300):
        text = " ".join("".join(rnd.choice(letters) for _ in range(rnd.randint(1, 6)))
                        for _ in range(rnd.randint(0, 12)))
        n = rnd.randint(161, 16000 * 5)
        start = rnd.randint(0, 10 ** 6)
        got = words(Span(start, start + n), fake_result(text, n))
        assert " ".join(w.text for w in got) == text
        assert all(start <= w.start < w.end <= start + n for w in got)
        assert all(a.end <= b.start for a, b in zip(got, got[1:]))
        assert all(w.end - w.start <= LAST_WORD_SAMPLES for w in got[-1:])


def test_align_concatenates_chunks_in_order():
    spans = [Span(0, 48000), Span(48000, 52000), Span(52000, 100000)]
    texts = ["một hai ba", "", "bốn năm"]
    chunks = [(s, fake_result(t, s.end - s.start)) for s, t in zip(spans, texts)]
    got = align(chunks)
    j = Joiner()
    for t in texts:
        j.add(t)
    assert " ".join(w.text for w in got) == j.text == "một hai ba bốn năm"
    assert [w.text for w in got[3:]] == ["bốn", "năm"] and got[3].start >= 52000
    assert all(a.end <= b.start for a, b in zip(got, got[1:]))


def test_align_rejects_overlapping_chunks():
    chunks = [(Span(0, 32000), fake_result("a", 32000)), (Span(16000, 48000), fake_result("b", 32000))]
    with pytest.raises(AlignmentError):
        align(chunks)


# -- segments -------------------------------------------------------------------------------------------------------

def w(text, start_s, end_s):
    return Word(text, round(start_s * 16000), round(end_s * 16000))


def test_segments_split_at_pauses():
    pause = PAUSE_SAMPLES / 16000
    ws = [w("a", 0, 0.2), w("b", 0.3, 0.5), w("c", 0.5 + pause, 1.5 + pause),  # gap == PAUSE: split
          w("d", 1.5 + pause + pause - 0.01, 3 + pause * 2)]  # gap just below PAUSE: same cue
    assert segments(ws) == [Segment("a b", 0, 8000), Segment("c d", ws[2].start, ws[3].end)]


def test_emission_gap_within_a_chunk_does_not_imply_silence():
    ws = words(Span(0, 80000), result("a b", [S + "a", S + "b"], [0, 48000]))
    assert ws[0].end == ws[1].start == 48000
    assert segments(ws) == [Segment("a b", 0, 48000 + LAST_WORD_SAMPLES)]


@pytest.mark.parametrize("gap", [PAUSE_SAMPLES - 1, PAUSE_SAMPLES])
def test_aligned_gaps_across_chunks_split_cues_at_threshold(gap):
    next_start = LAST_WORD_SAMPLES + gap
    ws = align([
        (Span(0, next_start), result("a", [S + "a"], [0])),
        (Span(next_start, next_start + LAST_WORD_SAMPLES), result("b", [S + "b"], [0])),
    ])
    expected = ([Segment("a", 0, LAST_WORD_SAMPLES), Segment("b", next_start, next_start + LAST_WORD_SAMPLES)]
                if gap == PAUSE_SAMPLES else [Segment("a b", 0, next_start + LAST_WORD_SAMPLES)])
    assert segments(ws) == expected


def test_segments_respect_max_duration():
    # Words every 1 s, no pauses: a word that would end past MAX_CUE_SAMPLES after the cue start opens a new cue.
    ws = [w(str(i), i, i + 0.9) for i in range(20)]
    got = segments(ws)
    assert all(s.end - s.start <= MAX_CUE_SAMPLES for s in got)
    assert len(got) > 1 and got[0].text == " ".join(str(i) for i in range(7))  # 0..6 ends at 6.9 s; 7 ends at 7.9 s
    assert " ".join(s.text for s in got) == " ".join(x.text for x in ws)


def test_segments_respect_max_chars():
    ws = [w("x" * 9, i * 0.3, i * 0.3 + 0.2) for i in range(20)]
    got = segments(ws)
    assert all(len(s.text) <= MAX_CUE_CHARS for s in got)
    assert got[0].text == " ".join(["x" * 9] * 8)  # 8 words = 79 chars; a 9th would make 89


def test_oversized_word_is_its_own_cue():
    ws = [w("a", 0, 0.1), w("y" * (MAX_CUE_CHARS + 5), 0.2, 9.0), w("b", 9.1, 9.2)]
    assert [s.text for s in segments(ws)] == ["a", "y" * (MAX_CUE_CHARS + 5), "b"]


def test_segments_of_nothing():
    assert segments([]) == []


def spaced(text, step=0.2):
    """Words of `text` back to back (no gap), `step` seconds each."""
    return [w(t, i * step, (i + 1) * step) for i, t in enumerate(text.split(" "))]


def cue_texts(text, step=0.2):
    return [s.text for s in segments(spaced(text, step))]


def test_sentence_end_closes_the_cue_without_a_pause():
    assert cue_texts("Dạ. Em chào anh, hôm nay thế nào?") == ["Dạ.", "Em chào anh, hôm nay thế nào?"]
    assert cue_texts("Một hai. Ba bốn.") == ["Một hai.", "Ba bốn."]


def test_one_word_sentence_is_a_cue():
    assert segments(spaced("Dạ.")) == [Segment("Dạ.", 0, 3200)]
    assert cue_texts("a Dạ. b") == ["a Dạ.", "b"]


@pytest.mark.parametrize("mark", [",", ";", ":"])
def test_clause_punctuation_does_not_split(mark):
    assert cue_texts(f"Ngân sách{mark} 50 triệu") == [f"Ngân sách{mark} 50 triệu"]


@pytest.mark.parametrize("mark", [".", "?", "!", "\u2026"])
def test_every_sentence_end_splits(mark):
    assert cue_texts(f"Thế à{mark} Vâng") == [f"Thế à{mark}", "Vâng"]


def test_long_sentence_still_splits_by_length_before_its_end():
    text = " ".join(["x" * 9] * 10) + "."
    got = cue_texts(text, step=0.3)
    assert got == [" ".join(["x" * 9] * 8), " ".join(["x" * 9] * 2) + "."]


def test_pause_still_splits_inside_a_sentence():
    pause = PAUSE_SAMPLES / 16000
    ws = [w("Em", 0, 0.2), w("chào", 0.2, 0.4), w("anh.", 0.4 + pause, 0.6 + pause), w("Vâng", 0.6 + pause, 0.8 + pause)]
    assert [s.text for s in segments(ws)] == ["Em chào", "anh.", "Vâng"]
