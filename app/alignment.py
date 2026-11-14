"""Word and cue timestamps from the token emission times of each chunk (docs/api-contract.vi.md §4).

The transducer reports when it emitted each token, so those times are the alignment: no extra inference, no forced
alignment. Times are absolute 16 kHz sample offsets (ints) until serialization.

- A word is a maximal run of non-whitespace characters of the final text (a Vietnamese syllable, not a compound).
  Token pieces start a new word with "▁" and never contain it elsewhere, so each word is a run of tokens.
- A word starts when its first token (the one carrying its "▁", even a lone "▁") was emitted, moved later by
  `EMISSION_SHIFT`, and ends where the next word of the chunk starts; the last word of a chunk lasts
  `LAST_WORD_SAMPLES` (or until the chunk ends). Near a chunk end the shift shrinks (down to 0) so the last word
  keeps `LAST_WORD_SAMPLES` inside the chunk; one shift per chunk keeps every other word's duration.
- Any inconsistency (the tokens do not spell the text, times out of order or outside the chunk, a word with no
  duration) raises `AlignmentError`: timestamps are never guessed or spread evenly.
"""

import re
from collections.abc import Sequence
from itertools import pairwise
from typing import NamedTuple

from app.chunking import Span
from app.config import SAMPLE_RATE
from app.engine import ChunkResult

# Values from scripts/probe_alignment.py (docs/baseline/alignment_probe.json, docs/baseline.vi.md §6); the probe exits 1
# if they drift. Emission times come in steps of 640 samples (40 ms).
# Constant emission delay compensation in samples: the first token after a known silence is emitted a median 205 ms
# before speech resumes (quartiles -375 / -65 ms, same sign), so word starts move 200 ms later.
EMISSION_SHIFT = 3200
# Duration given to the last word of a chunk: nothing marks where it ends. Median time from the last emission before
# a known silence to that silence (305 ms), minus EMISSION_SHIFT, rounded up to a step.
LAST_WORD_SAMPLES = 1920
WORD_START = "\u2581"

# Cue (subtitle segment) limits.
PAUSE_SAMPLES = 10240  # an aligned gap of at least this ends the cue: p90 of word-start intervals (640 ms), step-rounded
MAX_CUE_SAMPLES = 7 * SAMPLE_RATE  # subtitle conventions: at most 7 s on screen,
MAX_CUE_CHARS = 80  # two 40-character lines
SENTENCE_END = (
    ".",
    "?",
    "!",
    "\u2026",
)  # a word ending in one of these ends its cue ("," ";" ":" do not)

_WORD = re.compile(r"\S+")


class Word(NamedTuple):
    text: str
    start: int  # absolute samples at 16 kHz, end exclusive
    end: int


class Segment(NamedTuple):
    text: str
    start: int
    end: int


class AlignmentError(Exception):
    """The tokens and the text disagree; the response must fail rather than carry made-up times."""


def words(span: Span, result: ChunkResult) -> list[Word]:
    """Words of one chunk with absolute times inside `span`."""
    n = span.end - span.start
    if len(result.tokens) != len(result.times):
        raise AlignmentError("tokens and times differ in length")
    # Character range and emission time of each token that emits text.
    chars: list[str] = []
    ranges: list[tuple[int, int, int]] = []  # (char start, char end, time)
    pos, prev = 0, 0
    for token, t in zip(result.tokens, result.times, strict=True):
        if not prev <= t < n:
            raise AlignmentError("token times out of order or out of range")
        prev = t
        if WORD_START in token[1:] or any(c.isspace() for c in token):
            raise AlignmentError("token crosses a word boundary")
        piece = token.replace(WORD_START, " ")
        if piece:
            chars.append(piece)
            ranges.append((pos, pos + len(piece), t))
            pos += len(piece)
    spelled = "".join(chars)
    if spelled.strip() != result.text:
        raise AlignmentError("tokens do not spell the text")

    found: list[tuple[str, int]] = []
    j = 0
    for m in _WORD.finditer(spelled):
        # The space before a word is its "▁", so the token holding it is the word's first token.
        first = max(0, m.start() - 1)
        while ranges[j][1] <= first:
            j += 1
        found.append((m.group(), ranges[j][2]))
    if found:
        shift = min(EMISSION_SHIFT, max(0, n - LAST_WORD_SAMPLES - found[-1][1]))
        found = [(text, max(0, t + shift)) for text, t in found]
    out: list[Word] = []
    for i, (text, start) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else min(n, start + LAST_WORD_SAMPLES)
        if not start < end <= n:
            raise AlignmentError("word without duration or outside the chunk")
        out.append(Word(text, span.start + start, span.start + end))
    if " ".join(w.text for w in out) != result.text:
        raise AlignmentError("words do not rebuild the text")
    return out


def align(chunks: Sequence[tuple[Span, ChunkResult]]) -> list[Word]:
    """Words of a whole transcript, chunks in audio order; their texts joined by one space give the transcript."""
    out: list[Word] = []
    prev_end = 0
    for span, result in chunks:
        if span.start < prev_end:
            raise AlignmentError("chunks overlap or are out of order")
        prev_end = span.end
        out.extend(words(span, result))
    if any(b.start < a.start for a, b in pairwise(out)):
        raise AlignmentError("word starts go backwards")
    return out


def segments(ws: Sequence[Word]) -> list[Segment]:
    """Group words into subtitle cues: a new cue after an aligned gap of `PAUSE_SAMPLES`, or when the next word
    would exceed `MAX_CUE_SAMPLES` or `MAX_CUE_CHARS` (a single oversized word is a cue of its own). A word whose
    text ends a sentence (`SENTENCE_END`) closes its cue, so "Dạ." alone is a cue.

    Alignment leaves gaps only across chunks: within a chunk each word ends at the next word's start.
    Emission times alone cannot identify silence, so large emission gaps do not imply cue boundaries.
    """
    out: list[Segment] = []
    cue: list[Word] = []
    for w in ws:
        if cue and (
            w.start - cue[-1].end >= PAUSE_SAMPLES
            or w.end - cue[0].start > MAX_CUE_SAMPLES
            or sum(len(x.text) + 1 for x in cue) + len(w.text) > MAX_CUE_CHARS
        ):
            out.append(_cue(cue))
            cue = []
        cue.append(w)
        if w.text.endswith(SENTENCE_END):
            out.append(_cue(cue))
            cue = []
    if cue:
        out.append(_cue(cue))
    return out


def _cue(ws: list[Word]) -> Segment:
    return Segment(" ".join(w.text for w in ws), ws[0].start, ws[-1].end)
