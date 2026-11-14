"""Dependency-free scoring shared by the inference and server smoke gates."""

import unicodedata


def normalize(text: str) -> str:
    """Lower case without Unicode punctuation, applied to both sides: the model writes casing and punctuation, the
    FLEURS references do not. Digits vs spelled-out numbers are not reconciled (a known WER inflation)."""
    return "".join(c for c in text.lower() if not unicodedata.category(c).startswith("P"))


def errors(ref: str, hyp: str) -> tuple[int, int]:
    """(word edit distance, reference words) after `normalize`: sum both over a corpus for its WER."""
    r, h = normalize(ref).split(), normalize(hyp).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def wer(ref: str, hyp: str) -> float:
    e, n = errors(ref, hyp)
    return e / max(n, 1)
