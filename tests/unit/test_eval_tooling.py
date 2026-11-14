"""Eval-set tooling: corpus WER counts, the synthetic non-speech clips and the decision rules of the probes."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.baseline_offline import choose_decoding
from scripts.fetch_fixtures import EVAL_MANIFEST, nonspeech, NONSPEECH_KINDS, NONSPEECH_SECONDS
from scripts.probe_limits import derive
from scripts.scoring import errors

ROOT = Path(__file__).resolve().parents[2]


def test_errors_count_edits_and_reference_words_after_normalization():
    assert errors("Một hai, ba.", "một bốn ba năm") == (2, 3)
    assert errors("", "Không.") == (1, 0)


def test_nonspeech_clips_are_deterministic_and_at_their_level():
    a, b = nonspeech(), nonspeech()
    assert set(a) == {f"{k}_{s}s.wav" for k in NONSPEECH_KINDS for s in NONSPEECH_SECONDS}
    for name, x in a.items():
        assert np.array_equal(x, b[name]) and x.dtype == np.float32
    assert not a["zeros_5s.wav"].any()
    for kind in ("white_-45dB", "pink_-30dB"):
        rms = np.sqrt(np.mean(a[f"{kind}_5s.wav"].astype(np.float64) ** 2))
        assert 20 * np.log10(rms) == pytest.approx(NONSPEECH_KINDS[kind][1], abs=0.01)


def test_eval_manifest_holds_hashes_and_counts_only():
    m = json.loads(EVAL_MANIFEST.read_text())
    assert sum(n.startswith("fleurs/") for n in m["files"]) == 200
    assert sum(n.startswith("nonspeech/") for n in m["files"]) == len(NONSPEECH_KINDS) * len(NONSPEECH_SECONDS)
    assert all(set(e) == {"sha256", "samples", "sample_rate"} for e in m["files"].values())
    assert "reference" not in json.dumps(m["files"])


def _eval(wer, rate):
    return {"eval": {"speech": {"wer": wer}, "nonspeech": {"text_rate": rate}}}


@pytest.mark.parametrize(("beam", "greedy", "expected"), [
    (0.170, 0.175, "greedy_search"),  # loses exactly 0.5 point
    (0.170, 0.1751, "modified_beam_search"),
    (0.170, 0.168, "greedy_search"),
])
def test_default_decoding_follows_the_half_point_rule(beam, greedy, expected):
    rows = {"modified_beam_search": _eval(beam, 0.5), "greedy_search": _eval(greedy, 0.0)}
    assert choose_decoding(rows)["decoding_method"] == expected


def test_max_chunk_is_the_longest_length_tying_the_best_wer_within_bounds():
    chunks = [{"seconds": s, "latency_max_s": lat, "peak_rss_mb": rss, "peak_rss_load_mb": 400}
              for s, lat, rss in ((10, 0.3, 400), (20, 0.6, 410), (25, 0.8, 600), (30, 1.2, 420))]
    wers = [{"seconds": s, "eval": {"wer": w}} for s, w in ((10, 0.170), (20, 0.175), (25, 0.160), (30, 0.160))]
    out = derive(chunks, wers)
    assert out["within_bounds"] == [10, 20]  # 25 s takes too much RAM, 30 s is too slow
    assert out["max_chunk_seconds"] == 20  # 0.5 point above the best: a tie
    wers[1]["eval"]["wer"] = 0.1751
    assert derive(chunks, wers)["max_chunk_seconds"] == 10
