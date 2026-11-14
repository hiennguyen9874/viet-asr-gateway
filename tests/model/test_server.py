"""Seam 2: the real app on CPU over a real Uvicorn server: json, text and SSE transcription of the committed fixtures.

Transcripts through the HTTP API equal the reference script's output; every format carries the same text; concurrent
requests do not mix; silence -> ""; clips at and just above the short-audio floors never fail. Needs the staged model
and ffmpeg.

R2a cue gate (docs/api-contract.vi.md §4, docs/baseline.vi.md §6): verbose_json/srt/vtt agree with json on every
fixture; cues are ordered, inside the audio and within the cue limits; no cue straddles the silence where a long file
is cut into chunks and the cues next to it end/start close to it; cue times follow a shift of the audio; the long
fixture keeps its WER bound in every format. When every gate test passes, its metadata-only report is written to
docs/baseline/r2a_gate.json.
"""

import json
import subprocess
import threading
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.alignment import MAX_CUE_CHARS, MAX_CUE_SAMPLES
from app.chunking import MIN_AUDIO_SAMPLES
from app.config import SAMPLE_RATE
from app.engine import MIN_FORWARD_SAMPLES, SherpaEngine
from scripts.scoring import wer
from tests.fake_engine import config
from tests.live_server import parse_cues, serve, wav_file
from tests.model.conftest import FIXTURES, MODELS, reference_text, ROOT

URL = "/v1/audio/transcriptions"
MODEL = "viet-asr"
LONG = ROOT / "fixtures-large/vi_long.wav"
REPORT = ROOT / "docs/baseline/r2a_gate.json"
GATE_SECTIONS = {"fixtures", "chunk_cut", "shift", "long"}
STEP_S = 0.04  # token emission times come in 40 ms steps (alignment probe a)
# Alignment probe c: the last emission before a known silence is <= 1.16 s (p95) before it; the first word after
# one is stamped up to ~1 s early, inside the silence.
EDGE_TOLERANCE_S = 1.5
SHIFTS_S = (0.25, 0.5, 0.75)  # small enough that the first cue stays under the duration limit


@pytest.fixture(scope="module")
def srv():
    engine = SherpaEngine(MODELS, "fp32", "modified_beam_search", 4, 4, warmup_seconds=config().max_chunk_seconds)
    with serve(engine, ready_timeout=300, http_timeout=300, models_dir=MODELS) as s:
        yield s


def post(srv, path: Path, **fields):
    r = srv.client.post(URL, files={"file": (path.name, path.read_bytes())}, data={"model": MODEL, **fields})
    assert r.status_code == 200, r.text
    return r


def stream(srv, path: Path) -> tuple[list[str], str]:
    events = [json.loads(b.removeprefix("data: ")) for b in post(srv, path, stream="true").text.split("\n\n")
              if b.strip()]
    assert events[-1]["type"] == "transcript.text.done", events[-1]
    return [e["delta"] for e in events[:-1]], events[-1]["text"]


@pytest.fixture(scope="module")
def gate():
    sections: dict = {}
    yield sections
    if set(sections) == GATE_SECTIONS:  # each section is stored only at the end of a passing test
        import sherpa_onnx

        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        report = {"commit": commit + ("-dirty" if dirty else ""), "sherpa_onnx": sherpa_onnx.__version__,
                  "settings": "fp32, modified_beam_search, beam 4, 4 threads",
                  "max_chunk_seconds": config().max_chunk_seconds, **sections}
        REPORT.write_text(json.dumps(report, indent=2) + "\n")


def timed(srv, path: Path, fmt: str) -> list[tuple[float, float, str]]:
    return parse_cues(post(srv, path, response_format=fmt).text, fmt)


def cues_of(srv, path: Path, text: str, duration: float) -> list[tuple[float, float, str]]:
    """SRT and VTT cues of `path` (asserted identical): they rebuild `text`, are ordered, do not overlap, lie inside
    the audio and respect the cue limits (a single oversized word may exceed the character limit)."""
    srt, vtt = timed(srv, path, "srt"), timed(srv, path, "vtt")
    assert srt == vtt
    assert " ".join(c[2] for c in srt) == text
    assert all(0 <= s < e <= duration + 0.0005 for s, e, _ in srt)
    assert all(a[1] <= b[0] for a, b in pairwise(srt))
    assert all(e - s <= MAX_CUE_SAMPLES / SAMPLE_RATE + 0.0005 for s, e, _ in srt)
    assert all(len(t) <= MAX_CUE_CHARS or " " not in t for _, _, t in srt)
    return srt


def summary(cues: list[tuple[float, float, str]]) -> dict:
    """Metadata only: no cue text."""
    return {"cues": len(cues), "max_cue_s": round(max((e - s for s, e, _ in cues), default=0), 3),
            "max_cue_chars": max((len(t) for _, _, t in cues), default=0)}


def test_json_text_and_sse_equal_the_reference(srv):
    assert srv.client.get("/ready").status_code == 200
    for name in ("vi_a.wav", "vi_b.wav"):
        path = FIXTURES / name
        expected = reference_text(path, "fp32", "modified_beam_search")
        body = post(srv, path).json()
        assert body["text"] == expected and body["usage"]["seconds"] > 0
        assert post(srv, path, response_format="text").text == expected
        deltas, done = stream(srv, path)
        assert "".join(deltas) == done == expected


def test_silence_and_short_clip_match_the_reference(srv):
    # Digital silence is not forced to "": the model's own output is returned, as in the reference script.
    silence = FIXTURES / "silence_2s.wav"
    expected = reference_text(silence, "fp32", "modified_beam_search")
    assert post(srv, silence).json()["text"] == expected
    assert stream(srv, silence)[1] == expected
    assert isinstance(post(srv, FIXTURES / "short_200ms.wav").json()["text"], str)


@pytest.mark.parametrize("samples", [MIN_FORWARD_SAMPLES, MIN_AUDIO_SAMPLES, MIN_AUDIO_SAMPLES + 1, 2 * MIN_AUDIO_SAMPLES])
def test_clips_at_and_just_above_the_floors_succeed_in_every_format(srv, tmp_path, samples):
    # Speech cut mid-utterance: at or below the short-audio floor -> "" without the model, just above it -> the
    # model's own (possibly empty) output; never an error, timed formats included.
    x = sf.read(FIXTURES / "vi_a.wav", dtype="float32")[0][2 * SAMPLE_RATE:][:samples]
    path = tmp_path / "cut.wav"
    sf.write(path, x, SAMPLE_RATE, subtype="FLOAT")
    text = post(srv, path).json()["text"]
    assert samples > MIN_AUDIO_SAMPLES or text == ""
    assert post(srv, path, response_format="text").text == text
    assert stream(srv, path)[1] == text
    assert post(srv, path, response_format="verbose_json").json()["text"] == text
    for fmt in ("srt", "vtt"):
        post(srv, path, response_format=fmt)


def test_concurrent_requests_do_not_mix(srv):
    expected = {n: post(srv, FIXTURES / n).json()["text"] for n in ("vi_a.wav", "vi_b.wav")}
    results: dict[tuple[str, int], str] = {}

    def run(name: str, i: int):
        results[(name, i)] = post(srv, FIXTURES / name).json()["text"]

    threads = [threading.Thread(target=run, args=(n, i)) for i in range(3) for n in expected]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert results == {(n, i): expected[n] for i in range(3) for n in expected}


def test_timed_formats_agree_with_json_on_every_fixture(srv, gate):
    rows = {}
    for name in ("vi_a.wav", "vi_b.wav", "silence_2s.wav", "short_200ms.wav"):
        path = FIXTURES / name
        body = post(srv, path).json()
        duration = body["usage"]["seconds"]
        assert post(srv, path, response_format="verbose_json").json() == {
            "text": body["text"], "language": "vietnamese", "duration": duration, "usage": body["usage"]}
        cues = cues_of(srv, path, body["text"], duration)
        assert bool(cues) == bool(body["text"])
        rows[name] = {"audio_s": duration, **summary(cues)}
    gate["fixtures"] = rows


def test_no_cue_straddles_the_silence_at_a_chunk_cut(srv, tmp_path, gate):
    """vi_a | 1 s | vi_b | 1 s | vi_a | 1 s | vi_b (37.7 s): two chunks, cut inside the middle silence."""
    a, b = (sf.read(FIXTURES / n, dtype="float32")[0] for n in ("vi_a.wav", "vi_b.wav"))
    gap = np.zeros(SAMPLE_RATE, np.float32)
    parts = [a, gap, b, gap, a, gap, b]
    edges = np.cumsum([0] + [len(p) for p in parts]) / SAMPLE_RATE
    path = wav_file(tmp_path / "composite.wav", np.concatenate(parts))
    body = post(srv, path).json()
    duration = body["usage"]["seconds"]
    deltas, done = stream(srv, path)
    assert len(deltas) == 2 and done == body["text"]
    cues = cues_of(srv, path, body["text"], duration)
    g0, g1 = edges[3], edges[4]
    before = [c for c in cues if c[0] < g0]
    after = [c for c in cues if c[0] >= g0]
    assert before and after
    # Nothing inside the silence (times are whole ms); the cue edges are close to it.
    assert g0 - EDGE_TOLERANCE_S <= before[-1][1] <= g0 + 0.0005, before[-1][:2]
    assert g1 - EDGE_TOLERANCE_S <= after[0][0] <= g1 + EDGE_TOLERANCE_S, after[0][:2]
    # The second chunk starts inside the silence, so its first cue cannot start before the silence.
    assert after[0][0] >= g0
    gate["chunk_cut"] = {
        "audio_s": duration, "chunks": len(deltas), "silence_s": [round(g0, 3), round(g1, 3)],
        "last_cue_end_minus_silence_start_s": round(before[-1][1] - g0, 3),
        "first_cue_start_minus_silence_end_s": round(after[0][0] - g1, 3), **summary(cues)}


def edges(cues) -> dict[tuple[str, int], float]:
    """Cue edges keyed by the word index they sit at: ("start", i) where a cue starts with word i, ("end", i)
    where one ends with word i."""
    out, i = {}, 0
    for s, e, text in cues:
        out[("start", i)] = s
        i += len(text.split())
        out[("end", i - 1)] = e
    return out


def test_cue_times_follow_a_shift_of_the_audio(srv, tmp_path, gate):
    # Prepending silence moves every cue edge by the same amount (within one emission step), except the start of
    # the first word: the model stamps the first token at the chunk start (alignment probe b). That also lengthens
    # the first cue, which may then reach the duration limit and end earlier: only edges at the same word in both
    # cue lists are compared.
    a, b = (sf.read(FIXTURES / n, dtype="float32")[0] for n in ("vi_a.wav", "vi_b.wav"))
    x = np.concatenate([a, np.zeros(SAMPLE_RATE, np.float32), b])  # 18.3 s: one chunk, also when shifted
    base_path = wav_file(tmp_path / "base.wav", x)
    base_text = post(srv, base_path).json()["text"]
    base = cues_of(srv, base_path, base_text, len(x) / SAMPLE_RATE)
    worst, compared = 0.0, 0
    for shift in SHIFTS_S:
        y = np.concatenate([np.zeros(int(shift * SAMPLE_RATE), np.float32), x])
        path = wav_file(tmp_path / f"shift{shift}.wav", y)
        text = post(srv, path).json()["text"]
        assert text == base_text
        moved = cues_of(srv, path, text, len(y) / SAMPLE_RATE)
        assert moved[0][0] == 0.0
        old, new = edges(base), edges(moved)
        common = (old.keys() & new.keys()) - {("start", 0)}
        assert len(common) >= 2
        compared += len(common)
        worst = max(worst, *(abs(new[k] - old[k] - shift) for k in common))
    assert worst <= STEP_S + 0.0015, worst
    gate["shift"] = {"shifts_s": list(SHIFTS_S), "cues": len(base), "edges_compared": compared,
                     "max_edge_error_s": round(worst, 3)}


@pytest.mark.skipif(not LONG.exists(), reason="large fixture not fetched (scripts/fetch_fixtures.py)")
def test_long_m4a_keeps_its_wer_in_every_format(srv, tmp_path, gate):
    m4a = tmp_path / "long.m4a"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(LONG), "-c:a", "aac", str(m4a)],
                   check=True)
    reference = json.loads((FIXTURES / "manifest.json").read_text())["files"]["vi_long.wav"]["reference"]
    body = post(srv, m4a).json()
    deltas, done = stream(srv, m4a)
    assert done == body["text"] and len(deltas) > 1
    score = wer(reference, body["text"].lower())  # the model's vocabulary is upper case
    assert score <= 0.10
    cues = cues_of(srv, m4a, body["text"], body["usage"]["seconds"])
    gate["long"] = {"audio_s": body["usage"]["seconds"], "chunks": len(deltas), "wer": round(score, 4),
                    **summary(cues)}
