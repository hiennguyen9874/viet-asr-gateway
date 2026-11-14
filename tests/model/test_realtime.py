"""R3a realtime gate: `/v1/realtime` on the real model on CPU over a real Uvicorn server (websockets-sansio).

FLEURS fixtures (16 kHz) are resampled to 24 kHz PCM16 (soxr HQ, quantized) and sent in 20 ms packets, as a live
client would. (a) Each transcript equals REST on the exact committed 16 kHz waveform (same split/run_chunks/Joiner);
(b) WER vs the reference is at most REST-on-the-original + 2 points (soxr + PCM16, so no text parity with the
original); (c) items committed back to back, short turns included, complete in commit order with the same text as
alone; turns at or below the 100 ms floor complete with "" without a model call, turns just above it call the
model once; (d) concurrent sessions with different audio do not mix; (e) a turn longer than
GATEWAY_MAX_CHUNK_SECONDS yields one delta per chunk in one item. When every gate test passes, its metadata-only
report (no audio, no transcripts) is written to docs/baseline/r3a_gate.json.

Needs the staged model and ffmpeg: uv run pytest tests/model/test_realtime.py
"""

import json
import re
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import soxr

from app.chunking import MIN_AUDIO_SAMPLES, split
from app.config import SAMPLE_RATE
from app.engine import SherpaEngine
from app.realtime import PCM_RATE, TurnBuffer
from scripts.scoring import wer
from tests.fake_engine import config
from tests.live_server import serve, wav_file, WS
from tests.model.conftest import FIXTURES, MODELS, ROOT

MODEL = "viet-asr"
PACKET = PCM_RATE // 50 * 2  # 20 ms of PCM16
REPORT = ROOT / "docs/baseline/r3a_gate.json"
GATE_SECTIONS = {"turns", "fifo", "concurrent", "long"}
FLOOR_PCM = MIN_AUDIO_SAMPLES * PCM_RATE // SAMPLE_RATE  # 2,400 input samples = 100 ms
COMPLETED = "conversation.item.input_audio_transcription.completed"
DELTA = "conversation.item.input_audio_transcription.delta"


@pytest.fixture(scope="module")
def srv():
    engine = SherpaEngine(MODELS, "fp32", "modified_beam_search", 4, 4, warmup_seconds=config().max_chunk_seconds)
    with serve(engine, ready_timeout=300, http_timeout=300, models_dir=MODELS) as s:
        yield s


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
                  "max_chunk_seconds": config().max_chunk_seconds, "packet_ms": 20, **sections}
        REPORT.write_text(json.dumps(report, indent=2) + "\n")


def load(name: str) -> np.ndarray:
    return sf.read(FIXTURES / name, dtype="float32")[0]


def to_pcm24(wav16: np.ndarray) -> bytes:
    up = soxr.resample(wav16, SAMPLE_RATE, PCM_RATE, quality="HQ")
    return np.round(np.clip(up, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()


def committed(pcm: bytes) -> np.ndarray:
    """The 16 kHz waveform the server transcribes for this turn."""
    buf = TurnBuffer(len(pcm))
    buf.append(pcm)
    return buf.commit()[0]


def chunks(pcm: bytes) -> int:
    return len(split(committed(pcm), config().max_chunk_samples))


def speech_cut(pcm_samples: int) -> bytes:
    """`pcm_samples` of 24 kHz PCM16 speech cut mid-utterance (2 s into vi_a)."""
    return to_pcm24(load("vi_a.wav"))[2 * 2 * PCM_RATE:][:2 * pcm_samples]


def rest(srv, path: Path) -> str:
    r = srv.client.post("/v1/audio/transcriptions", files={"file": (path.name, path.read_bytes())},
                        data={"model": MODEL})
    assert r.status_code == 200, r.text
    return r.json()["text"]


def infer_calls(srv) -> int:
    """Model calls so far (warm-up excluded)."""
    m = re.search(r"^gateway_infer_seconds_count (\d+)$", srv.client.get("/metrics").text, re.M)
    return int(m.group(1)) if m else 0


def realtime(srv, turns: list[bytes]) -> list[tuple[list[str], dict]]:
    """One session, one turn at a time: each commit waits for its terminal event."""
    ws = WS(srv, timeout=120)
    try:
        out = []
        for pcm in turns:
            ws.append(pcm, packet=PACKET)
            ws.send("input_audio_buffer.commit")
            _, deltas, done = ws.turn()
            assert done["type"] == COMPLETED, done
            assert done["transcript"] == "".join(deltas)
            assert done["usage"] == {"type": "duration", "seconds": len(pcm) / 2 / PCM_RATE}
            out.append((deltas, done))
        return out
    finally:
        ws.close()


def test_turns_match_rest_on_the_committed_waveform_and_wer(srv, tmp_path, gate):
    refs = json.loads((FIXTURES / "manifest.json").read_text())["files"]
    names = ("vi_a.wav", "vi_b.wav", "silence_2s.wav", "short_200ms.wav")
    pcms = [to_pcm24(load(n)) for n in names]
    results = realtime(srv, pcms)  # one socket, several commits
    rows = {}
    for name, pcm, (deltas, done) in zip(names, pcms, results, strict=True):
        text = done["transcript"]
        assert text == rest(srv, wav_file(tmp_path / f"c_{name}", committed(pcm))), name  # (a)
        assert len(deltas) == chunks(pcm)
        row = {"audio_s": done["usage"]["seconds"], "chunks": len(deltas), "empty": not text}
        if ref := refs[name]["reference"]:
            score, base = wer(ref, text.lower()), wer(ref, rest(srv, FIXTURES / name).lower())
            assert score <= base + 0.02, (name, score, base)  # (b)
            row |= {"wer": round(score, 4), "rest_wer": round(base, 4)}
        rows[name] = row
    gate["turns"] = rows


def test_items_complete_in_commit_order_and_short_turns_follow_the_floor(srv, tmp_path, gate):
    # Four items (GATEWAY_MAX_PENDING_ITEMS) committed without waiting: speech, a turn at the floor, speech, a
    # turn just above it. The floor turn must not reach the model; every other chunk is one model call.
    cases = [("vi_a", to_pcm24(load("vi_a.wav"))), ("at_floor", speech_cut(FLOOR_PCM)),
             ("vi_b", to_pcm24(load("vi_b.wav"))), ("above_floor", speech_cut(FLOOR_PCM + 120))]
    assert len(committed(cases[1][1])) <= MIN_AUDIO_SAMPLES < len(committed(cases[3][1]))
    expected = {n: rest(srv, wav_file(tmp_path / f"{n}.wav", committed(p))) for n, p in cases}
    calls = infer_calls(srv)
    ws = WS(srv, timeout=120)
    try:
        for _, pcm in cases:
            ws.append(pcm, packet=PACKET)
            ws.send("input_audio_buffer.commit")
        events = []
        while sum(e["type"] == COMPLETED for e in events) < len(cases):
            events.append(ws.recv())
    finally:
        ws.close()
    ids = [e["item_id"] for e in events if e["type"] == "input_audio_buffer.committed"]
    previous = [e["previous_item_id"] for e in events if e["type"] == "input_audio_buffer.committed"]
    assert len(ids) == len(cases) and previous == [None, *ids[:-1]]
    # Per item: deltas then completed, items one after the other in commit order.
    results = [e for e in events if e["type"] in (DELTA, COMPLETED)]
    order = [ids.index(e["item_id"]) for e in results]
    assert order == sorted(order)
    done = [e for e in results if e["type"] == COMPLETED]
    assert [e["item_id"] for e in done] == ids
    for (name, pcm), e in zip(cases, done, strict=True):
        deltas = [d["delta"] for d in results if d["type"] == DELTA and d["item_id"] == e["item_id"]]
        assert e["transcript"] == "".join(deltas) == expected[name], name
        assert e["usage"] == {"type": "duration", "seconds": len(pcm) / 2 / PCM_RATE}
    assert done[1]["transcript"] == ""
    model_calls = infer_calls(srv) - calls
    assert model_calls == sum(chunks(p) for _, p in cases)
    assert chunks(cases[1][1]) == 0 and chunks(cases[3][1]) == 1
    gate["fifo"] = {"items": [{"case": n, "audio_s": len(p) / 2 / PCM_RATE, "committed_samples": len(committed(p)),
                               "chunks": chunks(p)} for n, p in cases],
                    "floor_samples": MIN_AUDIO_SAMPLES, "model_calls": model_calls,
                    "floor_turn_empty": done[1]["transcript"] == ""}


def test_concurrent_sessions_do_not_mix(srv, gate):
    pcms = {n: to_pcm24(load(n)) for n in ("vi_a.wav", "vi_b.wav")}
    expected = {n: realtime(srv, [p])[0][1]["transcript"] for n, p in pcms.items()}
    assert expected["vi_a.wav"] != expected["vi_b.wav"]
    results: dict[tuple[str, int], list[str]] = {}

    def run(name: str, i: int):
        results[(name, i)] = [d["transcript"] for _, d in realtime(srv, [pcms[name]] * 2)]

    threads = [threading.Thread(target=run, args=(n, i)) for i in range(2) for n in pcms]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    assert results == {(n, i): [expected[n]] * 2 for i in range(2) for n in pcms}  # (d)
    gate["concurrent"] = {"sessions": len(threads), "items_per_session": 2}


def test_long_turn_is_one_item_with_one_delta_per_chunk(srv, tmp_path, gate):
    a, b = load("vi_a.wav"), load("vi_b.wav")
    gap = np.zeros(SAMPLE_RATE, np.float32)
    pcm = to_pcm24(np.concatenate([a, gap, b, gap, a, gap, b]))  # 37.7 s > GATEWAY_MAX_CHUNK_SECONDS (20)
    [(deltas, done)] = realtime(srv, [pcm])
    assert len(deltas) == chunks(pcm) >= 2 and all(deltas)  # (e)
    assert done["transcript"] == rest(srv, wav_file(tmp_path / "long.wav", committed(pcm)))
    gate["long"] = {"audio_s": done["usage"]["seconds"], "chunks": len(deltas)}
