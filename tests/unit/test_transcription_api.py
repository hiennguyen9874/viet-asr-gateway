"""POST /v1/audio/transcriptions and /v1/models end to end on CPU: real Uvicorn, real FFmpeg, fake engine.

Helpers and the audio convention are in tests/live_server.py.
"""

import asyncio
import io
import json
import os
import socket
import threading
import time
from pathlib import Path

import httpx2
import numpy as np
import openai
import pytest
import soundfile as sf

from app import transcription
from app.alignment import PAUSE_SAMPLES
from app.chunking import MIN_AUDIO_SAMPLES, split
from app.engine import MIN_FORWARD_SAMPLES
from app.main import create_app
from app.transcription import MODEL_ID
from tests.fake_engine import config, FakeEngine
from tests.live_server import (
    AUTH,
    expected_deltas,
    KEY,
    name,
    noise,
    parse_cues,
    serve,
    SR,
    wait_for,
    wav_file,
)

URL = "/v1/audio/transcriptions"


def form(**fields) -> dict:
    return {"model": MODEL_ID, **fields}


def post(srv, path: Path, **fields) -> httpx2.Response:
    return srv.client.post(URL, files={"file": (path.name, path.read_bytes())}, data=form(**fields))


def events(response) -> list[dict]:
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    raw = response.text
    assert raw.endswith("\n\n")
    out = []
    for block in raw.split("\n\n")[:-1]:
        assert block.startswith("data: ") and "\n" not in block
        out.append(json.loads(block[len("data: "):]))
    return out


def envelope(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-request-id"].startswith("req_")
    err = response.json()["error"]
    assert set(err) == {"message", "type", "param", "code"} and err["code"] == code
    return err


def idle(app) -> bool:
    """No lane, job, decode or spooled byte left behind."""
    audio, scheduler = app.state.audio, app.state.scheduler
    return (scheduler._lanes == 0 and scheduler._jobs == 0 and audio._decoding == 0 and audio._held == 0
            and audio._spooled == 0 and not any(audio.spool_dir.iterdir()))


def outcomes(app, outcome: str, response_format="json", stream="false") -> float:
    return app.state.metrics.value("transcriptions_total", response_format=response_format, stream=stream,
                                   outcome=outcome)


class Blocker:
    """text_fn that blocks call number `at` (1-based) until released, or fails it."""

    def __init__(self, at: int, fail: bool = False):
        self.at, self.fail, self.calls = at, fail, 0
        self.entered, self.release = threading.Event(), threading.Event()

    def __call__(self, w: np.ndarray) -> str:
        self.calls += 1
        if self.calls == self.at:
            if self.fail:
                raise RuntimeError("inference error: simulated")
            self.entered.set()
            self.release.wait(10)
        return name(w)


def raw_post(port: int, body: bytes, boundary: str = "b0undary") -> socket.socket:
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall((f"POST {URL} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\n"
               f"Content-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {len(body)}\r\n\r\n"
               ).encode() + body)
    return s


def multipart(path: Path, **fields) -> bytes:
    out = b""
    for k, v in form(**fields).items():
        out += f'--b0undary\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    out += (b'--b0undary\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n' +
            path.read_bytes() + b"\r\n--b0undary--\r\n")
    return out


# -- happy paths ----------------------------------------------------------------------------------------------------

def test_sdk_models_json_text_and_stream(tmp_path):
    x = noise(3.5, seed=1)
    audio = wav_file(tmp_path / "a.wav", x)
    deltas = expected_deltas(x, 1.0)
    assert len(deltas) >= 3
    full = "".join(deltas)
    with serve(max_chunk_seconds=1.0) as srv:
        assert [m.id for m in srv.sdk.models.list()] == [MODEL_ID]
        m = srv.sdk.models.retrieve(MODEL_ID)
        assert (m.id, m.object, m.created, m.owned_by) == (MODEL_ID, "model", 1793867382, "dangvansam")
        for missing in ("whisper-1", "org/model", "quan-transcribe-1.0"):
            with pytest.raises(openai.NotFoundError) as e:
                srv.sdk.models.retrieve(missing)
            assert e.value.code == "model_not_found"

        with audio.open("rb") as f:
            t = srv.sdk.audio.transcriptions.create(file=f, model=MODEL_ID, language="vi", temperature=0)
        assert t.text == full and t.usage.type == "duration" and t.usage.seconds == len(x) / SR
        assert set(t.model_dump(exclude_none=True)) == {"text", "usage"}
        with audio.open("rb") as f:
            assert srv.sdk.audio.transcriptions.create(file=f, model=MODEL_ID, response_format="text") == full
        with audio.open("rb") as f:
            got = list(srv.sdk.audio.transcriptions.create(file=f, model=MODEL_ID, stream=True))
        assert [e.type for e in got] == ["transcript.text.delta"] * len(deltas) + ["transcript.text.done"]
        assert [e.delta for e in got[:-1]] == deltas and got[-1].text == full and got[-1].usage is None

        r = post(srv, audio, stream="true", response_format="text")
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
        assert r.headers["x-accel-buffering"] == "no" and "content-length" not in r.headers
        assert events(r)[-1] == {"type": "transcript.text.done", "text": full}
        r = post(srv, audio, response_format="text")
        assert r.headers["content-type"] == "text/plain; charset=utf-8" and r.text == full
        assert idle(srv.app)
        assert outcomes(srv.app, "ok") == 1 and outcomes(srv.app, "ok", "json", "true") == 1
        assert srv.app.state.metrics.value("transcription_audio_seconds_total") == pytest.approx(5 * len(x) / SR)


def test_timed_formats_agree_with_json_text_and_stream(tmp_path):
    """verbose_json/srt/vtt carry the same transcript as json/text/SSE; cues are ordered, inside the audio, and
    cross chunk boundaries with absolute times. A silent chunk adds no cue."""
    x = noise(9.5, seed=11)
    x[4 * SR:7 * SR] = 0  # digital silence: the chunks inside it transcribe to "" and form a pause between cues
    audio = wav_file(tmp_path / "a.wav", x)

    def text_fn(w):
        return "" if not w.any() else name(w)

    silent = [(s.start / SR, s.end / SR) for s in split(x, SR) if not x[s.start:s.end].any()]
    assert sum(b - a for a, b in silent) >= PAUSE_SAMPLES / SR
    with serve(FakeEngine(text_fn=text_fn), max_chunk_seconds=1.0) as srv:
        full = post(srv, audio).json()["text"]
        assert full and post(srv, audio, response_format="text").text == full
        assert events(post(srv, audio, stream="true"))[-1]["text"] == full
        with audio.open("rb") as f:
            v = srv.sdk.audio.transcriptions.create(file=f, model=MODEL_ID, response_format="verbose_json")
        assert isinstance(v, openai.types.audio.TranscriptionVerbose)
        assert v.model_dump(exclude_none=True) == {"text": full, "language": "vietnamese", "duration": len(x) / SR,
                                                   "usage": {"type": "duration", "seconds": len(x) / SR}}
        for fmt in ("srt", "vtt"):
            r = post(srv, audio, response_format=fmt)
            assert r.status_code == 200 and r.headers["content-type"] == "text/plain; charset=utf-8"
            with audio.open("rb") as f:
                assert srv.sdk.audio.transcriptions.create(file=f, model=MODEL_ID, response_format=fmt) == r.text
            cues = parse_cues(r.text, fmt)
            assert " ".join(c[2] for c in cues) == full
            assert all(0 <= a < b <= len(x) / SR for a, b, _ in cues)
            assert all(p[1] <= n[0] for p, n in zip(cues, cues[1:]))
            # No cue in a silent chunk (SRT/VTT times are rounded to whole ms).
            assert not any(a < s1 - 0.0005 and b > s0 + 0.0005 for a, b, _ in cues for s0, s1 in silent)
            assert cues[-1][1] > 5  # absolute times past the first chunks
        assert idle(srv.app)
        assert outcomes(srv.app, "ok", "srt") == 2 and outcomes(srv.app, "ok", "verbose_json") == 1


def test_written_form_text_is_returned_verbatim_in_every_format(tmp_path):
    written = "Dạ. Em chào anh, hôm nay thế nào? Ngân sách: 50 triệu…"
    audio = wav_file(tmp_path / "a.wav", noise(1.5, seed=5))
    with serve(FakeEngine(text_fn=lambda _: written), max_chunk_seconds=2.0) as srv:
        assert post(srv, audio).json()["text"] == written
        assert post(srv, audio, response_format="text").text == written
        assert events(post(srv, audio, stream="true"))[-1]["text"] == written
        assert post(srv, audio, response_format="verbose_json").json()["text"] == written
        for fmt in ("srt", "vtt"):
            assert " ".join(c[2] for c in parse_cues(post(srv, audio, response_format=fmt).text, fmt)) == written


def test_cues_end_at_sentence_ends(tmp_path):
    written = "Dạ. Em chào anh, hôm nay thế nào? Ngân sách: 50 triệu… Vâng"
    audio = wav_file(tmp_path / "a.wav", noise(3, seed=6))
    with serve(FakeEngine(text_fn=lambda _: written), max_chunk_seconds=5.0) as srv:
        verbose = post(srv, audio, response_format="verbose_json").json()
        assert verbose["text"] == written
        assert "segments" not in verbose
        for fmt in ("srt", "vtt"):
            cues = parse_cues(post(srv, audio, response_format=fmt).text, fmt)
            assert [c[2] for c in cues] == ["Dạ.", "Em chào anh, hôm nay thế nào?", "Ngân sách: 50 triệu…", "Vâng"]
            assert all(p[1] == n[0] for p, n in zip(cues, cues[1:]))  # one chunk: back to back, no pause


def test_word_timestamps_after_the_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(transcription, "WORD_TIMESTAMPS", True)
    x = noise(3.5, seed=12)
    audio = wav_file(tmp_path / "a.wav", x)
    full = "".join(expected_deltas(x, 1.0))
    with serve(max_chunk_seconds=1.0) as srv:
        with audio.open("rb") as f:
            v = srv.sdk.audio.transcriptions.create(file=f, model=MODEL_ID, response_format="verbose_json",
                                                    timestamp_granularities=["word"])
        assert v.text == full and " ".join(w.word for w in v.words) == full
        assert all(0 <= w.start < w.end <= len(x) / SR for w in v.words)
        assert all(a.end <= b.start for a, b in zip(v.words, v.words[1:]))
        assert v.segments is None


@pytest.mark.parametrize("failure", ["missing_tokens", "embedded_marker", "whitespace"])
def test_alignment_failure_is_an_error_only_for_timed_formats(tmp_path, failure):
    class Broken(FakeEngine):
        def transcribe(self, wav):
            r = super().transcribe(wav)
            if failure == "embedded_marker":
                return r._replace(text="ab c", tokens=("▁a", "b▁c"), times=(0, 1))
            if failure == "whitespace":
                return r._replace(text="a b", tokens=("▁a b",), times=(0,))
            return r._replace(tokens=())  # tokens do not spell the text

    audio = wav_file(tmp_path / "a.wav", noise(2.5, seed=13))
    with serve(Broken(text_fn=name), max_chunk_seconds=1.0) as srv:
        assert post(srv, audio).status_code == 200
        assert post(srv, audio, response_format="verbose_json").status_code == 200  # needs no timing
        for fmt in ("srt", "vtt"):
            err = envelope(post(srv, audio, response_format=fmt), 500, "alignment_failed")
            assert err["type"] == "server_error" and "0." not in err["message"]
        assert outcomes(srv.app, "alignment_failed", "srt") == 1
        assert idle(srv.app)


@pytest.mark.parametrize("samples", [0, 1, MIN_FORWARD_SAMPLES - 1, MIN_FORWARD_SAMPLES, MIN_AUDIO_SAMPLES])
def test_empty_and_short_audio_is_empty_text_without_inference(tmp_path, samples, monkeypatch):
    monkeypatch.setattr(transcription, "WORD_TIMESTAMPS", True)
    audio = wav_file(tmp_path / "a.wav", noise(1, seed=2)[:samples])
    with serve() as srv:
        r = post(srv, audio)
        assert r.json() == {"text": "", "usage": {"type": "duration", "seconds": samples / SR}}
        assert post(srv, audio, response_format="text").text == ""
        assert events(post(srv, audio, stream="true")) == [{"type": "transcript.text.done", "text": ""}]
        verbose = {"text": "", "language": "vietnamese", "duration": samples / SR,
                   "usage": {"type": "duration", "seconds": samples / SR}}
        assert post(srv, audio, response_format="verbose_json").json() == verbose
        assert post(srv, audio, response_format="verbose_json", **{"timestamp_granularities[]": "word"}).json() == {
            **verbose, "words": []}
        assert post(srv, audio, response_format="srt").text == ""
        assert post(srv, audio, response_format="vtt").text == "WEBVTT\n\n"
        assert srv.engine.calls == []
        assert idle(srv.app)


def test_audio_just_above_the_short_audio_floor_reaches_the_engine_once(tmp_path):
    x = noise(1, seed=3)[:MIN_AUDIO_SAMPLES + 1]
    with serve(FakeEngine(text_fn=name)) as srv:
        assert post(srv, wav_file(tmp_path / "a.wav", x)).json()["text"] == name(x)
        assert len(srv.engine.calls) == 1
        assert idle(srv.app)


def test_concurrent_requests_are_serialized_and_not_mixed(tmp_path):
    files = {seed: noise(2.5, seed) for seed in range(4)}
    paths = {seed: wav_file(tmp_path / f"{seed}.wav", x) for seed, x in files.items()}
    with serve(FakeEngine(delay=0.02, text_fn=name), max_chunk_seconds=1.0) as srv:
        results = {}

        def run(seed, stream):
            fields = {"stream": "true"} if stream else {}
            with httpx2.Client(base_url=srv.base_url, headers=AUTH, timeout=20) as c:
                r = c.post(URL, files={"file": ("a.wav", paths[seed].read_bytes())}, data=form(**fields))
            results[seed] = events(r)[-1]["text"] if stream else r.json()["text"]

        threads = [threading.Thread(target=run, args=(seed, seed % 2 == 1)) for seed in files]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert results == {seed: "".join(expected_deltas(x, 1.0)) for seed, x in files.items()}
        assert srv.engine.max_active == 1
        assert len(srv.engine.calls) == sum(len(split(x, SR)) for x in files.values())
        assert idle(srv.app)


# -- errors before the response -------------------------------------------------------------------------------------

def test_errors_before_headers_are_http_envelopes(tmp_path):
    audio = wav_file(tmp_path / "a.wav", noise(0.5, seed=3))  # 32 KB
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    with serve(max_upload_bytes=60_000, max_spool_bytes=60_000) as srv:
        err = envelope(post(srv, audio, language="en"), 400, "unsupported_language")
        assert err["param"] == "language"
        err = envelope(post(srv, audio, stream="true", include="x"), 400, "unsupported_parameter")
        assert err["param"] == "include"
        for model in ("nope", "whisper-1"):  # the alias is off by default
            err = envelope(post(srv, audio, model=model), 404, "model_not_found")
            assert err["param"] == "model"
        # Decoding fails before SSE headers: an HTTP error, not an event.
        err = envelope(post(srv, bad, stream="true"), 400, "invalid_audio")
        assert err["param"] == "file"
        big = wav_file(tmp_path / "big.wav", noise(2, seed=3))
        envelope(post(srv, big), 413, "request_too_large")
        no_file = f'--b0undary\r\nContent-Disposition: form-data; name="model"\r\n\r\n{MODEL_ID}\r\n--b0undary--\r\n'
        r = srv.client.post(URL, content=no_file.encode(),
                            headers={"Content-Type": "multipart/form-data; boundary=b0undary"})
        assert envelope(r, 400, "missing_required_parameter")["param"] == "file"
        envelope(srv.client.get(URL), 405, "method_not_allowed")
        envelope(post(srv, audio, response_format="diarized_json"), 400, "unsupported_response_format")
        with pytest.raises(openai.BadRequestError) as e, audio.open("rb") as f:
            srv.sdk.audio.transcriptions.create(file=f, model=MODEL_ID, response_format="srt", stream=True)
        assert (e.value.code, e.value.param) == ("unsupported_parameter", "stream")
        with pytest.raises(openai.BadRequestError) as e, audio.open("rb") as f:
            srv.sdk.audio.transcriptions.create(file=f, model=MODEL_ID, response_format="verbose_json",
                                                timestamp_granularities=["word"])
        assert (e.value.code, e.value.param) == ("unsupported_parameter", "timestamp_granularities[]")
        assert srv.engine.calls == []
        assert idle(srv.app)
        assert outcomes(srv.app, "unsupported_language", "unknown", "unknown") == 1
        assert outcomes(srv.app, "invalid_audio", "json", "true") == 1


def test_auth_is_required(tmp_path):
    audio = wav_file(tmp_path / "a.wav", noise(1, seed=3))
    with serve() as srv:
        for path in (URL, "/v1/models", f"/v1/models/{MODEL_ID}"):
            r = httpx2.post(srv.base_url + path, files={"file": ("a.wav", audio.read_bytes())},
                            data=form()) if path == URL else httpx2.get(srv.base_url + path)
            envelope(r, 401, "invalid_api_key")
        assert srv.engine.calls == []


def test_model_not_ready_is_refused_before_the_body():
    engine = FakeEngine(text_fn=name)
    engine.load_gate.clear()
    with serve(engine, wait_ready=False) as srv:
        with socket.create_connection(("127.0.0.1", srv.port)) as s:
            # A huge declared body that is never sent: the 503 must not wait for it.
            s.sendall((f"POST {URL} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\n"
                       f"Content-Type: multipart/form-data; boundary=b0undary\r\nContent-Length: 20000000\r\n\r\n"
                       ).encode())
            s.settimeout(5)
            head = b""
            while b"}}" not in head:
                head += s.recv(4096)
        assert head.startswith(b"HTTP/1.1 503") and b"model_not_ready" in head and b"retry-after: 5" in head


def test_busy_server_returns_429_before_decoding(tmp_path):
    audio = wav_file(tmp_path / "a.wav", noise(1.5, seed=4))
    blocker = Blocker(at=1)
    with serve(FakeEngine(text_fn=blocker), queue_max_lanes=1, max_sessions=1) as srv:
        first = threading.Thread(target=post, args=(srv, audio))
        first.start()
        assert blocker.entered.wait(5)
        r = post(srv, audio)
        envelope(r, 429, "server_busy")
        assert r.headers["retry-after"] == "1"
        assert srv.app.state.metrics.value("decodes_total", outcome="ok") == 1  # the refused one never decoded
        blocker.release.set()
        first.join(10)
        wait_for(lambda: idle(srv.app))


def test_decoded_audio_budget_returns_429(tmp_path):
    audio = wav_file(tmp_path / "a.wav", noise(2, seed=5))
    blocker = Blocker(at=1)
    with serve(FakeEngine(text_fn=blocker), max_audio_seconds=2, max_chunk_seconds=2,
               max_decoded_seconds=3) as srv:
        first = threading.Thread(target=post, args=(srv, audio))
        first.start()
        assert blocker.entered.wait(5)
        assert srv.app.state.audio._held == 2 * SR
        envelope(post(srv, audio), 429, "server_busy")
        blocker.release.set()
        first.join(10)
        wait_for(lambda: idle(srv.app))
        assert post(srv, audio).status_code == 200


# -- errors after headers -------------------------------------------------------------------------------------------

def test_inference_error_mid_stream_is_an_error_event_without_done(tmp_path):
    x = noise(3.5, seed=6)
    audio = wav_file(tmp_path / "a.wav", x)
    first = expected_deltas(x, 1.0)[0]
    with serve(FakeEngine(text_fn=Blocker(at=2, fail=True)), max_chunk_seconds=1.0) as srv:
        got = []
        with pytest.raises(openai.APIError) as e, audio.open("rb") as f:
            for event in srv.sdk.audio.transcriptions.create(file=f, model=MODEL_ID, stream=True):
                got.append(event)
        assert [(g.type, g.delta) for g in got] == [("transcript.text.delta", first)]
        assert e.value.code == "inference_failed" and e.value.type == "server_error"
        wait_for(lambda: idle(srv.app))
        assert outcomes(srv.app, "inference_failed", "json", "true") == 1


def test_deadline_mid_stream_is_a_request_timeout_event(tmp_path):
    x = noise(3.5, seed=7)
    audio = wav_file(tmp_path / "a.wav", x)
    blocker = Blocker(at=2)
    with serve(FakeEngine(text_fn=blocker), max_chunk_seconds=1.0, request_timeout_seconds=1.5) as srv:
        try:
            got = events(post(srv, audio, stream="true"))
        finally:
            blocker.release.set()
        assert got[0] == {"type": "transcript.text.delta", "delta": expected_deltas(x, 1.0)[0]}
        assert got[1:] == [{"error": {"message": "Request timed out.", "type": "server_error", "param": None,
                                      "code": "request_timeout"}}]
        wait_for(lambda: idle(srv.app))
        assert len(srv.engine.calls) == 2  # the queued third chunk was cancelled, never run


# -- disconnects, slow clients, shutdown ----------------------------------------------------------------------------

@pytest.mark.parametrize("stream", ["false", "true"])
def test_disconnect_while_waiting_for_inference_stops_the_request(tmp_path, stream):
    audio = wav_file(tmp_path / "a.wav", noise(5, seed=8))
    blocker = Blocker(at=1)
    with serve(FakeEngine(text_fn=blocker), max_chunk_seconds=1.0,
               max_audio_seconds=5, max_decoded_seconds=5) as srv:
        s = raw_post(srv.port, multipart(audio, stream=stream))
        assert blocker.entered.wait(5)
        assert srv.app.state.scheduler._jobs == 2  # pipelined: the next chunk is already queued
        s.close()
        wait_for(lambda: srv.app.state.scheduler._lanes == 0)
        assert srv.app.state.audio._held == 5 * SR  # running slice still pins the whole decoded file
        envelope(post(srv, audio), 429, "server_busy")
        blocker.release.set()
        wait_for(lambda: idle(srv.app))
        time.sleep(0.2)
        assert len(srv.engine.calls) == 1  # the queued chunk was cancelled, no later chunk was submitted
        assert outcomes(srv.app, "disconnected", "json", stream) == 1


def test_disconnect_while_decoding_reaps_ffmpeg(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pid_file = tmp_path / "ffmpeg.pid"
    fake = bin_dir / "ffmpeg"
    fake.write_text(f"#!/bin/sh\necho $$ > {pid_file}\nexec /bin/sleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    audio = wav_file(tmp_path / "a.wav", noise(1, seed=9))
    with serve() as srv:
        s = raw_post(srv.port, multipart(audio))
        wait_for(lambda: pid_file.exists() and pid_file.read_text().strip())
        pid = int(pid_file.read_text())
        assert Path(f"/proc/{pid}").exists()
        s.close()
        wait_for(lambda: not Path(f"/proc/{pid}").exists())  # killed and reaped (no zombie)
        wait_for(lambda: idle(srv.app))
        assert srv.app.state.metrics.value("decodes_total", outcome="cancelled") == 1
        assert outcomes(srv.app, "disconnected") == 1
        assert srv.engine.calls == []


def test_shutdown_drains_a_running_stream(tmp_path):
    x = noise(3.5, seed=10)
    audio = wav_file(tmp_path / "a.wav", x)
    blocker = Blocker(at=2)
    with serve(FakeEngine(text_fn=blocker), max_chunk_seconds=1.0) as srv:
        got = {}

        def run():
            got["r"] = post(srv, audio, stream="true")

        t = threading.Thread(target=run)
        t.start()
        assert blocker.entered.wait(5)
        srv.server.should_exit = True
        time.sleep(0.3)
        assert t.is_alive()  # the stream is still open while the server drains
        blocker.release.set()
        t.join(10)
        evs = events(got["r"])
        assert evs[-1] == {"type": "transcript.text.done", "text": "".join(expected_deltas(x, 1.0))}


@pytest.mark.parametrize("outcome", ["send_timeout", "request_timeout"])
def test_slow_send_is_bounded_by_send_and_request_deadlines(capfd, outcome):
    """A client that stops reading: called as ASGI directly, since a real socket buffers far more than an event."""
    x = noise(3.5, seed=11)
    elapsed = []

    async def main():
        app = create_app(config(max_chunk_seconds=1.0,
                                request_timeout_seconds=0.2 if outcome == "request_timeout" else 10,
                                send_timeout_seconds=0.2 if outcome == "send_timeout" else 2),
                         FakeEngine(text_fn=name))
        buf = io.BytesIO()
        sf.write(buf, x, SR, format="WAV", subtype="FLOAT")
        boundary = "b0undary"
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{MODEL_ID}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="stream"\r\n\r\ntrue\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n').encode()
        body += buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": URL, "raw_path": URL.encode(), "root_path": "",
                 "query_string": b"", "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80),
                 "headers": [(b"authorization", f"Bearer {KEY}".encode()),
                             (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
                             (b"content-length", str(len(body)).encode())]}
        messages = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            if messages:
                return messages.pop()
            await asyncio.Event().wait()  # never disconnects

        sent = []

        async def send(message):
            sent.append(message)
            if len(sent) == 2:
                await asyncio.sleep(0.4)  # inference can finish while the first delta is blocked

        async with app.router.lifespan_context(app):
            while not app.state.worker.ready:
                await asyncio.sleep(0.01)
            started = time.monotonic()
            await asyncio.wait_for(app(scope, receive, send), 5)
            elapsed.append(time.monotonic() - started)
            await asyncio.sleep(0.01)  # waveform finalization is accounted on the event loop
            assert idle(app)
            assert outcomes(app, outcome, "json", "true") == 1
        assert sent[0]["status"] == 200
        if outcome == "send_timeout":
            assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]
        else:
            assert len(sent) == 3
            assert json.loads(sent[-1]["body"].decode().removeprefix("data: "))["error"]["code"] == outcome
            assert sent[-1]["more_body"] is False

    asyncio.run(main())
    assert elapsed[0] < 3
    logs = [json.loads(line) for line in capfd.readouterr().err.splitlines() if line.startswith("{")]
    assert [e["outcome"] for e in logs if e["event"] == "transcription.request"] == [outcome]
