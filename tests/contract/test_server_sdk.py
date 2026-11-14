"""The real gateway (Uvicorn + FFmpeg, fake engine) returns the payload shapes of tests/contract/examples/, and both
pinned SDKs parse them: Python here, JS via `js/server.test.ts` (run against gateways this module starts: plain HTTP
for REST, HTTPS/WSS with a self-signed certificate for Realtime, since the JS realtime client only uses `wss:`).

The fake engine fails any chunk whose mean is below FAIL_MEAN, so `fail.wav` (quiet noise, then a loud constant)
streams at least one delta before an inference error.
"""

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import openai
import pydantic
import pytest
from openai.types.realtime import RealtimeServerEvent
from websockets.exceptions import InvalidStatus

from app.chunking import split
from app.transcription import MODEL_ID
from tests.fake_engine import FakeEngine
from tests.live_server import (
    expected_deltas,
    expected_turn,
    KEY,
    name,
    noise,
    parse_cues,
    pcm24,
    serve,
    SR,
    wait_for,
    wav_file,
)

EXAMPLES = Path(__file__).parent / "examples"
JS = Path(__file__).parent / "js"
CHUNK_SECONDS = 1.0
FAIL_MEAN = -0.5


def text_fn(w: np.ndarray) -> str:
    if w.mean() < FAIL_MEAN:
        raise RuntimeError("inference error: simulated")
    return name(w)


def shape(value):
    """Keys and JSON types, not values."""
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    return type(value).__name__


def example_events(name_: str) -> list[dict]:
    blocks = (EXAMPLES / name_).read_text().split("\n\n")
    return [json.loads(b.removeprefix("data: ")) for b in blocks if b.strip()]


@pytest.fixture
def audio(tmp_path) -> Path:
    speech = noise(3.5, seed=21)
    fail = np.concatenate([noise(1.5, seed=22) * 0.25, np.full(int(1.5 * SR), -0.95, np.float32)])
    means = [fail[s.start:s.end].mean() for s in split(fail, int(CHUNK_SECONDS * SR))]
    assert means[0] > FAIL_MEAN and min(means) < FAIL_MEAN  # first chunk succeeds, a later one fails
    wav_file(tmp_path / "speech.wav", speech)
    wav_file(tmp_path / "fail.wav", fail)
    deltas = expected_deltas(speech, CHUNK_SECONDS)
    turns = []
    for i, seconds in enumerate((2.5, 0.8)):  # realtime turns: several chunks, then one
        pcm = pcm24(seconds, seed=30 + i)
        (tmp_path / f"turn{i}.pcm").write_bytes(pcm)
        turns.append({"deltas": expected_turn(pcm, CHUNK_SECONDS), "seconds": len(pcm) / 2 / 24000})
    (tmp_path / "expected.json").write_text(json.dumps(
        {"text": "".join(deltas), "deltas": deltas, "seconds": len(speech) / SR, "turns": turns}))
    return tmp_path


@pytest.fixture
def srv():
    with serve(FakeEngine(text_fn=text_fn), max_chunk_seconds=CHUNK_SECONDS) as s:
        yield s


def post(srv, path: Path, **fields):
    return srv.client.post("/v1/audio/transcriptions", files={"file": (path.name, path.read_bytes())},
                           data={"model": MODEL_ID, **fields})


def sse(response) -> list[dict]:
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(b.removeprefix("data: ")) for b in response.text.split("\n\n") if b.strip()]


def test_payloads_match_examples(srv, audio):
    assert srv.client.get("/v1/models").json() == json.loads((EXAMPLES / "models.json").read_text())
    assert srv.client.get(f"/v1/models/{MODEL_ID}").json() == json.loads((EXAMPLES / "model.json").read_text())

    body = post(srv, audio / "speech.wav").json()
    assert shape(body) == shape(json.loads((EXAMPLES / "transcription.json").read_text()))
    r = post(srv, audio / "speech.wav", response_format="text")
    assert r.headers["content-type"] == "text/plain; charset=utf-8" and r.text == body["text"]
    assert not r.text.endswith("\n")
    verbose = post(srv, audio / "speech.wav", response_format="verbose_json").json()
    assert shape(verbose) == shape(json.loads((EXAMPLES / "transcription_verbose.json").read_text()))
    assert verbose["text"] == body["text"] and verbose["duration"] == body["usage"]["seconds"]
    for fmt in ("srt", "vtt"):
        r = post(srv, audio / "speech.wav", response_format=fmt)
        assert r.headers["content-type"] == "text/plain; charset=utf-8"
        assert " ".join(c[2] for c in parse_cues(r.text, fmt)) == body["text"]

    delta, _, done = map(shape, example_events("transcription_stream.sse"))
    got = sse(post(srv, audio / "speech.wav", stream="true"))
    assert [shape(e) for e in got] == [delta] * (len(got) - 1) + [done]
    _, error = map(shape, example_events("transcription_stream_error.sse"))
    got = sse(post(srv, audio / "fail.wav", stream="true"))
    assert [shape(e) for e in got] == [delta] * (len(got) - 1) + [error] and len(got) >= 2

    # Pipelined work can still be finishing/recovering after the error event reaches the client.
    wait_for(lambda: srv.app.state.scheduler._jobs == 0 and srv.app.state.worker.ready)
    err = srv.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"x")},
                          data={"model": MODEL_ID, "language": "en"}).json()
    assert shape(err) == shape(json.loads((EXAMPLES / "error.json").read_text()))


def test_python_sdk(srv, audio):
    expected = json.loads((audio / "expected.json").read_text())
    c = srv.sdk
    assert [m.model_dump(exclude_none=True) for m in c.models.list()] == json.loads((EXAMPLES / "models.json").read_text())["data"]
    with (audio / "speech.wav").open("rb") as f:
        t = c.audio.transcriptions.create(file=f, model=MODEL_ID, language="vi", temperature=0)
    assert isinstance(t, openai.types.audio.Transcription)
    assert t.model_dump(exclude_none=True) == {"text": expected["text"],
                                               "usage": {"type": "duration", "seconds": expected["seconds"]}}
    with (audio / "speech.wav").open("rb") as f:
        v = c.audio.transcriptions.create(file=f, model=MODEL_ID, response_format="verbose_json")
    assert isinstance(v, openai.types.audio.TranscriptionVerbose)
    assert v.model_dump(exclude_none=True) == {"text": expected["text"], "language": "vietnamese",
                                               "duration": expected["seconds"],
                                               "usage": {"type": "duration", "seconds": expected["seconds"]}}
    for fmt in ("srt", "vtt"):
        with (audio / "speech.wav").open("rb") as f:
            body = c.audio.transcriptions.create(file=f, model=MODEL_ID, response_format=fmt)
        assert isinstance(body, str) and " ".join(t for _, _, t in parse_cues(body, fmt)) == expected["text"]
    with pytest.raises(openai.BadRequestError) as e, (audio / "speech.wav").open("rb") as f:
        c.audio.transcriptions.create(file=f, model=MODEL_ID, response_format="verbose_json",
                                      timestamp_granularities=["segment"])
    assert (e.value.code, e.value.param) == ("unsupported_parameter", "timestamp_granularities[]")
    with (audio / "speech.wav").open("rb") as f:
        events = list(c.audio.transcriptions.create(file=f, model=MODEL_ID, stream=True))
    assert [e.delta for e in events[:-1]] == expected["deltas"] and events[-1].text == expected["text"]
    got = []
    with pytest.raises(openai.APIError) as e, (audio / "fail.wav").open("rb") as f:
        for event in c.audio.transcriptions.create(file=f, model=MODEL_ID, stream=True):
            got.append(event.type)
    assert got and set(got) == {"transcript.text.delta"}
    assert (e.value.code, e.value.type) == ("inference_failed", "server_error")


def realtime_shapes(event: dict) -> None:
    """A server event has the keys and JSON types of one example of its type (null and string are told apart)."""
    examples = [e for e in (json.loads(line) for line in (EXAMPLES / "realtime_server.jsonl").read_text().splitlines())
                if e["type"] == event["type"]]
    assert examples, event["type"]
    assert shape(event) in [shape(e) for e in examples], event


def test_python_sdk_realtime(srv, audio):
    expected = json.loads((audio / "expected.json").read_text())
    adapter = pydantic.TypeAdapter(RealtimeServerEvent)
    received = []

    def recv(conn) -> dict:
        raw = json.loads(conn.recv_bytes())
        event = adapter.validate_python(raw)  # strict validation, not just construction
        assert event.model_dump(exclude_unset=True) == raw
        realtime_shapes(raw)
        assert conn.parse_event(json.dumps(raw)).type == raw["type"]
        received.append(raw)
        return raw

    def until(conn, type_: str) -> list[dict]:
        got = [recv(conn)]
        while got[-1]["type"] != type_:
            got.append(recv(conn))
        return got

    with srv.sdk.realtime.connect(model=MODEL_ID) as conn:
        created = recv(conn)
        assert created["type"] == "session.created"
        conn.session.update(session=created["session"])
        assert recv(conn)["type"] == "session.updated"
        previous = None
        for i, turn in enumerate(expected["turns"]):
            pcm = (audio / f"turn{i}.pcm").read_bytes()
            for j in range(0, len(pcm), 4801):  # 100 ms packets, split samples
                conn.input_audio_buffer.append(audio=base64.b64encode(pcm[j:j + 4801]).decode())
            conn.input_audio_buffer.commit()
            events = until(conn, "conversation.item.input_audio_transcription.completed")
            committed, added, *deltas, done = events
            assert [committed["type"], added["type"]] == ["input_audio_buffer.committed", "conversation.item.added"]
            assert committed["previous_item_id"] == added["previous_item_id"] == previous
            assert [d["delta"] for d in deltas] == turn["deltas"] and done["transcript"] == "".join(turn["deltas"])
            assert done["usage"] == {"type": "duration", "seconds": turn["seconds"]}
            previous = committed["item_id"]
        conn.input_audio_buffer.append(audio=base64.b64encode(b"\0" * 960).decode())
        conn.input_audio_buffer.clear()
        assert recv(conn)["type"] == "input_audio_buffer.cleared"
        conn.input_audio_buffer.commit(event_id="client_evt_7")
        err = recv(conn)
        assert err["error"]["code"] == "input_audio_buffer_commit_empty" and err["error"]["event_id"] == "client_evt_7"
        conn.send({"type": "input_audio_buffer.append", "event_id": "client_evt_5", "audio": "not base64!"})
        assert recv(conn)["error"]["code"] == "invalid_audio"
    assert {e["type"] for e in received} >= {e["type"] for e in map(json.loads, (EXAMPLES / "realtime_server.jsonl")
                                                                     .read_text().splitlines())} - {
        "conversation.item.input_audio_transcription.failed"}
    with pytest.raises(InvalidStatus) as e, srv.sdk.realtime.connect(model="whisper-1"):
        pass
    assert e.value.response.status_code == 404


@pytest.fixture
def srv_tls(tmp_path):
    with serve(FakeEngine(text_fn=text_fn), max_chunk_seconds=CHUNK_SECONDS, tls_dir=tmp_path) as s:
        yield s


def test_js_sdk(srv, srv_tls, audio):
    node = shutil.which("node")
    if node is None or not (JS / "node_modules").is_dir():
        pytest.skip("needs node and `npm ci` in tests/contract/js")
    env = {**os.environ, "GATEWAY_TEST_BASE_URL": srv.base_url + "/v1", "GATEWAY_TEST_API_KEY": KEY,
           "GATEWAY_TEST_AUDIO_DIR": str(audio), "GATEWAY_TEST_WSS_BASE_URL": srv_tls.base_url + "/v1",
           "GATEWAY_TEST_CA": str(srv_tls.ca)}
    r = subprocess.run([node, "--test", "--test-reporter=tap", "server.test.ts"], cwd=JS, env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "# fail 0" in r.stdout and "# skipped 0" in r.stdout and "# pass 0" not in r.stdout, r.stdout
