"""The examples in tests/contract/examples/ (docs/api-contract.vi.md) parse with the pinned OpenAI Python SDK.

HTTP examples go through the real SDK client over a mock transport, so this also pins what the SDK sends
(path, auth header, multipart field names) and how it parses JSON, text, SSE and errors.
Realtime examples are validated (not just constructed) against the SDK's pydantic event unions.
"""

import base64
import email.parser
import json
from pathlib import Path

import httpx2
import openai
import pydantic
import pytest
from openai.types.realtime import (
    RealtimeClientEvent,
    RealtimeServerEvent,
    RealtimeTranscriptionSessionCreateRequest,
)

EXAMPLES = Path(__file__).parent / "examples"
FIXTURE = Path(__file__).parents[1] / "fixtures" / "vi_a.wav"
MODEL = "viet-asr"


def test_sdk_version_is_pinned():
    assert openai.__version__ == "3.31.0"


def parse_multipart(request) -> dict[str, list]:
    raw = b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.read()
    msg = email.parser.BytesParser().parsebytes(raw)
    fields: dict[str, list] = {}
    for part in msg.get_payload():
        name = part.get_param("name", header="content-disposition")
        value = part.get_filename() or part.get_payload(decode=True).decode()
        fields.setdefault(name, []).append(value)
    return fields


def client_with(status: int, body: bytes, content_type: str, seen: list, headers: dict | None = None):
    def handler(request):
        seen.append(request)
        return httpx2.Response(status, content=body, headers={"content-type": content_type, **(headers or {})})

    return openai.OpenAI(
        api_key="sk-test", base_url="http://gateway.test/v1", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def example(name: str) -> bytes:
    return (EXAMPLES / name).read_bytes()


def test_models_list_and_retrieve():
    seen = []
    c = client_with(200, example("models.json"), "application/json", seen)
    assert [m.id for m in c.models.list()] == [MODEL]
    c = client_with(200, example("model.json"), "application/json", seen)
    m = c.models.retrieve(MODEL)
    assert (m.id, m.object, m.owned_by, m.created) == (MODEL, "model", "dangvansam", 1793867382)
    assert [r.url.path for r in seen] == ["/v1/models", f"/v1/models/{MODEL}"]
    assert all(r.headers["authorization"] == "Bearer sk-test" for r in seen)


def test_json_transcription_and_request_encoding():
    seen = []
    c = client_with(200, example("transcription.json"), "application/json", seen)
    with FIXTURE.open("rb") as f:
        t = c.audio.transcriptions.create(file=f, model=MODEL, language="vi", temperature=0)
    assert isinstance(t, openai.types.audio.Transcription)
    assert t.text.startswith("Các sản phẩm") and t.usage.type == "duration" and t.usage.seconds == 8.88
    req = seen[0]
    assert req.method == "POST" and req.url.path == "/v1/audio/transcriptions"
    fields = parse_multipart(req)
    assert fields == {"file": ["vi_a.wav"], "model": [MODEL], "language": ["vi"], "temperature": ["0"]}


def test_text_transcription_returns_body_verbatim():
    c = client_with(200, example("transcription.txt"), "text/plain; charset=utf-8", [])
    with FIXTURE.open("rb") as f:
        t = c.audio.transcriptions.create(file=f, model=MODEL, response_format="text")
    assert t == example("transcription.txt").decode()


def test_verbose_json_parses_without_segments():
    raw = json.loads(example("transcription_verbose.json"))
    c = client_with(200, example("transcription_verbose.json"), "application/json", [])
    with FIXTURE.open("rb") as f:
        t = c.audio.transcriptions.create(file=f, model=MODEL, response_format="verbose_json")
    assert isinstance(t, openai.types.audio.TranscriptionVerbose)
    assert t.model_dump(exclude_unset=True) == raw  # no field dropped or renamed
    assert (t.language, t.duration, t.usage.seconds) == ("vietnamese", 8.88, 8.88)
    assert t.segments is None and t.words is None


def test_verbose_json_words_shape_parses():
    """R2b payload shape (served only after the alignment gate)."""
    raw = json.loads(example("transcription_verbose_words.json"))
    c = client_with(200, example("transcription_verbose_words.json"), "application/json", [])
    with FIXTURE.open("rb") as f:
        t = c.audio.transcriptions.create(file=f, model=MODEL, response_format="verbose_json",
                                          timestamp_granularities=["word"])
    assert t.model_dump(exclude_unset=True) == raw
    assert " ".join(w.word for w in t.words) == t.text
    assert all(0 <= w.start < w.end <= t.duration for w in t.words)
    assert all(a.end <= b.start for a, b in zip(t.words, t.words[1:]))


@pytest.mark.parametrize("fmt", ["srt", "vtt"])
def test_subtitles_are_returned_verbatim(fmt):
    from tests.live_server import parse_cues

    body = example(f"transcription.{fmt}").decode()
    c = client_with(200, body.encode(), "text/plain; charset=utf-8", [])
    with FIXTURE.open("rb") as f:
        assert c.audio.transcriptions.create(file=f, model=MODEL, response_format=fmt) == body
    cues = parse_cues(body, fmt)
    assert all(a < b for a, b, _ in cues) and all(p[1] <= n[0] for p, n in zip(cues, cues[1:]))


def test_every_format_example_has_the_same_text():
    from tests.live_server import parse_cues

    text = json.loads(example("transcription.json"))["text"]
    assert example("transcription.txt").decode() == text
    assert json.loads(example("transcription_verbose.json"))["text"] == text
    for fmt in ("srt", "vtt"):
        assert " ".join(c[2] for c in parse_cues(example(f"transcription.{fmt}").decode(), fmt)) == text


def test_array_fields_are_sent_as_repeated_bracket_fields():
    """The gateway must parse `include[]` / `timestamp_granularities[]` (repeated), not `include`."""
    seen = []
    c = client_with(200, example("transcription.json"), "application/json", seen)
    with FIXTURE.open("rb") as f:
        c.audio.transcriptions.create(
            file=f, model=MODEL, response_format="verbose_json",
            timestamp_granularities=["word", "segment"], include=["logprobs"],
        )
    fields = parse_multipart(seen[0])
    assert fields["timestamp_granularities[]"] == ["word", "segment"]
    assert fields["include[]"] == ["logprobs"]
    assert "include" not in fields and "timestamp_granularities" not in fields


def test_sse_stream_deltas_join_to_done():
    seen = []
    c = client_with(200, example("transcription_stream.sse"), "text/event-stream", seen)
    with FIXTURE.open("rb") as f:
        events = list(c.audio.transcriptions.create(file=f, model=MODEL, stream=True))
    assert parse_multipart(seen[0])["stream"] == ["true"]
    assert [e.type for e in events] == ["transcript.text.delta", "transcript.text.delta", "transcript.text.done"]
    assert "".join(e.delta for e in events[:-1]) == events[-1].text
    assert events[-1].usage is None  # done.usage only has a token variant; we omit it


def test_sse_error_after_headers_raises_and_has_no_done():
    c = client_with(200, example("transcription_stream_error.sse"), "text/event-stream", [])
    got = []
    with FIXTURE.open("rb") as f, pytest.raises(openai.APIError) as exc:
        for e in c.audio.transcriptions.create(file=f, model=MODEL, stream=True):
            got.append(e.type)
    assert got == ["transcript.text.delta"]
    assert exc.value.code == "inference_failed" and exc.value.type == "server_error"


def test_http_error_envelope():
    c = client_with(400, example("error.json"), "application/json", [])
    with FIXTURE.open("rb") as f, pytest.raises(openai.BadRequestError) as exc:
        c.audio.transcriptions.create(file=f, model=MODEL, language="en")
    e = exc.value
    assert (e.status_code, e.code, e.param, e.type) == (400, "unsupported_language", "language", "invalid_request_error")


@pytest.mark.parametrize("case", json.loads(example("service_errors.json")))
def test_service_errors_and_request_ids(case):
    classes = {
        400: openai.BadRequestError, 401: openai.AuthenticationError, 404: openai.NotFoundError,
        405: openai.APIStatusError, 413: openai.APIStatusError, 429: openai.RateLimitError,
        500: openai.InternalServerError, 503: openai.InternalServerError,
    }
    headers = {"x-request-id": "req_contract_example", **case["headers"]}
    c = client_with(case["status"], json.dumps({"error": case["error"]}).encode(),
                    "application/json", [], headers)
    with pytest.raises(classes[case["status"]]) as exc:
        c.models.list()
    e = exc.value
    expected = case["error"]
    assert (e.status_code, e.code, e.type, e.param) == (
        case["status"], expected["code"], expected["type"], expected["param"],
    )
    assert e.request_id == "req_contract_example"
    assert e.response.headers.get("retry-after") == case["headers"].get("Retry-After")


def test_service_examples_match_gateway_errors():
    from app import errors

    actual = [
        errors.invalid_request("invalid_value", "Invalid request."), errors.invalid_audio(),
        errors.audio_too_long(1800), errors.invalid_api_key(),
        errors.model_not_found("unknown"), errors.not_found(),
        errors.APIError(405, "invalid_request_error", "method_not_allowed", "Method not allowed."),
        errors.request_too_large(1024), errors.request_too_large(8192, what="Form field"), errors.server_busy(1), errors.inference_failed(),
        errors.alignment_failed(),
        errors.internal_error(), errors.model_not_ready(), errors.shutting_down(), errors.request_timeout(),
    ]
    assert [{"status": e.status, "headers": e.headers, **e.body()} for e in actual] == json.loads(
        example("service_errors.json"),
    )


def lines(name: str) -> list[dict]:
    return [json.loads(line) for line in example(name).decode().splitlines() if line.strip()]


def test_realtime_client_events_validate():
    adapter = pydantic.TypeAdapter(RealtimeClientEvent)
    events = [adapter.validate_python(e) for e in lines("realtime_client.jsonl")]
    assert isinstance(events[0].session, RealtimeTranscriptionSessionCreateRequest)
    pcm = base64.b64decode(events[1].audio, validate=True)
    assert len(pcm) % 2 == 0  # PCM16 little-endian


def test_realtime_server_events_validate_and_are_consistent():
    adapter = pydantic.TypeAdapter(RealtimeServerEvent)
    raw = lines("realtime_server.jsonl")
    events = [adapter.validate_python(e) for e in raw]
    for r, e in zip(raw, events):
        assert e.type == r["type"]
        assert e.model_dump(exclude_unset=True) == r  # no field silently dropped or renamed
    by_type = {}
    for e in events:
        by_type.setdefault(e.type, []).append(e)
    for e in by_type["session.created"] + by_type["session.updated"]:
        assert isinstance(e.session, RealtimeTranscriptionSessionCreateRequest)
        assert e.session.audio.input.format.rate == 24000
    committed = by_type["input_audio_buffer.committed"][0]
    added = by_type["conversation.item.added"][0]
    assert added.item.id == committed.item_id and added.previous_item_id == committed.previous_item_id
    deltas = by_type["conversation.item.input_audio_transcription.delta"]
    done = by_type["conversation.item.input_audio_transcription.completed"][0]
    assert all(d.item_id == done.item_id and d.content_index == done.content_index == 0 for d in deltas)
    assert "".join(d.delta for d in deltas) == done.transcript
    assert done.usage.type == "duration"
    assert len({e.event_id for e in events}) == len(events)


def test_realtime_error_examples_match_gateway_errors():
    from app import errors

    def event_error(e, event_id=None):
        body = {"type": e.type, "code": e.code, "message": e.message, "param": e.param}
        return body | ({"event_id": event_id} if event_id else {})

    raw = [e["error"] for e in lines("realtime_server.jsonl") if e["type"] == "error"]
    assert raw == [
        event_error(errors.invalid_request("invalid_audio", "input_audio_buffer.append: audio is not valid base64.",
                                           "audio"), "client_evt_5"),
        event_error(errors.unknown_parameter("session.audio.input.foo"), "client_evt_6"),
        event_error(errors.input_audio_buffer_commit_empty(), "client_evt_7"),
        event_error(errors.server_busy(1), "client_evt_8"),
        event_error(errors.idle_timeout(120)),
    ]
    failed = [e for e in lines("realtime_server.jsonl") if e["type"].endswith(".failed")]
    e = errors.inference_failed()
    assert [f["error"] for f in failed] == [{"type": e.type, "code": e.code, "message": e.message}]
