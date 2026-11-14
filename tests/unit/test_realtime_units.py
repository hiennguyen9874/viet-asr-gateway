"""Pure parts of the realtime adapter (docs/api-contract.vi.md §6): turn buffer, session.update, event dispatch."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soxr

from app.errors import APIError
from app.realtime import (
    CLIENT_EVENT_TYPES,
    parse_event,
    parse_session_update,
    PCM_RATE,
    samples_bound,
    Session,
    session_object,
    TurnBuffer,
)
from app.transcription import MODEL_ID, WHISPER_ALIAS

MODELS = (MODEL_ID,)


def pcm16(n: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1, 1, n) * 30000).astype("<i2").tobytes()


def reference(pcm: bytes) -> np.ndarray:
    """A one-shot soxr HQ resample of the same samples."""
    x = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768
    return soxr.resample(x, PCM_RATE, 16000).astype(np.float32)


def feed(buf: TurnBuffer, data: bytes, sizes) -> None:
    i = 0
    for n in sizes:
        buf.append(data[i:i + n])
        i += n
    buf.append(data[i:])


def code(exc_info) -> tuple[str, str | None]:
    return exc_info.value.code, exc_info.value.param


# -- TurnBuffer ---------------------------------------------------------------------------------------------------

def test_empty_appends_do_not_accumulate_reservations():
    reservations = []

    def reserve(n):
        reservations.append(n)
        return lambda: None

    session = Session.__new__(Session)
    session._ep = SimpleNamespace(audio=SimpleNamespace(reserve=reserve))
    session._buffer = TurnBuffer(PCM_RATE)
    session._pending_releases = []

    async def append():
        for _ in range(10_000):
            await session._append("")
        await session._append("AA==")  # one odd byte, no whole sample yet
        assert not session._pending_releases and not reservations
        await session._append("AA==")  # completes the sample without losing the odd byte
        assert session._buffer.samples == 1
        assert reservations == [1] and len(session._pending_releases) == 1

    asyncio.run(append())


def test_random_packetization_equals_one_shot_reference():
    data = pcm16(72_001, seed=1)  # 3 s + one sample
    rng = np.random.default_rng(2)
    buf = TurnBuffer(max_samples=10 * PCM_RATE)
    sizes = [int(rng.integers(1, 9_601)) for _ in range(40)]  # 1 byte .. 200 ms, odd sizes included
    feed(buf, data, sizes)
    wav, seconds = buf.commit()
    assert wav.dtype == np.float32 and np.array_equal(wav, reference(data))
    assert seconds == 72_001 / PCM_RATE


def test_odd_bytes_carry_over_and_trailing_half_sample_is_dropped():
    data = pcm16(4_800, seed=3)
    buf = TurnBuffer(max_samples=PCM_RATE)
    feed(buf, data, [1, 3, 5, 2, 7])
    buf.append(b"\x7f")  # half a sample at commit: dropped, not counted
    assert buf.samples == 4_800
    wav, seconds = buf.commit()
    assert np.array_equal(wav, reference(data)) and seconds == 0.2


def test_second_turn_does_not_leak_state():
    a, b = pcm16(24_000, seed=4), pcm16(30_000, seed=5)
    buf = TurnBuffer(max_samples=PCM_RATE * 10)
    feed(buf, a, [999, 1])
    buf.append(b"\x01")  # odd byte dropped by the commit
    buf.commit()
    feed(buf, b, [4_801, 333])
    assert np.array_equal(buf.commit()[0], reference(b))
    buf.append(a[:1001])
    buf.clear()  # drops samples, odd byte and resampler state
    assert buf.samples == 0
    feed(buf, b, [7])
    assert np.array_equal(buf.commit()[0], reference(b))


def test_empty_commit_is_refused_and_keeps_the_buffer():
    buf = TurnBuffer(max_samples=PCM_RATE)
    with pytest.raises(APIError) as e:
        buf.commit()
    assert code(e) == ("input_audio_buffer_commit_empty", None)
    buf.append(b"\x01")  # one byte: still no whole sample
    with pytest.raises(APIError):
        buf.commit()
    buf.append(b"\x02")  # the kept byte completes a sample
    wav, seconds = buf.commit()
    assert len(wav) == 1 and seconds == 1 / PCM_RATE


@pytest.mark.parametrize("n", [1, 2, 240, 241, 2_400, 2_401])
def test_short_turns_resample_like_the_reference(n):
    data = pcm16(n, seed=n)
    buf = TurnBuffer(max_samples=PCM_RATE)
    buf.append(data)
    wav, _ = buf.commit()
    assert np.array_equal(wav, reference(data))
    assert len(wav) <= samples_bound(n)


def test_reservation_bound_covers_streamed_output():
    data = pcm16(48_017, seed=6)
    rng = np.random.default_rng(7)
    buf, reserved, i = TurnBuffer(max_samples=PCM_RATE * 5), 0, 0
    while i < len(data):
        n = int(rng.integers(1, 3_000))
        reserved += samples_bound(buf.check(len(data[i:i + n])))
        buf.append(data[i:i + n])
        i += n
    assert len(buf.commit()[0]) <= reserved


def test_turn_cap_refuses_the_whole_append():
    buf = TurnBuffer(max_samples=1_000)
    buf.append(pcm16(999, seed=8) + b"\x01")
    assert buf.check(1) == 1  # the odd byte completes sample 1000: still within the cap
    with pytest.raises(APIError) as e:
        buf.append(b"\x00\x00\x00")  # would reach 1001 whole samples
    assert code(e) == ("input_audio_buffer_full", None)
    assert buf.samples == 999
    buf.append(b"\x00")
    assert buf.samples == 1_000


# -- session.update -----------------------------------------------------------------------------------------------

def update(**input_):
    return {"type": "transcription", "audio": {"input": input_}}


def test_session_object_matches_the_contract_example():
    example = json.loads(open("tests/contract/examples/realtime_server.jsonl").readline())
    assert session_object(MODEL_ID) == example["session"]


@pytest.mark.parametrize("session", [
    {"type": "transcription"},
    update(),
    update(format={"type": "audio/pcm", "rate": 24000}, turn_detection=None, noise_reduction=None),
    update(format={"type": "audio/pcm"}, transcription={"model": MODEL_ID, "language": "vi", "prompt": ""}),
    update(transcription={"model": None, "language": None, "prompt": None, "delay": None}),
    {"type": "transcription", "include": None, "audio": None},
])
def test_valid_updates_keep_the_only_supported_session(session):
    assert parse_session_update(session, session_object(MODEL_ID), MODELS) == session_object(MODEL_ID)


def test_update_switches_between_served_models():
    models = (MODEL_ID, WHISPER_ALIAS)
    new = parse_session_update(update(transcription={"model": WHISPER_ALIAS}), session_object(MODEL_ID), models)
    assert new == session_object(WHISPER_ALIAS)
    assert parse_session_update(update(), new, models) == new  # omitted = unchanged


P = "session.audio.input"


@pytest.mark.parametrize("session,expected", [
    ([], ("invalid_value", "session")),
    ({}, ("invalid_value", "session.type")),
    ({"type": "realtime"}, ("invalid_value", "session.type")),
    ({"type": "transcription", "voice": "x"}, ("unknown_parameter", "session.voice")),
    ({"type": "transcription", "include": ["item.input_audio_transcription.logprobs"]},
     ("unsupported_parameter", "session.include")),
    ({"type": "transcription", "audio": {"output": {}}}, ("unknown_parameter", "session.audio.output")),
    ({"type": "transcription", "audio": {"input": 1}}, ("invalid_value", P)),
    (update(foo=1), ("unknown_parameter", f"{P}.foo")),
    (update(format={"type": "audio/pcmu"}), ("unsupported_parameter", f"{P}.format.type")),
    (update(format={"type": "audio/pcma"}), ("unsupported_parameter", f"{P}.format.type")),
    (update(format={"type": "audio/wav"}), ("invalid_value", f"{P}.format.type")),
    (update(format={"type": "audio/pcm", "rate": 16000}), ("invalid_value", f"{P}.format.rate")),
    (update(format={"type": "audio/pcm", "rate": True}), ("invalid_value", f"{P}.format.rate")),
    (update(format={"type": "audio/pcm", "bits": 16}), ("unknown_parameter", f"{P}.format.bits")),
    (update(turn_detection={"type": "server_vad"}), ("unsupported_turn_detection", f"{P}.turn_detection")),
    (update(turn_detection={"type": "semantic_vad"}), ("unsupported_turn_detection", f"{P}.turn_detection")),
    (update(noise_reduction={"type": "near_field"}), ("unsupported_parameter", f"{P}.noise_reduction")),
    (update(transcription={"prompt": "xin chào"}), ("unsupported_parameter", f"{P}.transcription.prompt")),
    (update(transcription={"prompt": 1}), ("invalid_value", f"{P}.transcription.prompt")),
    (update(transcription={"keywords": ["a"]}), ("unsupported_parameter", f"{P}.transcription.keywords")),
    (update(transcription={"languages": ["vi"]}), ("unsupported_parameter", f"{P}.transcription.languages")),
    (update(transcription={"delay": "low"}), ("unsupported_parameter", f"{P}.transcription.delay")),
    (update(transcription={"language": "en"}), ("unsupported_language", f"{P}.transcription.language")),
    (update(transcription={"language": 1}), ("invalid_value", f"{P}.transcription.language")),
    (update(transcription={"model": "gpt-4o-transcribe"}), ("model_not_found", f"{P}.transcription.model")),
    (update(transcription={"model": WHISPER_ALIAS}), ("model_not_found", f"{P}.transcription.model")),
    (update(transcription={"model": 3}), ("invalid_value", f"{P}.transcription.model")),
    (update(transcription={"x" * 100: 1}), ("unknown_parameter", f"{P}.transcription." + "x" * 100)),
])
def test_rejected_updates(session, expected):
    with pytest.raises(APIError) as e:
        parse_session_update(session, session_object(MODEL_ID), MODELS)
    got = code(e)
    assert got[0] == expected[0] and got[1] == expected[1][:64]
    assert e.value.type == "invalid_request_error"


def test_partially_valid_update_changes_nothing():
    current = session_object(MODEL_ID)
    snapshot = json.dumps(current)
    session = update(transcription={"model": WHISPER_ALIAS}, turn_detection={"type": "server_vad"})
    with pytest.raises(APIError):
        parse_session_update(session, current, (MODEL_ID, WHISPER_ALIAS))
    assert json.dumps(current) == snapshot


# -- event dispatch -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("{", ("invalid_event", None)),
    ("[]", ("invalid_event", None)),
    ('"x"', ("invalid_event", None)),
    ("{}", ("invalid_event", "type")),
    ('{"type": 1}', ("invalid_event", "type")),
    ('{"type": "no.such.event"}', ("invalid_event", "type")),
    ('{"type": "transcription_session.update"}', ("invalid_event", "type")),  # beta, not in the GA union
    ('{"type": "response.create"}', ("unsupported_event", "type")),
    ('{"type": "conversation.item.create"}', ("unsupported_event", "type")),
    ('{"type": "output_audio_buffer.clear"}', ("unsupported_event", "type")),
    ('{"type": "input_audio_buffer.commit", "foo": 1}', ("unknown_parameter", "foo")),
    ('{"type": "input_audio_buffer.clear", "audio": ""}', ("unknown_parameter", "audio")),
    ('{"type": "input_audio_buffer.append"}', ("missing_required_parameter", "audio")),
    ('{"type": "input_audio_buffer.append", "audio": 5}', ("invalid_value", "audio")),
    ('{"type": "session.update"}', ("missing_required_parameter", "session")),
    ('{"type": "session.update", "session": {"type": "transcription"}, "event_id": 7}',
     ("invalid_value", "event_id")),
    ("[" * 100_000 + "]" * 100_000, ("invalid_event", None)),
])
def test_invalid_events(text, expected):
    with pytest.raises(APIError) as e:
        parse_event(text)
    assert code(e) == expected


def test_supported_events_parse():
    lines = open("tests/contract/examples/realtime_client.jsonl").read().splitlines()
    assert [parse_event(line)["type"] for line in lines] == [
        "session.update", "input_audio_buffer.append", "input_audio_buffer.commit", "input_audio_buffer.clear"]
    assert len(CLIENT_EVENT_TYPES) == 11


def test_client_event_types_match_the_pinned_sdk_union():
    import typing

    from openai.types.realtime import RealtimeClientEvent

    union = typing.get_args(typing.get_args(RealtimeClientEvent)[0])
    assert CLIENT_EVENT_TYPES == {typing.get_args(m.model_fields["type"].annotation)[0] for m in union}
