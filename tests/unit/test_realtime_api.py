"""Realtime WebSocket end to end (docs/api-contract.vi.md §6): real Uvicorn (`websockets-sansio`), fake engine,
raw `websockets` client (it can send what the SDKs never do: binary frames, bad JSON, half samples)."""

import base64
import json
import socket
import threading
import time

import pytest
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from app.realtime import WS_MAX_BYTES
from app.transcription import MODEL_ID
from tests.fake_engine import FakeEngine
from tests.live_server import AUTH, expected_turn, name, noise, pcm24, serve, wait_for, wav_file, WS

CHUNK = 1.0  # max_chunk_seconds: turns over 1 s become several chunks of one item


@pytest.fixture
def srv():
    with serve(max_chunk_seconds=CHUNK) as s:
        yield s


def error(event: dict) -> tuple[str, str | None]:
    assert event["type"] == "error", event
    return event["error"]["code"], event["error"]["param"]


def settled(srv) -> None:
    """No session, lane, job or audio reservation left."""
    state = srv.app.state
    wait_for(lambda: state.scheduler._lanes == 0 and state.scheduler._jobs == 0 and state.audio._held == 0)
    wait_for(lambda: "gateway_realtime_sessions 0.0\n" in srv.client.get("/metrics").text)


# -- handshake ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("query,headers,status,code", [
    ("", {}, 401, "invalid_api_key"),
    ("", {"Authorization": "Bearer wrong-key-0123456789"}, 401, "invalid_api_key"),
    ("?model=whisper-1", AUTH, 404, "model_not_found"),
    ("?model=", AUTH, 404, "model_not_found"),
    ("?intent=conversation", AUTH, 400, "invalid_value"),
    ("?call_id=rtc_1", AUTH, 400, "unsupported_parameter"),
    ("?model=viet-asr&model=viet-asr", AUTH, 400, "invalid_value"),
])
def test_handshake_refused_before_accept(srv, query, headers, status, code):
    with pytest.raises(InvalidStatus) as e:
        connect(srv.ws_url + "/v1/realtime" + query, additional_headers=headers, open_timeout=5)
    r = e.value.response
    assert r.status_code == status and r.headers["x-request-id"].startswith("req_")
    assert json.loads(r.body)["error"]["code"] == code
    settled(srv)


def test_intent_and_no_query_open_a_session(srv):
    for query in ("?intent=transcription", ""):
        ws = WS(srv, query)
        ws.close()
    settled(srv)


def test_full_not_ready_and_draining_are_refused():
    engine = FakeEngine(text_fn=name)
    with serve(engine, max_sessions=1) as s:
        ws = WS(s)
        with pytest.raises(InvalidStatus) as e:
            connect(s.ws_url + "/v1/realtime", additional_headers=AUTH, open_timeout=5)
        assert e.value.response.status_code == 429 and e.value.response.headers["retry-after"] == "1"
        ws.close()
        wait_for(lambda: s.app.state.scheduler._lanes == 0)
        s.app.state.scheduler._closing = True  # what the lifespan sets when draining
        with pytest.raises(InvalidStatus) as e:
            connect(s.ws_url + "/v1/realtime", additional_headers=AUTH, open_timeout=5)
        assert e.value.response.status_code == 503 and b"shutting_down" in e.value.response.body
    engine = FakeEngine(text_fn=name)
    engine.load_gate.clear()
    with serve(engine, wait_ready=False) as s:
        with pytest.raises(InvalidStatus) as e:
            connect(s.ws_url + "/v1/realtime", additional_headers=AUTH, open_timeout=5)
        assert e.value.response.status_code == 503 and b"model_not_ready" in e.value.response.body


# -- turns and items ----------------------------------------------------------------------------------------------

def test_commits_on_one_socket(srv):
    ws = WS(srv)
    ws.send("session.update", event_id="u1", session={"type": "transcription", "audio": {"input": {
        "format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None}}})
    assert ws.recv()["type"] == "session.updated"
    turns = [pcm24(0.7, seed=1), pcm24(3.5, seed=2), pcm24(1.3, seed=3)]  # 3.5 s: several chunks, one item
    ids, previous = [], None
    for pcm in turns:
        ws.append(pcm, packet=4801)  # odd packet sizes: samples split across messages
        ws.send("input_audio_buffer.commit")
        committed, deltas, done = ws.turn()
        assert committed["previous_item_id"] == previous
        assert done["type"] == "conversation.item.input_audio_transcription.completed"
        assert deltas == expected_turn(pcm, CHUNK) and "".join(deltas) == done["transcript"]
        assert done["usage"] == {"type": "duration", "seconds": len(pcm) / 2 / 24000}
        previous = committed["item_id"]
        ids.append(previous)
    assert len(expected_turn(turns[1], CHUNK)) >= 3
    assert all(i.startswith("item_") for i in ids) and len(set(ids)) == 3
    event_ids = [e["event_id"] for e in ws.events]
    assert len(set(event_ids)) == len(event_ids) and all(i.startswith("event_") for i in event_ids)
    ws.close()
    settled(srv)


def test_next_turn_is_accepted_while_the_previous_item_runs():
    engine = FakeEngine(text_fn=name)
    with serve(engine, max_chunk_seconds=CHUNK) as s:
        ws = WS(s)
        a, b = pcm24(2.5, seed=4), pcm24(0.5, seed=5)
        engine.gate.clear()
        ws.append(a)
        ws.send("input_audio_buffer.commit")
        first = ws.until("conversation.item.added")[0]
        assert engine.started.acquire(timeout=5)  # item A is on the GPU
        ws.append(b)
        ws.send("input_audio_buffer.commit")
        second = ws.until("conversation.item.added")[0]
        assert second["previous_item_id"] == first["item_id"]
        engine.gate.set()
        rest = ws.until("conversation.item.input_audio_transcription.completed")
        rest += ws.until("conversation.item.input_audio_transcription.completed")
        done = [e for e in rest if e["type"].endswith("completed")]
        assert [d["item_id"] for d in done] == [first["item_id"], second["item_id"]]  # FIFO
        assert [d["transcript"] for d in done] == ["".join(expected_turn(p, CHUNK)) for p in (a, b)]
        assert engine.max_active == 1
        ws.close()
        settled(s)


def test_clear_while_previous_item_infers():
    engine = FakeEngine(text_fn=name)
    with serve(engine, max_chunk_seconds=CHUNK) as s:
        ws = WS(s)
        a = pcm24(1.5, seed=6)
        engine.gate.clear()
        ws.append(a)
        ws.send("input_audio_buffer.commit")
        ws.until("conversation.item.added")
        ws.append(pcm24(0.4, seed=7) + b"\x01")
        ws.send("input_audio_buffer.clear")
        assert ws.recv()["type"] == "input_audio_buffer.cleared"
        ws.send("input_audio_buffer.commit")
        assert error(ws.recv()) == ("input_audio_buffer_commit_empty", None)
        engine.gate.set()
        done = ws.until("conversation.item.input_audio_transcription.completed")[-1]
        assert done["transcript"] == "".join(expected_turn(a, CHUNK))
        ws.close()
        settled(s)


def test_empty_and_short_commits(srv):
    ws = WS(srv)
    ws.send("input_audio_buffer.commit", event_id="c1")
    e = ws.recv()
    assert error(e) == ("input_audio_buffer_commit_empty", None) and e["error"]["event_id"] == "c1"
    ws.send("input_audio_buffer.append", audio=base64.b64encode(b"\x01").decode())
    ws.send("input_audio_buffer.commit")  # one byte: no whole sample
    assert error(ws.recv())[0] == "input_audio_buffer_commit_empty"
    calls = len(srv.engine.calls)
    for pcm in (b"\x02", pcm24(0.1, seed=8)):  # completes the kept byte (1 sample); exactly 100 ms
        ws.append(pcm)
        ws.send("input_audio_buffer.commit")
        committed, deltas, done = ws.turn()
        assert deltas == [] and done["transcript"] == ""
        assert done["usage"]["seconds"] == (1 if pcm == b"\x02" else 2400) / 24000
    assert len(srv.engine.calls) == calls  # the model is never called for <= 100 ms
    ws.close()
    settled(srv)


def test_bad_append_leaves_the_buffer_unchanged():
    with serve(max_chunk_seconds=CHUNK, max_turn_seconds=1.0) as s:
        ws = WS(s)
        pcm = pcm24(0.6, seed=9)
        ws.append(pcm[:4801])
        for audio, expected in (("***", "invalid_audio"), ("QUJD=", "invalid_audio"), ("ñ", "invalid_audio")):
            ws.send("input_audio_buffer.append", event_id="bad", audio=audio)
            e = ws.recv()
            assert error(e) == (expected, "audio") and e["error"]["event_id"] == "bad"
        ws.send("input_audio_buffer.append", audio=5)
        assert error(ws.recv()) == ("invalid_value", "audio")
        ws.append(pcm24(0.95, seed=10), packet=10**9)  # 0.1 s pending + 0.95 s > max_turn_seconds 1.0
        assert error(ws.recv()) == ("input_audio_buffer_full", None)
        ws.append(pcm[4801:])
        ws.send("input_audio_buffer.commit")
        _, deltas, done = ws.turn()
        assert done["transcript"] == "".join(expected_turn(pcm, CHUNK))
        ws.close()
        settled(s)


def test_invalid_and_unsupported_events_keep_the_socket(srv):
    ws = WS(srv)
    ws.conn.send(b"\x00\x01")
    assert error(ws.recv())[0] == "invalid_event"
    ws.conn.send("{not json")
    assert error(ws.recv()) == ("invalid_event", None)
    for type_ in ([], {}, None, 123):
        ws.conn.send(json.dumps({"type": type_}))
        assert error(ws.recv()) == ("invalid_event", "type")
    ws.send("response.create", event_id="r1")
    e = ws.recv()
    assert error(e) == ("unsupported_event", "type") and e["error"]["event_id"] == "r1"
    ws.send("transcription_session.update", session={})
    assert error(ws.recv()) == ("invalid_event", "type")
    ws.send("session.update", session={"type": "transcription", "audio": {"input": {
        "turn_detection": {"type": "server_vad"}}}})
    assert error(ws.recv()) == ("unsupported_turn_detection", "session.audio.input.turn_detection")
    ws.send("session.update", session={"type": "transcription", "audio": {"input": {"format": {
        "type": "audio/pcmu"}}}})
    assert error(ws.recv()) == ("unsupported_parameter", "session.audio.input.format.type")
    ws.send("session.update", session={"type": "transcription", "audio": {"input": {"transcription": {
        "model": MODEL_ID, "language": "vi"}}}})
    updated = ws.recv()
    assert updated["type"] == "session.updated"
    assert updated["session"]["audio"]["input"]["transcription"] == {"model": MODEL_ID, "language": "vi"}
    pcm = pcm24(0.5, seed=11)
    ws.append(pcm)
    ws.send("input_audio_buffer.commit")
    assert ws.turn()[2]["transcript"] == "".join(expected_turn(pcm, CHUNK))
    metrics = srv.client.get("/metrics").text
    assert 'gateway_realtime_client_events_total{type="invalid"} 7.0' in metrics
    assert 'gateway_realtime_client_events_total{type="response.create"} 1.0' in metrics
    ws.close()
    settled(srv)


def test_too_many_pending_items_keeps_the_buffer():
    engine = FakeEngine(text_fn=name)
    with serve(engine, max_chunk_seconds=CHUNK, max_pending_items=2) as s:
        ws = WS(s)
        turns = [pcm24(0.3, seed=s_) for s_ in (12, 13, 14)]
        engine.gate.clear()
        for pcm in turns[:2]:
            ws.append(pcm)
            ws.send("input_audio_buffer.commit")
            ws.until("conversation.item.added")
        ws.append(turns[2])
        ws.send("input_audio_buffer.commit")
        assert error(ws.recv())[0] == "too_many_pending_items"
        engine.gate.set()
        ws.until("conversation.item.input_audio_transcription.completed")
        ws.until("conversation.item.input_audio_transcription.completed")
        ws.send("input_audio_buffer.commit")  # the refused turn is still pending
        assert ws.turn()[2]["transcript"] == "".join(expected_turn(turns[2], CHUNK))
        ws.close()
        settled(s)


def test_failed_items_do_not_stop_the_session():
    engine = FakeEngine(text_fn=name)
    with serve(engine, max_chunk_seconds=CHUNK, request_timeout_seconds=0.5) as s:
        ws = WS(s)
        engine.fail_calls = 1
        ws.append(pcm24(0.5, seed=15))
        ws.send("input_audio_buffer.commit")
        _, _, failed = ws.turn()
        assert failed["type"] == "conversation.item.input_audio_transcription.failed"
        assert failed["error"] == {"type": "server_error", "code": "inference_failed", "message": "Inference failed."}
        wait_for(lambda: s.app.state.worker.ready)
        engine.gate.clear()
        ws.append(pcm24(2.5, seed=16))
        ws.send("input_audio_buffer.commit")
        _, _, failed = ws.turn()  # the 3-chunk item outlives its deadline
        assert failed["error"]["code"] == "request_timeout"
        held = s.app.state.audio._held
        assert held > 0  # the running call still reads the item's waveform
        engine.gate.set()
        pcm = pcm24(0.5, seed=17)
        ws.append(pcm)
        ws.send("input_audio_buffer.commit")
        assert ws.turn()[2]["transcript"] == "".join(expected_turn(pcm, CHUNK))
        assert engine.max_active == 1
        ws.close()
        settled(s)


def test_short_queued_item_obeys_its_commit_deadline():
    engine = FakeEngine(text_fn=name)
    with serve(engine, request_timeout_seconds=0.3) as s:
        ws = WS(s)
        engine.gate.clear()
        try:
            ws.append(pcm24(0.5, seed=31))
            ws.send("input_audio_buffer.commit")
            ws.until("conversation.item.added")
            assert engine.started.acquire(timeout=5)
            ws.append(pcm24(0.05, seed=32))
            ws.send("input_audio_buffer.commit")
            second = ws.until("conversation.item.added")[-1]
            first_failed = ws.until("conversation.item.input_audio_transcription.failed")[-1]
            assert first_failed["error"]["code"] == "request_timeout"
            time.sleep(0.4)  # second item expires while the first call still owns the lane
            engine.gate.set()
            second_failed = ws.until("conversation.item.input_audio_transcription.failed")[-1]
            assert second_failed["item_id"] == second["item"]["id"]
            assert second_failed["error"]["code"] == "request_timeout"
        finally:
            engine.gate.set()
            ws.close()
        settled(s)


def test_session_expiry_waits_for_offloaded_append(monkeypatch):
    from app.realtime import OFFLOAD_CHARS, TurnBuffer

    started, finish = threading.Event(), threading.Event()
    append = TurnBuffer.append

    def blocked_append(buffer, pcm):
        started.set()
        assert finish.wait(5)
        append(buffer, pcm)

    monkeypatch.setattr(TurnBuffer, "append", blocked_append)
    with serve(max_session_seconds=0.3) as s:
        ws = WS(s)
        try:
            ws.append(b"\x00\x00" * OFFLOAD_CHARS, packet=2 * OFFLOAD_CHARS)
            assert started.wait(5)
            time.sleep(0.5)  # expiry cancels the receiver, but its thread still owns audio
            assert s.app.state.audio._held > 0
            assert s.app.state.scheduler._lanes == 1
            finish.set()
            assert error(ws.recv())[0] == "session_expired"
        finally:
            finish.set()
            ws.close()
        settled(s)


def test_disconnect_during_inference_frees_everything_after_the_call():
    engine = FakeEngine(text_fn=name)
    with serve(engine, max_chunk_seconds=CHUNK) as s:
        ws = WS(s)
        engine.gate.clear()
        ws.append(pcm24(2.5, seed=18))
        ws.append(pcm24(0.5, seed=19))  # pending turn, never committed
        ws.send("input_audio_buffer.commit")
        ws.until("conversation.item.added")
        assert engine.started.acquire(timeout=5)
        ws.close()
        state = s.app.state
        wait_for(lambda: state.scheduler._lanes == 0)
        time.sleep(0.1)
        assert state.audio._held > 0 and state.scheduler._jobs == 1  # the running call keeps its audio
        engine.gate.set()
        settled(s)
        assert len(engine.calls) == 1 and engine.max_active == 1  # queued chunks were cancelled


def test_two_sessions_and_rest_do_not_mix(tmp_path):
    engine = FakeEngine(text_fn=name, delay=0.01)
    with serve(engine, max_chunk_seconds=CHUNK) as s:
        audio = [pcm24(2.2, seed=20), pcm24(1.7, seed=21)]
        results: dict[int, str] = {}
        rest = noise(2.5, seed=22)
        path = wav_file(tmp_path / "a.wav", rest)

        def session(i: int) -> None:
            ws = WS(s)
            ws.append(audio[i], packet=960)
            ws.send("input_audio_buffer.commit")
            results[i] = ws.turn()[2]["transcript"]
            ws.close()

        threads = [threading.Thread(target=session, args=(i,)) for i in (0, 1)]
        for t in threads:
            t.start()
        r = s.client.post("/v1/audio/transcriptions", files={"file": ("a.wav", path.read_bytes())},
                          data={"model": MODEL_ID})
        for t in threads:
            t.join(10)
        assert r.status_code == 200
        from tests.live_server import expected_deltas
        assert r.json()["text"] == "".join(expected_deltas(rest, CHUNK))
        assert results == {i: "".join(expected_turn(audio[i], CHUNK)) for i in (0, 1)}
        assert engine.max_active == 1
        settled(s)


# -- timeouts and limits ------------------------------------------------------------------------------------------

def test_idle_timeout_waits_for_pending_items():
    engine = FakeEngine(text_fn=name)
    with serve(engine, max_chunk_seconds=CHUNK, session_idle_timeout_seconds=0.3) as s:
        ws = WS(s)
        engine.gate.clear()
        ws.append(pcm24(0.5, seed=23))
        ws.send("input_audio_buffer.commit")
        ws.until("conversation.item.added")
        time.sleep(0.8)  # idle, but a transcript is pending: still open
        engine.gate.set()
        assert ws.until("conversation.item.input_audio_transcription.completed")
        start = time.monotonic()
        assert error(ws.recv(timeout=3))[0] == "idle_timeout"
        assert 0.2 < time.monotonic() - start < 1.5
        with pytest.raises(ConnectionClosed) as e:
            ws.recv()
        assert e.value.rcvd.code == 1000
        settled(s)


def test_session_expires():
    with serve(max_chunk_seconds=CHUNK, max_session_seconds=0.5) as s:
        ws = WS(s)
        ws.append(pcm24(0.2, seed=24))  # pending audio is dropped
        assert error(ws.recv(timeout=3))[0] == "session_expired"
        with pytest.raises(ConnectionClosed) as e:
            ws.recv()
        assert e.value.rcvd.code == 1000
        settled(s)
        assert 'gateway_realtime_sessions_total{outcome="session_expired"} 1.0' in s.client.get("/metrics").text


def test_message_over_the_limit_closes_1009(srv):
    ws = WS(srv)
    big = base64.b64encode(b"\x00" * (WS_MAX_BYTES * 3 // 4)).decode()
    ws.send("input_audio_buffer.append", audio=big)
    with pytest.raises(ConnectionClosed) as e:
        ws.recv()
    assert e.value.rcvd.code == 1009
    settled(srv)


def test_large_append_is_processed_off_the_event_loop(srv):
    """An append near the message limit (~32 s) goes through the thread offload and still resamples exactly."""
    ws = WS(srv)
    pcm = pcm24(30.0, seed=25)
    ws.append(pcm, packet=len(pcm))
    ws.send("input_audio_buffer.commit")
    _, deltas, done = ws.turn()
    assert deltas == expected_turn(pcm, CHUNK)
    ws.close()
    settled(srv)


def test_decoded_audio_budget_refuses_the_append():
    with serve(max_chunk_seconds=1.0, max_audio_seconds=1.0, max_decoded_seconds=1.0) as s:
        ws = WS(s)
        ws.append(pcm24(1.2, seed=26), packet=10**9)
        e = ws.recv()
        assert error(e) == ("server_busy", None) and e["error"]["type"] == "rate_limit_error"
        ws.send("input_audio_buffer.commit")
        assert error(ws.recv())[0] == "input_audio_buffer_commit_empty"
        ws.close()
        settled(s)


def test_client_that_stops_reading_is_dropped():
    """The session ends at the send timeout and frees its lane and audio. (Uvicorn then closes the transport only
    once its buffered bytes drain, so the test drops the stuck socket itself.)"""
    with serve(max_chunk_seconds=CHUNK, send_timeout_seconds=0.3) as s:
        sock = socket.create_connection(("127.0.0.1", s.port))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        ws = WS(s, sock=sock)
        ws.append(pcm24(0.5, seed=27))  # pending audio holds a reservation
        update = json.dumps({"type": "session.update", "session": {"type": "transcription"}})

        def flood() -> None:  # never reads the session.updated replies
            try:
                while True:
                    ws.conn.send(update)
            except (ConnectionClosed, OSError):
                pass

        sender = threading.Thread(target=flood, daemon=True)
        sender.start()
        metric = 'gateway_realtime_sessions_total{outcome="send_timeout"} 1.0'
        wait_for(lambda: metric in s.client.get("/metrics").text, timeout=30)
        settled(s)
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        sender.join(5)
