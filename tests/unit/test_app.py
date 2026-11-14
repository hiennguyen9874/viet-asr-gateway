"""Gateway app over a real Uvicorn server (fake engine): probes, auth, request ids, error envelope, drain.

Error bodies are also parsed by the pinned OpenAI SDK, the way clients will see them.
"""

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx2
import openai
import pytest
import uvicorn

from app.config import ConfigError
from app.engine import EngineState
from app.main import create_app
from tests.fake_engine import config, FakeEngine

KEY = "test-key-0123456789"
AUTH = {"Authorization": f"Bearer {KEY}"}


@contextmanager
def serve(engine: FakeEngine, wait_ready: bool = True, setup=None):
    app = create_app(config(), engine)
    if setup:
        setup(app)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "server did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    with httpx2.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
        if wait_ready:
            wait_for(lambda: client.get("/ready").status_code == 200)
        try:
            yield client, app
        finally:
            engine.gate.set()
            engine.load_gate.set()
            server.should_exit = True
            thread.join(10)
            assert not thread.is_alive()


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def assert_envelope(response, status: int, type_: str, code: str) -> dict:
    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-request-id"].startswith("req_")
    body = response.json()
    assert set(body) == {"error"} and set(body["error"]) == {"message", "type", "param", "code"}
    assert (body["error"]["type"], body["error"]["code"]) == (type_, code)
    return body["error"]


def test_readiness_follows_model_load_and_health_stays_up():
    engine = FakeEngine()
    engine.load_gate.clear()
    with serve(engine, wait_ready=False) as (client, _):
        assert client.get("/health").json() == {"status": "ok"}
        r = client.get("/ready")
        assert_envelope(r, 503, "server_error", "model_not_ready")
        assert r.headers["retry-after"] == "5"
        engine.load_gate.set()
        wait_for(lambda: client.get("/ready").status_code == 200)
        assert client.get("/ready").json() == {"status": "ready"}


def test_load_failure_fails_liveness_and_readiness():
    engine = FakeEngine()
    engine.load_error = RuntimeError("no weights")
    with serve(engine, wait_ready=False) as (client, app):
        wait_for(lambda: app.state.worker.state == EngineState.FAILED)
        assert_envelope(client.get("/health"), 503, "server_error", "unhealthy")
        assert_envelope(client.get("/ready"), 503, "server_error", "model_not_ready")


def test_api_key_required_except_probes():
    with serve(FakeEngine()) as (client, _):
        for headers in ({}, {"Authorization": "Bearer wrong-key-0123456789"}, {"Authorization": KEY},
                        {"Authorization": "Basic " + KEY}, {"Authorization": "Bearer"}):
            assert_envelope(client.get("/metrics", headers=headers), 401, "authentication_error", "invalid_api_key")
        # Unknown routes do not reveal themselves without a key.
        assert_envelope(client.get("/v1/nope"), 401, "authentication_error", "invalid_api_key")
        assert client.get("/metrics", headers={"Authorization": f"bearer {KEY}"}).status_code == 200


def test_error_envelopes_never_use_fastapi_defaults():
    def setup(app):
        @app.get("/v1/_probe")
        async def probe(n: int):
            return {"n": n}

        @app.get("/v1/_boom")
        async def boom():
            raise RuntimeError("unexpected")

    with serve(FakeEngine(), setup=setup) as (client, _):
        assert_envelope(client.get("/v1/nope", headers=AUTH), 404, "invalid_request_error", "not_found")
        assert_envelope(client.post("/health"), 405, "invalid_request_error", "method_not_allowed")
        err = assert_envelope(client.get("/v1/_probe?n=abc", headers=AUTH), 400, "invalid_request_error",
                              "invalid_value")
        assert err["param"] == "n"
        err = assert_envelope(client.get("/v1/_boom", headers=AUTH), 500, "server_error", "internal_error")
        assert "unexpected" not in err["message"]


def test_openai_sdk_parses_gateway_errors():
    with serve(FakeEngine()) as (client, _):
        base = str(client.base_url).rstrip("/") + "/v1"
        bad = openai.OpenAI(api_key="wrong-key-0123456789", base_url=base, max_retries=0)
        with pytest.raises(openai.AuthenticationError) as e:
            bad.get("/anything", cast_to=object)
        assert e.value.code == "invalid_api_key" and e.value.request_id.startswith("req_")
        good = openai.OpenAI(api_key=KEY, base_url=base, max_retries=0)
        with pytest.raises(openai.NotFoundError) as e:
            good.get("/anything", cast_to=object)
        assert e.value.code == "not_found"


def test_metrics_and_logs_carry_no_secrets(capfd):
    with serve(FakeEngine()) as (client, _):
        client.get("/metrics", headers={"Authorization": "Bearer wrong-key-0123456789"})
        text = client.get("/metrics", headers=AUTH).text
    assert "gateway_ready 1.0" in text and 'gateway_http_requests_total{status="401"}' in text
    logs = capfd.readouterr().err
    entries = [json.loads(line) for line in logs.splitlines() if line.startswith("{")]
    requests = [e for e in entries if e["event"] == "http.request"]
    assert requests and all(e["request_id"].startswith("req_") for e in requests)
    assert {"server.started", "engine.loaded", "server.stopped"} <= {e["event"] for e in entries}
    assert KEY not in logs + text and "wrong-key" not in logs + text


def test_shutdown_drains_and_stops_worker(capfd):
    engine = FakeEngine()
    with serve(engine) as (_, app):
        pass
    assert app.state.worker.state == EngineState.STOPPED
    stopped = [json.loads(line) for line in capfd.readouterr().err.splitlines() if '"server.stopped"' in line]
    assert stopped and stopped[0]["drained"] is True and stopped[0]["worker_stopped"] is True


@pytest.mark.parametrize("denial_extension", [True, False])
def test_websocket_handshake_refused_without_key(denial_extension):
    app = create_app(config(), FakeEngine())
    scope = {"type": "websocket", "path": "/v1/realtime", "headers": [], "query_string": b"",
             "extensions": {"websocket.http.response": {}} if denial_extension else {}}
    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    if denial_extension:
        assert sent[0]["type"] == "websocket.http.response.start" and sent[0]["status"] == 401
        assert any(k == b"x-request-id" for k, _ in sent[0]["headers"])
        assert json.loads(sent[1]["body"])["error"]["code"] == "invalid_api_key"
    else:
        assert sent == [{"type": "websocket.close", "code": 1008}]


def test_real_engine_requires_staged_models_dir(tmp_path: Path):
    with pytest.raises(ConfigError):
        create_app(config(models_dir=tmp_path / "missing"))


def test_startup_requires_spool_dir_and_ffmpeg(tmp_path: Path, monkeypatch):
    with pytest.raises(ConfigError, match="GATEWAY_SPOOL_DIR"):
        create_app(config(spool_dir=tmp_path / "missing"), FakeEngine())
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ConfigError, match="ffmpeg"):
        create_app(config(), FakeEngine())
