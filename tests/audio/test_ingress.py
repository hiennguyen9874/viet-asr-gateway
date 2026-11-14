"""Multipart ingress over a real Uvicorn server (fake engine): body limits before parsing (Content-Length and chunked),
field flooding, malformed forms, spool budget, and temp-file cleanup on success, error and client disconnect.

A test-only route runs the same ingress + decode path the transcription endpoint (M3) uses.
"""

import errno
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx2
import pytest
import uvicorn
from fastapi import Request

from app.main import create_app
from tests.fake_engine import config, FakeEngine

KEY = "test-key-0123456789"
AUTH = {"Authorization": f"Bearer {KEY}"}
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
WAV = (FIXTURES / "vi_a.wav").read_bytes()  # 568,364 bytes
LIMIT = 700_000


@contextmanager
def serve(**overrides):
    cfg = config(max_upload_bytes=LIMIT, max_spool_bytes=LIMIT, **overrides)
    app = create_app(cfg, FakeEngine())

    @app.post("/v1/_ingest")
    async def ingest(request: Request):
        audio = app.state.audio
        deadline = time.monotonic() + cfg.request_timeout_seconds
        async with audio.receive(request, deadline) as upload:
            wav = await audio.decode(upload.path, deadline)
            return {"fields": upload.fields, "size": upload.size, "samples": len(wav)}

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "server did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    spool = app.state.audio.spool_dir
    assert spool.is_dir() and (spool.stat().st_mode & 0o777) == 0o700
    with httpx2.Client(base_url=f"http://127.0.0.1:{port}", headers=AUTH, timeout=10) as client:
        try:
            yield client, app, port
        finally:
            server.should_exit = True
            thread.join(10)
    assert not thread.is_alive()
    assert not spool.exists()  # removed at shutdown


def spool_empty(app) -> bool:
    return not any(app.state.audio.spool_dir.iterdir()) and app.state.audio._spooled == 0


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    err = response.json()["error"]
    assert err["code"] == code
    return err


def multipart(parts: list[tuple[str, bytes, str | None]], boundary: str = "b0undary") -> bytes:
    out = b""
    for name, value, filename in parts:
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        out += f"--{boundary}\r\nContent-Disposition: {disp}\r\n\r\n".encode() + value + b"\r\n"
    return out + f"--{boundary}--\r\n".encode()


MP_HEADERS = {"Content-Type": "multipart/form-data; boundary=b0undary"}


@pytest.mark.parametrize("target", ["app.audio.tempfile.mkstemp", "app.audio.os.write"])
@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EDQUOT])
def test_disk_exhaustion_returns_busy_and_cleans_up(monkeypatch, target, error_number):
    def full(*args, **kwargs):
        raise OSError(error_number, "Disk capacity exhausted")

    with serve() as (client, app, _):
        monkeypatch.setattr(target, full)
        response = client.post("/v1/_ingest", files={"file": ("audio.wav", WAV)})
        error(response, 429, "server_busy")
        assert spool_empty(app)


def test_upload_fields_and_decode():
    with serve() as (client, app, _):
        r = client.post("/v1/_ingest", files={"file": ("../../etc/passwd", WAV, "audio/wav")},
                        data={"model": "viet-asr", "include[]": ["a", "b"]})
        assert r.status_code == 200, r.text
        assert r.json() == {"fields": [["model", "viet-asr"], ["include[]", "a"], ["include[]", "b"]],
                            "size": len(WAV), "samples": 142080}
        assert spool_empty(app)
        assert app.state.metrics.value("uploads_total", outcome="ok") == 1


def test_chunked_body_without_content_length():
    def body(data: bytes):
        for i in range(0, len(data), 10_000):
            yield data[i:i + 10_000]

    with serve() as (client, app, _):
        ok = multipart([("model", b"m", None), ("file", WAV, "a.wav")])
        r = client.post("/v1/_ingest", content=body(ok), headers=MP_HEADERS)
        assert "content-length" not in r.request.headers and r.request.headers["transfer-encoding"] == "chunked"
        assert r.status_code == 200 and r.json()["samples"] == 142080
        big = multipart([("file", WAV + b"\0" * 200_000, "a.wav")])
        error(client.post("/v1/_ingest", content=body(big), headers=MP_HEADERS), 413, "request_too_large")
        assert spool_empty(app)


def test_declared_oversize_body_is_refused_before_reading():
    with serve() as (client, app, port):
        with socket.create_connection(("127.0.0.1", port)) as s:
            s.sendall((f"POST /v1/_ingest HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\n"
                       f"Content-Type: multipart/form-data; boundary=b0undary\r\n"
                       f"Content-Length: {LIMIT + 1}\r\n\r\n").encode())
            # No body sent: the 413 must come from the declared length alone.
            s.settimeout(5)
            assert s.recv(4096).startswith(b"HTTP/1.1 413")
        assert spool_empty(app)


def test_field_flooding_and_oversized_fields():
    with serve() as (client, app, _):
        flood = multipart([(f"f{i}", b"x", None) for i in range(40)] + [("file", WAV, "a.wav")])
        error(client.post("/v1/_ingest", content=flood, headers=MP_HEADERS), 400, "invalid_value")
        big = multipart([("prompt", b"x" * 10_000, None), ("file", WAV, "a.wav")])
        error(client.post("/v1/_ingest", content=big, headers=MP_HEADERS), 413, "request_too_large")
        headers = (b"--b0undary\r\n" + b"".join(b"X-H%d: v\r\n" % i for i in range(20)) +
                   b'Content-Disposition: form-data; name="model"\r\n\r\nm\r\n--b0undary--\r\n')
        error(client.post("/v1/_ingest", content=headers, headers=MP_HEADERS), 400, "invalid_value")
        assert spool_empty(app)


@pytest.mark.parametrize("body,headers,code,param", [
    (multipart([("model", b"m", None)]), MP_HEADERS, "missing_required_parameter", "file"),
    (multipart([("file", WAV[:1000], "a.wav"), ("file", WAV[:1000], "b.wav")]), MP_HEADERS, "invalid_value", "file"),
    (multipart([("file", WAV, "a.wav")])[:-30], MP_HEADERS, "invalid_value", None),  # no end boundary
    (multipart([("model", b"\xff\xfe", None), ("file", WAV, "a.wav")]), MP_HEADERS, "invalid_value", None),
    (b'{"model": "m"}', {"Content-Type": "application/json"}, "invalid_value", None),
    (b"x", {"Content-Type": "multipart/form-data"}, "invalid_value", None),
    (multipart([("file", b"not audio at all", "a.wav")]), MP_HEADERS, "invalid_audio", "file"),
], ids=["missing-file", "two-files", "truncated", "non-utf8-field", "json", "no-boundary", "not-audio"])
def test_malformed_forms(body, headers, code, param):
    with serve() as (client, app, _):
        err = error(client.post("/v1/_ingest", content=body, headers=headers), 400, code)
        assert err["param"] == param
        assert spool_empty(app)


def partial_upload(port: int, sent: int) -> socket.socket:
    """Open an upload that declares the full WAV but sends only `sent` bytes of it."""
    head = b'--b0undary\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
    total = len(multipart([("file", WAV, "a.wav")]))
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall((f"POST /v1/_ingest HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\n"
               f"Content-Type: multipart/form-data; boundary=b0undary\r\nContent-Length: {total}\r\n\r\n").encode())
    s.sendall(head + WAV[:sent])
    return s


def test_client_disconnect_mid_upload_removes_temp_file():
    with serve() as (_, app, port):
        s = partial_upload(port, 300_000)
        audio = app.state.audio
        wait_for(lambda: audio._spooled == 300_000 and len(list(audio.spool_dir.iterdir())) == 1)
        s.close()
        wait_for(lambda: spool_empty(app))
        wait_for(lambda: app.state.metrics.value("uploads_total", outcome="disconnected") == 1)


def test_spool_budget_is_shared_by_uploads_in_flight():
    with serve() as (client, app, port):
        s = partial_upload(port, 400_000)  # holds 400 KB of the 700 KB budget
        wait_for(lambda: app.state.audio._spooled == 400_000)
        r = client.post("/v1/_ingest", files={"file": ("a.wav", WAV, "audio/wav")})
        error(r, 429, "server_busy")
        assert r.headers["retry-after"] == "1"
        s.close()
        wait_for(lambda: spool_empty(app))
        assert client.post("/v1/_ingest", files={"file": ("a.wav", WAV, "audio/wav")}).status_code == 200


def test_slow_upload_hits_the_request_deadline():
    with serve(request_timeout_seconds=0.5) as (_, app, port):
        s = partial_upload(port, 1000)
        s.settimeout(5)
        assert s.recv(4096).startswith(b"HTTP/1.1 503")
        s.close()
        wait_for(lambda: spool_empty(app))
        assert app.state.metrics.value("uploads_total", outcome="timeout") == 1
