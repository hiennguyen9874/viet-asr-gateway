"""Real Uvicorn server around `create_app` (fake engine by default), shared by CPU end-to-end and contract tests.

Audio is written as 16 kHz float WAV, which FFmpeg decodes bit-exactly, so tests can split it the way the server
does. `name` (the default fake `text_fn`) names a chunk by its first sample, identifying file and chunk position.

WebSockets use the production settings (`websockets-sansio`, `WS_MAX_BYTES`). `tls=True` serves HTTPS/WSS with a
fresh self-signed certificate for 127.0.0.1 (`ca` = its path): the JS realtime client only connects to `wss:`.
"""

import base64
import json
import ssl
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx2
import numpy as np
import openai
import soundfile as sf
import uvicorn
from websockets.sync.client import connect

from app.chunking import split
from app.engine import Engine
from app.main import create_app
from app.realtime import WS_MAX_BYTES
from app.transcription import Joiner
from scripts.subtitles import parse_cues as _parse_cues
from tests.fake_engine import config, FakeEngine

KEY = "test-key-0123456789"
AUTH = {"Authorization": f"Bearer {KEY}"}
SR = 16000


def name(w: np.ndarray) -> str:
    """Written-form text identifying a chunk by its first sample (casing and punctuation, like the model)."""
    return f"Mẫu {float(w[0]):.7f}, xong."


def noise(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    envelope = 0.2 + 0.2 * np.sin(np.arange(n) / SR * 2 * np.pi * 0.7)  # quiet points for the splitter
    return (rng.uniform(-1, 1, n) * envelope).astype(np.float32)


def wav_file(path: Path, samples: np.ndarray) -> Path:
    sf.write(path, samples, SR, subtype="FLOAT")
    return path


def expected_deltas(x: np.ndarray, max_chunk_seconds: float) -> list[str]:
    """SSE deltas the server sends for `x` with the `name` text_fn (empty chunks never occur with noise)."""
    j = Joiner()
    return [j.add(name(x[s.start:s.end])) for s in split(x, int(max_chunk_seconds * SR))]


def pcm24(seconds: float, seed: int) -> bytes:
    """Realtime input: PCM16 LE mono 24 kHz noise with quiet points."""
    rng = np.random.default_rng(seed)
    n = int(seconds * 24000)
    envelope = 0.2 + 0.2 * np.sin(np.arange(n) / 24000 * 2 * np.pi * 0.7)
    return (rng.uniform(-1, 1, n) * envelope * 32767).astype("<i2").tobytes()


def expected_turn(pcm: bytes, max_chunk_seconds: float) -> list[str]:
    """Realtime deltas the server sends for one committed turn with the `name` text_fn."""
    from app.realtime import TurnBuffer

    buf = TurnBuffer(len(pcm))
    buf.append(pcm)
    return expected_deltas(buf.commit()[0], max_chunk_seconds)


class WS:
    """Raw realtime client (`websockets`): sends what the SDKs never do and checks per-item event order."""

    def __init__(self, srv, query: str = "?model=viet-asr", headers=AUTH, timeout: float = 5, **kwargs):
        self.conn = connect(srv.ws_url + "/v1/realtime" + query, additional_headers=headers, max_size=None,
                            open_timeout=5, **kwargs)
        self.timeout = timeout  # default for recv/until
        self.events: list[dict] = []
        created = self.recv()
        assert created["type"] == "session.created"

    def send(self, type_: str, **fields) -> None:
        self.conn.send(json.dumps({"type": type_, **fields}))

    def append(self, pcm: bytes, packet: int = 9600, **fields) -> None:
        for i in range(0, len(pcm), packet):
            self.send("input_audio_buffer.append", audio=base64.b64encode(pcm[i:i + packet]).decode(), **fields)

    def recv(self, timeout: float | None = None) -> dict:
        event = json.loads(self.conn.recv(self.timeout if timeout is None else timeout))
        self.events.append(event)
        return event

    def until(self, type_: str, timeout: float | None = None) -> list[dict]:
        """Events up to and including the next one of `type_`."""
        got = []
        while not got or got[-1]["type"] != type_:
            got.append(self.recv(timeout))
        return got

    def turn(self) -> tuple[dict, list[str], dict]:
        """After a commit: (committed, deltas, terminal event), checking the per-item order."""
        committed, added = self.recv(), self.recv()
        assert (committed["type"], added["type"]) == ("input_audio_buffer.committed", "conversation.item.added")
        assert added["item"]["id"] == committed["item_id"]
        assert added["previous_item_id"] == committed["previous_item_id"]
        return (committed, *self.result(committed["item_id"]))

    def result(self, item_id: str) -> tuple[list[str], dict]:
        deltas = []
        while True:
            e = self.recv()
            assert e["item_id"] == item_id, e
            if e["type"] != "conversation.item.input_audio_transcription.delta":
                return deltas, e
            assert e["content_index"] == 0
            deltas.append(e["delta"])

    def close(self) -> None:
        self.conn.close()


def parse_cues(body: str, fmt: str) -> list[tuple[float, float, str]]:
    """Strictly parse a gateway SRT/VTT body into (start s, end s, text) cues."""
    cues = _parse_cues(body, fmt)
    assert cues is not None, body
    return cues


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def self_signed(directory: Path) -> tuple[Path, Path]:
    """(certificate, key) for 127.0.0.1, valid one day."""
    cert, key = directory / "cert.pem", directory / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=127.0.0.1",
                    "-addext", "subjectAltName=IP:127.0.0.1", "-keyout", str(key), "-out", str(cert)],
                   check=True, capture_output=True)
    return cert, key


@contextmanager
def serve(engine: Engine | None = None, wait_ready: bool = True, ready_timeout: float = 5.0,
          http_timeout: float = 10.0, tls_dir: Path | None = None, **overrides) -> Iterator[SimpleNamespace]:
    """Yield `client` (httpx, authorized), `sdk` (OpenAI client), `base_url`, `ws_url`, `app`, `port`, `server`,
    `engine`, `ca` (certificate path with `tls_dir`, else None).

    `engine=None` is a `FakeEngine(text_fn=name)`; config overrides apply to tests.fake_engine.config()."""
    engine = engine or FakeEngine(text_fn=name)
    app = create_app(config(**overrides), engine)
    cert = key = None
    if tls_dir is not None:
        cert, key = self_signed(tls_dir)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", access_log=False,
                                           ws="websockets-sansio", ws_max_size=WS_MAX_BYTES,
                                           ssl_certfile=cert, ssl_keyfile=key))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "server did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    scheme = "https" if cert else "http"
    base = f"{scheme}://127.0.0.1:{port}"
    verify = ssl.create_default_context(cafile=cert) if cert else True
    with httpx2.Client(base_url=base, headers=AUTH, timeout=http_timeout, verify=verify) as client:
        if wait_ready:
            wait_for(lambda: client.get("/ready").status_code == 200, ready_timeout)
        sdk = openai.OpenAI(api_key=KEY, base_url=base + "/v1", max_retries=0, timeout=http_timeout)
        try:
            yield SimpleNamespace(client=client, sdk=sdk, base_url=base, app=app, port=port, server=server,
                                  engine=engine, ca=cert, ws_url=f"{'wss' if cert else 'ws'}://127.0.0.1:{port}")
        finally:
            if isinstance(engine, FakeEngine):
                engine.gate.set()
                engine.load_gate.set()
            client.close()  # an idle keep-alive TLS connection would delay the shutdown (SSL close_notify)
            server.should_exit = True
            thread.join(10)
            assert not thread.is_alive()
