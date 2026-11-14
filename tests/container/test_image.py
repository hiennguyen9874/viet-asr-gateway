"""R1 container gate (docs/container-gate.vi.md): build, non-root, no secrets or extra files in the image, offline
start to /ready with the models mounted read-only, smoke against the baseline, graceful SIGTERM, restart.

Needs Docker and staged models: uv run pytest tests/container
The gateway runs with `--network none`; requests come from a sidecar container sharing its network namespace, with
this repository mounted read-only (scripts/smoke_server.py + fixtures). The smoke report is written to
docs/baseline/server-container.json (metadata only).
"""

import json
import os
import secrets
import selectors
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("GATEWAY_MODELS_DIR", ROOT / "models")).resolve()
IMAGE = "viet-asr-gateway:test"
KEY = "ctr-" + secrets.token_hex(16)
pytestmark = [pytest.mark.container,
              pytest.mark.skipif(shutil.which("docker") is None, reason="needs docker")]


def docker(*args: str, check: bool = True, timeout: float = 120, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True, timeout=timeout, **kw)


@pytest.fixture(scope="module")
def image() -> str:
    docker("build", "-t", IMAGE, str(ROOT), timeout=3600)
    return IMAGE


def run_in_image(image: str, script: str) -> str:
    return docker("run", "--rm", "--network", "none", "--entrypoint", "sh", image, "-c", script).stdout


def test_runs_as_non_root(image):
    assert run_in_image(image, "id -u").strip() == "10001"


def test_image_contains_only_the_app(image):
    listing = run_in_image(image, "ls -A /srv").split()
    assert sorted(listing) == [".venv", "app"]
    for path in ("/srv/assets", "/srv/references", "/srv/models", "/srv/tests", "/srv/.git", "/srv/fixtures-large", "/models"):
        assert run_in_image(image, f"test -e {path} && echo present || echo absent").strip() == "absent", path
    versions = run_in_image(image, "python --version; ffmpeg -version | head -1")
    assert "Python 3.12.14" in versions and "ffmpeg version 6.1.1" in versions
    assert run_in_image(image, "python -c 'import sherpa_onnx'") == ""
    for module in ("torch", "transformers"):
        assert run_in_image(image, f"python -c 'import {module}' 2>/dev/null && echo present || echo absent").strip() == "absent"


def test_no_secrets_in_image_metadata(image):
    meta = docker("inspect", image).stdout + docker("history", "--no-trunc", image).stdout
    config_env = json.loads(docker("inspect", image).stdout)[0]["Config"]["Env"]
    assert not any(e.startswith(("HF_TOKEN=", "GATEWAY_API_KEYS=", "GATEWAY_API_KEYS_FILE=")) for e in config_env)
    assert "HF_TOKEN" not in meta and "GATEWAY_API_KEYS" not in meta
    hf_token = os.environ.get("HF_TOKEN")
    assert not hf_token or hf_token not in meta


def models_staged() -> None:
    assert (MODELS / "gipformer1.5-68M-rnnt").is_dir(), "models not staged; run scripts/stage_models.py"


def wait_ready(cid: str, timeout: float = 600) -> None:
    probe = "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18001/ready', timeout=4)"
    deadline = time.monotonic() + timeout
    while docker("exec", cid, "python", "-c", probe, check=False).returncode != 0:
        state = json.loads(docker("inspect", cid).stdout)[0]["State"]
        logs = docker("logs", cid, check=False).stderr
        assert state["Running"], logs[-4000:]
        assert '"event": "engine.load_failed"' not in logs, logs[-6000:]
        assert time.monotonic() < deadline, "container did not become ready"
        time.sleep(2)


def sidecar(cid: str, image: str, *argv: str, timeout: float = 900, detach: bool = False):
    cmd = ["docker", "run", "--rm", "--network", f"container:{cid}", "-v", f"{ROOT}:/w:ro",
           "-e", f"OPENAI_API_KEY={KEY}", "--entrypoint", "python", image, *argv]
    if detach:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# Streams 6 x vi_long.wav (~640 s, ~22 chunks) and prints the event types.
STREAM_LONG = """
import io, json, os, urllib.request, uuid, wave
src = wave.open('/w/fixtures-large/vi_long.wav'); frames = src.readframes(src.getnframes())
buf = io.BytesIO(); out = wave.open(buf, 'wb'); out.setparams(src.getparams()); out.writeframes(frames * 6); out.close()
b = uuid.uuid4().hex
body = (f'--{b}\\r\\nContent-Disposition: form-data; name="model"\\r\\n\\r\\nviet-asr\\r\\n'
        f'--{b}\\r\\nContent-Disposition: form-data; name="stream"\\r\\n\\r\\ntrue\\r\\n'
        f'--{b}\\r\\nContent-Disposition: form-data; name="file"; filename="a.wav"\\r\\n\\r\\n').encode()
body += buf.getvalue() + f'\\r\\n--{b}--\\r\\n'.encode()
req = urllib.request.Request('http://127.0.0.1:18001/v1/audio/transcriptions', data=body, method='POST', headers={
    'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'], 'Content-Type': f'multipart/form-data; boundary={b}'})
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        if line.strip():
            e = json.loads(line.decode().removeprefix('data: '))
            print(e.get('type', 'error'), flush=True)
"""


@contextmanager
def running_container(image, *env):
    models_staged()
    # Exercise the same file-secret path as compose, without putting the key in container metadata.
    with tempfile.TemporaryDirectory(prefix="gateway-gate-") as tmp:
        key_file = Path(tmp) / "api_keys"
        key_file.write_text(KEY + "\n")
        key_file.chmod(0o444)  # bind mount must be readable by uid 10001
        args = ["run", "-d", "--network", "none", "--read-only",
                "--tmpfs", "/tmp:size=512m,mode=1777", "--mount", f"type=bind,src={MODELS},dst=/models,readonly",
                "--mount", f"type=bind,src={key_file},dst=/run/secrets/gateway_api_keys,readonly",
                "-e", "GATEWAY_API_KEYS_FILE=/run/secrets/gateway_api_keys"]
        for value in env:
            args += ["-e", value]
        cid = docker(*args, image).stdout.strip()
        try:
            wait_ready(cid)
            assert KEY not in docker("inspect", cid).stdout
            yield cid
        finally:
            docker("rm", "-f", cid, check=False)


def test_offline_ready_and_smoke(image):
    with running_container(image) as cid:
        r = sidecar(cid, image, "/w/scripts/smoke_server.py", "--base-url", "http://127.0.0.1:18001")
        assert r.returncode == 0, r.stderr[-4000:]
        report = json.loads(r.stdout)
        report["image"] = json.loads(docker("inspect", image).stdout)[0]["Id"]
        report["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                          text=True, check=True).stdout.strip()
        report["source_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
        (ROOT / "docs/baseline/server-container.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


# Two idle realtime sessions deterministically occupy both lanes: no timing-dependent saturation.
ERRORS_AND_ISOLATION = """
import concurrent.futures, hashlib, json, os, sys, urllib.error
sys.path.insert(0, '/w')
from scripts.smoke_server import Gateway
from websockets.sync.client import connect
gw = Gateway('http://127.0.0.1:18001', os.environ['OPENAI_API_KEY'], 120)
def error(call, status, code):
    try:
        call()
    except urllib.error.HTTPError as e:
        assert e.code == status, e.code
        assert json.load(e)['error']['code'] == code
        assert e.headers['x-request-id'].startswith('req_')
    else:
        raise AssertionError('expected HTTP error')
error(lambda: gw.get_json('/v1/models', auth=False), 401, 'invalid_api_key')
error(lambda: gw.get_json('/v1/models/not-a-model'), 404, 'model_not_found')
audio = open('/w/tests/fixtures/vi_a.wav', 'rb').read()
error(lambda: gw.transcribe(audio, model='not-a-model'), 404, 'model_not_found')
error(lambda: gw.transcribe(b'not audio'), 400, 'invalid_audio')
headers = {'Authorization': 'Bearer ' + gw.key}
url = 'ws://127.0.0.1:18001/v1/realtime?model=viet-asr'
with connect(url, additional_headers=headers) as a, connect(url, additional_headers=headers) as b:
    assert json.loads(a.recv(30))['type'] == 'session.created'
    assert json.loads(b.recv(30))['type'] == 'session.created'
    error(lambda: gw.transcribe(audio), 429, 'server_busy')
# Closing a socket releases its lane asynchronously; wait for the active-lane gauge, not a fixed sleep.
import time
for _ in range(100):
    metrics = gw.request('GET', '/metrics').read().decode()
    if 'gateway_open_lanes 0' in metrics:
        break
    time.sleep(.1)
else:
    raise AssertionError('lanes not released')
offline = json.load(open('/w/docs/baseline/offline.json'))
baseline = offline['a_settings']['rows'][offline['default']]
def transcribe(name):
    data = open('/w/tests/fixtures/' + name, 'rb').read()
    with gw.transcribe(data) as r:
        assert hashlib.sha256(json.load(r)['text'].encode()).hexdigest() == baseline[name]['text_sha256']
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    list(pool.map(transcribe, ['vi_a.wav', 'vi_b.wav']))
print('auth/errors/overload/isolation passed')
"""


def test_errors_overload_and_isolation(image):
    with running_container(image, "GATEWAY_QUEUE_MAX_LANES=2", "GATEWAY_MAX_SESSIONS=2") as cid:
        r = sidecar(cid, image, "-c", ERRORS_AND_ISOLATION)
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        assert KEY not in docker("logs", cid).stderr


def test_stream_drain_and_restart(image):
    if not (ROOT / "fixtures-large/vi_long.wav").is_file():
        pytest.fail("drain/restart gate needs fixtures-large/vi_long.wav; run scripts/fetch_fixtures.py")
    with running_container(image) as cid:
        stream = sidecar(cid, image, "-c", STREAM_LONG, detach=True)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(stream.stdout, selectors.EVENT_READ)
                assert selector.select(timeout=120), "stream produced no first event within 120s"
                assert stream.stdout.readline().strip() == "transcript.text.delta"  # stream is running
            started = time.monotonic()
            docker("stop", "-t", "70", cid, timeout=120)
            out, err = stream.communicate(timeout=120)
            assert stream.returncode == 0, err[-4000:]
            assert out.strip().splitlines()[-1] == "transcript.text.done"  # finished while draining
            assert time.monotonic() - started < 70
            state = json.loads(docker("inspect", cid).stdout)[0]["State"]
            assert state["ExitCode"] == 0
            logs = docker("logs", cid).stderr
            stopped = [json.loads(line) for line in logs.splitlines() if '"server.stopped"' in line]
            assert stopped and stopped[-1]["drained"] is True
            assert KEY not in logs

            docker("start", cid)
            wait_ready(cid)
        finally:
            if stream.poll() is None:
                stream.kill()
            stream.communicate(timeout=120)
