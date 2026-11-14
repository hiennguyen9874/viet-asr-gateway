"""Smoke test of a running gateway (local process or container): readiness, models, json/text/SSE and
verbose_json/srt/vtt on the fixtures, transcripts compared with the offline baseline, timings -> metadata-only JSON
report (counts, timings, WER and SHA256 of transcripts; no text).

    OPENAI_API_KEY=<gateway key> uv run scripts/smoke_server.py --base-url http://127.0.0.1:18001 \
        --out docs/baseline/server.json

Runs with `uv run` from the host against a container port as well (or with the image's venv). Checks (exit 1 if one
fails):
- every fixture: json text == text body == "".join(SSE deltas) == SSE done text == verbose_json text == SRT/VTT cue
  texts joined by spaces; usage.seconds == verbose_json duration == samples / 16 kHz; cues ordered, inside the audio;
- fixtures of one chunk (<= GATEWAY_MAX_CHUNK_SECONDS, 16 kHz WAV decodes bit-exactly): SHA256 of the text == the
  default setting of docs/baseline/offline.json (scripts/baseline_offline.py), so the server must run the default
  GATEWAY_PRECISION and GATEWAY_DECODING_METHOD;
- fixtures-large/vi_long.wav, if present: WER against the FLEURS reference <= 0.10;
- Realtime (`/v1/realtime`, one socket, one commit per fixture, 24 kHz PCM16 in 20 ms packets): events in order,
  transcript == joined deltas, usage == committed seconds, WER <= REST WER + 0.02 (soxr + PCM16 vs no resampling),
  silence == REST (digital zeros survive resampling; the model's own text, not forced to ""). This part imports runtime dependencies (`websockets`, `numpy`, `soundfile`, `soxr`); the rest is
  stdlib only.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.scoring import wer  # noqa: E402
from scripts.subtitles import parse_cues  # noqa: E402

FIXTURES = ROOT / "tests/fixtures"
LONG = ROOT / "fixtures-large/vi_long.wav"
MODEL = "viet-asr"
LONG_MAX_WER = 0.10
SMALL = ("vi_a.wav", "vi_b.wav", "silence_2s.wav", "short_200ms.wav")
TEXT_FIELDS = (
    "text",
    "text_body",
    "sse_joined",
    "sse_done",
)  # verified in memory, never written to the report
REALTIME_WER_SLACK = 0.02
PCM_RATE = 24000


class Gateway:
    def __init__(self, base_url: str, api_key: str, timeout: float, ssl_context=None):
        self.base, self.key, self.timeout = base_url.rstrip("/"), api_key, timeout
        self.ssl_context = ssl_context

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
        auth: bool = True,
    ):
        h = dict(headers or {})
        if auth:
            h["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.base + path, data=body, method=method, headers=h)
        return urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context)

    def get_json(self, path: str, auth: bool = True):
        with self.request("GET", path, auth=auth) as r:
            return json.load(r)

    def transcribe(self, audio: bytes, **fields: str):
        boundary = uuid.uuid4().hex
        body = b""
        for k, v in {"model": MODEL, **fields}.items():
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        body += (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode()
            + audio
            + f"\r\n--{boundary}--\r\n".encode()
        )
        return self.request(
            "POST",
            "/v1/audio/transcriptions",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )


def wait_ready(gw: Gateway, timeout: float) -> float:
    started = time.monotonic()
    while True:
        try:
            if gw.get_json("/ready", auth=False) == {"status": "ready"}:
                return time.monotonic() - started
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if time.monotonic() - started > timeout:
            raise SystemExit(f"gateway not ready after {timeout:g} s")
        time.sleep(1)


def run_fixture(gw: Gateway, audio: bytes) -> dict:
    t0 = time.monotonic()
    with gw.transcribe(audio) as r:
        body = json.load(r)
    t1 = time.monotonic()
    with gw.transcribe(audio, response_format="text") as r:
        text = r.read().decode()
        text_type = r.headers["content-type"]
    t2 = time.monotonic()
    deltas, done, first, errors = [], None, None, []
    with gw.transcribe(audio, stream="true") as r:
        sse_type = r.headers["content-type"]
        for line in r:
            line = line.decode().rstrip("\n")
            if not line:
                continue
            first = first or time.monotonic()
            event = json.loads(line.removeprefix("data: "))
            if "error" in event:
                errors.append(event["error"]["code"])
            elif event["type"] == "transcript.text.delta":
                deltas.append(event["delta"])
            elif event["type"] == "transcript.text.done":
                done = event["text"]
    t3 = time.monotonic()
    with gw.transcribe(audio, response_format="verbose_json") as r:
        verbose = json.load(r)
    t4 = time.monotonic()
    timed = {}
    for fmt in ("srt", "vtt"):
        with gw.transcribe(audio, response_format=fmt) as r:
            timed[fmt] = (r.headers["content-type"], parse_cues(r.read().decode(), fmt))
    t5 = time.monotonic()
    return {
        "text": body["text"],
        "usage": body["usage"],
        "text_body": text,
        "text_content_type": text_type,
        "sse_content_type": sse_type,
        "deltas": len(deltas),
        "sse_joined": "".join(deltas),
        "sse_done": done,
        "sse_errors": errors,
        "verbose": verbose,
        "cue_content_types": {f: t for f, (t, _) in timed.items()},
        "cues": {f: c for f, (_, c) in timed.items()},
        "timings_s": {
            "json": round(t1 - t0, 3),
            "text": round(t2 - t1, 3),
            "sse_total": round(t3 - t2, 3),
            "sse_first_event": round((first or t3) - t2, 3),
            "verbose_json": round(t4 - t3, 3),
            "srt_vtt": round(t5 - t4, 3),
        },
    }


def run_realtime(gw: Gateway, paths: list[Path]) -> tuple[dict, list[str]]:
    """One socket, one commit per fixture; returns per-fixture results and failures."""
    import base64

    import numpy as np
    import soundfile as sf
    import soxr
    from websockets.sync.client import connect

    url = "ws" + gw.base.removeprefix("http") + f"/v1/realtime?model={MODEL}"
    results, failures = {}, []
    with connect(
        url,
        additional_headers={"Authorization": f"Bearer {gw.key}"},
        max_size=None,
        open_timeout=30,
    ) as ws:

        def recv() -> dict:
            return json.loads(ws.recv(gw.timeout))

        if recv()["type"] != "session.created":
            failures.append("realtime: no session.created")
        previous = None
        for path in paths:
            wav, sr = sf.read(path, dtype="float32")
            up = soxr.resample(wav, sr, PCM_RATE, quality="HQ")
            pcm = np.round(np.clip(up, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
            t0 = time.monotonic()
            for i in range(0, len(pcm), PCM_RATE // 50 * 2):
                ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm[i : i + PCM_RATE // 50 * 2]).decode(),
                        }
                    )
                )
            t1 = time.monotonic()
            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            events = [recv()]
            first_delta = None
            while not events[-1]["type"].startswith(
                "conversation.item.input_audio_transcription."
            ) or events[-1]["type"].endswith(".delta"):
                events.append(recv())
                if first_delta is None and events[-1]["type"].endswith(".delta"):
                    first_delta = time.monotonic()
            t2 = time.monotonic()
            types = [e["type"] for e in events]
            deltas = [e["delta"] for e in events if e["type"].endswith(".delta")]
            done = events[-1]
            name = path.name
            if (
                types[:2] != ["input_audio_buffer.committed", "conversation.item.added"]
                or events[0]["previous_item_id"] != previous
                or not types[-1].endswith(".completed")
            ):
                failures.append(f"realtime {name}: events {types[:2]}..{types[-1]}")
                results[name] = {"events": types}
                continue
            previous = events[0]["item_id"]
            seconds = len(pcm) / 2 / PCM_RATE
            if done["transcript"] != "".join(deltas) or done["usage"] != {
                "type": "duration",
                "seconds": seconds,
            }:
                failures.append(f"realtime {name}: transcript/usage disagree")
            results[name] = {
                "text": done["transcript"],
                "deltas": len(deltas),
                "audio_s": seconds,
                "timings_s": {
                    "append_send": round(t1 - t0, 3),
                    "commit_to_first_delta": round((first_delta or t2) - t1, 3),
                    "commit_to_completed": round(t2 - t1, 3),
                },
            }
            print(
                f"realtime {name}: {results[name]['timings_s']} deltas={len(deltas)}",
                file=sys.stderr,
            )
    return results, failures


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:18001")
    ap.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="environment variable holding the gateway key",
    )
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--timeout", type=float, default=600.0, help="per HTTP request")
    ap.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=20.0,
        help="the server's GATEWAY_MAX_CHUNK_SECONDS",
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"set {args.api_key_env} to a gateway API key")

    gw = Gateway(args.base_url, key, args.timeout)
    failures: list[str] = []
    report: dict = {
        "base_url": args.base_url,
        "commit": git_commit(),
        "started_unix": round(time.time()),
    }
    report["ready_wait_s"] = round(wait_ready(gw, args.ready_timeout), 3)
    report["health"] = gw.get_json("/health", auth=False)
    models = gw.get_json("/v1/models")
    report["models"] = [m["id"] for m in models["data"]]
    if MODEL not in report["models"] or gw.get_json(f"/v1/models/{MODEL}")["id"] != MODEL:
        failures.append("models: served model missing")

    offline = json.loads((ROOT / "docs/baseline/offline.json").read_text())
    baseline = offline["a_settings"]["rows"][offline["default"]]
    manifest = json.loads((FIXTURES / "manifest.json").read_text())["files"]
    inputs = [FIXTURES / n for n in SMALL] + ([LONG] if LONG.exists() else [])
    report["results"] = {}
    for path in inputs:
        info = manifest[path.name]
        res = run_fixture(gw, path.read_bytes())
        res["audio_s"] = info["samples"] / info["sample_rate"]
        name = path.name
        if (
            not (res["text"] == res["text_body"] == res["sse_joined"] == res["sse_done"])
            or res["sse_errors"]
        ):
            failures.append(f"{name}: json/text/SSE disagree or SSE error {res['sse_errors']}")
        if res["usage"] != {"type": "duration", "seconds": res["audio_s"]}:
            failures.append(f"{name}: usage {res['usage']}")
        if res["verbose"] != {
            "text": res["text"],
            "language": "vietnamese",
            "duration": res["audio_s"],
            "usage": res["usage"],
        }:
            failures.append(f"{name}: verbose_json disagrees with json")
        for fmt, cues in res["cues"].items():
            if (
                cues is None
                or " ".join(c[2] for c in cues) != res["text"]
                or not all(0 <= a < b <= res["audio_s"] + 0.0005 for a, b, _ in cues)
                or not all(p[1] <= n[0] for p, n in pairwise(cues))
                or res["cue_content_types"][fmt] != "text/plain; charset=utf-8"
            ):
                failures.append(f"{name}: {fmt} malformed, unordered or disagrees with json")
        res["cues"] = {f: len(c or []) for f, c in res["cues"].items()}  # counts only in the report
        del res["verbose"]
        if res["text_content_type"] != "text/plain; charset=utf-8" or not res[
            "sse_content_type"
        ].startswith("text/event-stream"):
            failures.append(f"{name}: content types")
        if res["audio_s"] <= args.max_chunk_seconds:
            res["text_sha256"] = hashlib.sha256(res["text"].encode()).hexdigest()
            res["matches_baseline"] = res["text_sha256"] == baseline[name]["text_sha256"]
            if not res["matches_baseline"]:
                failures.append(f"{name}: differs from the offline baseline")
        else:
            # The model's vocabulary is upper case; FLEURS references are lower case.
            res["wer"] = round(wer(info["reference"], res["text"].lower()), 4)
            if res["wer"] > LONG_MAX_WER:
                failures.append(f"{name}: WER {res['wer']} > {LONG_MAX_WER}")
        report["results"][name] = res
        print(f"{name}: {res['timings_s']} deltas={res['deltas']}", file=sys.stderr)
    report["realtime"], rt_failures = run_realtime(gw, inputs)
    failures += rt_failures
    for name, res in report["realtime"].items():
        if "text" not in res:
            continue
        ref = manifest[name]["reference"]
        if ref:
            res["wer"] = round(wer(ref, res["text"].lower()), 4)
            res["rest_wer"] = round(wer(ref, report["results"][name]["text"].lower()), 4)
            if res["wer"] > res["rest_wer"] + REALTIME_WER_SLACK:
                failures.append(
                    f"realtime {name}: WER {res['wer']} > REST {res['rest_wer']} + {REALTIME_WER_SLACK}"
                )
        elif name == "silence_2s.wav" and res["text"] != report["results"][name]["text"]:
            failures.append(f"realtime {name}: differs from REST")
    report["metrics_excerpt"] = [
        line
        for line in gw.request("GET", "/metrics").read().decode().splitlines()
        if line.startswith(
            (
                "gateway_transcriptions_total",
                "gateway_jobs_total",
                "gateway_decodes_total",
                "gateway_realtime_",
            )
        )
    ]
    report["failures"] = failures
    for group in ("results", "realtime"):
        for res in report[group].values():
            for field in TEXT_FIELDS:
                res.pop(field, None)
    out = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(out + "\n")
    print(out)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
