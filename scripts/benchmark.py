"""Bounded, closed-loop capacity probe of a running gateway; never records text/audio/keys.

OPENAI_API_KEY=... uv run scripts/benchmark.py --engine cpu --hardware <label> --out report.json
Capacity gate and how to run it: docs/roadmap.md (R4). A report is not a capacity claim.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import re
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.smoke_server import Gateway, git_commit, MODEL, wait_ready  # noqa: E402


@dataclass(frozen=True)
class Clip:
    body: bytes
    pcm: bytes
    seconds: float
    sha256: str


def load_clip(path: Path) -> Clip:
    import numpy as np
    import soundfile as sf
    import soxr

    body = path.read_bytes()
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if not len(wav):
        raise ValueError("empty benchmark input")
    mono = wav.mean(axis=1)
    up = soxr.resample(mono, sr, 24000, quality="HQ")
    pcm = np.round(np.clip(up, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
    return Clip(body, pcm, len(wav) / sr, hashlib.sha256(body).hexdigest())


def percentiles(values: list[float]) -> dict:
    """Nearest-rank percentiles; missing observations remain null, never zero."""
    values = sorted(values)
    return {
        f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)] if values else None
        for p in (50, 95)
    }


def parse_metrics(text: str) -> dict[str, float]:
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.rsplit(None, 1)
        result[key] = float(value)
    return result


def metrics(gw: Gateway) -> dict:
    with gw.request("GET", "/metrics") as response:
        return parse_metrics(response.read().decode())


def metric_delta(before: dict, after: dict, name: str) -> float | None:
    if name not in after:
        return None
    value = after[name] - before.get(name, 0)
    if value < 0:
        raise RuntimeError("server metrics reset during benchmark; discard this stage")
    return value


def histogram(before: dict, after: dict, name: str) -> dict:
    """Interval quantiles are upper bucket bounds, NOT exact/interpolated latencies."""

    def total(suffix):
        keys = [k for k in after if k == name + suffix or k.startswith(name + suffix + "{")]
        return sum(metric_delta(before, after, k) for k in keys) if keys else None

    count = total("_count") or 0
    bounds = Counter()
    for key in after:
        if key.startswith(name + "_bucket{"):
            bound = float(re.search(r'le="([^"]+)"', key)[1])
            bounds[bound] += metric_delta(before, after, key)
    buckets = list(bounds.items())
    result = {
        "count": count,
        "sum_s": total("_sum"),
        "overflow_count": count - bounds.get(120.0, 0) if buckets else 0,
    }
    for p in (50, 95):
        bound = next(
            (b for b, n in sorted(buckets) if count and n >= math.ceil(count * p / 100)), None
        )
        result[f"p{p}_upper_s"] = bound if bound is not None and math.isfinite(bound) else None
    return result


class ProtocolFailure(Exception):
    def __init__(self, overload: bool = False):
        self.overload = overload


class Client:
    def __init__(self, gw: Gateway, pace: float):
        self.gw, self.pace, self.ws = gw, pace, None

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def run(self, kind: str, clip: Clip) -> dict:
        start = time.monotonic()
        result = {
            "kind": kind,
            "audio_s": clip.seconds,
            "ok": False,
            "overload": False,
            "first_delta_s": None,
            "final_s": None,
            "commit_to_final_s": None,
            "commit_to_first_delta_s": None,
        }
        try:
            if kind == "ws":
                self._ws(clip, result, start)
            else:
                self._rest(kind, clip, result, start)
            result["ok"] = True
        except urllib.error.HTTPError as e:
            result["error"] = f"http_{e.code}"
            result["overload"] = e.code == 429
            e.close()
        except ProtocolFailure as e:
            result["error"] = "protocol_error"
            result["overload"] = e.overload
        except Exception as e:
            # Exception messages can contain server payloads, transcripts, URLs or keys.
            result["error"] = type(e).__name__
        finally:
            result["elapsed_s"] = time.monotonic() - start
            if not result["ok"]:
                self.close()
        return result

    def _rest(self, kind: str, clip: Clip, result: dict, start: float):
        with self.gw.transcribe(
            clip.body, **({"stream": "true"} if kind == "sse" else {})
        ) as response:
            if kind != "sse":
                event = json.load(response)
                if not isinstance(event.get("text"), str):
                    raise ProtocolFailure()
            else:
                done, deltas = None, []
                for line in response:
                    if time.monotonic() - start > self.gw.timeout:
                        raise TimeoutError()
                    if not line.startswith(b"data: "):
                        continue
                    event = json.loads(line[6:])
                    if "error" in event:
                        raise ProtocolFailure(
                            event["error"].get("code") in ("server_busy", "queue_full")
                        )
                    if event.get("type") == "transcript.text.delta":
                        if result["first_delta_s"] is None:
                            result["first_delta_s"] = time.monotonic() - start
                        deltas.append(event["delta"])
                    elif event.get("type") == "transcript.text.done":
                        done = event["text"]
                        result["final_s"] = time.monotonic() - start
                if done is None or done != "".join(deltas):
                    raise ProtocolFailure()
        if result["final_s"] is None:
            result["final_s"] = time.monotonic() - start

    def _ws(self, clip: Clip, result: dict, start: float):
        from websockets.exceptions import InvalidStatus
        from websockets.sync.client import connect

        if self.ws is None:
            try:
                self.ws = connect(
                    "ws" + self.gw.base.removeprefix("http") + f"/v1/realtime?model={MODEL}",
                    additional_headers={"Authorization": f"Bearer {self.gw.key}"},
                    open_timeout=self.gw.timeout,
                    close_timeout=2,
                    max_size=2097152,
                    **({"ssl": self.gw.ssl_context} if self.gw.base.startswith("https:") else {}),
                )
            except InvalidStatus as e:
                raise ProtocolFailure(e.response.status_code == 429) from None
            if json.loads(self.ws.recv(self.gw.timeout)).get("type") != "session.created":
                raise ProtocolFailure()
        append_start = time.monotonic()
        for offset in range(0, len(clip.pcm), 960):  # PCM16 24 kHz, 20 ms packets
            if self.pace:
                time.sleep(max(0, append_start + offset / 48000 * self.pace - time.monotonic()))
            self.ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(clip.pcm[offset : offset + 960]).decode(),
                    }
                )
            )
        if self.pace:
            time.sleep(max(0, append_start + len(clip.pcm) / 48000 * self.pace - time.monotonic()))
        commit = time.monotonic()
        self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        deltas, item = [], None
        while True:
            remaining = self.gw.timeout - (time.monotonic() - commit)
            if remaining <= 0:
                raise TimeoutError()
            event = json.loads(self.ws.recv(remaining))
            type_ = event.get("type", "")
            if type_ == "error" or type_.endswith(".failed"):
                raise ProtocolFailure(
                    event.get("error", {}).get("code") in ("server_busy", "queue_full")
                )
            if type_ == "input_audio_buffer.committed":
                item = event["item_id"]
            if type_.endswith((".delta", ".completed")):
                if item is None or event.get("item_id") != item:
                    raise ProtocolFailure()
                if type_.endswith(".delta"):
                    if result["commit_to_first_delta_s"] is None:
                        result["commit_to_first_delta_s"] = time.monotonic() - commit
                    deltas.append(event["delta"])
                else:
                    if event["transcript"] != "".join(deltas):
                        raise ProtocolFailure()
                    result["commit_to_final_s"] = time.monotonic() - commit
                    result["final_s"] = time.monotonic() - start
                    return


class Sampler:
    """Optional Linux server RSS + per-PID NVML CLI VRAM sampling; no dependency on torch."""

    def __init__(self, pid: int | None):
        self.pid, self.rss, self.vram = pid, [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                status = Path(f"/proc/{self.pid}/status").read_text()
                self.rss.append(int(re.search(r"VmRSS:\s+(\d+)", status)[1]) * 1024)
            except (OSError, TypeError):
                pass
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                values = [
                    int(line.split(",")[1]) * 1024**2
                    for line in proc.stdout.splitlines()
                    if line.split(",")[0].strip() == str(self.pid)
                ]
                if values:
                    self.vram.append(sum(values))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.5)

    def __enter__(self):
        if self.pid is not None:
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.pid is not None:
            self.thread.join(3)

    def report(self):
        return {
            "rss_sampled_peak_bytes": max(self.rss, default=None),
            "vram_sampled_peak_bytes": max(self.vram, default=None),
            "rss_samples": len(self.rss),
            "vram_samples": len(self.vram),
        }


def summarize(rows: list[dict], wall: float) -> dict:
    ok = [r for r in rows if r["ok"]]
    return {
        "requests": len(rows),
        "successes": len(ok),
        "error_rate": 1 - len(ok) / len(rows),
        "overload_rate": sum(r["overload"] for r in rows) / len(rows),
        "errors": dict(Counter(r["error"] for r in rows if not r["ok"])),
        "successful_audio_s_per_wall_s": sum(r["audio_s"] for r in ok) / wall,
        "rtf": percentiles([r["final_s"] / r["audio_s"] for r in ok]),
        **{
            k: percentiles([r[k] for r in ok if r[k] is not None])
            for k in ("final_s", "first_delta_s", "commit_to_final_s", "commit_to_first_delta_s")
        },
    }


def stage(
    gw: Gateway,
    clips: dict[str, Clip],
    workload: str,
    clients: int,
    requests: int,
    duration: float,
    pace: float,
    pid: int | None,
) -> dict:
    tasks = {
        "short": [("rest", "short")],
        "long": [("rest", "long")],
        "sse": [("sse", "short"), ("sse", "long")],
        "ws": [("ws", "short")],
        "mixed": [
            ("rest", "short"),
            ("rest", "long"),
            ("sse", "short"),
            ("sse", "long"),
            ("ws", "short"),
        ],
    }[workload]
    before = metrics(gw)
    barrier = threading.Barrier(clients)

    def worker(index):
        client = Client(gw, pace)
        rows = []
        barrier.wait()
        until = time.monotonic() + duration
        try:
            i = 0
            while i < 10000 and (i < requests or time.monotonic() < until):
                kind, size = tasks[(index + i) % len(tasks)]
                row = client.run(kind, clips[size])
                row["input"] = size
                rows.append(row)
                i += 1
        finally:
            client.close()
        return rows, time.monotonic() >= until

    start = time.monotonic()
    with (
        Sampler(pid) as sampler,
        concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool,
    ):
        batches = list(pool.map(worker, range(clients)))
        rows = [row for batch, _ in batches for row in batch]
    wall = time.monotonic() - start
    after = metrics(gw)
    return {
        "clients": clients,
        "wall_s": wall,
        "duration_target_s": duration,
        "duration_target_met": all(met for _, met in batches),
        "request_cap_per_client": 10000,
        "summary": summarize(rows, wall),
        "by_workload": {
            name: summarize([r for r in rows if r["kind"] + "_" + r["input"] == name], wall)
            for name in sorted({r["kind"] + "_" + r["input"] for r in rows})
        },
        "server": {
            "queue_wait": histogram(before, after, "gateway_queue_wait_seconds"),
            "decode_wall": histogram(before, after, "gateway_decode_seconds"),
            "infer": histogram(before, after, "gateway_infer_seconds"),
            **{
                n: metric_delta(before, after, "gateway_" + n)
                for n in ("process_cpu_seconds_total", "children_cpu_seconds_total")
            },
            **sampler.report(),
        },
    }


def positive(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:18001")
    ap.add_argument(
        "--ca", type=Path, help="trusted CA PEM for HTTPS/WSS (verification is never disabled)"
    )
    ap.add_argument("--short", type=Path, default=ROOT / "tests/fixtures/vi_a.wav")
    ap.add_argument("--long", type=Path, default=ROOT / "fixtures-large/vi_long.wav")
    ap.add_argument("--workload", choices=("short", "long", "sse", "ws", "mixed"), default="mixed")
    ap.add_argument("--clients", default="1,2,4,8,16,32")
    ap.add_argument("--requests-per-client", type=int, default=10)
    ap.add_argument(
        "--duration",
        type=positive,
        help="minimum seconds per stage (soak); requests also remains a minimum",
    )
    ap.add_argument("--timeout", type=positive, default=600.0)
    ap.add_argument("--ws-pace", choices=("realtime", "burst"), default="realtime")
    ap.add_argument("--phase", choices=("cold", "warm"), default="warm")
    ap.add_argument("--server-pid", type=int, help="local host PID, not a container-namespace PID")
    ap.add_argument(
        "--hardware", required=True, help="non-secret hardware/profile label; stored verbatim"
    )
    ap.add_argument("--engine", choices=("fake", "cpu"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        clients = [int(v) for v in args.clients.split(",")]
    except ValueError:
        ap.error("--clients must be comma-separated integers")
    if not clients or any(c < 1 or c > 256 for c in clients) or clients != sorted(set(clients)):
        ap.error("--clients must be strictly increasing, in [1,256]")
    if not 1 <= args.requests_per_client <= 10000:
        ap.error("--requests-per-client must be in [1,10000]")
    if args.server_pid is not None and args.server_pid < 1:
        ap.error("--server-pid must be positive")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        ap.error("set OPENAI_API_KEY (never pass keys on the command line)")
    gw = Gateway(
        args.base_url, key, args.timeout, ssl_context=ssl.create_default_context(cafile=args.ca)
    )
    ready = wait_ready(gw, args.timeout)
    sizes = (
        {"short", "long"}
        if args.workload in ("mixed", "sse")
        else {"short" if args.workload == "ws" else args.workload}
    )
    clips = {size: load_clip(getattr(args, size)) for size in sizes}
    report = {
        "schema_version": 1,
        "commit": git_commit(),
        "started_unix": time.time(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_manifest_sha256": hashlib.sha256(
            (ROOT / "assets/manifest.json").read_bytes()
        ).hexdigest(),
        "client_python": platform.python_version(),
        "timeout_s": args.timeout,
        "minimum_requests_per_client": args.requests_per_client,
        "minimum_duration_per_stage_s": args.duration or 0,
        "engine": args.engine,
        "hardware": args.hardware,
        "client_platform": platform.platform(),
        "phase": args.phase,
        "readiness_wait_s": ready,
        "lm": "off (greedy only)",
        "load_model": "closed-loop; one outstanding request/commit per client; no automatic retries",
        "workload": args.workload,
        "ws_pace": args.ws_pace,
        "inputs": {k: {"sha256": c.sha256, "seconds": c.seconds} for k, c in clips.items()},
        "stages": [],
        "production_gate": "not evaluated; see runbook",
    }
    if args.phase == "warm":
        client = Client(gw, 0)
        try:
            for clip in clips.values():
                if not client.run("rest", clip)["ok"]:
                    raise RuntimeError("warmup request failed")
        finally:
            client.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for count in clients:
        result = stage(
            gw,
            clips,
            args.workload,
            count,
            args.requests_per_client,
            args.duration or 0,
            float(args.ws_pace == "realtime"),
            args.server_pid,
        )
        report["stages"].append(result)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            f"clients={count} successes={result['summary']['successes']}/{result['summary']['requests']}"
        )
    return int(
        any(
            s["summary"]["error_rate"] > 0 or not s["duration_target_met"] for s in report["stages"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
