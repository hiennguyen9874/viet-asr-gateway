"""M7 client diagnostics over real HTTP/SSE/WS; fake timings are NOT capacity evidence."""

import io
import json
import os
import ssl
import subprocess
import sys
import urllib.error

import pytest

from app.observability import Metrics
from scripts.benchmark import (
    Client,
    Clip,
    histogram,
    load_clip,
    metric_delta,
    parse_metrics,
    percentiles,
    stage,
)
from scripts.smoke_server import Gateway
from tests.fake_engine import FakeEngine
from tests.live_server import KEY, noise, serve, wav_file


def test_statistics_and_histogram_interval_not_lifetime():
    assert percentiles([]) == {"p50": None, "p95": None}
    assert percentiles([4, 1, 3, 2]) == {"p50": 2, "p95": 4}
    m = Metrics()
    m.observe("queue_wait_seconds", "Wait.", 50)
    before = parse_metrics(m.render())
    for value in (.01, .03):
        m.observe("queue_wait_seconds", "Wait.", value)
    result = histogram(before, parse_metrics(m.render()), "gateway_queue_wait_seconds")
    assert result["count"] == 2
    assert result["sum_s"] == pytest.approx(.04)
    assert result["p50_upper_s"] == .01 and result["p95_upper_s"] == .05
    with pytest.raises(RuntimeError, match="reset"):
        metric_delta({"counter": 4}, {"counter": 1}, "counter")
    assert metric_delta({}, {}, "missing") is None


def test_empty_and_overflow_histograms():
    assert histogram({}, {}, "missing")["p95_upper_s"] is None
    m = Metrics()
    m.observe("latency", "Latency.", 200)
    result = histogram({}, parse_metrics(m.render()), "gateway_latency")
    assert result["count"] == 1 and result["p95_upper_s"] is None


def test_process_resource_metrics():
    m = Metrics()
    m.process_resources()
    values = parse_metrics(m.render())
    assert values["gateway_process_max_rss_bytes"] > 0
    assert values["gateway_process_cpu_seconds_total"] > 0
    assert values["gateway_children_cpu_seconds_total"] >= 0


@pytest.mark.parametrize("tls", [False, True])
def test_mixed_real_transports_no_sensitive_report(tmp_path, monkeypatch, tls):
    clip = load_clip(wav_file(tmp_path / "audio.wav", noise(1.2, 100)))
    with serve(max_chunk_seconds=1, tls_dir=tmp_path if tls else None) as srv:
        if tls:
            monkeypatch.setenv("SSL_CERT_FILE", str(srv.ca))
        result = stage(Gateway(srv.base_url, KEY, 5, ssl_context=ssl.create_default_context(cafile=srv.ca)),
                       {"short": clip, "long": clip}, "mixed",
                       clients=5, requests=2, duration=0, pace=0, pid=None)
        assert result["summary"]["successes"] == 10
        assert result["summary"]["error_rate"] == 0
        assert result["server"]["queue_wait"]["count"] >= 10
        assert result["server"]["children_cpu_seconds_total"] >= 0
        assert result["server"]["vram_sampled_peak_bytes"] is None
        assert result["by_workload"]["sse_short"]["first_delta_s"]["p50"] > 0
        assert result["by_workload"]["ws_short"]["commit_to_final_s"]["p50"] > 0
        assert result["by_workload"]["rest_short"]["first_delta_s"]["p50"] is None
    serialized = json.dumps(result)
    assert KEY not in serialized and "transcript" not in serialized and "audio.wav" not in serialized


def test_http_overload_not_success_or_retry():
    class Busy:
        timeout = 1
        calls = 0

        def transcribe(self, *args, **kwargs):
            self.calls += 1
            raise urllib.error.HTTPError("http://local", 429, "secret payload", {}, None)

    gw = Busy()
    result = Client(gw, 0).run("rest", Clip(b"", b"", 1, ""))
    assert result["overload"] and not result["ok"] and gw.calls == 1
    assert "secret" not in json.dumps(result)


def test_cli_report(tmp_path):
    clip = wav_file(tmp_path / "input.wav", noise(.2, 101))
    output = tmp_path / "report.json"
    with serve(queue_max_lanes=1, max_sessions=1) as srv:
        command = [sys.executable, "scripts/benchmark.py", "--base-url", srv.base_url,
                   "--short", str(clip), "--workload", "short", "--clients", "1", "--requests-per-client", "1",
                   "--phase", "cold", "--engine", "fake", "--hardware", "CPU test", "--out", str(output)]
        process = subprocess.run(command, env={**os.environ, "OPENAI_API_KEY": KEY}, capture_output=True,
                                 text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    report = json.loads(output.read_text())
    assert report["engine"] == "fake" and report["phase"] == "cold"
    assert report["stages"][0]["summary"]["requests"] == 1
    assert KEY not in output.read_text() + process.stdout + process.stderr
    assert "production_gate" in report


def test_cli_overload_writes_report_and_exits_one(tmp_path):
    clip = wav_file(tmp_path / "input.wav", noise(.2, 102))
    output = tmp_path / "overload.json"
    with serve(FakeEngine(delay=.2), queue_max_lanes=1, max_sessions=1) as srv:
        process = subprocess.run([sys.executable, "scripts/benchmark.py", "--base-url", srv.base_url,
                                  "--short", str(clip), "--workload", "short", "--clients", "4",
                                  "--requests-per-client", "1", "--phase", "cold", "--engine", "fake",
                                  "--hardware", "CPU test", "--out", str(output)],
                                 env={**os.environ, "OPENAI_API_KEY": KEY}, capture_output=True,
                                 text=True, timeout=30)
    assert process.returncode == 1, process.stderr
    summary = json.loads(output.read_text())["stages"][0]["summary"]
    assert summary["requests"] == 4 and summary["overload_rate"] > 0
    assert summary["errors"]["http_429"] > 0


@pytest.mark.parametrize("events,ok", [
    ([{"type": "transcript.text.done", "text": ""}], True),
    ([{"type": "transcript.text.delta", "delta": "sensitive text"}], False),
    ([{"error": {"code": "server_busy", "message": "sensitive error"}}], False),
])
def test_sse_empty_done_truncation_and_inband_error(events, ok):
    class Stream:
        timeout = 2

        def transcribe(self, *args, **kwargs):
            return io.BytesIO(b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events))

    result = Client(Stream(), 0).run("sse", Clip(b"", b"", 1, ""))
    assert result["ok"] == ok
    if ok:
        assert result["first_delta_s"] is None
    assert "sensitive" not in json.dumps(result)


def test_cli_rejects_unbounded_load(tmp_path):
    process = subprocess.run([sys.executable, "scripts/benchmark.py", "--clients", "0,1000000",
                              "--engine", "fake", "--hardware", "CPU", "--out", str(tmp_path / "out.json")],
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 2 and "[1,256]" in process.stderr
