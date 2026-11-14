"""M7 fault injection: real probes/traffic, simulated OOM/hang, restart, bounded mixed soak.

These tests prove control-plane semantics, not real model memory recovery or CPU capacity.
"""

import concurrent.futures
import secrets

import pytest

from app.engine import EngineState
from app.realtime import decode_base64
from scripts.benchmark import load_clip, stage
from scripts.smoke_server import Gateway
from tests.fake_engine import FakeEngine
from tests.live_server import KEY, name, noise, serve, wait_for, wav_file


@pytest.mark.parametrize("audio", ["QUJD=", "QUJD====", "Zh==", "***", "ñ"])
def test_strict_base64_rejects_noncanonical_packets(audio):
    with pytest.raises(ValueError):
        decode_base64(audio)
    assert decode_base64("QUJD") == b"ABC"


def request(srv, clip):
    return srv.client.post("/v1/audio/transcriptions", data={"model": "viet-asr"},
                           files={"file": ("input.wav", clip, "audio/wav")})


class OOMEngine(FakeEngine):
    oom = False

    def transcribe(self, wav):
        if self.oom:
            self.oom = False
            raise MemoryError("simulated allocator OOM")
        return super().transcribe(wav)


def test_oom_recovery_probes_and_no_traffic_until_warm(tmp_path):
    clip = wav_file(tmp_path / "input.wav", noise(.3, 201)).read_bytes()
    engine = OOMEngine(text_fn=name)
    with serve(engine) as srv:
        engine.warmup_gate.clear()
        engine.oom = True
        try:
            response = request(srv, clip)
            assert response.status_code == 500 and response.json()["error"]["code"] == "inference_failed"
            wait_for(lambda: srv.app.state.worker.state == EngineState.RECOVERING)
            assert srv.client.get("/ready").status_code == 503
            assert request(srv, clip).status_code == 503
            assert engine.calls == []  # no inference admitted during recovery
            assert srv.client.get("/health").status_code == 200
        finally:
            engine.warmup_gate.set()
        wait_for(lambda: srv.client.get("/ready").status_code == 200)
        assert request(srv, clip).status_code == 200
        assert engine.max_active == 1


def test_hung_worker_needs_restart_and_fresh_instance_warmup_barrier(tmp_path):
    clip = wav_file(tmp_path / "input.wav", noise(.3, 202)).read_bytes()
    engine = FakeEngine(text_fn=name)
    with serve(engine, infer_hang_seconds=.1) as srv:
        engine.gate.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(request, srv, clip)
            try:
                assert engine.started.acquire(timeout=5)
                wait_for(lambda: srv.client.get("/health").status_code == 503)
                assert srv.client.get("/ready").status_code == 503
                assert request(srv, clip).status_code == 503
                assert engine.max_active == 1
            finally:
                engine.gate.set()
            assert pending.result(timeout=5).status_code == 200
    # New instance must independently load/warm; old ready state cannot leak into restart.
    restarted = FakeEngine(text_fn=name)
    restarted.load_gate.clear()
    with serve(restarted, wait_ready=False) as srv:
        assert srv.client.get("/ready").status_code == 503
        assert request(srv, clip).status_code == 503
        assert restarted.calls == []
        restarted.load_gate.set()
        wait_for(lambda: srv.client.get("/ready").status_code == 200)
        assert request(srv, clip).status_code == 200


def test_secret_rotation_requires_restart_and_overlapping_keys():
    new = secrets.token_urlsafe(24)
    with serve(api_keys=(KEY, new)) as srv:
        assert srv.client.get("/v1/models").status_code == 200
        assert srv.client.get("/v1/models", headers={"Authorization": f"Bearer {new}"}).status_code == 200
    with serve(api_keys=(new,)) as srv:
        assert srv.client.get("/v1/models").status_code == 401
        assert srv.client.get("/v1/models", headers={"Authorization": f"Bearer {new}"}).status_code == 200


def test_logs_exclude_transcript_filename_query_and_arbitrary_paths(tmp_path, capfd):
    private = "private-" + secrets.token_hex(8)
    clip = wav_file(tmp_path / "input.wav", noise(.3, 204)).read_bytes()
    with serve(FakeEngine(text_fn=lambda _: private)) as srv:
        response = srv.client.post("/v1/audio/transcriptions", data={"model": "viet-asr"},
                                   files={"file": (private + ".wav", clip, "audio/wav")})
        assert response.json()["text"] == private
        srv.client.get("/" + private + "?token=" + private)
        srv.client.get("/v1/models/" + private)
        metrics = srv.client.get("/metrics").text
    logs = capfd.readouterr().err
    assert private not in logs + metrics and KEY not in logs + metrics
    assert "<unmatched>" in logs and "/v1/models/{model}" in logs


def test_bounded_mixed_soak_releases_all_budgets(tmp_path):
    clip = load_clip(wav_file(tmp_path / "input.wav", noise(1.2, 203)))
    engine = FakeEngine(delay=.002, text_fn=name)
    with serve(engine, max_chunk_seconds=1) as srv:
        result = stage(Gateway(srv.base_url, KEY, 5), {"short": clip, "long": clip}, "mixed",
                       clients=5, requests=10, duration=1, pace=0, pid=None)
        assert result["summary"]["requests"] >= 50
        assert result["summary"]["error_rate"] == 0
        wait_for(lambda: srv.app.state.scheduler._jobs == 0 and srv.app.state.scheduler._lanes == 0)
        wait_for(lambda: srv.app.state.audio._held == 0 and srv.app.state.audio._spooled == 0)
        assert srv.app.state.audio._decoding == 0
        assert engine.max_active == 1
        assert srv.client.get("/ready").status_code == 200
