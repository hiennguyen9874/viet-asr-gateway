"""Seam 3: the real SherpaEngine against the reference inference script, and behind the worker and scheduler.

CPU only; needs the staged model (scripts/stage_models.py).
"""

import asyncio
import time

import numpy as np
import pytest
import soundfile as sf

from app.alignment import words
from app.chunking import Span
from app.config import SAMPLE_RATE
from app.engine import EngineState, EngineWorker, MIN_FORWARD_SAMPLES, SherpaEngine
from app.observability import Metrics
from app.scheduler import JobKind, Scheduler
from tests.fake_engine import config
from tests.model.conftest import FIXTURES, MODELS, reference_text

CLIPS = ("vi_a.wav", "vi_b.wav")


def loaded(precision="fp32", decoding="modified_beam_search", beam=4) -> SherpaEngine:
    engine = SherpaEngine(MODELS, precision, decoding, beam, 4, warmup_seconds=1.0)
    engine.load()
    return engine


@pytest.mark.parametrize("precision", ["fp32", "int8"])
@pytest.mark.parametrize("decoding", ["modified_beam_search", "greedy_search"])
def test_engine_text_equals_reference_and_tokens_are_consistent(precision, decoding):
    engine = loaded(precision, decoding)
    for name in CLIPS:
        wav = sf.read(FIXTURES / name, dtype="float32")[0]
        result = engine.transcribe(wav)
        assert result.text == reference_text(FIXTURES / name, precision, decoding), (name, precision, decoding)
        assert result.text and len(result.tokens) == len(result.times)
        assert "".join(result.tokens).replace("\u2581", " ").strip() == result.text
        assert list(result.times) == sorted(result.times) and 0 <= result.times[0] and result.times[-1] <= len(wav)
        assert result.tokens[0].startswith("\u2581")
        # Raw recognizer times (not repaired by the engine) must pass alignment.
        assert " ".join(w.text for w in words(Span(0, len(wav)), result)) == result.text


def test_silence_decodes_without_error_and_tokens_spell_the_text():
    # The model may hallucinate a filler on digital silence; the result must still be self-consistent.
    result = loaded().transcribe(np.zeros(2 * SAMPLE_RATE, dtype=np.float32))
    assert "".join(result.tokens).replace("\u2581", " ").strip() == result.text
    assert len(result.tokens) == len(result.times)


@pytest.mark.parametrize("precision", ["fp32", "int8"])
@pytest.mark.parametrize("decoding", ["modified_beam_search", "greedy_search"])
def test_hard_floor_is_the_shortest_input_the_recognizer_accepts(precision, decoding):
    # Measured by scripts/probe_limits.py (docs/baseline/limits_probe.json); a sherpa-onnx or model change that
    # moves it must move MIN_FORWARD_SAMPLES too.
    engine = loaded(precision, decoding)
    speech = sf.read(FIXTURES / "vi_a.wav", dtype="float32")[0]
    for x in (np.zeros(MIN_FORWARD_SAMPLES + 1, np.float32), speech[2 * SAMPLE_RATE:][:MIN_FORWARD_SAMPLES + 1]):
        with pytest.raises(RuntimeError):
            engine.transcribe(x[:MIN_FORWARD_SAMPLES - 1])
        for n in (MIN_FORWARD_SAMPLES, MIN_FORWARD_SAMPLES + 1):
            result = engine.transcribe(x[:n])
            assert "".join(result.tokens).replace("\u2581", " ").strip() == result.text
            assert len(result.tokens) == len(result.times)


def test_missing_model_files_fail_load(tmp_path):
    with pytest.raises(FileNotFoundError):
        SherpaEngine(tmp_path, "fp32", "greedy_search", 4, 4, 1.0).load()


def test_worker_serves_concurrent_lanes_and_is_ready_only_after_warmup():
    clips = {n: sf.read(FIXTURES / n, dtype="float32")[0] for n in CLIPS}
    expected = {n: reference_text(FIXTURES / n, "fp32", "modified_beam_search") for n in CLIPS}

    async def main():
        cfg = config(models_dir=MODELS, max_chunk_seconds=10.0)
        worker = EngineWorker(
            SherpaEngine(MODELS, "fp32", "modified_beam_search", 4, 4, cfg.max_chunk_seconds),
            cfg.infer_hang_seconds,
            Metrics(),
        )
        scheduler = Scheduler(worker, cfg, Metrics())
        assert not worker.ready
        worker.start()
        scheduler.start()
        deadline = time.monotonic() + 300
        while worker.state == EngineState.LOADING:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.1)
        assert worker.state == EngineState.READY and worker.ready

        async def lane_run(kind, name):
            async with scheduler.open_lane(kind) as lane:
                return [(await (await lane.submit(clips[name]))).text for _ in range(3)]

        try:
            return await asyncio.gather(
                lane_run(JobKind.FILE, "vi_a.wav"), lane_run(JobKind.LIVE, "vi_b.wav"), lane_run(JobKind.FILE, "vi_b.wav")
            )
        finally:
            assert await scheduler.close(30)
            assert await asyncio.to_thread(worker.stop, 30)

    a, b_live, b_file = asyncio.run(main())
    assert a == [expected["vi_a.wav"]] * 3
    assert b_live == b_file == [expected["vi_b.wav"]] * 3
