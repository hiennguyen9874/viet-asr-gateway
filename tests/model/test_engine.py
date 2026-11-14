"""Seam 3: the real SherpaEngine against the minimal sherpa-onnx reference (tests/model/reference.py), and behind the
worker and scheduler.

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
from tests.model import reference
from tests.model.conftest import FIXTURES, MODEL_DIR, MODELS, reference_text

CLIPS = ("vi_a.wav", "vi_b.wav")
DECODINGS = ("modified_beam_search", "greedy_search")


def loaded(decoding="modified_beam_search", beam=4) -> SherpaEngine:
    engine = SherpaEngine(MODELS, decoding, beam, 4, warmup_seconds=1.0)
    engine.load()
    return engine


@pytest.mark.parametrize("decoding", DECODINGS)
def test_engine_equals_reference_and_tokens_are_consistent(decoding):
    engine, ref = loaded(decoding), reference.recognizer(decoding, MODEL_DIR)
    for name in CLIPS:
        wav = sf.read(FIXTURES / name, dtype="float32")[0]
        result, expected = engine.transcribe(wav), reference.transcribe(ref, FIXTURES / name)
        assert result.text == expected.text.strip(), (name, decoding)
        assert result.tokens == tuple(t.replace(" ", "\u2581") for t in expected.tokens), (name, decoding)
        assert result.times == tuple(round(t * SAMPLE_RATE) for t in expected.times), (name, decoding)
        assert result.text and len(result.tokens) == len(result.times)
        assert "".join(result.tokens).replace("\u2581", " ").strip() == result.text
        assert list(result.times) == sorted(result.times) and 0 <= result.times[0] and result.times[-1] <= len(wav)
        assert result.tokens[0].startswith("\u2581")
        # Raw recognizer times (not repaired by the engine) must pass alignment.
        assert " ".join(w.text for w in words(Span(0, len(wav)), result)) == result.text
    # Written form: casing and punctuation come from the model, verbatim.
    text = engine.transcribe(sf.read(FIXTURES / "vi_a.wav", dtype="float32")[0]).text
    assert text[0].isupper() and text.endswith(".") and "," in text


@pytest.mark.parametrize("decoding", DECODINGS)
def test_byte_fallback_tokens_are_merged_into_the_characters_they_spell(decoding):
    # Probe (sherpa-onnx 1.13.8): the recognizer decodes byte-fallback tokens in its text ("Út") but returns them as raw
    # tokens ("<0xC3>", "<0x9A>", "t"); the engine merges each run so the tokens spell the text.
    path = FIXTURES / "vi_bytes.wav"
    wav = sf.read(path, dtype="float32")[0]
    result = loaded(decoding).transcribe(wav)
    raw = reference.transcribe(reference.recognizer(decoding, MODEL_DIR), path)
    first = next(i for i, t in enumerate(raw.tokens) if t.startswith("<0x"))
    assert raw.tokens[first:first + 2] == ["<0xC3>", "<0x9A>"]
    assert result.text == raw.text.strip() and "Út" in result.text
    assert not any(t.startswith("<0x") for t in result.tokens)
    i = result.tokens.index("Ú")
    assert result.times[i] == round(raw.times[first] * SAMPLE_RATE)
    assert "".join(result.tokens).replace("\u2581", " ").strip() == result.text
    assert " ".join(w.text for w in words(Span(0, len(wav)), result)) == result.text


def test_silence_decodes_without_error_and_tokens_spell_the_text():
    # The model may hallucinate a filler on digital silence; the result must still be self-consistent.
    result = loaded().transcribe(np.zeros(2 * SAMPLE_RATE, dtype=np.float32))
    assert "".join(result.tokens).replace("\u2581", " ").strip() == result.text
    assert len(result.tokens) == len(result.times)


@pytest.mark.parametrize("decoding", DECODINGS)
def test_hard_floor_is_the_shortest_input_the_recognizer_accepts(decoding):
    # A sherpa-onnx or model change that moves it must move MIN_FORWARD_SAMPLES too.
    engine = loaded(decoding)
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
        SherpaEngine(tmp_path, "greedy_search", 4, 4, 1.0).load()


def test_worker_serves_concurrent_lanes_and_is_ready_only_after_warmup():
    clips = {n: sf.read(FIXTURES / n, dtype="float32")[0] for n in CLIPS}
    expected = {n: reference_text(FIXTURES / n, "modified_beam_search") for n in CLIPS}

    async def main():
        cfg = config(models_dir=MODELS, max_chunk_seconds=10.0)
        worker = EngineWorker(
            SherpaEngine(MODELS, "modified_beam_search", 4, 4, cfg.max_chunk_seconds),
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
