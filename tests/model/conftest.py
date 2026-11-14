"""Real-model tests run the sherpa-onnx recognizer on CPU and need the staged model (scripts/stage_models.py)."""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("GATEWAY_MODELS_DIR", ROOT / "models"))
MODEL_DIR = MODELS / "viet-asr/int8"
FIXTURES = ROOT / "tests/fixtures"


def default_engine():
    """The engine the server builds with the default settings (decoding, beam, threads, warm-up length)."""
    from app.config import Config
    from app.engine import SherpaEngine

    return SherpaEngine(MODELS, Config.decoding_method, Config.beam_size, Config.num_threads,
                        warmup_seconds=Config.max_chunk_seconds)


def default_settings() -> str:
    """Report label of `default_engine`."""
    from app.config import Config

    beam = f", beam {Config.beam_size}" if Config.decoding_method == "modified_beam_search" else ""
    return f"viet-asr int8, {Config.decoding_method}{beam}, {Config.num_threads} threads"


def pytest_collection_modifyitems(items):
    skip = pytest.mark.skip(reason="model not staged (scripts/stage_models.py)")
    for item in items:
        if "tests/model/" in item.nodeid:
            item.add_marker(pytest.mark.model)
            if not (MODEL_DIR / "tokens.txt").is_file():
                item.add_marker(skip)


def reference_text(audio: Path, decoding_method: str) -> str:
    """Text of the minimal sherpa-onnx reference (tests/model/reference.py) for the same settings."""
    from tests.model import reference

    return reference.transcribe(reference.recognizer(decoding_method, MODEL_DIR), audio).text.strip()
