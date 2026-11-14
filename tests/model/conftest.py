"""Real-model tests run the sherpa-onnx recognizer on CPU and need the staged model (scripts/stage_models.py)."""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("GATEWAY_MODELS_DIR", ROOT / "models"))
MODEL_DIR = MODELS / "gipformer1.5-68M-rnnt"
FIXTURES = ROOT / "tests/fixtures"


def pytest_collection_modifyitems(items):
    skip = pytest.mark.skip(reason="model not staged (scripts/stage_models.py)")
    for item in items:
        if "tests/model/" in item.nodeid:
            item.add_marker(pytest.mark.model)
            if not (MODEL_DIR / "tokens.txt").is_file():
                item.add_marker(skip)


def reference_text(audio: Path, precision: str, decoding_method: str) -> str:
    """Output of the reference inference script (references/gipformer/infer_onnx.py) for the same settings."""
    spec = importlib.util.spec_from_file_location("infer_onnx", ROOT / "references/gipformer/infer_onnx.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    paths = ref.load_model(precision, model_dir=str(MODEL_DIR))
    recognizer = ref.create_recognizer(paths, num_threads=4, decoding_method=decoding_method)
    return ref.transcribe(recognizer, str(audio))
