"""Minimal sherpa-onnx reference for the engine parity test: the staged viet-asr int8 files, called directly.

Independent of app/: same files and recognizer settings as `SherpaEngine`, raw recognizer output (tokens as
sherpa-onnx returns them, times in seconds).

    uv run python -m tests.model.reference tests/fixtures/vi_a.wav [--decoding greedy_search]
"""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/viet-asr/int8"


class Output(NamedTuple):
    text: str
    tokens: list[str]
    times: list[float]


def recognizer(decoding_method: str, model_dir: Path = MODEL_DIR, beam_size: int = 4, num_threads: int = 4):
    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=decoding_method,
        max_active_paths=beam_size,
    )


def transcribe(rec, audio: Path) -> Output:
    wav, sr = sf.read(audio, dtype="float32")
    assert sr == 16000 and wav.ndim == 1, "16 kHz mono expected"
    stream = rec.create_stream()
    stream.accept_waveform(sr, wav)
    rec.decode_streams([stream])
    r = stream.result
    return Output(r.text, list(r.tokens), list(r.timestamps))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--decoding", default="modified_beam_search", choices=("modified_beam_search", "greedy_search"))
    args = ap.parse_args()
    print(json.dumps(transcribe(recognizer(args.decoding), args.audio)._asdict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
