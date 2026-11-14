"""Probe the real recognizer for the short-audio floor and the maximum chunk length (docs/baseline/limits_probe.json).

    uv run scripts/probe_limits.py --out docs/baseline/limits_probe.json

CPU only; needs the staged model (scripts/stage_models.py) and, for part (c), fixtures-large/vi_long.wav
(scripts/fetch_fixtures.py). Reports numbers and counts only: no audio, no transcripts.

(a) Shortest input: for every precision x decoding method, the smallest waveform length (samples at 16 kHz) the
    recognizer accepts, for digital zeros and for noise, found by bisection and then checked exhaustively around it
    and on a grid up to 3 s above it (a length below it must fail, every length checked above it must succeed).
(b) Short speech: cuts of the speech fixtures at several lengths and offsets -> how many cuts yield any text.
(c) Chunk length: for each length, a fresh process loads the default engine (fp32, modified_beam_search, beam 4,
    `--threads`), warms up on that length (as the gateway does with GATEWAY_MAX_CHUNK_SECONDS) and decodes speech
    of that length (vi_long.wav, tiled when shorter): latency, real-time factor, RSS after load and peak RSS.
(d) Accuracy against chunk length: vi_long.wav split by app.chunking.split at each length -> WER.

Exit 1 unless the floor is the same for every setting and input, equals `app.engine.MIN_FORWARD_SAMPLES`, and the
short-audio rule (`app.chunking.MIN_AUDIO_SAMPLES`) never forwards less. The constants and the decision drawn from
this report: docs/baseline.vi.md §5.
"""

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app.config import SAMPLE_RATE as SR  # noqa: E402

FIXTURES = ROOT / "tests/fixtures"
LONG = ROOT / "fixtures-large/vi_long.wav"
PRECISIONS = ("fp32", "int8")
DECODINGS = ("modified_beam_search", "greedy_search")
SPEECH = ("vi_a.wav", "vi_b.wav")
SHORT_LENGTHS = (1360, 1600, 1601, 2400, 3200, 4800, 8000)
SHORT_OFFSETS = 20  # cuts per clip and length, evenly spaced over the clip
CHUNK_SECONDS = (5, 10, 15, 20, 30, 45, 60, 90, 120, 180)
REPEATS = 3


def models_dir() -> Path:
    return Path(os.environ.get("GATEWAY_MODELS_DIR", ROOT / "models"))


def engine(precision="fp32", decoding="modified_beam_search", threads=4, warmup_seconds=1.0):
    from app.engine import SherpaEngine

    e = SherpaEngine(models_dir(), precision, decoding, 4, threads, warmup_seconds)
    e.load()
    return e


def accepts(e, n: int, kind: str) -> bool:
    x = np.zeros(n, np.float32)
    if kind == "noise":
        x = np.random.default_rng(n).normal(0, 0.1, n).astype(np.float32)
    try:
        e.transcribe(x)
    except RuntimeError:  # onnxruntime: the encoder's input convolution has too few feature frames
        return False
    return True


def shortest(e, kind: str) -> dict:
    lo, hi = 0, SR  # lo is rejected (empty input), hi accepted (1 s)
    assert not accepts(e, lo, kind) and accepts(e, hi, kind)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if accepts(e, mid, kind) else (mid, hi)
    below = [n for n in range(0, hi, 7) if accepts(e, n, kind)]
    above = list(range(hi, hi + 400)) + list(range(hi + 400, 3 * SR, 997))
    failed_above = [n for n in above if not accepts(e, n, kind)]
    return {
        "first_accepted": hi,
        "accepted_below": below,  # expected empty
        "checked_above": len(above),
        "rejected_above": failed_above,  # expected empty
    }


def probe_floor() -> dict:
    out = {}
    for precision in PRECISIONS:
        for decoding in DECODINGS:
            e = engine(precision, decoding)
            out[f"{precision}/{decoding}"] = {
                kind: shortest(e, kind) for kind in ("zeros", "noise")
            }
    return out


def probe_short_speech() -> dict:
    e = engine()
    clips = [sf.read(FIXTURES / name, dtype="float32")[0] for name in SPEECH]
    out = {}
    for n in SHORT_LENGTHS:
        cuts = nonempty = 0
        for x in clips:
            for start in np.linspace(0, len(x) - n, SHORT_OFFSETS).astype(int):
                cuts += 1
                nonempty += bool(e.transcribe(x[start : start + n]).text)
        out[str(n)] = {"ms": n * 1000 // SR, "cuts": cuts, "with_text": nonempty}
    return out


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def child(seconds: float, threads: int) -> dict:
    """One chunk length in a fresh process, so the peak RSS belongs to that length only."""
    t0 = time.monotonic()
    e = engine(threads=threads, warmup_seconds=seconds)
    load_s, load_rss = time.monotonic() - t0, rss_mb()
    load_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.monotonic()
    e.warmup()
    warmup_s = time.monotonic() - t0
    n = int(seconds * SR)
    x = sf.read(LONG, dtype="float32")[0]
    x = np.tile(x, -(-n // len(x)))[:n]
    times = []
    for _ in range(REPEATS):
        t0 = time.monotonic()
        tokens = len(e.transcribe(x).tokens)
        times.append(time.monotonic() - t0)
    median = statistics.median(times)
    return {
        "seconds": seconds,
        "load_s": round(load_s, 3),
        "warmup_s": round(warmup_s, 3),
        "latency_median_s": round(median, 3),
        "latency_max_s": round(max(times), 3),
        "rtf": round(median / seconds, 4),
        "tokens": tokens,
        "rss_after_load_mb": round(load_rss),
        "peak_rss_load_mb": round(load_peak),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }


def probe_chunks(threads: int) -> list[dict]:
    rows = []
    for seconds in CHUNK_SECONDS:
        cmd = [sys.executable, __file__, "--child", str(seconds), "--threads", str(threads)]
        rows.append(
            json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
        )
        print(rows[-1], file=sys.stderr)
    return rows


def probe_chunk_wer() -> list[dict]:
    from app.chunking import split
    from scripts.scoring import wer

    reference = json.loads((FIXTURES / "manifest.json").read_text())["files"]["vi_long.wav"][
        "reference"
    ]
    x = sf.read(LONG, dtype="float32")[0]
    e = engine()
    rows = []
    for seconds in CHUNK_SECONDS:
        spans = split(x, int(seconds * SR))
        hyp = " ".join(t for s in spans if (t := e.transcribe(x[s.start : s.end]).text))
        # The model's vocabulary is upper case; FLEURS references are lower case.
        rows.append(
            {"seconds": seconds, "chunks": len(spans), "wer": round(wer(reference, hyp.lower()), 4)}
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--threads", type=int, default=4, help="GATEWAY_NUM_THREADS for part (c)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--child", type=float, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child is not None:
        print(json.dumps(child(args.child, args.threads)))
        return 0
    if not LONG.is_file():
        sys.exit(f"{LONG} missing: run scripts/fetch_fixtures.py")

    import sherpa_onnx

    from app.chunking import MIN_AUDIO_SAMPLES
    from app.config import Config
    from app.engine import MIN_FORWARD_SAMPLES

    report = {
        "sherpa_onnx": sherpa_onnx.__version__,
        "constants": {
            "MIN_FORWARD_SAMPLES": MIN_FORWARD_SAMPLES,
            "MIN_AUDIO_SAMPLES": MIN_AUDIO_SAMPLES,
            "max_chunk_seconds_default": Config.max_chunk_seconds,
        },
        "cpu": {"count": os.cpu_count(), "model": _cpu_model()},
        "a_shortest_input": probe_floor(),
        "b_short_speech": probe_short_speech(),
        "c_chunk_length": {
            "threads": args.threads,
            "repeats": REPEATS,
            "rows": probe_chunks(args.threads),
        },
        "d_chunk_wer_vi_long": probe_chunk_wer(),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    floors = {
        k: v[kind]["first_accepted"] for k, v in report["a_shortest_input"].items() for kind in v
    }
    consistent = all(
        not v[kind]["accepted_below"] and not v[kind]["rejected_above"]
        for v in report["a_shortest_input"].values()
        for kind in v
    )
    # The engine's hard floor must be the measured one, and the short-audio rule must never forward less.
    ok = (
        consistent
        and set(floors.values()) == {MIN_FORWARD_SAMPLES}
        and MIN_AUDIO_SAMPLES >= MIN_FORWARD_SAMPLES
    )
    return 0 if ok else 1


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


if __name__ == "__main__":
    sys.exit(main())
