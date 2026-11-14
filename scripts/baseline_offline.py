"""Offline CPU baseline of the real recognizer (docs/baseline/offline.json): speed and accuracy for every precision x
decoding method, a thread sweep for the default settings, and the capacity defaults derived from them.

    uv run scripts/baseline_offline.py --out docs/baseline/offline.json

CPU only; needs the staged model (scripts/stage_models.py) and fixtures-large/vi_long.wav (scripts/fetch_fixtures.py).
Every measurement runs in a fresh process, so load time and peak RSS belong to that setting only. The engine is
used exactly as the gateway uses it: warm-up on GATEWAY_MAX_CHUNK_SECONDS of zeros, fixtures as one chunk each,
vi_long.wav split by `app.chunking.split` and joined by `app.transcription.Joiner`. The report holds numbers, counts
and SHA256 of transcripts only (no audio, no text); scripts/smoke_server.py compares the gateway's transcripts with
the hashes of the default setting.

(a) Settings: fp32/int8 x modified_beam_search/greedy_search at `--threads`: load, warm-up, per fixture median latency
    over the repeats, real-time factor (RTF = latency / audio seconds), determinism, WER against FLEURS.
(b) Threads: the default setting at 1, 2, 4, 8, 16 threads: latency of one max-length chunk (the worst single call:
    the longest a realtime job waits behind a running file chunk) and RTF of vi_long.wav.
(c) Derived: the default thread count and the bounds the queue defaults must respect (rules in `derive`,
    docs/baseline.vi.md §4).

This is a single-stream offline measurement on a shared host, not a capacity claim (docs/roadmap.md §1). Exit 1
when the defaults in `app.config.Config` differ from the derived ones, or a decode is not deterministic.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402

from app.config import Config  # noqa: E402
from app.config import SAMPLE_RATE as SR  # noqa: E402

FIXTURES = ROOT / "tests/fixtures"
LONG = ROOT / "fixtures-large/vi_long.wav"
SMALL = ("vi_a.wav", "vi_b.wav", "silence_2s.wav", "short_200ms.wav")
PRECISIONS = ("fp32", "int8")
DECODINGS = ("modified_beam_search", "greedy_search")
THREADS = (1, 2, 4, 8, 16)
REPEATS = 5
LONG_REPEATS = 3
# (c): the most threads that still run at this parallel efficiency (speed-up over 1 thread / threads) on the max
# chunk. More threads buy little on one stream and take cores from FFmpeg decodes and the event loop.
MIN_THREAD_EFFICIENCY = 0.5
# (c): a call this many times the worst max-chunk latency measured (any setting, any thread count) counts as hung.
HANG_MARGIN = 20


def models_dir() -> Path:
    return Path(os.environ.get("GATEWAY_MODELS_DIR", ROOT / "models"))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def peak_rss_mb() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def timed(fn, repeats: int) -> tuple[list[float], list]:
    times, outs = [], []
    for _ in range(repeats):
        t0 = time.monotonic()
        outs.append(fn())
        times.append(time.monotonic() - t0)
    return times, outs


def latency(times: list[float], audio_s: float) -> dict:
    median = statistics.median(times)
    return {
        "median_s": round(median, 4),
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
        "rtf": round(median / audio_s, 4),
        "rtf_max": round(max(times) / audio_s, 4),
    }


def child(precision: str, decoding: str, threads: int, part: str) -> dict:
    """One setting in this (fresh) process."""
    from app.chunking import split
    from app.engine import SherpaEngine
    from app.transcription import Joiner
    from scripts.scoring import wer

    cfg = Config(api_keys=("x" * 16,))
    refs = json.loads((FIXTURES / "manifest.json").read_text())["files"]
    t0 = time.monotonic()
    e = SherpaEngine(
        models_dir(), precision, decoding, cfg.beam_size, threads, cfg.max_chunk_seconds
    )
    e.load()
    load_s, load_peak = time.monotonic() - t0, peak_rss_mb()
    t0 = time.monotonic()
    e.warmup()
    out = {
        "load_s": round(load_s, 3),
        "warmup_s": round(time.monotonic() - t0, 3),
        "peak_rss_load_mb": load_peak,
    }

    x = sf.read(LONG, dtype="float32")[0]
    max_chunk = x[: cfg.max_chunk_samples]  # speech, the longest single call the gateway makes
    times, outs = timed(lambda: e.transcribe(max_chunk).text, REPEATS)
    out["max_chunk"] = {
        "audio_s": len(max_chunk) / SR,
        **latency(times, len(max_chunk) / SR),
        "deterministic": len(set(outs)) == 1,
    }

    def transcribe_file() -> tuple[str, int]:
        joiner = Joiner()
        spans = split(x, cfg.max_chunk_samples)
        for s in spans:
            joiner.add(e.transcribe(x[s.start : s.end]).text)
        return joiner.text, len(spans)

    times, outs = timed(transcribe_file, LONG_REPEATS)
    text, chunks = outs[0]
    out["vi_long.wav"] = {
        "audio_s": len(x) / SR,
        "chunks": chunks,
        **latency(times, len(x) / SR),
        "deterministic": len(set(outs)) == 1,
        "wer": round(wer(refs["vi_long.wav"]["reference"], text.lower()), 4),
        "text_sha256": sha256(text),
    }
    if part == "settings":
        for name in SMALL:
            wav = sf.read(FIXTURES / name, dtype="float32")[0]
            times, outs = timed(lambda wav=wav: e.transcribe(wav).text, REPEATS)
            row = {
                "audio_s": len(wav) / SR,
                **latency(times, len(wav) / SR),
                "deterministic": len(set(outs)) == 1,
                "empty": outs[0] == "",
                "text_sha256": sha256(outs[0]),
            }
            if refs[name].get("reference"):
                # The model's vocabulary is upper case; FLEURS references are lower case.
                row["wer"] = round(wer(refs[name]["reference"], outs[0].lower()), 4)
            out[name] = row
    out["peak_rss_mb"] = peak_rss_mb()
    return out


def run_child(precision: str, decoding: str, threads: int, part: str) -> dict:
    cmd = [sys.executable, __file__, "--child", precision, decoding, str(threads), part]
    row = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    print(f"{part} {precision}/{decoding} x{threads}: {row['max_chunk']}", file=sys.stderr)
    return row


def derive(threads: dict, worst_max_chunk_s: float) -> dict:
    """Default thread count and the bounds of the queue defaults (docs/baseline.vi.md §4)."""
    one = threads["1"]["max_chunk"]["median_s"]
    efficiency = {
        t: round(one / (int(t) * r["max_chunk"]["median_s"]), 3) for t, r in threads.items()
    }
    num_threads = max(int(t) for t, e in efficiency.items() if e >= MIN_THREAD_EFFICIENCY)
    chosen = threads[str(num_threads)]
    rtf = max(chosen["max_chunk"]["rtf_max"], chosen["vi_long.wav"]["rtf_max"])
    cfg = Config(api_keys=("x" * 16,))
    # An admitted chunk may wait for everything buffered ahead of it (queued + running audio). That wait must not
    # exceed the time a client is made to wait for admission (GATEWAY_QUEUE_TIMEOUT_SECONDS).
    return {
        "rules": {"min_thread_efficiency": MIN_THREAD_EFFICIENCY, "hang_margin": HANG_MARGIN},
        "thread_efficiency": efficiency,
        "num_threads": num_threads,
        "rtf_worst_default": rtf,
        "queue_max_buffered_seconds_max": math.floor(cfg.queue_timeout_seconds / rtf),
        "queue_wait_at_default_s": round(cfg.queue_max_buffered_seconds * rtf, 1),
        "live_wait_behind_file_chunk_s": chosen["max_chunk"]["max_s"],
        "worst_max_chunk_s": worst_max_chunk_s,
        "infer_hang_seconds_min": math.ceil(worst_max_chunk_s * HANG_MARGIN),
        # The longest file alone, at the worst default RTF: request_timeout leaves room for this many in parallel.
        "max_audio_alone_s": round(cfg.max_audio_seconds * rtf, 1),
        "full_length_files_per_request_timeout": int(
            cfg.request_timeout_seconds // (cfg.max_audio_seconds * rtf)
        ),
    }


def git_commit() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--threads", type=int, default=Config.num_threads, help="threads for part (a)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--child", nargs=4, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        precision, decoding, threads, part = args.child
        print(json.dumps(child(precision, decoding, int(threads), part)))
        return 0
    if not LONG.is_file():
        sys.exit(f"{LONG} missing: run scripts/fetch_fixtures.py")

    import sherpa_onnx

    cfg = Config(api_keys=("x" * 16,))
    default = f"{cfg.precision}/{cfg.decoding_method}"
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    report = {
        "commit": git_commit(),
        "sherpa_onnx": sherpa_onnx.__version__,
        "python": platform.python_version(),
        "model_revision": manifest["models"]["gipformer1.5-68M-rnnt"]["revision"],
        "cpu": {"count": os.cpu_count(), "model": cpu_model()},
        "load_average_start": [round(v, 1) for v in os.getloadavg()],
        "beam_size": cfg.beam_size,
        "max_chunk_seconds": cfg.max_chunk_seconds,
        "repeats": {"fixtures": REPEATS, "vi_long": LONG_REPEATS},
        "default": default,
        "a_settings": {
            "threads": args.threads,
            "rows": {
                f"{p}/{d}": run_child(p, d, args.threads, "settings")
                for p in PRECISIONS
                for d in DECODINGS
            },
        },
        "b_threads": {
            "setting": default,
            "rows": {
                str(t): run_child(cfg.precision, cfg.decoding_method, t, "threads") for t in THREADS
            },
        },
    }
    worst = max(
        r["max_chunk"]["max_s"]
        for part in ("a_settings", "b_threads")
        for r in report[part]["rows"].values()
    )
    report["c_derived"] = derive(report["b_threads"]["rows"], worst)
    report["load_average_end"] = [round(v, 1) for v in os.getloadavg()]

    d = report["c_derived"]
    checks = {
        "deterministic": all(
            v["deterministic"]
            for part in ("a_settings", "b_threads")
            for r in report[part]["rows"].values()
            for v in r.values()
            if isinstance(v, dict) and "deterministic" in v
        ),
        "num_threads": cfg.num_threads == d["num_threads"],
        "queue_max_buffered_seconds": cfg.queue_max_buffered_seconds
        <= d["queue_max_buffered_seconds_max"],
        "infer_hang_seconds": cfg.infer_hang_seconds >= d["infer_hang_seconds_min"],
    }
    report["checks"] = checks
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
