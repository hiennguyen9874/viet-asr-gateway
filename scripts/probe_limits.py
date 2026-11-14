"""Probe the real recognizer for the short-audio floor and the maximum chunk length (docs/baseline/limits_probe.json).

    uv run scripts/probe_limits.py --out docs/baseline/limits_probe.json

CPU only; needs the staged model (scripts/stage_models.py), fixtures-large/vi_long.wav and the local eval set
(scripts/fetch_fixtures.py [--eval]). Reports numbers and counts only: no audio, no transcripts. Parts (b)-(d) use the
default engine settings of `app.config.Config` (decoding method, beam size, threads).

(a) Shortest input: for every decoding method, the smallest waveform length (samples at 16 kHz) the
    recognizer accepts, for digital zeros and for noise, found by bisection and then checked exhaustively around it
    and on a grid up to 3 s above it (a length below it must fail, every length checked above it must succeed).
(b) Short speech: cuts of the speech fixtures at several lengths and offsets -> how many cuts yield any text.
(c) Chunk length: for each length (up to 30 s, the model's training length), a fresh process loads the default
    engine, warms up on that length (as the gateway does with GATEWAY_MAX_CHUNK_SECONDS) and decodes speech of that
    length (vi_long.wav, tiled when shorter): latency, real-time factor, RSS after load and peak RSS.
(d) Accuracy against chunk length: the eval set's FLEURS clips joined into recordings of EVAL_GROUP utterances
    (0.3 s of digital zeros between them, as vi_long.wav) and vi_long.wav, split by app.chunking.split at each length
    -> normalized corpus WER, and at each chunk cut whether the chunk before ends in punctuation and the chunk after
    starts with a capital letter, for cuts inside an utterance and cuts between utterances.
(e) Derived GATEWAY_MAX_CHUNK_SECONDS (rules in `derive`, docs/baseline.vi.md §5).

Exit 1 unless the floor is the same for every setting and input, equals `app.engine.MIN_FORWARD_SAMPLES`, the
short-audio rule (`app.chunking.MIN_AUDIO_SAMPLES`) never forwards less, and `Config.max_chunk_seconds` equals (e).
"""

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app.config import SAMPLE_RATE as SR  # noqa: E402

FIXTURES = ROOT / "tests/fixtures"
LONG = ROOT / "fixtures-large/vi_long.wav"
DECODINGS = ("modified_beam_search", "greedy_search")
SPEECH = ("vi_a.wav", "vi_b.wav")
SHORT_LENGTHS = (1360, 1600, 1601, 2400, 3200, 4800, 8000)
SHORT_OFFSETS = 20  # cuts per clip and length, evenly spaced over the clip
CHUNK_SECONDS = (5, 10, 15, 20, 25, 30)  # the model was trained on utterances of at most 30 s
REPEATS = 3
EVAL_GROUP = 10  # utterances per eval recording in (d)
GAP = int(0.3 * SR)
# A cut is inside a sentence when there is speech (a 50 ms window within SPEECH_DB of the clip's loudest one) on both
# sides of it in the same utterance (each FLEURS clip is one sentence), at least MARGIN away from the cut.
SPEECH_DB = -30.0
MARGIN = SR // 5
# (e): a LIVE job may wait behind one running FILE chunk, so a chunk's worst latency must stay under this.
MAX_CHUNK_LATENCY_S = 1.0
# (e): a chunk length may raise the process's peak RSS by at most this factor over the load peak.
MAX_RSS_GROWTH = 1.25
# (e): corpus WERs this close to the best one are a tie (one 200-clip set); the longest tied length wins, since every
# chunk cut costs a written-form artifact (the model ends each chunk as a sentence).
WER_TIE = 0.005


def models_dir() -> Path:
    return Path(os.environ.get("GATEWAY_MODELS_DIR", ROOT / "models"))


def engine(decoding=None, threads=None, warmup_seconds=1.0):
    from app.config import Config
    from app.engine import SherpaEngine

    decoding = decoding or Config.decoding_method
    threads = threads or Config.num_threads
    e = SherpaEngine(models_dir(), decoding, Config.beam_size, threads, warmup_seconds)
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
    for decoding in DECODINGS:
        e = engine(decoding)
        out[decoding] = {kind: shortest(e, kind) for kind in ("zeros", "noise")}
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


def recordings() -> list[tuple[np.ndarray, str, list[tuple[int, int]]]]:
    """(audio, reference, utterance intervals): the eval clips in groups of EVAL_GROUP, then vi_long.wav."""
    from scripts.fetch_fixtures import eval_set

    speech, _ = eval_set()
    items = list(speech.items())
    out = []
    for i in range(0, len(items), EVAL_GROUP):
        parts, refs, utts, pos = [], [], [], 0
        for path, ref in items[i : i + EVAL_GROUP]:
            x = sf.read(path, dtype="float32")[0]
            utts.append((pos, pos + len(x)))
            parts += [x, np.zeros(GAP, np.float32)]
            refs.append(ref["reference"])
            pos += len(x) + GAP
        out.append((np.concatenate(parts[:-1]), " ".join(refs), utts))
    x = sf.read(LONG, dtype="float32")[0]
    ref = json.loads((FIXTURES / "manifest.json").read_text())["files"]["vi_long.wav"]["reference"]
    z = np.diff(np.concatenate(([0], (x == 0).astype(np.int8), [0])))
    gaps = [
        (a, b)
        for a, b in zip(np.flatnonzero(z == 1), np.flatnonzero(z == -1), strict=True)
        if b - a >= GAP
    ]
    bounds = [0] + [int(v) for g in gaps for v in g] + [len(x)]
    out.append((x, ref, list(zip(bounds[::2], bounds[1::2], strict=True))))
    return out


def loudest(x: np.ndarray) -> float:
    """RMS of the loudest 50 ms window (0 for less than one window)."""
    win = SR // 20
    n = len(x) // win
    if n == 0:
        return 0.0
    return float(np.sqrt((x[: n * win].astype(np.float64).reshape(n, win) ** 2).mean(1)).max())


def wer_child(seconds: float) -> dict:
    """(d) at one chunk length, in a fresh process (the lengths run in parallel)."""
    from app.chunking import split
    from scripts.scoring import errors

    e = engine()
    cuts = {k: {"cuts": 0, "punct_before": 0, "capital_after": 0} for k in ("inside", "between")}
    totals = {"eval": [0, 0, 0], "vi_long": [0, 0, 0]}  # edits, reference words, chunks
    recs = recordings()
    for i, (x, ref, utts) in enumerate(recs):
        spans = split(x, int(seconds * SR))
        texts = [e.transcribe(x[s.start : s.end]).text for s in spans]
        edits, words = errors(ref, " ".join(t for t in texts if t))
        t = totals["vi_long" if i == len(recs) - 1 else "eval"]
        t[0], t[1], t[2] = t[0] + edits, t[1] + words, t[2] + len(spans)
        for k in range(1, len(spans)):
            c, before, after = spans[k].start, texts[k - 1], texts[k]
            inside = any(
                a < c < b
                and min(loudest(x[a : c - MARGIN]), loudest(x[c + MARGIN : b]))
                >= loudest(x[a:b]) * 10 ** (SPEECH_DB / 20)
                for a, b in utts
            )
            row = cuts["inside" if inside else "between"]
            row["cuts"] += 1
            row["punct_before"] += bool(before) and unicodedata.category(before[-1]).startswith("P")
            row["capital_after"] += bool(after) and after[0].isupper()
    return {
        "seconds": seconds,
        **{
            name: {"chunks": t[2], "words": t[1], "wer": round(t[0] / max(t[1], 1), 4)}
            for name, t in totals.items()
        },
        "eval_recordings": len(recs) - 1,
        "cuts": cuts,
    }


def probe_chunk_wer() -> list[dict]:
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, "--wer-child", str(s)], stdout=subprocess.PIPE, text=True
        )
        for s in CHUNK_SECONDS
    ]
    rows = []
    for p in procs:
        out, _ = p.communicate()
        if p.returncode:
            raise SystemExit(f"wer child failed: {p.args}")
        rows.append(json.loads(out))
        print(rows[-1], file=sys.stderr)
    return rows


def derive(chunks: list[dict], wers: list[dict]) -> dict:
    """Default GATEWAY_MAX_CHUNK_SECONDS: among the lengths whose worst latency and peak RSS stay in bounds, the
    longest whose eval WER ties the best one (fewer chunk cuts)."""
    wer = {r["seconds"]: r["eval"]["wer"] for r in wers}
    ok = [
        r["seconds"]
        for r in chunks
        if r["latency_max_s"] <= MAX_CHUNK_LATENCY_S
        and r["peak_rss_mb"] <= MAX_RSS_GROWTH * r["peak_rss_load_mb"]
    ]
    best = min((wer[s] for s in ok), default=None)
    chosen = max((s for s in ok if wer[s] <= best + WER_TIE), default=None)
    return {
        "rules": {
            "max_chunk_latency_s": MAX_CHUNK_LATENCY_S,
            "max_rss_growth": MAX_RSS_GROWTH,
            "wer_tie": WER_TIE,
        },
        "within_bounds": ok,
        "best_eval_wer": best,
        "max_chunk_seconds": chosen,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--threads", type=int, help="GATEWAY_NUM_THREADS for part (c) (default: the config's)"
    )
    ap.add_argument("--out", type=Path)
    ap.add_argument("--child", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--wer-child", type=float, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child is not None:
        print(json.dumps(child(args.child, args.threads)))
        return 0
    if args.wer_child is not None:
        print(json.dumps(wer_child(args.wer_child)))
        return 0
    if not LONG.is_file():
        sys.exit(f"{LONG} missing: run scripts/fetch_fixtures.py")

    import sherpa_onnx

    from app.chunking import MIN_AUDIO_SAMPLES
    from app.config import Config
    from app.engine import MIN_FORWARD_SAMPLES

    threads = args.threads or Config.num_threads
    report = {
        "sherpa_onnx": sherpa_onnx.__version__,
        "settings": {
            "decoding_method": Config.decoding_method,
            "beam_size": Config.beam_size,
            "threads": threads,
        },
        "constants": {
            "MIN_FORWARD_SAMPLES": MIN_FORWARD_SAMPLES,
            "MIN_AUDIO_SAMPLES": MIN_AUDIO_SAMPLES,
            "max_chunk_seconds_default": Config.max_chunk_seconds,
        },
        "cpu": {"count": os.cpu_count(), "model": _cpu_model()},
        "a_shortest_input": probe_floor(),
        "b_short_speech": probe_short_speech(),
        "c_chunk_length": {
            "threads": threads,
            "repeats": REPEATS,
            "rows": probe_chunks(threads),
        },
        "d_chunk_wer": probe_chunk_wer(),
    }
    report["e_derived"] = derive(report["c_chunk_length"]["rows"], report["d_chunk_wer"])
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
        and report["e_derived"]["max_chunk_seconds"] == Config.max_chunk_seconds
    )
    report["checks_ok"] = ok
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
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
