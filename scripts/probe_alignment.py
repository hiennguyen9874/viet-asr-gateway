"""Probe the real recognizer's token emission times for the timed formats (docs/baseline/alignment_probe.json).

    uv run scripts/probe_alignment.py --out docs/baseline/alignment_probe.json

CPU only; needs the staged model (scripts/stage_models.py) and fixtures-large/vi_long.wav (scripts/fetch_fixtures.py).
Reports numbers and counts only: no audio, no transcripts.

(a) Resolution: emission times of the speech fixtures for every decoding method -> their common step.
(b) Shift equivariance: prepend `s` digital zeros; when the tokens are unchanged, each time should move by `s`.
(c) Known silences: insert 1 s of digital zeros at quiet points inside utterances (vi_a, vi_b and the ten utterances
    of vi_long), where speech stops at the gap start g0 and resumes at the gap end g1: first token after the gap - g1
    (emission delay at a word onset) and g0 - last token before it (how long the last word lasts after its emission).
(d) Word-start intervals inside utterances (the spoken rhythm, with no silence inserted).
(e) Cues (app.alignment with its current constants) of vi_long and of the composite vi_a | 1 s | vi_b | 1 s | vi_a |
    1 s | vi_b cut into chunks by app.chunking.split: durations, characters, reading rate, gaps across chunks.

Constants derived from the report (docs/api-contract.vi.md §4, docs/baseline.vi.md §6):
- EMISSION_SHIFT = -median of (c) onset delays, in whole steps, only if the delay is constant (its 25th and 75th
  percentiles have the same sign); otherwise 0.
- LAST_WORD_SAMPLES = median of (c) "g0 - last token" minus EMISSION_SHIFT (the last word starts at its shifted
  emission), rounded up to a step, at least one step.
- PAUSE_SAMPLES = 90th percentile of (d), rounded up to a step: a gap longer than normal rhythm.
- MAX_CUE_CHARS (80) and MAX_CUE_SAMPLES (7 s) are subtitle conventions (two 40-character lines, at most 7 s on
  screen), not model properties; (e) shows which of them binds on real speech.
Exit 1 unless app.alignment carries exactly these values, token times follow a shift of the audio (>= 90% within one
step) and no cue straddles the silence where the composite is cut into chunks.
"""

import argparse
import json
import math
import os
import statistics
import sys
from itertools import pairwise
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
SHIFTS = (160, 320, 640, 1000, 1280, 3200, 8000, 16000)
LONG_GAP = (
    4800  # vi_long.wav joins its utterances with 0.3 s of digital zeros (scripts/fetch_fixtures.py)
)
CUT_DB = -35.0  # quiet point: a 10 ms window this far below the clip's loudest one
CUTS_PER_CLIP = 8


def ms(samples: float) -> float:
    return round(samples * 1000 / SR, 1)


def stats(values) -> dict:
    v = sorted(values)
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "min": round(v[0], 1),
        **{f"p{p}": round(float(np.percentile(v, p)), 1) for p in (10, 25, 50, 75, 90, 95)},
        "max": round(v[-1], 1),
        "mean": round(statistics.fmean(v), 1),
    }


def engine(decoding=None, warmup_seconds=1.0):
    from app.config import Config
    from app.engine import SherpaEngine

    models = Path(os.environ.get("GATEWAY_MODELS_DIR", ROOT / "models"))
    e = SherpaEngine(
        models,
        decoding or Config.decoding_method,
        Config.beam_size,
        Config.num_threads,
        warmup_seconds,
    )
    e.load()
    return e


def utterances(x: np.ndarray) -> list[np.ndarray]:
    z = np.diff(np.concatenate(([0], (x == 0).astype(np.int8), [0])))
    runs = [
        (a, b)
        for a, b in zip(np.flatnonzero(z == 1), np.flatnonzero(z == -1), strict=True)
        if b - a >= LONG_GAP
    ]
    bounds = [0] + [int(v) for run in runs for v in run] + [len(x)]
    return [x[a:b] for a, b in zip(bounds[::2], bounds[1::2], strict=True)]


def quiet_points(x: np.ndarray) -> list[int]:
    """Up to CUTS_PER_CLIP quiet 10 ms windows, >= 1 s from the clip edges and from each other."""
    win = SR // 100
    n = len(x) // win
    rms = np.sqrt((x[: n * win].astype(np.float64).reshape(n, win) ** 2).mean(1) + 1e-20)
    db = 20 * np.log10(rms / rms.max())
    out: list[int] = []
    for i in np.argsort(db, kind="stable"):
        if db[i] > CUT_DB or len(out) == CUTS_PER_CLIP:
            break
        p = int(i) * win + win // 2
        if SR <= p <= len(x) - SR and all(abs(p - q) >= SR for q in out):
            out.append(p)
    return sorted(out)


def probe_resolution(clips: dict) -> dict:
    out, steps = {}, []
    for decoding in DECODINGS:
        e = engine(decoding)
        times = [t for x in clips.values() for t in e.transcribe(x).times]
        step = math.gcd(*times)
        steps.append(step)
        out[decoding] = {"tokens": len(times), "step_samples": step}
    return {"step_samples": math.gcd(*steps), "settings": out}


def probe_shift(e, clips: dict, step: int) -> dict:
    rows = []
    for name in SPEECH:
        x = clips[name]
        base = e.transcribe(x)
        for s in SHIFTS:
            r = e.transcribe(np.concatenate([np.zeros(s, np.float32), x]))
            row = {"clip": name, "shift_samples": s, "same_tokens": r.tokens == base.tokens}
            if row["same_tokens"]:
                d = [b - a - s for a, b in zip(base.times, r.times, strict=True)]
                row["within_1_step"] = round(sum(abs(v) <= step for v in d) / len(d), 4)
                row["delta_ms"] = stats([ms(v) for v in d])
                # The first token is stamped at the chunk start whatever the leading silence.
                row["first_token_ms"] = ms(r.times[0])
            rows.append(row)
    same = [r for r in rows if r["same_tokens"]]
    return {
        "rows": rows,
        "same_tokens": f"{len(same)}/{len(rows)}",
        "min_within_1_step": min(r["within_1_step"] for r in same),
    }


def probe_gaps(e, clips: list[np.ndarray]) -> dict:
    onset, tail, same_text, inside = [], [], 0, 0
    for x in clips:
        base = e.transcribe(x).text
        for p in quiet_points(x):
            r = e.transcribe(np.concatenate([x[:p], np.zeros(SR, np.float32), x[p:]]))
            same_text += r.text == base
            g0, g1 = p, p + SR
            before = [t for t in r.times if t < g0]
            after = [t for t in r.times if t >= g0]
            if before and after:
                onset.append(ms(after[0] - g1))
                tail.append(ms(g0 - before[-1]))
                inside += after[0] < g1
    return {
        "gaps": len(onset),
        "text_unchanged": same_text,
        "first_after_gap_minus_g1_ms": stats(onset),
        "first_after_gap_inside_gap": inside,
        "g0_minus_last_before_gap_ms": stats(tail),
    }


def word_starts(r) -> list[int]:
    from app.alignment import WORD_START

    return [t for tok, t in zip(r.tokens, r.times, strict=True) if tok.startswith(WORD_START)]


def probe_rhythm(e, clips: list[np.ndarray]) -> dict:
    gaps = [ms(b - a) for x in clips for a, b in pairwise(word_starts(e.transcribe(x)))]
    return {"word_start_interval_ms": stats(gaps)}


def cues(e, x: np.ndarray, max_chunk: int, known_gaps: list[tuple[int, int]] = ()) -> dict:
    from app.alignment import align, segments
    from app.chunking import split

    spans = split(x, max_chunk)
    ws = align([(s, e.transcribe(x[s.start : s.end])) for s in spans])
    cs = segments(ws)
    # Gaps between the last word of a chunk and the first word of the next one (the only aligned gaps).
    starts = {s.start for s in spans[1:]}
    cross = [
        ms(b.start - a.end) for a, b in pairwise(ws) if any(a.start < c <= b.start for c in starts)
    ]
    out = {
        "audio_ms": ms(len(x)),
        "chunks": len(spans),
        "words": len(ws),
        "cues": len(cs),
        "cue_ms": stats([ms(c.end - c.start) for c in cs]),
        "cue_chars": stats([len(c.text) for c in cs]),
        "cue_chars_per_s": stats(
            [len(c.text) * SR / (c.end - c.start) for c in cs if len(c.text) >= 40]
        ),
        "cross_chunk_gap_ms": stats(cross),
    }
    if known_gaps:
        out["known_gaps"] = [
            {
                "gap_ms": [ms(g0), ms(g1)],
                "chunk_cut_inside": any(g0 <= c <= g1 for c in starts),
                "cues_straddling": sum(c.start < g0 and c.end > g1 for c in cs),
                "first_word_after_minus_g1_ms": ms(min(w.start for w in ws if w.start >= g0) - g1),
            }
            for g0, g1 in known_gaps
        ]
    return out


def probe_cues(e, clips: dict, max_chunk: int) -> dict:
    parts = [
        clips["vi_a.wav"],
        np.zeros(SR, np.float32),
        clips["vi_b.wav"],
        np.zeros(SR, np.float32),
    ] * 2
    parts = parts[:-1]
    edges = np.cumsum([0] + [len(p) for p in parts])
    known = [(int(edges[i]), int(edges[i + 1])) for i in (1, 3, 5)]
    return {
        "composite": cues(e, np.concatenate(parts), max_chunk, known),
        "vi_long": cues(e, clips["vi_long.wav"], max_chunk),
    }


def up(samples_ms: float, step: int) -> int:
    return math.ceil(samples_ms * SR / 1000 / step) * step


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if not LONG.is_file():
        sys.exit(f"{LONG} missing: run scripts/fetch_fixtures.py")

    import sherpa_onnx

    from app import alignment
    from app.config import Config

    clips = {n: sf.read(FIXTURES / n, dtype="float32")[0] for n in SPEECH}
    clips["vi_long.wav"] = sf.read(LONG, dtype="float32")[0]
    utts = [clips[n] for n in SPEECH] + utterances(clips["vi_long.wav"])
    max_chunk = int(Config.max_chunk_seconds * SR)

    resolution = probe_resolution({n: clips[n] for n in SPEECH})
    step = resolution["step_samples"]
    e = engine(warmup_seconds=Config.max_chunk_seconds)
    report = {
        "sherpa_onnx": sherpa_onnx.__version__,
        "settings": {
            "model": "viet-asr int8",
            "decoding_method": Config.decoding_method,
            "beam_size": Config.beam_size,
            "threads": Config.num_threads,
            "note": "(a) runs every decoding method",
        },
        "max_chunk_seconds": Config.max_chunk_seconds,
        "a_resolution": resolution,
        "b_shift": probe_shift(e, clips, step),
        "c_known_silences": probe_gaps(e, utts),
        "d_rhythm": probe_rhythm(e, utts),
    }
    onset = report["c_known_silences"]["first_after_gap_minus_g1_ms"]
    constant = (onset["p25"] > 0) == (onset["p75"] > 0)
    shift = -round(onset["p50"] * SR / 1000 / step) * step if constant else 0
    tail_ms = report["c_known_silences"]["g0_minus_last_before_gap_ms"]["p50"]
    derived = {
        "EMISSION_SHIFT": shift,
        "LAST_WORD_SAMPLES": max(step, up(tail_ms - ms(shift), step)),
        "PAUSE_SAMPLES": up(report["d_rhythm"]["word_start_interval_ms"]["p90"], step),
        "MAX_CUE_SAMPLES": 7 * SR,
        "MAX_CUE_CHARS": 80,
    }
    current = {k: getattr(alignment, k) for k in derived}
    report["constants"] = {
        "derived": derived,
        "app.alignment": current,
        "onset_delay_constant": constant,
    }
    report["e_cues"] = probe_cues(e, clips, max_chunk)

    failures = []
    if current != derived:
        failures.append("app.alignment constants differ from the derived ones")
    if report["b_shift"]["min_within_1_step"] < 0.9:
        failures.append("token times do not follow a shift of the audio")
    for g in report["e_cues"]["composite"]["known_gaps"]:
        if g["chunk_cut_inside"] and g["cues_straddling"]:
            failures.append("a cue straddles the silence where the chunks were cut")
    report["failures"] = failures

    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
