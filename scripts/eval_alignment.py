"""M4 P4: evaluate word timestamps against hand-annotated boundaries (the R2b gate, docs/roadmap.md).

    uv run scripts/eval_alignment.py --out docs/baseline/alignment_eval.json

Annotations: tests/fixtures/alignment/*.json, one utterance each, labelled by hand (Praat/Audacity), never by
another aligner:

    {"audio": "<path relative to the repo>", "sha256": "<of the audio file>", "split": "dev" | "test",
     "speaker": "<optional, e.g. female/male>", "words": [{"word": "các", "start": 0.95, "end": 1.21}, ...]}

Words are whitespace units of the transcript (A3). Hypothesis words (real SherpaEngine, server chunking, app.alignment)
are matched to reference words by an edit alignment of the word strings; unmatched words are ASR errors, reported
apart and not counted in the timing error. The thresholds below were fixed before measuring; do not tune them.
The gate is computed on the `test` split only (`dev` is for choosing at most one end-extension rule). Exit 1 if
the gate fails or the data is insufficient.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANNOTATIONS = ROOT / "tests/fixtures/alignment"
MIN_WORDS, MIN_UTTERANCES = 200, 10
TOLERANCE_S = 0.100
MIN_WITHIN = 0.90
MAX_MEDIAN_S = 0.050


def match(ref: Sequence[dict], hyp: Sequence[dict]) -> list[tuple[dict, dict]]:
    """(reference, hypothesis) word pairs with equal text, by edit alignment of the word sequences."""
    # Full matrix permits deterministic minimum-edit traceback; annotations are short utterances.
    d = [list(range(len(hyp) + 1))]
    for i, r in enumerate(ref, 1):
        row = [i]
        for j, h in enumerate(hyp, 1):
            row.append(
                min(d[i - 1][j] + 1, row[j - 1] + 1, d[i - 1][j - 1] + (r["word"] != h["word"]))
            )
        d.append(row)
    pairs = []
    i, j = len(ref), len(hyp)
    while i or j:
        # Prefer diagonal steps on ties, then reference deletions, then insertions.
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1]["word"] != hyp[j - 1]["word"]):
            if ref[i - 1]["word"] == hyp[j - 1]["word"]:
                pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def valid(hyp: Sequence[dict], duration: float) -> bool:
    return all(0 <= w["start"] < w["end"] <= duration for w in hyp) and all(
        a["start"] <= b["start"] for a, b in pairwise(hyp)
    )


def score(utterances: Sequence[tuple[Sequence[dict], Sequence[dict], float]]) -> dict:
    """utterances: (reference words, hypothesis words, duration s). Returns metrics and the gate verdict."""
    ds, de = [], []
    ref_words = hyp_words = 0
    all_valid = True
    for ref, hyp, duration in utterances:
        ref_words += len(ref)
        hyp_words += len(hyp)
        all_valid &= valid(hyp, duration)
        for r, h in match(ref, hyp):
            ds.append(abs(h["start"] - r["start"]))
            de.append(abs(h["end"] - r["end"]))
    n = len(ds)
    out = {
        "utterances": len(utterances),
        "reference_words": ref_words,
        "hypothesis_words": hyp_words,
        "matched_words": n,
        "unmatched_reference_share": round(1 - n / ref_words, 4) if ref_words else None,
        "all_spans_valid_and_monotonic": all_valid,
    }
    timing_passed = False
    if n:
        start_within = sum(d <= TOLERANCE_S for d in ds) / n
        end_within = sum(d <= TOLERANCE_S for d in de) / n
        start_median, end_median = statistics.median(ds), statistics.median(de)
        timing_passed = (
            start_within >= MIN_WITHIN
            and end_within >= MIN_WITHIN
            and start_median <= MAX_MEDIAN_S
            and end_median <= MAX_MEDIAN_S
        )
        out |= {
            "start_within_100ms": round(start_within, 4),
            "end_within_100ms": round(end_within, 4),
            "start_median_abs_ms": round(start_median * 1000, 1),
            "end_median_abs_ms": round(end_median * 1000, 1),
        }
    enough = len(utterances) >= MIN_UTTERANCES and ref_words >= MIN_WORDS
    out["enough_data"] = enough
    out["gate_passed"] = bool(enough and all_valid and timing_passed)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ.get("GATEWAY_MODELS_DIR", ROOT / "models")),
    )
    ap.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    ap.add_argument("--max-chunk-seconds", type=float, default=20.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    files = sorted(args.annotations.glob("*.json")) if args.annotations.is_dir() else []
    if not files:
        print(f"no annotations in {args.annotations} (docs/roadmap.md, R2b)", file=sys.stderr)
        return 1

    import soundfile as sf

    from app.alignment import align
    from app.chunking import split
    from app.engine import SherpaEngine
    from app.formats import seconds

    engine = SherpaEngine(
        args.models_dir, "fp32", "modified_beam_search", 4, 4, warmup_seconds=args.max_chunk_seconds
    )
    engine.load()
    max_chunk = int(args.max_chunk_seconds * 16000)
    by_split: dict[str, list] = {}
    for path in files:
        ann = json.loads(path.read_text())
        audio = ROOT / ann["audio"]
        if hashlib.sha256(audio.read_bytes()).hexdigest() != ann["sha256"]:
            raise SystemExit(f"{path.name}: audio hash mismatch")
        x, sr = sf.read(audio, dtype="float32")
        if sr != 16000 or x.ndim != 1:
            raise SystemExit(f"{path.name}: audio must be 16 kHz mono")
        chunks = [(s, engine.transcribe(x[s.start : s.end])) for s in split(x, max_chunk)]
        hyp = [
            {"word": w.text, "start": seconds(w.start), "end": seconds(w.end)}
            for w in align(chunks)
        ]
        by_split.setdefault(ann["split"], []).append((ann["words"], hyp, len(x) / sr))

    report = {
        "annotations": [p.name for p in files],
        "thresholds": {
            "tolerance_ms": TOLERANCE_S * 1000,
            "min_within": MIN_WITHIN,
            "max_median_ms": MAX_MEDIAN_S * 1000,
            "min_words": MIN_WORDS,
            "min_utterances": MIN_UTTERANCES,
        },
        "splits": {name: score(u) for name, u in sorted(by_split.items())},
    }
    passed = report["splits"].get("test", {}).get("gate_passed", False)
    report["gate_passed"] = passed
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
