"""Build the Vietnamese audio fixtures from FLEURS vi_vn (CC BY 4.0) at a pinned dataset revision.

    uv run scripts/fetch_fixtures.py            # small set -> tests/fixtures/, large set -> fixtures-large/
    uv run scripts/fetch_fixtures.py --verify   # offline: check files against tests/fixtures/manifest.json

The small set is committed. The large set (long multi-utterance recording) is rebuilt locally and is gitignored.
Every file is derived deterministically from the pinned source; the manifest records SHA256 and reference text.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SMALL_DIR = ROOT / "tests" / "fixtures"
LARGE_DIR = ROOT / "fixtures-large"
MANIFEST = SMALL_DIR / "manifest.json"

DATASET = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/vi_vn"
SR = 16000

CLIP_A = "10282809127467658620.wav"  # two clips with different content, for isolation tests
CLIP_B = "10467396475561744099.wav"
LONG_COUNT = 10  # utterances in the long recording (> 30 s, so it exercises splitting)
LONG_GAP_S = 0.3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str, cache: Path) -> Path:
    path = cache / REVISION / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name}", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", tmp)
        tmp.rename(path)
    return path


def write_wav(path: Path, x: np.ndarray) -> None:
    # PCM_16, not FLOAT: libsndfile adds a timestamped PEAK chunk to float WAV, which breaks reproducible hashes.
    sf.write(path, x, SR, subtype="PCM_16")


def build(cache: Path) -> dict:
    rows = list(
        csv.reader(
            fetch("dev.tsv", cache).open(encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE
        )
    )
    by_file = {r[1]: r for r in rows}
    clean = sorted((r for r in rows if not any(c.isdigit() for c in r[2])), key=lambda r: r[1])
    long_rows = [r for r in clean if r[1] not in (CLIP_A, CLIP_B)]
    seen, picked = set(), []
    for r in long_rows:  # one recording per sentence id
        if r[0] not in seen:
            seen.add(r[0])
            picked.append(r)
        if len(picked) == LONG_COUNT:
            break

    wanted = {CLIP_A, CLIP_B, *(r[1] for r in picked)}
    audio = {}
    with tarfile.open(fetch("audio/dev.tar.gz", cache)) as tar:
        for member in tar:
            name = member.name.rsplit("/", 1)[-1]
            if name in wanted:
                audio[name] = tar.extractfile(member).read()
    assert set(audio) == wanted, wanted - set(audio)

    def source(*names):
        return {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "vi_vn/dev",
            "files": list(names),
        }

    SMALL_DIR.mkdir(parents=True, exist_ok=True)
    LARGE_DIR.mkdir(parents=True, exist_ok=True)
    entries = {}
    for out, name in (("vi_a.wav", CLIP_A), ("vi_b.wav", CLIP_B)):
        (SMALL_DIR / out).write_bytes(audio[name])  # verbatim copy of the source file
        entries[out] = {"set": "small", "source": source(name), "reference": by_file[name][3]}

    a, sr = sf.read(io.BytesIO(audio[CLIP_A]), dtype="float32")
    assert sr == SR
    write_wav(SMALL_DIR / "silence_2s.wav", np.zeros(2 * SR, np.float32))
    entries["silence_2s.wav"] = {"set": "small", "source": "digital zeros", "reference": ""}
    # A short cut from the middle of an utterance (speech, not leading silence).
    write_wav(SMALL_DIR / "short_200ms.wav", a[2 * SR : 2 * SR + SR // 5])
    entries["short_200ms.wav"] = {
        "set": "small",
        "source": source(CLIP_A) | {"samples": [2 * SR, 2 * SR + SR // 5]},
        "reference": None,
    }

    gap = np.zeros(int(LONG_GAP_S * SR), np.float32)
    parts = []
    for r in picked:
        x, sr = sf.read(io.BytesIO(audio[r[1]]), dtype="float32")
        assert sr == SR and x.ndim == 1
        parts += [x, gap]
    write_wav(LARGE_DIR / "vi_long.wav", np.concatenate(parts[:-1]))
    entries["vi_long.wav"] = {
        "set": "large",
        "source": source(*(r[1] for r in picked)) | {"gap_s": LONG_GAP_S},
        "reference": " ".join(r[3] for r in picked),
    }

    for name, e in entries.items():
        path = (SMALL_DIR if e["set"] == "small" else LARGE_DIR) / name
        info = sf.info(path)
        e.update(sha256=sha256(path), samples=info.frames, sample_rate=info.samplerate)
    return {
        "license": "FLEURS: CC BY 4.0 (https://huggingface.co/datasets/google/fleurs). Derived clips keep the "
        "same license; reference = FLEURS normalized `transcription` column.",
        "files": entries,
    }


def verify(manifest: dict, include_large: bool) -> list[str]:
    errors = []
    for name, e in manifest["files"].items():
        if e["set"] == "large" and not include_large:
            continue
        path = (SMALL_DIR if e["set"] == "small" else LARGE_DIR) / name
        if not path.is_file():
            errors.append(f"missing {path}")
        elif sha256(path) != e["sha256"]:
            errors.append(f"sha256 mismatch {path}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cache", type=Path, default=Path.home() / ".cache" / "viet-asr-gateway-fixtures"
    )
    ap.add_argument(
        "--verify", action="store_true", help="only verify existing files (no download)"
    )
    ap.add_argument("--small-only", action="store_true", help="with --verify: skip the large set")
    args = ap.parse_args()

    if args.verify:
        errors = verify(json.loads(MANIFEST.read_text()), not args.small_only)
    else:
        manifest = build(args.cache)
        if MANIFEST.exists():  # rebuilding must reproduce the committed files exactly
            errors = verify(json.loads(MANIFEST.read_text()), True)
        else:
            errors = []
        if not errors:
            MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    print("OK: fixtures match manifest" if not errors else "FAILED")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
