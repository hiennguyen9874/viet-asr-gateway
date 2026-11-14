"""Build the Vietnamese audio fixtures from FLEURS vi_vn (CC BY 4.0) at a pinned dataset revision.

    uv run scripts/fetch_fixtures.py            # small set -> tests/fixtures/, large set -> fixtures-large/
    uv run scripts/fetch_fixtures.py --verify   # offline: check files against tests/fixtures/manifest.json
    uv run scripts/fetch_fixtures.py --eval     # local eval set -> fixtures-large/eval/ (~550 MB download)
    uv run scripts/fetch_fixtures.py --verify --eval   # offline: check it against tests/fixtures/eval_manifest.json

The small set is committed. The large set (long multi-utterance recording) is rebuilt locally and is gitignored.
Every file is derived deterministically from the pinned source; the manifest records SHA256 and reference text.

The eval set (gitignored, used by scripts/baseline_offline.py and scripts/probe_limits.py) is EVAL_CLIPS FLEURS
vi_vn test recordings (one per sentence id, in file-name order, verbatim copies) plus synthetic non-speech clips
generated from fixed seeds (`nonspeech`). Its committed manifest holds hashes and counts only: the references live in
fixtures-large/eval/references.json, whose SHA256 the manifest records.
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
EVAL_DIR = LARGE_DIR / "eval"
EVAL_MANIFEST = SMALL_DIR / "eval_manifest.json"
EVAL_REFERENCES = EVAL_DIR / "references.json"

DATASET = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/vi_vn"
SR = 16000

CLIP_A = "10282809127467658620.wav"  # two clips with different content, for isolation tests
CLIP_B = "10467396475561744099.wav"
# Makes viet-asr emit byte-fallback tokens (<0xC3><0x9A> = "Ú", absent from its vocabulary) with both decodings.
CLIP_BYTES = "2335344692138890377.wav"
LONG_COUNT = 10  # utterances in the long recording (> 30 s, so it exercises splitting)
LONG_GAP_S = 0.3
EVAL_CLIPS = 200
# Non-speech clips: kind -> (noise colour, RMS level in dBFS); each kind at every duration, one fixed seed per clip.
NONSPEECH_KINDS = {
    "zeros": (None, None),
    "white_-60dB": ("white", -60),
    "white_-45dB": ("white", -45),
    "white_-30dB": ("white", -30),
    "pink_-60dB": ("pink", -60),
    "pink_-45dB": ("pink", -45),
    "pink_-30dB": ("pink", -30),
    "room_tone": ("room", -65),  # low pink background plus a 50 Hz hum at -60 dBFS
}
NONSPEECH_SECONDS = (2, 5, 10)


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
    long_rows = [r for r in clean if r[1] not in (CLIP_A, CLIP_B, CLIP_BYTES)]
    seen, picked = set(), []
    for r in long_rows:  # one recording per sentence id
        if r[0] not in seen:
            seen.add(r[0])
            picked.append(r)
        if len(picked) == LONG_COUNT:
            break

    wanted = {CLIP_A, CLIP_B, CLIP_BYTES, *(r[1] for r in picked)}
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

    # The part of CLIP_BYTES around "Út" (6.5 s, PCM_16 to stay small); its FLEURS reference covers the whole clip.
    x, sr = sf.read(io.BytesIO(audio[CLIP_BYTES]), dtype="float32")
    assert sr == SR
    cut = [int(4.5 * SR), 11 * SR]
    write_wav(SMALL_DIR / "vi_bytes.wav", x[cut[0] : cut[1]])
    entries["vi_bytes.wav"] = {
        "set": "small",
        "source": source(CLIP_BYTES) | {"samples": cut},
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


def _noise(colour: str, n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    if colour == "white":
        return white
    spectrum = np.fft.rfft(white)
    spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))  # 1/f power
    spectrum[0] = 0
    return np.fft.irfft(spectrum, n)


def nonspeech() -> dict[str, np.ndarray]:
    """Synthetic clips without speech, identical on every run (fixed seeds): name -> float32 waveform at 16 kHz."""
    out = {}
    for i, (kind, (colour, db)) in enumerate(NONSPEECH_KINDS.items()):
        for seconds in NONSPEECH_SECONDS:
            n = seconds * SR
            rng = np.random.default_rng(1000 * i + seconds)
            if colour is None:
                x = np.zeros(n)
            else:
                x = _noise("pink" if colour == "room" else colour, n, rng)
                x *= 10 ** (db / 20) / np.sqrt(np.mean(x**2))
                if colour == "room":
                    x += 10 ** (-60 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 50 * np.arange(n) / SR)
            out[f"{kind}_{seconds}s.wav"] = x.astype(np.float32)
    return out


def build_eval(cache: Path) -> dict:
    rows = list(
        csv.reader(
            fetch("test.tsv", cache).open(encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE
        )
    )
    seen, picked = set(), []
    for r in sorted(rows, key=lambda r: r[1]):  # one recording per sentence id
        if r[0] not in seen:
            seen.add(r[0])
            picked.append(r)
    picked = picked[:EVAL_CLIPS]
    assert len(picked) == EVAL_CLIPS
    wanted = {r[1] for r in picked}
    (EVAL_DIR / "fleurs").mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "nonspeech").mkdir(parents=True, exist_ok=True)
    found = set()
    with tarfile.open(fetch("audio/test.tar.gz", cache)) as tar:
        for member in tar:
            name = member.name.rsplit("/", 1)[-1]
            if name in wanted:
                (EVAL_DIR / "fleurs" / name).write_bytes(tar.extractfile(member).read())
                found.add(name)
    assert found == wanted, wanted - found
    for name, x in nonspeech().items():
        write_wav(EVAL_DIR / "nonspeech" / name, x)
    # Normalized `transcription` column, as the fixtures. It keeps digits, so after scoring.normalize it equals the
    # written `raw_transcription` on every eval clip.
    refs = {f"fleurs/{r[1]}": {"reference": r[3]} for r in picked}
    EVAL_REFERENCES.write_text(json.dumps(refs, indent=2, ensure_ascii=False) + "\n")
    files = {}
    for path in sorted(EVAL_DIR.glob("*/*.wav")):
        info = sf.info(path)
        files[path.relative_to(EVAL_DIR).as_posix()] = {
            "sha256": sha256(path),
            "samples": info.frames,
            "sample_rate": info.samplerate,
        }
    return {
        "license": "FLEURS: CC BY 4.0 (https://huggingface.co/datasets/google/fleurs); verbatim copies.",
        "source": {"dataset": DATASET, "revision": REVISION, "split": "vi_vn/test"},
        "selection": f"first {EVAL_CLIPS} recordings in file-name order, one per sentence id",
        "nonspeech": {
            "kinds": {k: {"noise": c, "dbfs": d} for k, (c, d) in NONSPEECH_KINDS.items()},
            "seconds": list(NONSPEECH_SECONDS),
            "generator": "scripts/fetch_fixtures.py:nonspeech (fixed seeds), PCM_16",
        },
        "references_sha256": sha256(EVAL_REFERENCES),
        "files": files,
    }


def verify_eval(manifest: dict) -> list[str]:
    errors = []
    if not EVAL_REFERENCES.is_file() or sha256(EVAL_REFERENCES) != manifest["references_sha256"]:
        errors.append(f"missing or changed {EVAL_REFERENCES}")
    for name, e in manifest["files"].items():
        path = EVAL_DIR / name
        if not path.is_file():
            errors.append(f"missing {path}")
        elif sha256(path) != e["sha256"]:
            errors.append(f"sha256 mismatch {path}")
    return errors


def eval_set() -> tuple[dict[str, dict], dict[str, Path]]:
    """The verified local eval set: ({FLEURS path: references}, {non-speech name: path}). Exits when it is missing."""
    errors = verify_eval(json.loads(EVAL_MANIFEST.read_text()))
    if errors:
        sys.exit(f"eval set missing or changed ({errors[0]}): run scripts/fetch_fixtures.py --eval")
    refs = json.loads(EVAL_REFERENCES.read_text())
    speech = {EVAL_DIR / name: r for name, r in refs.items()}
    silent = {p.name: p for p in sorted((EVAL_DIR / "nonspeech").glob("*.wav"))}
    return speech, silent


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
    ap.add_argument(
        "--eval", action="store_true", help="the local eval set instead of the fixtures"
    )
    args = ap.parse_args()

    if args.eval:
        if args.verify:
            errors = verify_eval(json.loads(EVAL_MANIFEST.read_text()))
        else:
            manifest = build_eval(args.cache)
            errors = (
                verify_eval(json.loads(EVAL_MANIFEST.read_text())) if EVAL_MANIFEST.exists() else []
            )
            if not errors:
                EVAL_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    elif args.verify:
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
