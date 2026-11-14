"""Build a listening page: Vietnamese test audio (clean, hard and edge cases) sent through a running gateway, with
the transcripts, WER and word diffs in one static HTML file.

    OPENAI_API_KEY=<gateway key> uv run scripts/demo_page.py --base-url http://127.0.0.1:18001
    # -> outputs/demo/index.html  (open it in a browser; audio/ and results.json sit next to it)

    # add your own files (a sidecar <name>.txt next to the audio is used as the reference, if present)
    ... scripts/demo_page.py --add ~/recordings/ --add meeting.m4a --out outputs/my-demo

Audio comes from FLEURS vi_vn dev (CC BY 4.0) at the revision pinned in scripts/fetch_fixtures.py, downloaded once to
the same cache. Cases, all deterministic:
- FLEURS clips: clean, same sentence by two speakers, numbers, foreign names, shortest, longest;
- one base clip degraded: white noise, babble, music, reverb, telephone (8 kHz mu-law), low-bitrate MP3/Opus,
  stereo 44.1 kHz AAC, fast/slow tempo, very quiet, clipped, long silence padding;
- a > 1 min recording (several 20 s chunks), and edge cases: silence, pure noise, a tone, 200 ms and 80 ms cuts.

Each case is sent as `verbose_json` (text, duration) and `srt` (cues). Request errors are recorded on the case, not
raised. Output stays local: `outputs/` is gitignored (no audio or transcripts in commits).
"""

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fetch_fixtures import fetch, REVISION, SR  # noqa: E402
from scripts.scoring import normalize  # noqa: E402
from scripts.smoke_server import Gateway, wait_ready  # noqa: E402
from scripts.subtitles import parse_cues  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".webm", ".mpga"}
RNG_SEED = 1234


# ---------------------------------------------------------------- FLEURS selection


def load_fleurs(cache: Path) -> list[dict]:
    rows = [
        {"id": r[0], "file": r[1], "raw": r[2], "ref": r[3], "samples": int(r[5])}
        for r in csv.reader(
            fetch("dev.tsv", cache).open(encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE
        )
    ]
    rows.sort(key=lambda r: r["file"])
    return rows


def pick(rows: list[dict]) -> dict[str, list[dict]]:
    def has_digit(r):
        return any(c.isdigit() for c in r["ref"])

    def foreign(r):  # f, j, w, z are not in the Vietnamese alphabet
        return any(c in "fjwz" for c in r["ref"])

    clean = [r for r in rows if not has_digit(r) and not foreign(r)]
    mid = [r for r in clean if 7 * SR <= r["samples"] <= 10 * SR]
    by_id: dict[str, list[dict]] = {}
    for r in clean:
        by_id.setdefault(r["id"], []).append(r)
    taken = {r["file"] for r in mid[:4]}
    pair = next(
        v[:2]
        for _, v in sorted(by_id.items())
        if len(v) >= 2 and not taken & {r["file"] for r in v[:2]}
    )
    used = taken | {r["file"] for r in pair}
    rest = [r for r in clean if r["file"] not in used]
    return {
        "base": mid[0],
        "clean": mid[1:4],
        "pair": pair,
        "numbers": [r for r in rows if has_digit(r)][:3],
        "foreign": [r for r in rows if foreign(r) and not has_digit(r)][:2],
        "shortest": [min(rest, key=lambda r: r["samples"])],
        "longest": [max(rest, key=lambda r: r["samples"])],
        "babble": rest[1:4],
        "long": rest[5:13],
    }


def load_audio(cache: Path, files: set[str]) -> dict[str, np.ndarray]:
    audio = {}
    with tarfile.open(fetch("audio/dev.tar.gz", cache)) as tar:
        for member in tar:
            name = member.name.rsplit("/", 1)[-1]
            if name in files:
                x, sr = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
                assert sr == SR and x.ndim == 1, name
                audio[name] = x
    missing = files - set(audio)
    assert not missing, missing
    return audio


# ---------------------------------------------------------------- signal helpers


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = np.resize(noise, speech.shape)
    noise = noise * (rms(speech) / rms(noise)) / (10 ** (snr_db / 20))
    return fit(speech + noise)


def fit(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return (x * (peak / m) if m > peak else x).astype(np.float32)


def music(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    out = np.zeros(n)
    chords = [(220, 277, 330), (196, 247, 294), (175, 220, 262), (247, 311, 370)]
    step = SR // 2
    for i in range(0, n, step):
        seg = slice(i, min(n, i + step))
        env = np.exp(-3 * (t[seg] - t[i]))
        for f in chords[(i // step) % len(chords)]:
            for h, a in ((1, 1.0), (2, 0.4), (3, 0.2)):
                out[seg] += a * env * np.sin(2 * np.pi * f * h * t[seg])
    beat = (np.arange(n) % (SR // 2)) < SR // 50  # short kick-like noise bursts
    out += beat * rng.standard_normal(n) * 0.8
    return out.astype(np.float32)


def reverb(x: np.ndarray, rt60: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(rt60 * SR)) / SR
    ir = rng.standard_normal(t.size) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    n = x.size + ir.size - 1
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)[: x.size]
    return fit(y * rms(x) / rms(y))


def ffmpeg(src: Path, dst: Path, *args: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            *args,
            str(dst),
        ],
        check=True,
    )


def write_wav(path: Path, x: np.ndarray) -> None:
    sf.write(path, x, SR, subtype="PCM_16")


# ---------------------------------------------------------------- case building


def build_cases(cache: Path, audio_dir: Path, extra: list[Path]) -> list[dict]:
    rows = load_fleurs(cache)
    sel = pick(rows)
    files = {r["file"] for v in sel.values() for r in (v if isinstance(v, list) else [v])}
    print(f"reading {len(files)} FLEURS clips", flush=True)
    audio = load_audio(cache, files)
    rng = np.random.default_rng(RNG_SEED)
    cases: list[dict] = []

    def add(cid, category, title, note, ref, x=None, path=None, preview=None, **meta):
        if path is None:
            path = audio_dir / f"{cid}.wav"
            write_wav(path, x)
        cases.append(
            {
                "id": cid,
                "category": category,
                "title": title,
                "note": note,
                "reference": ref,
                "file": path.name,
                "preview": preview.name if preview else None,
                **meta,
            }
        )

    def fleurs(r):
        return {"source": f"FLEURS vi_vn dev {r['file']} (sentence {r['id']})"}

    # FLEURS as-is
    base = sel["base"]
    add(
        "clean_base",
        "Sạch",
        "Câu gốc (dùng cho các biến thể bên dưới)",
        "Bản ghi FLEURS nguyên bản.",
        base["ref"],
        audio[base["file"]],
        **fleurs(base),
    )
    for i, r in enumerate(sel["clean"], 1):
        add(
            f"clean_{i}",
            "Sạch",
            f"Câu sạch #{i}",
            "Bản ghi FLEURS nguyên bản.",
            r["ref"],
            audio[r["file"]],
            **fleurs(r),
        )
    for i, r in enumerate(sel["pair"], 1):
        add(
            f"pair_{i}",
            "Cùng câu, khác bản ghi",
            f"Cùng nội dung, bản ghi #{i}",
            "Cùng một câu do hai lượt đọc khác nhau: so sánh độ ổn định.",
            r["ref"],
            audio[r["file"]],
            **fleurs(r),
        )
    for i, r in enumerate(sel["numbers"], 1):
        add(
            f"numbers_{i}",
            "Số và năm",
            f"Câu có số #{i}",
            "Reference FLEURS giữ chữ số, model xuất số bằng chữ: WER cao là do chuẩn hoá, không hẳn là lỗi nhận "
            "dạng. Nghe để đánh giá.",
            r["ref"],
            audio[r["file"]],
            raw=r["raw"],
            **fleurs(r),
        )
    for i, r in enumerate(sel["foreign"], 1):
        add(
            f"foreign_{i}",
            "Tên riêng nước ngoài",
            f"Tên riêng / từ mượn #{i}",
            "Tên nước ngoài đọc theo kiểu Việt: model chỉ có âm tiết tiếng Việt.",
            r["ref"],
            audio[r["file"]],
            raw=r["raw"],
            **fleurs(r),
        )
    for key, title in (
        ("shortest", "Câu ngắn nhất trong tập dev"),
        ("longest", "Câu dài nhất trong tập dev"),
    ):
        r = sel[key][0]
        add(key, "Độ dài", title, "Một lượt đọc FLEURS.", r["ref"], audio[r["file"]], **fleurs(r))

    # Degradations of the base clip
    x = audio[base["file"]]
    src = audio_dir / "clean_base.wav"
    deg = "Biến thể (nhiễu, kênh, tốc độ)"
    ref = base["ref"]
    for snr in (10, 0):
        add(
            f"white_noise_{snr}db",
            deg,
            f"Nhiễu trắng SNR {snr} dB",
            "Nhiễu Gauss cộng vào câu gốc.",
            ref,
            mix(x, rng.standard_normal(x.size).astype(np.float32), snr),
        )
    babble = np.concatenate([audio[r["file"]] for r in sel["babble"]])
    add(
        "babble_5db",
        deg,
        "Nhiều người nói nền (babble) SNR 5 dB",
        "Ba câu FLEURS khác chồng lên làm nền: model có thể chép cả giọng nền.",
        ref,
        mix(x, babble, 5),
    )
    add(
        "music_5db",
        deg,
        "Nhạc nền SNR 5 dB",
        "Hợp âm tổng hợp + nhịp trống.",
        ref,
        mix(x, music(x.size, rng), 5),
    )
    add(
        "reverb_0.8s",
        deg,
        "Vang phòng RT60 ≈ 0.8 s",
        "Đáp ứng xung tổng hợp (phòng họp lớn, hội trường).",
        ref,
        reverb(x, 0.8, rng),
    )
    add(
        "quiet_-35db",
        deg,
        "Rất nhỏ (−35 dB)",
        "Giảm biên độ 35 dB, không chuẩn hoá lại.",
        ref,
        (x * 10 ** (-35 / 20)).astype(np.float32),
    )
    add(
        "clipped",
        deg,
        "Méo do clipping (+18 dB rồi cắt)",
        "Mic quá tải.",
        ref,
        np.clip(x * 10 ** (18 / 20), -1, 1).astype(np.float32),
    )
    for tempo in (1.5, 0.7):
        p = audio_dir / f"tempo_{tempo}x.wav"
        ffmpeg(src, p, "-af", f"atempo={tempo}", "-ar", str(SR), "-c:a", "pcm_s16le")
        add(
            f"tempo_{tempo}x",
            deg,
            f"Tốc độ {tempo}x",
            "Đổi tốc độ, giữ cao độ (ffmpeg atempo).",
            ref,
            path=p,
        )
    p = audio_dir / "telephone_8k_mulaw.wav"
    ffmpeg(src, p, "-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-c:a", "pcm_mulaw")
    pv = audio_dir / "telephone_8k_mulaw.preview.wav"
    ffmpeg(p, pv, "-c:a", "pcm_s16le")  # browsers do not play mu-law WAV
    add(
        "telephone",
        deg,
        "Điện thoại: 300–3400 Hz, 8 kHz G.711 μ-law",
        "WAV μ-law 8 kHz gửi nguyên.",
        ref,
        path=p,
        preview=pv,
        format="wav pcm_mulaw 8 kHz",
    )
    p = audio_dir / "mp3_16kbps.mp3"
    ffmpeg(src, p, "-c:a", "libmp3lame", "-b:a", "16k")
    add(
        "mp3_16k",
        deg,
        "MP3 16 kbps",
        "Nén mạnh, artefact codec.",
        ref,
        path=p,
        format="mp3 16 kbps",
    )
    p = audio_dir / "opus_8kbps.ogg"
    ffmpeg(src, p, "-c:a", "libopus", "-b:a", "8k")
    add(
        "opus_8k",
        deg,
        "Opus 8 kbps (kiểu tin nhắn thoại)",
        "Ogg/Opus băng thông hẹp.",
        ref,
        path=p,
        format="ogg opus 8 kbps",
    )
    p = audio_dir / "stereo_44k.m4a"
    ffmpeg(src, p, "-ac", "2", "-ar", "44100", "-c:a", "aac", "-b:a", "96k")
    add(
        "stereo_m4a",
        deg,
        "Stereo 44.1 kHz AAC (.m4a)",
        "Kiểm tra downmix + resample phía server.",
        ref,
        path=p,
        format="m4a aac stereo 44.1 kHz",
    )
    pad = np.zeros(5 * SR, np.float32)
    add(
        "silence_pad_5s",
        deg,
        "5 s im lặng hai đầu",
        "Kiểm tra chunking và cue không lệch.",
        ref,
        np.concatenate([pad, x, pad]),
    )

    # Long recording: several 20 s chunks, utterances separated by 1.5 s gaps
    gap = np.zeros(int(1.5 * SR), np.float32)
    parts = []
    for r in sel["long"]:
        parts += [audio[r["file"]], gap]
    long_x = np.concatenate(parts[:-1])
    add(
        "long_multi",
        "Bản ghi dài",
        f"Bản ghi dài {long_x.size / SR:.0f} s ({len(sel['long'])} câu)",
        "Nối nhiều câu: server chia chunk ≤ 20 s ở điểm lặng; xem chỗ nối chunk có mất/lặp từ không.",
        " ".join(r["ref"] for r in sel["long"]),
        long_x,
    )

    # Edge cases: expected output is empty
    edge = "Biên (kỳ vọng rỗng)"
    add("silence_3s", edge, "Im lặng 3 s", "Số 0 tuyệt đối.", "", np.zeros(3 * SR, np.float32))
    add(
        "noise_3s",
        edge,
        "Chỉ có nhiễu trắng 3 s",
        "Model có 'ảo giác' ra chữ không?",
        "",
        (rng.standard_normal(3 * SR) * 0.05).astype(np.float32),
    )
    add(
        "tone_2s",
        edge,
        "Âm thuần 1 kHz 2 s",
        "Tín hiệu không phải tiếng nói.",
        "",
        (0.3 * np.sin(2 * np.pi * 1000 * np.arange(2 * SR) / SR)).astype(np.float32),
    )
    add(
        "cut_200ms",
        "Biên (rất ngắn)",
        "Đoạn cắt 200 ms giữa câu",
        "Ngắn hơn một âm tiết: kết quả nào cũng chấp nhận.",
        None,
        x[2 * SR : 2 * SR + SR // 5],
    )
    add(
        "cut_80ms",
        edge,
        "Đoạn cắt 80 ms",
        "≤ 100 ms: server trả rỗng, không gọi model.",
        "",
        x[2 * SR : 2 * SR + int(0.08 * SR)],
    )

    # User files
    for i, path in enumerate(extra, 1):
        dst = audio_dir / f"custom_{i}_{path.name}"
        shutil.copyfile(path, dst)
        txt = path.with_suffix(".txt")
        ref = txt.read_text(encoding="utf-8").strip().lower() if txt.is_file() else None
        add(f"custom_{i}", "Tự thêm", path.name, f"File người dùng: {path}", ref, path=dst)
    return cases


def collect_extra(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"--add: not found: {p}")
    return out


# ---------------------------------------------------------------- scoring


def align(ref: str, hyp: str) -> tuple[list[list[str]], int]:
    """Levenshtein word alignment -> ops [kind, ref_word, hyp_word] with kind in eq/sub/del/ins, and edit count."""
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(
                d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1])
            )
    ops, i, j = [], len(r), len(h)
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(["eq" if r[i - 1] == h[j - 1] else "sub", r[i - 1], h[j - 1]])
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(["del", r[i - 1], ""])
            i -= 1
        else:
            ops.append(["ins", "", h[j - 1]])
            j -= 1
    return ops[::-1], int(d[len(r), len(h)])


# ---------------------------------------------------------------- transcription


def transcribe(gw: Gateway, case: dict, audio_dir: Path) -> None:
    data = (audio_dir / case["file"]).read_bytes()
    try:
        t0 = time.monotonic()
        with gw.transcribe(data, response_format="verbose_json", language="vi") as r:
            verbose = json.load(r)
        case["latency_s"] = round(time.monotonic() - t0, 3)
        with gw.transcribe(data, response_format="srt", language="vi") as r:
            cues = parse_cues(r.read().decode(), "srt")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            msg = json.loads(body)["error"]
            case["error"] = f"HTTP {e.code} {msg.get('code')}: {msg.get('message')}"
        except (ValueError, KeyError, TypeError):
            case["error"] = f"HTTP {e.code}: {body[:300]}"
        return
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        case["error"] = f"{type(e).__name__}: {e}"
        return
    case["hypothesis"] = verbose["text"]
    case["duration_s"] = verbose.get("duration")
    case["cues"] = [[round(s, 3), round(e, 3), t] for s, e, t in cues or []]
    if case["duration_s"]:
        case["rtf"] = round(case["latency_s"] / case["duration_s"], 4)
    ref = case["reference"]
    if ref is None:
        case["verdict"] = "info"
    elif ref == "":
        case["verdict"] = "pass" if not case["hypothesis"].strip() else "fail"
    else:
        ops, edits = align(
            normalize(ref), normalize(case["hypothesis"])
        )  # same scoring as scripts/scoring.py
        case["diff"] = ops
        case["wer"] = round(edits / max(len(normalize(ref).split()), 1), 4)
        case["verdict"] = (
            "pass" if case["wer"] <= 0.15 else "warn" if case["wer"] <= 0.4 else "fail"
        )


# ---------------------------------------------------------------- HTML


PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>viet-asr-gateway: nghe thử</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mute:#6b7385;--line:#e3e6ec;--ok:#1f9d55;--warn:#d68a00;--bad:#d64545;
--info:#3b6fd6;--sub:#fde2e2;--ins:#dff5e6;--del:#eceef2;--hl:#fff3c4}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--card:#1d2129;--ink:#e6e9ef;--mute:#9aa3b5;--line:#2c323d;
--sub:#5a2626;--ins:#1f4a30;--del:#2c323d;--hl:#4d4320}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:20px 28px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:2}
h1{margin:0 0 4px;font-size:20px}.meta{color:var(--mute);font-size:13px}
.stats{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;font-size:13px}.stats b{font-size:16px}
.filters{margin-top:12px;display:flex;gap:6px;flex-wrap:wrap}
.filters button{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:999px;padding:4px 12px;cursor:pointer;font-size:13px}
.filters button.on{background:var(--ink);color:var(--bg)}
main{padding:20px 28px;display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(560px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;display:flex;flex-direction:column;gap:10px}
.top{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}
.title{font-weight:600}.cat{font-size:12px;color:var(--mute)}.note{font-size:13px;color:var(--mute)}
.badge{font-weight:700;font-size:13px;padding:2px 10px;border-radius:6px;color:#fff;white-space:nowrap}
.pass{background:var(--ok)}.warn{background:var(--warn)}.fail{background:var(--bad)}.info{background:var(--info)}
audio{width:100%}.tags{display:flex;gap:6px;flex-wrap:wrap}.tags span{font-size:12px;background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:0 6px;color:var(--mute)}
.lbl{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute);margin-bottom:2px}
.txt{background:var(--bg);border-radius:6px;padding:8px 10px;word-wrap:break-word}
.empty{color:var(--mute);font-style:italic}
.diff span{border-radius:3px;padding:0 2px}.d-sub{background:var(--sub)}.d-ins{background:var(--ins)}
.d-del{background:var(--del);text-decoration:line-through;color:var(--mute)}.d-sub s{color:var(--mute);margin-right:3px}
.cues{max-height:180px;overflow:auto;font-size:13px;border:1px solid var(--line);border-radius:6px}
.cue{display:flex;gap:10px;padding:3px 8px;cursor:pointer}.cue:hover{background:var(--bg)}.cue.on{background:var(--hl)}
.cue time{color:var(--mute);font-variant-numeric:tabular-nums;white-space:nowrap}
.err{color:var(--bad);font-family:monospace;font-size:13px}
details summary{cursor:pointer;font-size:13px;color:var(--mute)}
footer{padding:10px 28px 30px;color:var(--mute);font-size:12px}
</style></head><body>
<header><h1>viet-asr-gateway: nghe thử kết quả</h1><div class="meta" id="meta"></div><div class="stats" id="stats"></div>
<div class="filters" id="filters"></div></header>
<main id="cards"></main>
<footer>Âm thanh: FLEURS vi_vn (CC BY 4.0) và biến thể tổng hợp. Model xuất chữ HOA, không dấu câu, số bằng chữ (WER
tính sau khi hạ về chữ thường); reference FLEURS đã chuẩn hoá nhưng giữ chữ số. Ngưỡng màu: WER ≤ 15% xanh, ≤ 40% vàng, còn lại đỏ. Click một cue để
nhảy tới thời điểm đó.</footer>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const $=(t,c,txt)=>{const e=document.createElement(t);if(c)e.className=c;if(txt!=null)e.textContent=txt;return e};
const fmt=s=>{const m=Math.floor(s/60);return m+':'+(s%60).toFixed(2).padStart(5,'0')};
const pct=x=>(100*x).toFixed(1)+'%';
document.getElementById('meta').textContent=`${D.base_url} · model ${D.model} · ${D.created} · ${D.cases.length} trường hợp`;
const done=D.cases.filter(c=>!c.error),scored=done.filter(c=>c.wer!=null);
const words=scored.reduce((a,c)=>a+c.reference.split(' ').length,0);
const edits=scored.reduce((a,c)=>a+c.wer*c.reference.split(' ').length,0);
const dur=done.reduce((a,c)=>a+(c.duration_s||0),0),lat=done.reduce((a,c)=>a+(c.latency_s||0),0);
const st=[['WER tổng (theo từ)',words?pct(edits/words):'–'],['Đạt',D.cases.filter(c=>c.verdict==='pass').length+'/'+D.cases.length],
['Lỗi request',D.cases.length-done.length],['Tổng audio',dur.toFixed(1)+' s'],['RTF trung bình',dur?(lat/dur).toFixed(3):'–']];
for(const[k,v]of st){const e=$('div');e.append($('b',null,v),' ',k);document.getElementById('stats').append(e)}
const cats=['Tất cả',...new Set(D.cases.map(c=>c.category))];let cur='Tất cả';
const F=document.getElementById('filters');
for(const c of cats){const b=$('button',c===cur?'on':'',c);b.onclick=()=>{cur=c;[...F.children].forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('.card').forEach(el=>el.style.display=(c==='Tất cả'||el.dataset.cat===c)?'':'none')};F.append(b)}
function textBox(label,content){const w=$('div');w.append($('div','lbl',label));const t=$('div','txt');
if(typeof content==='string'){if(content)t.textContent=content;else{t.textContent='(rỗng)';t.classList.add('empty')}}else t.append(content);w.append(t);return w}
for(const c of D.cases){
 const card=$('section','card');card.dataset.cat=c.category;card.id=c.id;
 const top=$('div','top'),hd=$('div');hd.append($('div','cat',c.category),$('div','title',c.title));top.append(hd);
 let badge;if(c.error)badge=$('span','badge fail','LỖI');else if(c.wer!=null)badge=$('span','badge '+c.verdict,'WER '+pct(c.wer));
 else if(c.verdict==='info')badge=$('span','badge info','tham khảo');else badge=$('span','badge '+c.verdict,c.verdict==='pass'?'rỗng ✓':'không rỗng ✗');
 top.append(badge);card.append(top);if(c.note)card.append($('div','note',c.note));
 const au=$('audio');au.controls=true;au.preload='none';au.src='audio/'+encodeURIComponent(c.preview||c.file);card.append(au);
 const tags=$('div','tags');for(const t of[c.file,c.format,c.duration_s!=null&&c.duration_s.toFixed(2)+' s',
  c.latency_s!=null&&'latency '+c.latency_s.toFixed(2)+' s',c.rtf!=null&&'RTF '+c.rtf.toFixed(3),c.source])if(t)tags.append($('span',null,t));card.append(tags);
 if(c.error){card.append($('div','err',c.error))}
 if(c.reference!=null)card.append(textBox('Reference',c.reference));
 if(c.raw)card.append(textBox('Văn bản gốc (chưa chuẩn hoá)',c.raw));
 if(c.hypothesis!=null){if(c.diff){const d=$('div','diff');for(const[k,r,h]of c.diff){let s;
   if(k==='eq')s=$('span',null,h);else if(k==='sub'){s=$('span','d-sub');s.append($('s',null,r),h);s.title=r+' → '+h}
   else if(k==='del'){s=$('span','d-del',r);s.title='thiếu'}else{s=$('span','d-ins',h);s.title='thừa'}d.append(s,' ')}
   card.append(textBox('Kết quả (so với reference)',d))}else card.append(textBox('Kết quả',c.hypothesis))}
 if(c.cues&&c.cues.length){const det=$('details');det.open=c.cues.length<=12;det.append($('summary',null,c.cues.length+' cue (SRT)'));
  const box=$('div','cues');const rows=c.cues.map(([s,e,t])=>{const r=$('div','cue');r.append($('time',null,fmt(s)+' → '+fmt(e)),$('span',null,t));
   r.onclick=()=>{au.currentTime=s;au.play()};box.append(r);return r});det.append(box);card.append(det);
  au.ontimeupdate=()=>{const t=au.currentTime;c.cues.forEach(([s,e],i)=>rows[i].classList.toggle('on',t>=s&&t<e))}}
 au.onplay=()=>document.querySelectorAll('audio').forEach(a=>{if(a!==au)a.pause()});
 document.getElementById('cards').append(card)}
</script></body></html>
"""


def write_page(out: Path, report: dict) -> Path:
    data = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    page = out / "index.html"
    page.write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
    return page


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:18001")
    ap.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY"), help="default: $OPENAI_API_KEY"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "demo")
    ap.add_argument(
        "--add", type=Path, action="append", default=[], help="extra audio file or directory"
    )
    ap.add_argument("--only", help="comma-separated case ids or categories to run")
    ap.add_argument(
        "--cache", type=Path, default=Path.home() / ".cache" / "viet-asr-gateway-fixtures"
    )
    ap.add_argument("--timeout", type=float, default=600.0, help="per HTTP request")
    ap.add_argument("--ready-timeout", type=float, default=60.0)
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("set OPENAI_API_KEY or --api-key (a gateway key)")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    gw = Gateway(args.base_url, args.api_key, args.timeout)
    wait_ready(gw, args.ready_timeout)
    model = gw.get_json("/v1/models")["data"][0]["id"]

    audio_dir = args.out / "audio"
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True)
    cases = build_cases(args.cache, audio_dir, collect_extra(args.add))
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in keep or c["category"] in keep]

    for i, case in enumerate(cases, 1):
        transcribe(gw, case, audio_dir)
        status = case.get("error") or (
            f"WER {case['wer']:.3f}" if "wer" in case else case.get("verdict")
        )
        print(f"[{i:2}/{len(cases)}] {case['id']:<20} {status}", flush=True)

    report = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "base_url": args.base_url,
        "model": model,
        "fleurs_revision": REVISION,
        "cases": cases,
    }
    (args.out / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    page = write_page(args.out, report)
    print(f"\nopen {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
