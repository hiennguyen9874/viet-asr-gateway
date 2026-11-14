"""FFmpeg decoding: formats, sample rates/channels, hostile inputs, and bounds (samples, time, concurrency, stderr).

Every test checks that no FFmpeg child process is left behind. Needs `ffmpeg` on PATH (CPU lane).
Format fixtures are generated from `tests/fixtures/vi_a.wav` (FLEURS, CC BY 4.0) into a temp dir.
"""

import asyncio
import os
import socket
import stat
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio import AudioIngress
from app.errors import APIError
from app.observability import Metrics
from tests.fake_engine import config

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SOURCE = FIXTURES / "vi_a.wav"
REF = sf.read(SOURCE, dtype="float32")[0]  # 16 kHz mono float32, 142080 samples


def children() -> set[str]:
    """Child pids of this process, zombies included (a zombie means a decoder was not reaped)."""
    pids: set[str] = set()
    for task in Path(f"/proc/{os.getpid()}/task").iterdir():
        try:
            pids |= set((task / "children").read_text().split())
        except FileNotFoundError:
            pass
    return pids


@pytest.fixture(autouse=True)
def no_leftover_processes():
    before = children()
    yield
    deadline = time.monotonic() + 2
    while children() - before and time.monotonic() < deadline:
        time.sleep(0.02)
    assert children() - before == set(), "decoder process left behind"


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="session")
def media(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("media")
    src = str(SOURCE)
    specs = {
        "wav_f32_16k": None,
        "wav_s16_8k.wav": ["-i", src, "-ar", "8000", "-c:a", "pcm_s16le"],
        "wav_s16_48k_left_only.wav": ["-i", src, "-af", "pan=stereo|c0=c0|c1=0*c0", "-ar", "48000",
                                      "-c:a", "pcm_s16le"],
        "flac_stereo_44k.flac": ["-i", src, "-ac", "2", "-ar", "44100"],
        "mp3_44k.mp3": ["-i", src, "-ar", "44100", "-c:a", "libmp3lame"],
        "m4a_aac.m4a": ["-i", src, "-c:a", "aac"],
        "mp4_video_aac.mp4": ["-f", "lavfi", "-i", "testsrc=d=9:s=64x64", "-i", src, "-map", "0:v", "-map", "1:a",
                              "-c:v", "mpeg4", "-c:a", "aac", "-shortest"],
        "ogg_vorbis.ogg": ["-i", src, "-c:a", "libvorbis"],
        "ogg_opus.ogg": ["-i", src, "-c:a", "libopus"],
        "webm_opus.webm": ["-i", src, "-c:a", "libopus"],
        "ogg_flac.ogg": ["-i", src, "-c:a", "flac", "-f", "ogg"],
        "matroska_aac.mka": ["-i", src, "-c:a", "aac", "-f", "matroska"],
        # hostile / unsupported
        "video_only.mp4": ["-f", "lavfi", "-i", "testsrc=d=1:s=64x64", "-c:v", "mpeg4"],
        "adpcm.wav": ["-i", src, "-c:a", "adpcm_ms"],  # codec not on the allowlist
        "adts.aac": ["-i", src, "-c:a", "aac", "-f", "adts"],  # demuxer not on the allowlist
        "silence_600s.flac": ["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "600"],  # decode bomb
    }
    out = {}
    for name, args in specs.items():
        if args is None:
            out[name] = SOURCE
            continue
        out[name] = d / name
        ffmpeg(*args, str(d / name))
    out["random.bin"] = d / "random.bin"
    out["random.bin"].write_bytes(np.random.default_rng(0).integers(0, 256, 50_000, dtype=np.uint8).tobytes())
    out["empty"] = d / "empty"
    out["empty"].write_bytes(b"")
    out["text.txt"] = d / "text.txt"
    out["text.txt"].write_text("xin chào " * 1000)
    out["truncated.m4a"] = d / "truncated.m4a"
    out["truncated.m4a"].write_bytes(out["m4a_aac.m4a"].read_bytes()[:30_000])  # moov atom at the end: lost
    out["truncated.mp3"] = d / "truncated.mp3"
    out["truncated.mp3"].write_bytes(out["mp3_44k.mp3"].read_bytes()[:30_000])
    return out


def decode(path: Path, ffmpeg_bin: str | None = None, deadline_s: float = 30.0, metrics: Metrics | None = None,
           **overrides) -> np.ndarray:
    async def main():
        ingress = AudioIngress(config(**overrides), metrics or Metrics(), ffmpeg=ffmpeg_bin)
        return await ingress.decode(path, time.monotonic() + deadline_s)

    return asyncio.run(main())


def api_error(fn, *args, **kwargs) -> APIError:
    with pytest.raises(APIError) as exc:
        fn(*args, **kwargs)
    return exc.value


def fake_ffmpeg(tmp_path: Path, body: str) -> str:
    """A stand-in `ffmpeg` executable (shell script ignoring its argv) to exercise hostile process behaviour."""
    path = tmp_path / "fake-ffmpeg"
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


# -- formats ---------------------------------------------------------------------------------------------------------


def test_wav_at_16k_float_is_bit_exact():
    wav = decode(SOURCE)
    assert wav.dtype == np.float32 and wav.ndim == 1
    np.testing.assert_array_equal(wav, REF)


@pytest.mark.parametrize("name", ["wav_s16_8k.wav", "wav_s16_48k_left_only.wav", "flac_stereo_44k.flac"])
def test_lossless_rates_and_channels_resample_to_16k_mono(media, name):
    wav = decode(media[name])
    assert abs(len(wav) - len(REF)) <= 16  # resampler rounding only
    n = min(len(wav), len(REF))
    corr = np.corrcoef(wav[:n], REF[:n])[0, 1]
    assert corr > (0.95 if "8k" in name else 0.99)  # 8 kHz loses the upper band


@pytest.mark.parametrize("name", ["mp3_44k.mp3", "m4a_aac.m4a", "mp4_video_aac.mp4", "ogg_vorbis.ogg",
                                  "ogg_opus.ogg", "webm_opus.webm", "ogg_flac.ogg", "matroska_aac.mka"])
def test_lossy_containers_decode(media, name):
    wav = decode(media[name])
    assert wav.dtype == np.float32 and np.isfinite(wav).all()
    assert abs(len(wav) - len(REF)) < 0.02 * len(REF)  # encoder priming/padding only
    rms = np.sqrt(np.mean(wav ** 2)) / np.sqrt(np.mean(REF ** 2))
    assert 0.8 < rms < 1.2


# -- hostile and unsupported input ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["random.bin", "empty", "text.txt", "truncated.m4a", "video_only.mp4", "adpcm.wav",
                                  "adts.aac"])
def test_undecodable_or_disallowed_input_is_invalid_audio(media, name):
    metrics = Metrics()
    err = api_error(decode, media[name], metrics=metrics)
    assert (err.status, err.code, err.param) == (400, "invalid_audio", "file")
    assert metrics.value("decodes_total", outcome="invalid_audio") == 1


def test_truncated_stream_decodes_the_readable_prefix(media):
    wav = decode(media["truncated.mp3"])
    assert 0 < len(wav) < len(REF)


def test_playlists_and_concat_lists_are_refused_without_network_or_file_access(media, tmp_path):
    hits = []
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    server.settimeout(0.2)
    port = server.getsockname()[1]

    def accept():
        try:
            conn, _ = server.accept()
            hits.append(1)
            conn.close()
        except OSError:
            pass

    t = threading.Thread(target=accept)
    t.start()
    hls = tmp_path / "upload"
    hls.write_text(f"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nhttp://127.0.0.1:{port}/x.ts\n#EXT-X-ENDLIST\n")
    assert api_error(decode, hls).code == "invalid_audio"
    concat = tmp_path / "upload2"
    concat.write_text(f"ffconcat version 1.0\nfile '{media['mp3_44k.mp3']}'\n")
    assert api_error(decode, concat).code == "invalid_audio"
    t.join()
    server.close()
    assert hits == []


# -- bounds ----------------------------------------------------------------------------------------------------------


def test_decode_bomb_stops_at_max_audio_seconds(media):
    assert media["silence_600s.flac"].stat().st_size < 200_000  # tiny file, 600 s of audio
    err = api_error(decode, media["silence_600s.flac"], max_audio_seconds=5.0, max_chunk_seconds=5.0)
    assert (err.status, err.code, err.param) == (400, "audio_too_long", "file")
    wav = decode(media["silence_600s.flac"], max_audio_seconds=600.0)  # exactly at the limit is accepted
    assert len(wav) == 600 * 16000


def test_endless_output_is_cut_off_while_decoding(tmp_path):
    # Never terminates on its own: passes only if the decoder is killed once the sample limit is exceeded.
    endless = fake_ffmpeg(tmp_path, "exec cat /dev/zero")
    t0 = time.monotonic()
    err = api_error(decode, tmp_path, endless, max_audio_seconds=2.0, max_chunk_seconds=2.0,
                    decode_timeout_seconds=30.0)
    assert err.code == "audio_too_long"
    assert time.monotonic() - t0 < 5


def test_decode_timeout_kills_the_process(tmp_path):
    t0 = time.monotonic()
    err = api_error(decode, tmp_path, fake_ffmpeg(tmp_path, "exec sleep 30"), decode_timeout_seconds=0.3)
    assert (err.status, err.code) == (400, "invalid_audio") and "time limit" in err.message
    assert time.monotonic() - t0 < 5


def test_request_deadline_bounds_decode(tmp_path):
    err = api_error(decode, tmp_path, fake_ffmpeg(tmp_path, "exec sleep 30"), deadline_s=0.3)
    assert (err.status, err.code) == (503, "request_timeout")


def test_stderr_flood_is_drained_and_bounded(tmp_path):
    flood = fake_ffmpeg(tmp_path, "head -c 20000000 /dev/zero | tr '\\0' x >&2; exit 1")
    assert api_error(decode, tmp_path, flood, decode_timeout_seconds=20.0).code == "invalid_audio"


def test_non_finite_samples_are_rejected(tmp_path):
    nan = fake_ffmpeg(tmp_path, "i=0; while [ $i -lt 1000 ]; do printf '\\000\\000\\300\\177'; i=$((i+1)); done")
    err = api_error(decode, tmp_path, nan)
    assert err.code == "invalid_audio" and "non-finite" in err.message


def test_decoder_gets_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("GATEWAY_API_KEYS", "gateway-secret-0123456789")
    dump = tmp_path / "env.txt"
    decode(tmp_path, fake_ffmpeg(tmp_path, f"env > {dump}"))
    env = dump.read_text()
    assert "hf_secret_value" not in env and "gateway-secret" not in env


def test_concurrent_decodes_are_bounded_and_cancellation_reaps(tmp_path):
    slow = fake_ffmpeg(tmp_path, "exec sleep 30")

    async def main():
        metrics = Metrics()
        ingress = AudioIngress(config(max_concurrent_decodes=1, queue_timeout_seconds=0.3), metrics, ffmpeg=slow)
        deadline = time.monotonic() + 30
        first = asyncio.create_task(ingress.decode(tmp_path, deadline))
        await asyncio.sleep(0.2)
        assert len(children()) == 1
        with pytest.raises(APIError) as exc:
            await ingress.decode(tmp_path, deadline)  # waits for a slot, then gives up
        assert (exc.value.status, exc.value.code) == (429, "server_busy")
        assert len(children()) == 1  # the waiting decode never started a process
        first.cancel()  # e.g. client disconnected
        with pytest.raises(asyncio.CancelledError):
            await first
        assert children() == set()  # killed and reaped before the cancellation completed
        assert metrics.value("decodes_total", outcome="cancelled") == 1
        assert metrics.value("decodes_total", outcome="busy") == 1

    asyncio.run(main())
