"""Audio ingress: bounded multipart upload spooling and bounded FFmpeg decoding to 16 kHz mono float32.

Upload (`AudioIngress.receive`):
- The body is read from the ASGI stream and counted *before* it reaches the multipart parser, so a body over
  `max_upload_bytes` is refused with 413 whether or not it declares a Content-Length (chunked uploads included).
- Form fields are bounded in number and size; exactly one `file` part is spooled to a server-named temp file
  (the client filename is never used). Spooled bytes of all uploads share a disk budget (`max_spool_bytes`, 429).
- The temp file is removed when the `receive()` context exits: success, error, cancellation or client disconnect.

Decode (`AudioIngress.decode`):
- FFmpeg runs as a subprocess (argv, no shell, minimal environment) on the local temp file only: `file` protocol,
  demuxer and decoder allowlists, so playlists/concat lists/network references are refused.
- At most `max_concurrent_decodes` processes; waiting for a slot is bounded by `queue_timeout_seconds` (429).
- PCM is read in blocks and the process is killed as soon as the output exceeds `max_audio_seconds`; stderr is
  drained but only its first bytes are kept; wall time is bounded by `decode_timeout_seconds` and the request
  deadline. The process is killed and reaped on every exit path.

Upload and decode both stop at the request `deadline` (time.monotonic), the budget shared with queue and inference.

Decoded audio (`AudioIngress.hold`): the waveforms that requests keep in RAM until their transcript is done share a
budget (`max_decoded_seconds`, 429).
"""

import asyncio
import errno
import logging
import os
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from python_multipart.multipart import MultipartParseError, MultipartParser, parse_options_header
from starlette.requests import ClientDisconnect, Request

from app.config import Config, SAMPLE_RATE
from app.errors import (
    APIError,
    audio_too_long,
    invalid_audio,
    invalid_request,
    request_timeout,
    request_too_large,
    server_busy,
)
from app.observability import log, Metrics

FILE_FIELD = "file"
MAX_FORM_PARTS = 32  # file + fields, repeated `include[]`/`timestamp_granularities[]` included
MAX_FIELD_BYTES = 8 * 1024
MAX_PART_HEADERS = 8
MAX_PART_HEADER_BYTES = 4 * 1024

# Containers named by OpenAI (wav, flac, mp3/mpga, mp4/m4a, ogg, webm). FFmpeg matches a demuxer whose name list
# contains one of these: `mov` = mov,mp4,m4a,3gp,3g2,mj2; `matroska` = matroska,webm.
DEMUXERS = ("wav", "flac", "mp3", "mov", "ogg", "matroska")
DECODERS = (
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_u8",
    "pcm_f32le",
    "pcm_f64le",
    "pcm_alaw",
    "pcm_mulaw",
    "flac",
    "alac",
    "mp1float",
    "mp2float",
    "mp3float",
    "aac",
    "vorbis",
    "libvorbis",
    "opus",
    "libopus",
)
READ_BYTES = 64 * 1024
STDERR_KEEP_BYTES = 4 * 1024
SAMPLE_BYTES = 4  # f32le


@dataclass
class Upload:
    fields: list[tuple[str, str]]  # form fields in order, repeated names kept (e.g. `include[]`)
    path: Path  # spooled `file` part (a request without one is rejected)
    size: int  # bytes of the file part


class AudioIngress:
    def __init__(self, config: Config, metrics: Metrics, ffmpeg: str | None = None):
        self._max_body = config.max_upload_bytes
        self._max_spool = config.max_spool_bytes
        self._max_samples = config.max_audio_samples
        self._max_audio_seconds = config.max_audio_seconds
        self._decode_timeout = config.decode_timeout_seconds
        self._slot_timeout = config.queue_timeout_seconds
        self._spool_parent = config.spool_dir
        self._ffmpeg = ffmpeg or shutil.which("ffmpeg")
        if not self._ffmpeg:
            raise FileNotFoundError("ffmpeg not found on PATH")
        self._decode_slots = asyncio.Semaphore(config.max_concurrent_decodes)
        self._spool_dir: Path | None = None
        self._spooled = 0
        self._decoding = 0
        self._max_held = int(config.max_decoded_seconds * SAMPLE_RATE)
        self._held = 0
        self._metrics = metrics
        metrics.gauge_fn("spool_bytes", "Upload bytes held in temp files.", lambda: self._spooled)
        metrics.gauge_fn(
            "active_decodes", "Running FFmpeg decode processes.", lambda: self._decoding
        )
        metrics.gauge_fn(
            "decoded_audio_seconds",
            "Decoded audio held in RAM by requests in flight.",
            lambda: self._held / SAMPLE_RATE,
        )

    @property
    def spool_dir(self) -> Path | None:
        return self._spool_dir

    def start(self) -> None:
        """Create a private spool directory (0700) for this process."""
        self._spool_dir = Path(tempfile.mkdtemp(prefix="gateway-spool-", dir=self._spool_parent))

    def stop(self) -> None:
        """Remove the spool directory and anything left in it."""
        if self._spool_dir is not None:
            shutil.rmtree(self._spool_dir, ignore_errors=True)
            self._spool_dir = None

    # -- upload ----------------------------------------------------------------------------------------------------

    @asynccontextmanager
    async def receive(self, request: Request, deadline: float) -> AsyncIterator[Upload]:
        """Read and parse the multipart body; the spooled file exists only inside the context."""
        state = _FormState(self)
        outcome = "error"
        try:
            try:
                async with asyncio.timeout(deadline - time.monotonic()):
                    await self._read_body(request, state)
            except TimeoutError:
                outcome = "timeout"
                raise request_timeout() from None
            except APIError as e:
                outcome = e.code
                raise
            except ClientDisconnect:
                outcome = "disconnected"
                raise
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            outcome = "ok"
            yield Upload(state.fields, state.path, state.file_size)
        finally:
            state.close()
            self._metrics.inc("uploads_total", "Multipart uploads by outcome.", outcome=outcome)

    async def _read_body(self, request: Request, state: "_FormState") -> None:
        content_type, params = parse_options_header(request.headers.get("content-type", ""))
        boundary = params.get(b"boundary")
        if content_type != b"multipart/form-data" or not boundary:
            raise invalid_request(
                "invalid_value", "Content-Type must be multipart/form-data with a boundary."
            )
        declared = request.headers.get("content-length")  # the server has validated its syntax
        if declared is not None and int(declared) > self._max_body:
            raise request_too_large(self._max_body)

        parser = MultipartParser(
            boundary,
            state.callbacks(),
            max_header_count=MAX_PART_HEADERS,
            max_header_size=MAX_PART_HEADER_BYTES,
        )
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > self._max_body:  # counted before the parser sees the bytes
                raise request_too_large(self._max_body)
            try:
                parser.write(chunk)
            except MultipartParseError:
                state.raise_pending()
                raise invalid_request("invalid_value", "Malformed multipart body.") from None
            state.raise_pending()
        if not state.ended:
            raise invalid_request("invalid_value", "Malformed multipart body.")
        if state.path is None:
            raise invalid_request(
                "missing_required_parameter", "Missing required parameter: 'file'.", FILE_FIELD
            )

    def _reserve_spool(self, n: int) -> None:
        if self._spooled + n > self._max_spool:
            raise server_busy(retry_after=1)
        self._spooled += n

    def _release_spool(self, n: int) -> None:
        self._spooled -= n

    def _new_spool_file(self) -> tuple[int, Path]:
        if self._spool_dir is None:
            raise RuntimeError("AudioIngress not started")
        try:
            fd, name = tempfile.mkstemp(
                prefix="upload-", dir=self._spool_dir
            )  # server-generated name, mode 0600
        except OSError as e:
            if e.errno in (errno.ENOSPC, errno.EDQUOT):
                raise server_busy(retry_after=1) from None
            raise
        return fd, Path(name)

    # -- decode ----------------------------------------------------------------------------------------------------

    def hold(self, wav: np.ndarray) -> Callable[[], None]:
        """Reserve decoded RAM (429 if full). Release only after the request AND its jobs finish."""
        return self.reserve(len(wav))

    def reserve(self, n: int) -> Callable[[], None]:
        """Reserve `n` samples of the decoded-audio budget (realtime turns reserve per append). Call the returned
        release exactly once."""
        if self._held + n > self._max_held:
            raise server_busy(retry_after=1)
        self._held += n

        def release() -> None:
            self._held -= n

        return release

    async def decode(self, path: Path, deadline: float) -> np.ndarray:
        """Decode a spooled upload to 16 kHz mono float32. Raises `APIError` (400 invalid_audio/audio_too_long,
        429 server_busy, 503 request_timeout)."""
        started = time.monotonic()
        outcome = "error"
        try:
            try:
                async with asyncio.timeout(min(self._slot_timeout, deadline - started)):
                    await self._decode_slots.acquire()
            except TimeoutError:
                outcome = "busy"
                if time.monotonic() >= deadline:
                    outcome = "timeout"
                    raise request_timeout() from None
                raise server_busy(retry_after=1) from None
            try:
                self._decoding += 1
                wav = await self._run_ffmpeg(path, deadline)
            finally:
                self._decoding -= 1
                self._decode_slots.release()
            outcome = "ok"
            return wav
        except APIError as e:
            if outcome == "error":
                outcome = e.code
            raise
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            self._metrics.inc(
                "decodes_total",
                "FFmpeg decodes by outcome (anything but ok is a failure).",
                outcome=outcome,
            )
            self._metrics.observe(
                "decode_seconds",
                "Wall time of one decode, slot wait included.",
                time.monotonic() - started,
            )

    def _argv(self, path: Path) -> list[str]:
        return [
            self._ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "file",
            "-format_whitelist",
            ",".join(DEMUXERS),
            "-codec_whitelist",
            ",".join(DECODERS),
            "-threads",
            "1",
            "-i",
            f"file:{path}",
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ]

    async def _run_ffmpeg(self, path: Path, deadline: float) -> np.ndarray:
        timeout = min(self._decode_timeout, deadline - time.monotonic())
        limit = self._max_samples * SAMPLE_BYTES
        # Minimal environment: the gateway's secrets (API keys, HF_TOKEN) never reach the child.
        proc = await asyncio.create_subprocess_exec(
            *self._argv(path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            start_new_session=True,
        )
        pcm = bytearray()
        stderr = bytearray()
        too_long = False

        async def read_stdout() -> None:
            nonlocal too_long
            while block := await proc.stdout.read(READ_BYTES):
                if len(pcm) + len(block) > limit:
                    too_long = True
                    proc.kill()  # stop decoding now instead of after the whole file
                    return
                pcm.extend(block)

        async def read_stderr() -> None:
            while block := await proc.stderr.read(READ_BYTES):
                stderr.extend(block[: STDERR_KEEP_BYTES - len(stderr)])

        try:
            async with asyncio.timeout(timeout):
                await asyncio.gather(read_stdout(), read_stderr())
                returncode = await proc.wait()
        except TimeoutError:
            log_failure("timeout", None, stderr)
            if time.monotonic() >= deadline:
                raise request_timeout() from None
            raise invalid_audio("Audio decoding exceeded the time limit.") from None
        finally:
            if proc.returncode is None:
                with suppress(ProcessLookupError):
                    proc.kill()
                # Keep the decode slot until reaping finishes, even across repeated cancellation.
                reaper = asyncio.create_task(proc.wait())
                cancelled = False
                while not reaper.done():
                    try:
                        await asyncio.shield(reaper)
                    except asyncio.CancelledError:
                        cancelled = True
                reaper.result()
                if cancelled:
                    raise asyncio.CancelledError

        if too_long:
            raise audio_too_long(self._max_audio_seconds)
        if returncode != 0:
            log_failure("ffmpeg_error", returncode, stderr)
            raise invalid_audio()
        usable = len(pcm) - len(pcm) % SAMPLE_BYTES
        wav = np.frombuffer(pcm, dtype=np.float32, count=usable // SAMPLE_BYTES)
        if not np.isfinite(wav).all():
            raise invalid_audio("Audio contains non-finite samples.")
        return wav


def log_failure(reason: str, returncode: int | None, stderr: bytearray) -> None:
    # FFmpeg's error text names the server temp file and the failing component; never audio or client metadata
    # at -loglevel error. Kept short and printable.
    lines = stderr.decode("utf-8", "replace").strip().splitlines()
    detail = "".join(c if c.isprintable() else "?" for c in (lines[-1] if lines else ""))[:200]
    log(logging.INFO, "audio.decode_failed", reason=reason, returncode=returncode, detail=detail)


class _FormState:
    """Multipart callbacks (synchronous, called by the parser) plus the spooled file of one upload."""

    def __init__(self, ingress: AudioIngress):
        self._ingress = ingress
        self.fields: list[tuple[str, str]] = []
        self.path: Path | None = None
        self.file_size = 0
        self.ended = False
        self._fd: int | None = None
        self._parts = 0
        self._header_name = bytearray()
        self._header_value = bytearray()
        self._disposition: bytes | None = None
        self._name: str | None = None
        self._is_file = False
        self._value = bytearray()
        self._reserved = 0
        self._error: APIError | None = None

    def callbacks(self) -> dict:
        return {
            "on_part_begin": self._part_begin,
            "on_header_field": self._header_field,
            "on_header_value": self._header_value_cb,
            "on_header_end": self._header_end,
            "on_headers_finished": self._headers_finished,
            "on_part_data": self._part_data,
            "on_part_end": self._part_end,
            "on_end": self._end,
        }

    def raise_pending(self) -> None:
        if self._error is not None:
            raise self._error

    def _fail(self, error: APIError) -> None:
        # Callbacks cannot raise through the parser cleanly; remember the first error and stop consuming data.
        if self._error is None:
            self._error = error

    def _part_begin(self) -> None:
        self._parts += 1
        if self._parts > MAX_FORM_PARTS:
            self._fail(
                invalid_request("invalid_value", f"Too many form fields (max {MAX_FORM_PARTS}).")
            )
        self._disposition, self._name, self._is_file = None, None, False
        self._value.clear()

    def _header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name += data[start:end]

    def _header_value_cb(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _header_end(self) -> None:
        if bytes(self._header_name).lower() == b"content-disposition":
            self._disposition = bytes(self._header_value)
        self._header_name.clear()
        self._header_value.clear()

    def _headers_finished(self) -> None:
        kind, params = parse_options_header(self._disposition or b"")
        raw_name = params.get(b"name")
        if kind != b"form-data" or raw_name is None:
            self._fail(invalid_request("invalid_value", "Malformed multipart body."))
            return
        self._name = raw_name.decode("utf-8", "replace")
        if self._name == FILE_FIELD:
            if self.path is not None:
                self._fail(
                    invalid_request("invalid_value", "Only one 'file' part is allowed.", FILE_FIELD)
                )
                return
            self._is_file = True
            self._fd, self.path = self._ingress._new_spool_file()

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        if self._error is not None:
            return
        n = end - start
        if self._is_file:
            try:
                self._ingress._reserve_spool(n)
            except APIError as e:
                self._fail(e)
                return
            self._reserved += n
            self.file_size += n
            # Synchronous write of one body chunk (<= ~64 KiB) into the page cache; no fd outlives `close()`.
            try:
                _write_all(self._fd, memoryview(data)[start:end])
            except OSError as e:
                if e.errno in (errno.ENOSPC, errno.EDQUOT):
                    self._fail(server_busy(retry_after=1))
                    return
                raise
        else:
            if len(self._value) + n > MAX_FIELD_BYTES:
                self._fail(request_too_large(MAX_FIELD_BYTES, what="Form field"))
                return
            self._value += data[start:end]

    def _part_end(self) -> None:
        if self._error is not None or self._name is None or self._is_file:
            return
        try:
            self.fields.append((self._name, self._value.decode("utf-8")))
        except UnicodeDecodeError:
            self._fail(
                invalid_request(
                    "invalid_value", f"Form field '{self._name[:64]}' is not valid UTF-8."
                )
            )

    def _end(self) -> None:
        self.ended = True

    def close(self) -> None:
        """Close and delete the spool file and give back its budget. Idempotent."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self.path is not None:
            with suppress(FileNotFoundError):
                self.path.unlink()
        self._ingress._release_spool(self._reserved)
        self._reserved = 0


def _write_all(fd: int, view: memoryview) -> None:
    while view:
        view = view[os.write(fd, view) :]
