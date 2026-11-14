"""The only code that calls the model, and the single thread allowed to run it.

Every load/warm-up/inference on the recognizer runs on one dedicated worker thread (`EngineWorker`). Endpoints and sessions never touch the model; they go through `app.scheduler`, which
hands the worker at most one job at a time.

Cancellation is soft: a call that has started always runs to completion (a native decode cannot be interrupted safely);
the caller discards the result. A call running longer than `hang_seconds` marks the worker hung: `/health` then fails
so the supervisor restarts the process.
"""

import asyncio
import concurrent.futures
import enum
import logging
import queue
import re
import threading
import time
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np

from app.config import SAMPLE_RATE
from app.errors import inference_failed, model_not_ready
from app.observability import log, Metrics

# The recognizer fails below 1,360 samples (85 ms: fewer than 7 feature frames for the encoder's input convolution),
# for every decoding method (tests/model/test_engine.py). Audio policy for short clips (return "" without inference)
# belongs to the callers; this is the hard safety floor.
MIN_FORWARD_SAMPLES = 1360
MODEL_DIR = (
    "viet-asr/int8"  # under the staging root of scripts/stage_models.py (assets/manifest.json)
)
FEATURE_DIM = 80
WORD_START = "\u2581"
_BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>")  # SentencePiece byte fallback


class ChunkResult(NamedTuple):
    """Result of one chunk: the text and the tokens it was decoded from, with their emission times (no tensors).

    `tokens` are the decoded pieces ("▁" starts a word, so their concatenation spells the text); token i was emitted
    at `times[i]`, a sample offset (16 kHz) relative to the chunk start, exactly as the recognizer reported it.
    Times are not repaired here: alignment rejects times that go backwards or fall outside the chunk.
    """

    text: str
    tokens: tuple[str, ...]
    times: tuple[int, ...]


class Engine(Protocol):
    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def transcribe(self, wav: np.ndarray) -> ChunkResult: ...


def merge_byte_tokens(
    tokens: tuple[str, ...], times: tuple[int, ...]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Decode each run of byte-fallback tokens (`<0xNN>`) into the UTF-8 text it spells, as one token.

    sherpa-onnx already decodes the bytes in its text but returns them as raw tokens, which would not spell the text.
    The bytes of a run are concatenated and decoded as UTF-8 (upstream viet-asr detokenizer rule); an invalid sequence
    becomes U+FFFD, as in the recognizer's text. The merged token keeps the emission time of its first byte.
    """
    out_tokens: list[str] = []
    out_times: list[int] = []
    run = bytearray()
    for token, t in zip(tokens, times, strict=True):
        m = _BYTE_TOKEN.fullmatch(token)
        if m:
            if not run:
                out_times.append(t)
            run.append(int(m.group(1), 16))
            continue
        if run:
            out_tokens.append(run.decode("utf-8", "replace"))
            run.clear()
        out_tokens.append(token)
        out_times.append(t)
    if run:
        out_tokens.append(run.decode("utf-8", "replace"))
    return tuple(out_tokens), tuple(out_times)


class SherpaEngine:
    """One sherpa-onnx offline transducer recognizer on CPU (int8 weights); one 16 kHz mono chunk per call."""

    def __init__(
        self,
        models_dir: Path,
        decoding_method: str,
        beam_size: int,
        num_threads: int,
        warmup_seconds: float,
    ):
        self._dir = Path(models_dir) / MODEL_DIR
        self._decoding_method = decoding_method
        self._beam_size, self._num_threads, self._warmup_seconds = (
            beam_size,
            num_threads,
            warmup_seconds,
        )
        self._recognizer = None

    def load(self) -> None:
        """Local files only: nothing is resolved against the Hub."""
        import sherpa_onnx

        paths = {n: self._dir / f"{n}.int8.onnx" for n in ("encoder", "decoder", "joiner")}
        paths["tokens"] = self._dir / "tokens.txt"
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(
                    f"model file not staged: {path} (run scripts/stage_models.py)"
                )
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            joiner=str(paths["joiner"]),
            tokens=str(paths["tokens"]),
            num_threads=self._num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=FEATURE_DIM,
            decoding_method=self._decoding_method,
            max_active_paths=self._beam_size,
        )

    def warmup(self) -> None:
        """Also used to recover after a failed call. Running the longest allowed chunk reserves its peak memory."""
        self.transcribe(np.zeros(int(self._warmup_seconds * SAMPLE_RATE), dtype=np.float32))

    def transcribe(self, wav: np.ndarray) -> ChunkResult:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, wav)
        self._recognizer.decode_streams([stream])
        result = stream.result
        # sherpa-onnx renders the word marker "▁" as a leading space; times are seconds from the chunk start.
        tokens, times = merge_byte_tokens(
            tuple(t.replace(" ", WORD_START) for t in result.tokens),
            tuple(round(t * SAMPLE_RATE) for t in result.timestamps),
        )
        return ChunkResult(result.text.strip(), tokens, times)


def check_waveform(wav: np.ndarray) -> None:
    if not isinstance(wav, np.ndarray) or wav.dtype != np.float32 or wav.ndim != 1:
        raise ValueError("waveform must be a 1-D float32 numpy array (16 kHz mono)")
    if len(wav) < MIN_FORWARD_SAMPLES:
        raise ValueError(
            f"waveform shorter than {MIN_FORWARD_SAMPLES} samples must not reach the model"
        )
    if not np.isfinite(wav).all():
        raise ValueError("waveform contains non-finite samples")


class EngineState(enum.StrEnum):
    LOADING = "loading"
    READY = "ready"
    RECOVERING = "recovering"
    FAILED = "failed"
    STOPPED = "stopped"


class EngineWorker:
    def __init__(self, engine: Engine, hang_seconds: float, metrics: Metrics):
        self._engine, self._hang_seconds, self._metrics = engine, hang_seconds, metrics
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._main, name="gateway-inference", daemon=True)
        self._state = EngineState.LOADING
        self._call_started: float | None = None
        metrics.gauge_fn(
            "ready",
            "1 when the model is loaded, warmed up and the worker is not hung.",
            lambda: float(self.ready),
        )
        metrics.gauge_fn(
            "worker_busy",
            "1 while an inference call is running.",
            lambda: float(self._call_started is not None),
        )
        metrics.gauge_fn(
            "worker_hung",
            "1 when the running call exceeded GATEWAY_INFER_HANG_SECONDS.",
            lambda: float(self.hung),
        )

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def hung(self) -> bool:
        started = self._call_started
        return started is not None and time.monotonic() - started > self._hang_seconds

    @property
    def ready(self) -> bool:
        return self._state == EngineState.READY and not self.hung

    @property
    def healthy(self) -> bool:
        """Liveness: false when the process needs a restart (load/recovery failed or a call hangs)."""
        return self._state not in (EngineState.FAILED, EngineState.STOPPED) and not self.hung

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float) -> bool:
        """Ask the thread to exit after its current call; False if it is still running after `timeout`."""
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout)
        return not self._thread.is_alive()

    async def transcribe(self, wav: np.ndarray) -> ChunkResult:
        check_waveform(wav)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((wav, fut))
        try:
            return await asyncio.wrap_future(fut)
        finally:
            wav = fut = None  # exception tracebacks must not keep completed audio alive

    def _set_state(self, state: EngineState) -> None:
        if state != self._state:
            self._state = state
            log(logging.INFO, "engine.state", state=state.value)

    def _main(self) -> None:
        t0 = time.monotonic()
        try:
            self._engine.load()
            self._warmup()
        except Exception:
            log(logging.ERROR, "engine.load_failed", exc_info=True)
            self._metrics.inc(
                "worker_failures_total", "Load, inference or recovery failures.", stage="load"
            )
            self._set_state(EngineState.FAILED)
        else:
            log(logging.INFO, "engine.loaded", seconds=round(time.monotonic() - t0, 3))
            self._set_state(EngineState.READY)

        while (item := self._queue.get()) is not None:
            wav, fut = item
            if not fut.set_running_or_notify_cancel():
                wav = item = fut = None
                continue
            if self._state != EngineState.READY:
                fut.set_exception(model_not_ready())
                wav = item = fut = None
                continue
            self._call_started = started = time.monotonic()
            try:
                result = self._engine.transcribe(wav)
            except Exception as e:
                self._call_started = None
                log(logging.ERROR, "engine.inference_failed", exc_info=True, samples=len(wav))
                self._metrics.inc(
                    "worker_failures_total", "Load, inference or recovery failures.", stage="infer"
                )
                err = inference_failed()
                err.__cause__ = e.with_traceback(
                    None
                )  # already logged; the traceback can retain audio
                fut.set_exception(err)
                self._recover()
            else:
                self._call_started = None
                self._metrics.observe(
                    "infer_seconds", "Duration of one inference call.", time.monotonic() - started
                )
                fut.set_result(result)
            finally:
                wav = item = fut = err = (
                    None  # idle worker must not pin completed audio or failure tracebacks
                )
        self._set_state(EngineState.STOPPED)

    def _warmup(self) -> None:
        self._call_started = time.monotonic()
        try:
            self._engine.warmup()
        finally:
            self._call_started = None

    def _recover(self) -> None:
        """Re-run warm-up on this thread; queued calls wait. If it fails the model stays unavailable."""
        self._set_state(EngineState.RECOVERING)
        try:
            self._warmup()
        except Exception:
            log(logging.ERROR, "engine.recovery_failed", exc_info=True)
            self._metrics.inc(
                "worker_failures_total", "Load, inference or recovery failures.", stage="recover"
            )
            self._set_state(EngineState.FAILED)
        else:
            self._set_state(EngineState.READY)
