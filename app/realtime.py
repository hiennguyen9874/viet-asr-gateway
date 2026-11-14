"""Realtime transcription over WebSocket `/v1/realtime`, explicit commit (R3a; docs/api-contract.vi.md §6).

One session = one accepted WebSocket = one `LIVE` scheduler lane held for the whole session:
  handshake (auth in GatewayMiddleware; query, readiness and capacity here; every refusal is an HTTP response
  before accept) -> `session.created` -> three tasks until the first one ends:
  - receiver: client events in order. append: base64 -> `TurnBuffer` (PCM16 24 kHz -> streaming soxr -> float32
    16 kHz) under a decoded-audio reservation; commit: an immutable item for the processor (FIFO); clear.
  - processor: one item at a time: split -> run_chunks -> Joiner -> delta* -> completed | failed.
  - watchdog: idle timeout (no client event and no unfinished item); the session lifetime is bounded too.
- Every send goes through one lock and `send_timeout_seconds`; no outbound queue (Uvicorn flow control bounds it).
  A send that fails or times out ends the session as a disconnect.
- On every exit the lane is closed (queued jobs cancelled, the running job's result discarded) and each audio
  reservation is released once no job runs on it.
- Logs and metrics carry metadata only: never audio, base64, transcripts or prompts.

Run Uvicorn with `--ws websockets-sansio --ws-max-size 2097152` (`WS_MAX_BYTES`): `auto` silently disables
WebSockets when the library is missing, and the default 16 MiB message limit is too large a per-session buffer.
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from urllib.parse import parse_qsl

import numpy as np
import soxr
from fastapi import FastAPI
from starlette.types import Message, Receive, Scope, Send

from app.audio import AudioIngress
from app.chunking import split
from app.config import Config, SAMPLE_RATE
from app.engine import EngineWorker
from app.errors import (
    APIError,
    idle_timeout,
    input_audio_buffer_commit_empty,
    input_audio_buffer_full,
    internal_error,
    invalid_event,
    invalid_request,
    model_not_found,
    model_not_ready,
    request_timeout,
    send_error,
    server_busy,
    session_expired,
    shutting_down,
    too_many_pending_items,
    unknown_parameter,
    unsupported_event,
)
from app.observability import log, Metrics
from app.scheduler import JobKind, Lane, Scheduler
from app.transcription import Joiner, MAX_PARAM_CHARS, MODEL_ID, run_chunks, served_models

PCM_RATE = 24000  # audio/pcm: 16-bit little-endian mono at 24 kHz, the only accepted input format
WS_MAX_BYTES = (
    2 * 1024 * 1024
)  # Uvicorn --ws-max-size: one client message (~32 s of base64 audio); larger -> 1009
OFFLOAD_CHARS = (
    256 * 1024
)  # JSON/base64/resample of a larger message runs in a thread (~3 ms per 256 KiB inline)
LANGUAGE = "vi"

# Client event types of the pinned GA union (openai 3.31.0 `RealtimeClientEvent`); anything else is invalid.
CLIENT_EVENT_TYPES = frozenset(
    {
        "conversation.item.create",
        "conversation.item.delete",
        "conversation.item.retrieve",
        "conversation.item.truncate",
        "input_audio_buffer.append",
        "input_audio_buffer.clear",
        "output_audio_buffer.clear",
        "input_audio_buffer.commit",
        "response.cancel",
        "response.create",
        "session.update",
    }
)
EVENT_FIELDS = {  # supported events -> their fields besides `type` and `event_id` (all required)
    "session.update": ("session",),
    "input_audio_buffer.append": ("audio",),
    "input_audio_buffer.commit": (),
    "input_audio_buffer.clear": (),
}
QUERY_PARAMS = ("model", "intent")


# -- turn buffer --------------------------------------------------------------------------------------------------


def samples_bound(samples_in: int) -> int:
    """Upper bound of the 16 kHz samples `samples_in` input samples become (reserved per append)."""
    return -(-samples_in * SAMPLE_RATE // PCM_RATE)


class TurnBuffer:
    """Pending (uncommitted) audio of one session: PCM16 LE 24 kHz in, float32 16 kHz out.

    Resampling is streamed at append (soxr HQ); the
    committed waveform equals one-shot `soxr.resample` of the whole turn, whatever the packetization. A trailing
    odd byte waits for the next append; at commit it is dropped (half a sample). The turn limit counts whole input
    samples; an append over it is refused whole and leaves the buffer unchanged.
    """

    def __init__(self, max_samples: int):
        self._max = max_samples
        self.clear()

    @property
    def samples(self) -> int:
        """Whole input samples pending."""
        return self._samples

    def clear(self) -> None:
        self._stream = soxr.ResampleStream(PCM_RATE, SAMPLE_RATE, 1, dtype="float32", quality="HQ")
        self._parts: list[np.ndarray] = []
        self._odd = b""
        self._samples = 0

    def check(self, nbytes: int) -> int:
        """Whole input samples an append of `nbytes` adds; raises `input_audio_buffer_full` over the limit."""
        added = (len(self._odd) + nbytes) // 2
        if self._samples + added > self._max:
            raise input_audio_buffer_full(self._max / PCM_RATE)
        return added

    def append(self, pcm: bytes) -> None:
        added = self.check(len(pcm))
        data = self._odd + pcm if self._odd else pcm
        self._odd = bytes(data[2 * added :])
        if added:
            x = np.frombuffer(data, dtype="<i2", count=added).astype(np.float32) / 32768
            out = self._stream.resample_chunk(x)
            if len(out):
                self._parts.append(out)
        self._samples += added

    def commit(self) -> tuple[np.ndarray, float]:
        """Flush the resampler; return the 16 kHz waveform and the committed seconds. The buffer starts empty."""
        if not self._samples:
            raise input_audio_buffer_commit_empty()
        self._parts.append(self._stream.resample_chunk(np.zeros(0, np.float32), last=True))
        wav, seconds = np.concatenate(self._parts), self._samples / PCM_RATE
        self.clear()
        return wav, seconds


# -- session.update -----------------------------------------------------------------------------------------------


def session_object(model: str) -> dict:
    """The full session; optional fields the JS types do not allow as null are omitted (contract §6)."""
    return {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": PCM_RATE},
                "transcription": {"model": model, "language": LANGUAGE},
                "turn_detection": None,
            }
        },
    }


def _invalid(path: str, message: str) -> APIError:
    return invalid_request("invalid_value", message, path)


def _unsupported(path: str) -> APIError:
    return invalid_request(
        "unsupported_parameter", "This parameter is not supported by this server.", path
    )


def _object(parent: dict, key: str, path: str, keys: tuple[str, ...]) -> dict:
    """`parent[key]` as an object with only `keys` (absent or null = {})."""
    value = parent.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _invalid(f"{path}.{key}", f"'{path}.{key}' must be an object.")
    for k in value:
        if k not in keys:
            raise unknown_parameter(f"{path}.{key}.{k}")
    return value


def parse_session_update(session: object, current: dict, models: tuple[str, ...]) -> dict:
    """Validate a `session.update` session (absent or null field = unchanged) and return the new full session.
    Raises `APIError` on the first problem; `current` is never modified."""
    if not isinstance(session, dict):
        raise _invalid("session", "'session' must be an object.")
    if session.get("type") != "transcription":
        raise _invalid("session.type", "Only 'transcription' sessions are supported.")
    for k in session:
        if k not in ("type", "audio", "include"):
            raise unknown_parameter(f"session.{k}")
    if session.get("include") is not None:
        raise _unsupported("session.include")
    audio = _object(session, "audio", "session", ("input",))
    p = "session.audio.input"
    input_ = _object(
        audio,
        "input",
        "session.audio",
        ("format", "noise_reduction", "transcription", "turn_detection"),
    )

    fmt = _object(input_, "format", p, ("type", "rate"))
    if fmt.get("type") in ("audio/pcmu", "audio/pcma"):
        raise _unsupported(f"{p}.format.type")
    if fmt.get("type") not in (None, "audio/pcm"):
        raise _invalid(f"{p}.format.type", "Supported audio format: 'audio/pcm'.")
    rate = fmt.get("rate")
    if rate is not None and (type(rate) is not int or rate != PCM_RATE):
        raise _invalid(f"{p}.format.rate", f"Supported rate: {PCM_RATE}.")
    if input_.get("noise_reduction") is not None:
        raise _unsupported(f"{p}.noise_reduction")
    if input_.get("turn_detection") is not None:
        raise invalid_request(
            "unsupported_turn_detection",
            "Only turn_detection null (explicit commit) is supported.",
            f"{p}.turn_detection",
        )

    t = f"{p}.transcription"
    transcription = _object(
        input_,
        "transcription",
        p,
        ("delay", "keywords", "language", "languages", "model", "prompt"),
    )
    for k in ("delay", "keywords", "languages"):
        if transcription.get(k) is not None:
            raise _unsupported(f"{t}.{k}")
    prompt = transcription.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise _invalid(f"{t}.prompt", "'prompt' must be a string.")
    if prompt:
        raise _unsupported(f"{t}.prompt")
    language = transcription.get("language")
    if language is not None and not isinstance(language, str):
        raise _invalid(f"{t}.language", "'language' must be a string.")
    if language not in (None, LANGUAGE):
        raise invalid_request(
            "unsupported_language", f"Only language '{LANGUAGE}' is supported.", f"{t}.language"
        )
    model = transcription.get("model")
    if model is not None and not isinstance(model, str):
        raise _invalid(f"{t}.model", "'model' must be a string.")
    if model is not None and model not in models:
        e = model_not_found(model)
        raise invalid_request(e.code, e.message, f"{t}.model")
    return session_object(model or current["audio"]["input"]["transcription"]["model"])


# -- client events ------------------------------------------------------------------------------------------------


def load_event(text: str) -> object:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        raise invalid_event("Event is not valid JSON.") from None


def validate_event(event: object) -> dict:
    """A supported client event with exactly its fields, or `APIError` (invalid_event, unsupported_event, ...)."""
    if not isinstance(event, dict):
        raise invalid_event("Event must be a JSON object.")
    type_ = event.get("type")
    if not isinstance(type_, str) or type_ not in CLIENT_EVENT_TYPES:
        raise invalid_event("Unknown or missing event type.", "type")
    if type_ not in EVENT_FIELDS:
        raise unsupported_event()
    fields = EVENT_FIELDS[type_]
    for k in event:
        if k not in ("type", "event_id", *fields):
            raise unknown_parameter(k)
    if "event_id" in event and not isinstance(event["event_id"], str):
        raise _invalid("event_id", "'event_id' must be a string.")
    for k in fields:
        if k not in event:
            raise invalid_request(
                "missing_required_parameter", f"Missing required parameter: '{k}'.", k
            )
    if type_ == "input_audio_buffer.append" and not isinstance(event["audio"], str):
        raise _invalid("audio", "'audio' must be a base64 string.")
    return event


def parse_event(text: str) -> dict:
    return validate_event(load_event(text))


def client_event_id(event: object) -> str | None:
    """The client's `event_id`, echoed in `error.event_id` only when it is a short printable string."""
    eid = event.get("event_id") if isinstance(event, dict) else None
    if isinstance(eid, str) and len(eid) <= MAX_PARAM_CHARS and eid.isprintable():
        return eid
    return None


def decode_base64(audio: str) -> bytes:
    """Canonical padded base64 only (Python's validate=True still accepts excess padding)."""
    pcm = base64.b64decode(audio, validate=True)
    if base64.b64encode(pcm).decode("ascii") != audio:
        raise ValueError("non-canonical base64")
    return pcm


# -- session ------------------------------------------------------------------------------------------------------


class SessionGone(Exception):
    """A send failed: the client disconnected or stopped reading (`outcome`)."""

    def __init__(self, outcome: str):
        super().__init__(outcome)
        self.outcome = outcome


@dataclass(eq=False)
class Item:
    id: str
    wav: np.ndarray | None
    seconds: float
    committed_at: float
    releases: list[Callable[[], None]] = field(default_factory=list)  # decoded-audio reservations


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class Session:
    def __init__(
        self, endpoint: "RealtimeEndpoint", lane: Lane, model: str, receive: Receive, send: Send
    ):
        self._ep, self._lane, self._receive, self._raw_send = endpoint, lane, receive, send
        config = endpoint.config
        self._config = config
        self._session = session_object(model)
        self._buffer = TurnBuffer(int(config.max_turn_seconds * PCM_RATE))
        self._pending_releases: list[
            Callable[[], None]
        ] = []  # reservations of the uncommitted turn
        self._queue: asyncio.Queue[Item] = asyncio.Queue()
        self._items: list[Item] = []  # committed, reservations not yet released
        self._unfinished = 0  # committed items without a terminal event
        self._previous_item_id: str | None = None
        self._send_lock = asyncio.Lock()
        self._started = time.monotonic()
        self._last_activity = self._started
        self.committed = 0
        self.audio_s = 0.0

    # -- sending ---------------------------------------------------------------------------------------------------

    async def _send(self, message: Message) -> None:
        try:
            async with asyncio.timeout(self._config.send_timeout_seconds):
                async with self._send_lock:
                    await self._raw_send(message)
        except TimeoutError:
            raise SessionGone("send_timeout") from None
        except OSError:  # uvicorn ClientDisconnected
            raise SessionGone("disconnected") from None

    async def send_event(self, type_: str, **fields: object) -> None:
        event = {"type": type_, "event_id": _new_id("event"), **fields}
        await self._send(
            {
                "type": "websocket.send",
                "text": json.dumps(event, ensure_ascii=False, separators=(",", ":")),
            }
        )

    async def _send_error(self, e: APIError, event_id: str | None) -> None:
        error = {"type": e.type, "code": e.code, "message": e.message, "param": e.param}
        if event_id is not None:
            error["event_id"] = event_id
        await self.send_event("error", error=error)

    # -- lifecycle -------------------------------------------------------------------------------------------------

    async def run(self) -> str:
        """Serve the session until it ends; return the outcome. Always closes the lane and releases audio."""
        tasks: list[asyncio.Task] = []
        try:
            await self._send({"type": "websocket.accept"})
            await self.send_event("session.created", session=self._session)
            tasks = [
                asyncio.create_task(self._receive_loop(), name="gateway-rt-receive"),
                asyncio.create_task(self._process_loop(), name="gateway-rt-process"),
                asyncio.create_task(self._watch_idle(), name="gateway-rt-idle"),
            ]
            remaining = self._started + self._config.max_session_seconds - time.monotonic()
            done, _ = await asyncio.wait(
                tasks, timeout=max(remaining, 0), return_when=asyncio.FIRST_COMPLETED
            )
            outcome = done.pop().result() if done else "session_expired"
            await self._stop(tasks)
            if outcome in ("idle_timeout", "session_expired"):
                c = self._config
                error = (
                    idle_timeout(c.session_idle_timeout_seconds)
                    if outcome == "idle_timeout"
                    else session_expired(c.max_session_seconds)
                )
                await self._send_error(error, None)
                await self._send({"type": "websocket.close", "code": 1000})
            return outcome
        except SessionGone as e:
            return e.outcome
        finally:
            await self._stop(tasks)
            self._close()

    @staticmethod
    async def _stop(tasks: list[asyncio.Task]) -> None:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _close(self) -> None:
        self._lane.close()
        releases = self._pending_releases + [r for item in self._items for r in item.releases]
        for item in self._items:
            item.wav = None
        self._pending_releases, self._items = [], []

        def release_all() -> None:
            for release in releases:
                release()

        # A running job may still read an item's waveform: its reservation outlives the session until it ends.
        self._lane.release_audio_when_idle(release_all)

    async def _watch_idle(self) -> str:
        idle = self._config.session_idle_timeout_seconds
        while True:
            wait = self._last_activity + idle - time.monotonic()
            if wait <= 0:
                if not self._unfinished:
                    return "idle_timeout"
                wait = idle  # the client is waiting for a transcript; the terminal event counts as activity
            await asyncio.sleep(wait)

    # -- receiver --------------------------------------------------------------------------------------------------

    async def _receive_loop(self) -> str:
        while True:
            message = await self._receive()
            if message["type"] == "websocket.disconnect":
                return (
                    "disconnected"  # client close, message too big (1009) or server shutdown (1012)
                )
            self._last_activity = time.monotonic()
            await self._handle(message.get("text"))

    async def _handle(self, text: str | None) -> None:
        event_id = None
        label = "invalid"
        try:
            if text is None:
                raise invalid_event("Binary frames are not supported; send JSON text events.")
            raw = await _offload(len(text), load_event, text)
            event_id = client_event_id(raw)
            if (
                isinstance(raw, dict)
                and isinstance(raw.get("type"), str)
                and raw["type"] in CLIENT_EVENT_TYPES
            ):
                label = raw["type"]
            event = validate_event(raw)
            type_ = event["type"]
            if type_ == "session.update":
                self._session = parse_session_update(
                    event["session"], self._session, self._ep.models
                )
                await self.send_event("session.updated", session=self._session)
            elif type_ == "input_audio_buffer.append":
                await self._append(event["audio"])
            elif type_ == "input_audio_buffer.commit":
                await self._commit()
            else:
                self._clear()
                await self.send_event("input_audio_buffer.cleared")
        except APIError as e:
            await self._send_error(e, event_id)
        finally:
            self._ep.metrics.inc(
                "realtime_client_events_total", "Realtime client events by type.", type=label
            )

    async def _append(self, audio: str) -> None:
        try:
            pcm = await _offload(len(audio), decode_base64, audio)
        except ValueError:  # binascii.Error, or non-ASCII characters
            raise invalid_request(
                "invalid_audio", "input_audio_buffer.append: audio is not valid base64.", "audio"
            ) from None
        added = self._buffer.check(len(pcm))
        if added:
            self._pending_releases.append(self._ep.audio.reserve(samples_bound(added)))
        await _offload(len(pcm), self._buffer.append, pcm)

    async def _commit(self) -> None:
        if not self._buffer.samples:
            raise input_audio_buffer_commit_empty()
        if self._unfinished >= self._config.max_pending_items:
            raise too_many_pending_items(self._config.max_pending_items)
        wav, seconds = self._buffer.commit()
        item = Item(_new_id("item"), wav, seconds, time.monotonic(), self._pending_releases)
        self._pending_releases = []
        self._items.append(item)
        self._unfinished += 1
        previous, self._previous_item_id = self._previous_item_id, item.id
        self.committed += 1
        self.audio_s += seconds
        self._ep.metrics.inc(
            "realtime_audio_seconds_total", "Committed realtime audio.", amount=seconds
        )
        await self.send_event(
            "input_audio_buffer.committed", item_id=item.id, previous_item_id=previous
        )
        await self.send_event(
            "conversation.item.added",
            previous_item_id=previous,
            item={
                "id": item.id,
                "object": "realtime.item",
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_audio"}],
            },
        )
        self._queue.put_nowait(item)

    def _clear(self) -> None:
        self._buffer.clear()
        releases, self._pending_releases = self._pending_releases, []
        for release in releases:
            release()

    # -- processor -------------------------------------------------------------------------------------------------

    async def _process_loop(self) -> None:
        while True:
            item = await self._queue.get()
            await self._transcribe(item)
            self._unfinished -= 1
            self._last_activity = time.monotonic()
            if self._lane.unfinished:  # a failed item's running job still reads its waveform
                idle = asyncio.get_running_loop().create_future()
                self._lane.release_audio_when_idle(
                    lambda idle=idle: idle.done() or idle.set_result(None)
                )
                await idle
            self._items.remove(item)
            item.wav = None
            for release in item.releases:
                release()

    async def _transcribe(self, item: Item) -> None:
        c = self._config
        spans = split(item.wav, c.max_chunk_samples)
        joiner = Joiner()
        outcome = "completed"
        try:
            if time.monotonic() >= item.committed_at + c.request_timeout_seconds:
                raise request_timeout()
            results = run_chunks(
                self._lane,
                item.wav,
                spans,
                item.committed_at + c.request_timeout_seconds,
                c.queue_max_jobs_per_lane,
            )
            async with aclosing(results):
                async for _, result in results:
                    if (delta := joiner.add(result.text)) is not None:
                        await self.send_event(
                            "conversation.item.input_audio_transcription.delta",
                            item_id=item.id,
                            content_index=0,
                            delta=delta,
                        )
        except SessionGone:
            outcome = "disconnected"
            raise
        except Exception as e:
            if not isinstance(e, APIError):
                log(logging.ERROR, "realtime.item_error", exc_info=True)
                e = internal_error()
            outcome = e.code
            await self.send_event(
                "conversation.item.input_audio_transcription.failed",
                item_id=item.id,
                content_index=0,
                error={"type": e.type, "code": e.code, "message": e.message},
            )
        else:
            await self.send_event(
                "conversation.item.input_audio_transcription.completed",
                item_id=item.id,
                content_index=0,
                transcript=joiner.text,
                usage={"type": "duration", "seconds": item.seconds},
            )
            self._ep.metrics.observe(
                "realtime_commit_to_completed_seconds",
                "From commit to the completed event.",
                time.monotonic() - item.committed_at,
            )
        finally:
            m = self._ep.metrics
            m.inc("realtime_items_total", "Committed realtime items by outcome.", outcome=outcome)
            log(
                logging.INFO,
                "realtime.item",
                audio_s=round(item.seconds, 3),
                chunks=len(spans),
                outcome=outcome,
                latency_ms=round((time.monotonic() - item.committed_at) * 1000, 1),
            )


async def _offload(size: int, fn, *args, **kwargs):
    """Run CPU work on large client messages in a thread so the event loop keeps serving other sessions."""
    if size > OFFLOAD_CHARS:
        task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # A running thread cannot be cancelled; retain its audio reservation until it finishes.
            await asyncio.gather(task, return_exceptions=True)
            raise
    return fn(*args, **kwargs)


# -- endpoint -----------------------------------------------------------------------------------------------------


class RealtimeEndpoint:
    """ASGI app for the WebSocket route `/v1/realtime`."""

    def __init__(
        self,
        config: Config,
        audio: AudioIngress,
        scheduler: Scheduler,
        worker: EngineWorker,
        metrics: Metrics,
    ):
        self.config, self.audio, self.metrics = config, audio, metrics
        self._scheduler, self._worker = scheduler, worker
        self.models = served_models(config)
        self.sessions = 0
        metrics.gauge_fn("realtime_sessions", "Open realtime sessions.", lambda: self.sessions)

    def _admit(self, scope: Scope) -> tuple[str, Lane]:
        """Handshake checks in contract order; raises `APIError` (rendered before accept)."""
        pairs = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
        for name, _ in pairs:
            if name not in QUERY_PARAMS:
                shown = name[:MAX_PARAM_CHARS] if name.isprintable() else "<invalid>"
                raise invalid_request(
                    "unsupported_parameter", f"Unsupported query parameter: '{shown}'.", shown
                )
        values = dict(pairs)
        if len(values) != len(pairs):
            raise invalid_request("invalid_value", "Query parameters must be sent once.")
        if values.get("intent", "transcription") != "transcription":
            raise invalid_request(
                "invalid_value", "Only intent 'transcription' is supported.", "intent"
            )
        model = values.get("model", MODEL_ID)
        if model not in self.models:
            raise model_not_found(model)
        if self._scheduler.closing:
            raise shutting_down()
        if not self._worker.ready:
            raise model_not_ready()
        if self.sessions >= self.config.max_sessions:
            raise server_busy(retry_after=1)
        return model, self._scheduler.open_lane(JobKind.LIVE)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        started = time.monotonic()
        if (await receive())["type"] != "websocket.connect":
            return
        try:
            model, lane = self._admit(scope)
        except APIError as e:
            self.metrics.inc(
                "realtime_sessions_total",
                "Realtime sessions by outcome.",
                outcome=f"rejected_{e.code}",
            )
            await send_error(scope, send, e)
            return
        self.sessions += 1
        session = Session(self, lane, model, receive, send)
        outcome = "internal_error"
        try:
            outcome = await session.run()
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception:
            log(logging.ERROR, "realtime.unhandled_error", exc_info=True)
        finally:
            self.sessions -= 1
            self.metrics.inc(
                "realtime_sessions_total", "Realtime sessions by outcome.", outcome=outcome
            )
            log(
                logging.INFO,
                "realtime.session",
                outcome=outcome,
                items=session.committed,
                audio_s=round(session.audio_s, 3),
                duration_s=round(time.monotonic() - started, 3),
            )


def install(
    app: FastAPI,
    config: Config,
    audio: AudioIngress,
    scheduler: Scheduler,
    worker: EngineWorker,
    metrics: Metrics,
) -> None:
    app.router.add_websocket_route(
        "/v1/realtime", RealtimeEndpoint(config, audio, scheduler, worker, metrics)
    )
