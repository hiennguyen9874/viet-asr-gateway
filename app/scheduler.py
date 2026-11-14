"""Bounded, fair job queue in front of the single inference worker.

- A *lane* is one REST request or one realtime session; a *job* is one chunk of 16 kHz audio to transcribe.
- Producers submit chunks one at a time (`await lane.submit(wav)`); a lane holds at most `max_jobs_per_lane`
  unfinished jobs, so long files are never enqueued all at once.
- Global bounds: open lanes, unfinished jobs and buffered samples (audio held by queued + running jobs). A new lane
  over the limit gets 429 at once; a submit waits up to `queue_timeout` for capacity, then gets 429.
- Dispatch is round-robin across lanes; live lanes take priority, but at most `live_priority_ratio` live jobs run
  in a row while a file job waits.
- One dispatcher task feeds the worker one job at a time, so at most one inference runs, whatever the callers do.
- Cancelling a queued job (`job.cancel()`, cancelling the task awaiting it, or closing its lane) removes it and frees
  its budget. A running job is never interrupted: its result is discarded and its budget is freed only when the call
  ends, so a timed-out request cannot cause a second inference to start alongside it.

All state is owned by the event loop thread; nothing here is thread-safe.
"""

import asyncio
import enum
import itertools
import logging
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from app.config import Config, SAMPLE_RATE
from app.engine import check_waveform, ChunkResult, EngineState, EngineWorker
from app.errors import (
    APIError,
    inference_failed,
    model_not_ready,
    request_timeout,
    server_busy,
    shutting_down,
)
from app.observability import log, Metrics, request_id


class JobKind(enum.StrEnum):
    FILE = "file"
    LIVE = "live"


class JobState(enum.Enum):
    QUEUED = 1
    RUNNING = 2
    DONE = 3


class Job:
    """Awaitable result of one chunk. Awaiting it returns a `ChunkResult` or raises an `APIError`."""

    def __init__(self, lane: "Lane", audio: np.ndarray, deadline: float | None):
        self.lane, self.audio, self.samples, self.deadline = lane, audio, len(audio), deadline
        self.future: asyncio.Future[ChunkResult] = asyncio.get_running_loop().create_future()
        self.state = JobState.QUEUED
        self.expired = False
        self.enqueued_at = time.monotonic()

    def cancel(self) -> None:
        self.future.cancel()

    def __await__(self):
        return self.future.__await__()


class Lane:
    _ids = itertools.count(1)

    def __init__(self, scheduler: "Scheduler", kind: JobKind):
        self.id, self.kind, self._scheduler = next(self._ids), kind, scheduler
        self.request_id = request_id.get()
        self.queue: deque[Job] = deque()
        self.running: Job | None = None
        self.unfinished = 0  # queued + running
        self.closed = False
        self._release_audio: list[Callable[[], None]] = []

    def release_audio_when_idle(self, release: Callable[[], None]) -> None:
        """Call `release` (a decoded-audio reservation) once no job of this lane is unfinished, at once if idle.
        Every registered callback runs once, in registration order."""
        if self.unfinished:
            self._release_audio.append(release)
        else:
            release()

    async def submit(self, wav: np.ndarray, deadline: float | None = None) -> Job:
        """Queue one chunk. `deadline` (time.monotonic) expires its result, never the running call."""
        return await self._scheduler._submit(self, wav, deadline)

    def close(self) -> None:
        """Cancel queued jobs and discard the running one's result. Idempotent."""
        self._scheduler._close_lane(self)

    async def __aenter__(self) -> "Lane":
        return self

    async def __aexit__(self, *_) -> None:
        self.close()


class Scheduler:
    def __init__(self, worker: EngineWorker, config: Config, metrics: Metrics):
        self._worker, self._metrics = worker, metrics
        self._max_lanes = config.queue_max_lanes
        self._max_jobs = config.queue_max_jobs
        self._max_jobs_per_lane = config.queue_max_jobs_per_lane
        self._max_samples = config.queue_max_buffered_samples
        self._max_chunk_samples = config.max_chunk_samples
        self._queue_timeout = config.queue_timeout_seconds
        self._live_ratio = config.live_priority_ratio
        self._rings = {
            JobKind.LIVE: deque(),
            JobKind.FILE: deque(),
        }  # lanes with queued jobs, round-robin order
        self._live_streak = 0
        self._lanes = 0
        self._jobs = 0
        self._samples = 0
        self._closing = False
        self._closed: bool | None = None  # result of close()
        self._waiters: list[asyncio.Future] = []
        self._task: asyncio.Task | None = None
        metrics.gauge_fn(
            "open_lanes", "Open requests/sessions holding a scheduler lane.", lambda: self._lanes
        )
        metrics.gauge_fn("queued_jobs", "Unfinished jobs (queued + running).", lambda: self._jobs)
        metrics.gauge_fn(
            "buffered_audio_seconds",
            "Audio held by unfinished jobs.",
            lambda: self._samples / SAMPLE_RATE,
        )

    @property
    def closing(self) -> bool:
        return self._closing

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(
            self._dispatch(), name="gateway-dispatcher"
        )

    def open_lane(self, kind: JobKind) -> Lane:
        if self._closing:
            raise shutting_down()
        if not self._worker.ready:
            raise model_not_ready()
        if self._lanes >= self._max_lanes or not self._has_capacity(1):
            self._count("rejected", kind)
            raise server_busy(retry_after=1)
        self._lanes += 1
        return Lane(self, kind)

    async def close(self, timeout: float) -> bool:
        """Stop accepting work, wait up to `timeout` for unfinished jobs, then fail the rest.

        Returns False if a job was still running (the worker may be hung; the supervisor must restart the process).
        Idempotent: later calls return the first result.
        """
        if self._closed is not None:
            return self._closed
        self._closing = True
        self._wake()
        drained = True
        try:
            async with asyncio.timeout(timeout):
                while self._jobs:
                    await self._wait_change()
        except TimeoutError:
            drained = False
        for ring in self._rings.values():
            for lane in list(ring):
                for job in list(lane.queue):
                    if not job.future.done():
                        job.future.set_exception(shutting_down())  # done callback releases it
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._closed = drained
        return drained

    # -- submission ------------------------------------------------------------------------------------------------

    def _has_capacity(self, samples: int) -> bool:
        return self._jobs < self._max_jobs and self._samples + samples <= self._max_samples

    async def _submit(self, lane: Lane, wav: np.ndarray, deadline: float | None) -> Job:
        check_waveform(wav)
        if len(wav) > self._max_chunk_samples:
            raise ValueError("job exceeds max_chunk_seconds; chunk it first")
        if len(wav) > self._max_samples:
            raise ValueError("job larger than the queue budget; chunk it first")
        try:
            async with asyncio.timeout(self._queue_timeout):
                while True:
                    if lane.closed:
                        raise RuntimeError("lane is closed")
                    if self._closing:
                        raise shutting_down()
                    if self._worker.state in (EngineState.FAILED, EngineState.STOPPED):
                        raise model_not_ready()
                    if lane.unfinished < self._max_jobs_per_lane and self._has_capacity(len(wav)):
                        break
                    await self._wait_change()
        except TimeoutError:
            self._count("rejected", lane.kind)
            raise server_busy(retry_after=1) from None

        job = Job(lane, wav, deadline)
        job.future.add_done_callback(lambda _: self._drop_queued(job))
        lane.queue.append(job)
        lane.unfinished += 1
        self._jobs += 1
        self._samples += job.samples
        if len(lane.queue) == 1:
            self._rings[lane.kind].append(lane)
        self._wake()
        return job

    def _drop_queued(self, job: Job) -> None:
        """Release a job whose future was completed while queued (cancelled, or failed by close()). Idempotent."""
        if job.state != JobState.QUEUED:
            return
        lane, ring = job.lane, self._rings[job.lane.kind]
        if job in lane.queue:
            lane.queue.remove(job)
        if not lane.queue and lane in ring:
            ring.remove(lane)
        self._count("cancelled" if job.future.cancelled() else "aborted", lane.kind)
        self._release(job)

    def _release(self, job: Job) -> None:
        job.state = JobState.DONE
        job.audio = None
        job.lane.unfinished -= 1
        if job.lane.unfinished == 0 and job.lane._release_audio:
            releases, job.lane._release_audio = job.lane._release_audio, []
            for release in releases:
                release()
        self._jobs -= 1
        self._samples -= job.samples
        self._wake()

    def _close_lane(self, lane: Lane) -> None:
        if lane.closed:
            return
        lane.closed = True
        self._lanes -= 1
        for job in list(lane.queue):
            job.cancel()
        if lane.running is not None:
            lane.running.cancel()  # result discarded when the call ends
        self._wake()

    def _count(self, outcome: str, kind: JobKind) -> None:
        self._metrics.inc("jobs_total", "Jobs by outcome.", kind=kind.value, outcome=outcome)

    # -- change notification (callbacks cannot await a Condition) ----------------------------------------------------

    def _wake(self) -> None:
        for w in self._waiters:
            if not w.done():
                w.set_result(None)
        self._waiters.clear()

    async def _wait_change(self) -> None:
        w = asyncio.get_running_loop().create_future()
        self._waiters.append(w)
        await w

    # -- dispatch ---------------------------------------------------------------------------------------------------

    def _pick(self) -> Job | None:
        live, file = self._rings[JobKind.LIVE], self._rings[JobKind.FILE]
        while live or file:
            if live and (not file or self._live_streak < self._live_ratio):
                ring = live
                self._live_streak = self._live_streak + 1 if file else 0
            else:
                ring = file
                self._live_streak = 0
            lane = ring.popleft()
            job = lane.queue.popleft()
            if lane.queue:
                ring.append(lane)
            if not job.future.done():
                return job
            self._drop_queued(job)  # cancelled, its done callback has not run yet
        return None

    async def _dispatch(self) -> None:
        while True:
            job = self._pick()
            if job is None:
                await self._wait_change()
                continue
            lane, kind = job.lane, job.lane.kind
            now = time.monotonic()
            if job.deadline is not None and now > job.deadline:
                self._count("expired", kind)
                self._release(job)
                job.future.set_exception(request_timeout())
                continue
            self._metrics.observe(
                "queue_wait_seconds",
                "Time from submit to start of inference.",
                now - job.enqueued_at,
                kind=kind.value,
            )
            job.state, lane.running = JobState.RUNNING, job
            error: APIError | None = None

            def expire(job=job) -> None:
                if not job.future.done():
                    job.expired = True
                    job.future.set_exception(request_timeout())

            timer = None
            if job.deadline is not None:
                timer = asyncio.get_running_loop().call_later(max(0, job.deadline - now), expire)
            try:
                result = await self._worker.transcribe(job.audio)
            except APIError as e:
                error = e
            except Exception:  # bug in the adapter: never leak details to the client
                log(logging.ERROR, "scheduler.job_error", exc_info=True, request_id=lane.request_id)
                error = inference_failed()
            finally:
                if timer is not None:
                    timer.cancel()
            # The worker completion callback can win the race against the deadline timer.
            if job.deadline is not None and time.monotonic() >= job.deadline:
                expire()
            lane.running = None
            self._release(job)
            infer_s = time.monotonic() - now
            if job.expired:
                outcome = "expired"
            elif job.future.done():
                outcome = "discarded"
            elif error is not None:
                outcome = "failed"
                job.future.set_exception(error)
            else:
                outcome = "ok"
                job.future.set_result(result)
            self._count(outcome, kind)
            log(
                logging.DEBUG,
                "scheduler.job",
                request_id=lane.request_id,
                lane=lane.id,
                kind=kind.value,
                outcome=outcome,
                audio_s=round(job.samples / SAMPLE_RATE, 3),
                queue_wait_s=round(now - job.enqueued_at, 4),
                infer_s=round(infer_s, 4),
            )
