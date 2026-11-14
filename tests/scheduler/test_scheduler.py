"""M1 gate: the scheduler + single worker thread with a fake engine (no GPU).

Covers: at most one inference at a time for file and live jobs, queue/lane/sample bounds (429), per-lane quota,
round-robin and live priority without file starvation, cancellation of queued jobs, timeouts that must not free the
worker early, deadlines, shutdown/drain, worker failure/recovery and hang detection.
"""

import asyncio
import functools
import time
from contextlib import asynccontextmanager

import pytest

from app.engine import EngineState, EngineWorker
from app.errors import APIError
from app.observability import Metrics
from app.scheduler import JobKind, Scheduler
from tests.fake_engine import config, FakeEngine, wav

FILE, LIVE = JobKind.FILE, JobKind.LIVE


def run_async(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(asyncio.wait_for(fn(*args, **kwargs), 30))

    return wrapper


async def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


async def call_started(engine: FakeEngine, n: int = 1) -> None:
    for _ in range(n):
        assert await asyncio.to_thread(engine.started.acquire, True, 5), "inference call did not start"


@asynccontextmanager
async def serving(engine: FakeEngine, **overrides):
    cfg = config(**overrides)
    metrics = Metrics()
    worker = EngineWorker(engine, cfg.infer_hang_seconds, metrics)
    scheduler = Scheduler(worker, cfg, metrics)
    worker.start()
    await wait_until(lambda: worker.state != EngineState.LOADING)
    scheduler.start()
    try:
        yield scheduler, worker, metrics
    finally:
        engine.gate.set()
        engine.warmup_gate.set()
        await scheduler.close(5)
        assert worker.stop(5)


def jobs(metrics: Metrics, outcome: str, kind: JobKind = FILE) -> float:
    return metrics.value("jobs_total", kind=kind.value, outcome=outcome)


async def transcribe_all(scheduler: Scheduler, kind: JobKind, values: list[int]) -> list[str]:
    """A producer as M3/M5 will use it: submit chunk by chunk, await results in order."""
    async with scheduler.open_lane(kind) as lane:
        pending = [await lane.submit(wav(v)) for v in values[:1]]
        results = []
        for v in values[1:]:
            pending.append(await lane.submit(wav(v)))  # waits while the lane quota is full
            results.append((await pending.pop(0)).text)
        results += [(await j).text for j in pending]
        return results


@run_async
async def test_single_inference_at_a_time_across_file_and_live_lanes():
    engine = FakeEngine(delay=0.002)
    async with serving(engine, queue_max_jobs_per_lane=2) as (scheduler, _, _):
        lanes = [(FILE if i % 2 else LIVE, [i * 100 + k for k in range(6)]) for i in range(1, 9)]
        results = await asyncio.gather(*(transcribe_all(scheduler, kind, values) for kind, values in lanes))
    # Each lane gets exactly its own transcripts, in order.
    assert results == [[str(v) for v in values] for _, values in lanes]
    assert engine.max_active == 1
    assert len(engine.calls) == 48


@run_async
async def test_timed_out_caller_does_not_free_the_worker():
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine) as (scheduler, worker, metrics):
        a = scheduler.open_lane(FILE)
        job_a = await a.submit(wav(1))
        await call_started(engine)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await job_a  # soft deadline: the call keeps running
        b = scheduler.open_lane(LIVE)
        job_b = await b.submit(wav(2))
        await asyncio.sleep(0.2)
        assert engine.calls == [1] and engine.active == 1, "second inference started while first still runs"
        assert scheduler._jobs == 2  # the running job still holds its budget
        engine.gate.set()
        assert (await job_b).text == "2"
        assert engine.max_active == 1
        assert jobs(metrics, "discarded") == 1
        assert worker.ready


@run_async
async def test_new_lane_rejected_when_lanes_full():
    async with serving(FakeEngine(), queue_max_lanes=2, max_sessions=2) as (scheduler, _, metrics):
        scheduler.open_lane(FILE)
        lane = scheduler.open_lane(LIVE)
        with pytest.raises(APIError) as e:
            scheduler.open_lane(FILE)
        assert (e.value.status, e.value.code, e.value.headers) == (429, "server_busy", {"Retry-After": "1"})
        lane.close()
        scheduler.open_lane(FILE)  # slot freed
        assert jobs(metrics, "rejected") == 1


@run_async
async def test_submit_gets_429_when_queue_jobs_stay_full():
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine, queue_max_jobs=2, queue_max_jobs_per_lane=2, queue_timeout_seconds=0.2) as (
        scheduler, _, _,
    ):
        a, b = scheduler.open_lane(FILE), scheduler.open_lane(FILE)
        await a.submit(wav(1))
        await a.submit(wav(2))
        with pytest.raises(APIError) as e:
            scheduler.open_lane(FILE)  # no capacity left: fail fast
        assert e.value.status == 429
        t0 = time.monotonic()
        with pytest.raises(APIError) as e:
            await b.submit(wav(3))  # admitted lane waits up to queue_timeout, then 429
        assert e.value.code == "server_busy" and time.monotonic() - t0 >= 0.2


@run_async
async def test_buffered_samples_bound_and_waiting_submit_admitted_when_capacity_frees():
    engine = FakeEngine()
    engine.gate.clear()
    # 1 s budget = 16000 samples; two 10000-sample jobs do not fit together.
    async with serving(engine, queue_max_buffered_seconds=1.0, max_chunk_seconds=1.0, max_turn_seconds=1.0) as (
        scheduler, _, _,
    ):
        a, b = scheduler.open_lane(FILE), scheduler.open_lane(FILE)
        job_a = await a.submit(wav(1, 10000))
        submit_b = asyncio.create_task(b.submit(wav(2, 10000)))
        await asyncio.sleep(0.1)
        assert not submit_b.done() and scheduler._samples == 10000
        engine.gate.set()
        assert (await job_a).text == "1"
        assert (await (await submit_b)).text == "2"
        with pytest.raises(ValueError):
            await a.submit(wav(3, 16001))  # larger than the whole budget: caller must chunk


@run_async
async def test_lane_quota_backpressure_does_not_enqueue_whole_file():
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine, queue_max_jobs_per_lane=2) as (scheduler, _, _):
        lane = scheduler.open_lane(FILE)
        await lane.submit(wav(1))
        await lane.submit(wav(2))
        third = asyncio.create_task(lane.submit(wav(3)))
        await asyncio.sleep(0.1)
        assert not third.done() and lane.unfinished == 2
        engine.gate.set()
        assert (await (await third)).text == "3"


@run_async
async def test_round_robin_between_file_lanes():
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine, queue_max_jobs_per_lane=3) as (scheduler, _, _):
        blocker = scheduler.open_lane(FILE)
        await blocker.submit(wav(0))
        await call_started(engine)
        a, b, c = (scheduler.open_lane(FILE) for _ in range(3))
        pending = [await lane.submit(wav(v)) for lane, v in ((a, 11), (a, 12), (a, 13), (b, 21), (b, 22), (c, 31))]
        engine.gate.set()
        await asyncio.gather(*pending)
    assert engine.calls == [0, 11, 21, 31, 12, 22, 13]


@run_async
async def test_live_priority_is_bounded_so_files_are_not_starved():
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine, queue_max_jobs_per_lane=10, live_priority_ratio=2) as (scheduler, _, _):
        blocker = scheduler.open_lane(FILE)
        await blocker.submit(wav(0))
        await call_started(engine)
        f, live = scheduler.open_lane(FILE), scheduler.open_lane(LIVE)
        pending = [await f.submit(wav(100 + i)) for i in range(4)]
        pending += [await live.submit(wav(200 + i)) for i in range(5)]
        engine.gate.set()
        await asyncio.gather(*pending)
    assert engine.calls == [0, 200, 201, 100, 202, 203, 101, 204, 102, 103]


@run_async
async def test_cancel_queued_job_frees_budget_and_never_runs():
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine, queue_max_jobs_per_lane=3) as (scheduler, _, metrics):
        lane = scheduler.open_lane(FILE)
        running = await lane.submit(wav(1))
        await call_started(engine)
        by_method = await lane.submit(wav(2))
        by_task = await lane.submit(wav(3))
        waiter = asyncio.create_task(_await(by_task))
        await asyncio.sleep(0)
        by_method.cancel()
        waiter.cancel()  # cancelling the awaiting task cancels the queued job
        await wait_until(lambda: scheduler._jobs == 1 and scheduler._samples == 1600)
        assert lane.unfinished == 1
        engine.gate.set()
        assert (await running).text == "1"
        follow_up = await lane.submit(wav(4))
        assert (await follow_up).text == "4"
        assert engine.calls == [1, 4]
        assert jobs(metrics, "cancelled") == 2


async def _await(job):
    return await job


@run_async
async def test_closing_lane_cancels_queued_and_discards_running():
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine) as (scheduler, _, metrics):
        lane = scheduler.open_lane(FILE)
        running, queued = await lane.submit(wav(1)), await lane.submit(wav(2))
        await call_started(engine)
        lane.close()
        assert running.future.cancelled() and queued.future.cancelled()
        await asyncio.sleep(0.05)
        assert scheduler._jobs == 1  # the running call still owns the worker
        other = scheduler.open_lane(FILE)
        job = await other.submit(wav(3))
        engine.gate.set()
        assert (await job).text == "3"
        assert engine.calls == [1, 3] and engine.max_active == 1
        assert jobs(metrics, "discarded") == 1 and jobs(metrics, "cancelled") == 1
        with pytest.raises(RuntimeError):
            await lane.submit(wav(4))


@run_async
async def test_job_past_deadline_is_dropped_before_inference():
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine) as (scheduler, _, metrics):
        lane = scheduler.open_lane(LIVE)
        await lane.submit(wav(1))
        await call_started(engine)
        late = await scheduler.open_lane(FILE).submit(wav(2), deadline=time.monotonic() + 0.05)
        await asyncio.sleep(0.1)
        engine.gate.set()
        with pytest.raises(APIError) as e:
            await late
        assert (e.value.status, e.value.code) == (503, "request_timeout")
        assert engine.calls == [1]
        assert jobs(metrics, "expired") == 1


@run_async
async def test_shutdown_drains_in_flight_jobs_and_rejects_new_work():
    engine = FakeEngine(delay=0.05)
    async with serving(engine, queue_max_jobs_per_lane=3) as (scheduler, _, _):
        lane = scheduler.open_lane(FILE)
        pending = [await lane.submit(wav(v)) for v in (1, 2, 3)]
        assert await scheduler.close(5) is True
        assert [(await j).text for j in pending] == ["1", "2", "3"]
        with pytest.raises(APIError) as e:
            scheduler.open_lane(FILE)
        assert (e.value.status, e.value.code) == (503, "shutting_down")
        with pytest.raises(APIError) as e:
            await lane.submit(wav(4))
        assert e.value.code == "shutting_down"


@run_async
async def test_shutdown_timeout_fails_queued_jobs_and_reports_stuck_worker():
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine) as (scheduler, worker, _):
        lane = scheduler.open_lane(FILE)
        running, queued = await lane.submit(wav(1)), await lane.submit(wav(2))
        await call_started(engine)
        assert await scheduler.close(0.1) is False
        with pytest.raises(APIError) as e:
            await queued
        assert e.value.code == "shutting_down"
        assert not running.future.done()  # never reported as success or failure by a fake path
        assert worker.stop(0.05) is False  # stuck in the call; the supervisor restarts the process
        engine.gate.set()
    assert engine.calls == [1]


@run_async
async def test_inference_error_returns_500_then_worker_recovers():
    engine = FakeEngine()
    engine.fail_calls = 1
    async with serving(engine) as (scheduler, worker, metrics):
        lane = scheduler.open_lane(LIVE)
        with pytest.raises(APIError) as e:
            await (await lane.submit(wav(1)))
        assert (e.value.status, e.value.code) == (500, "inference_failed")
        assert "simulated" not in e.value.message
        assert (await (await lane.submit(wav(2)))).text == "2"
        assert worker.state == EngineState.READY and engine.warmups == 2  # load + recovery
        assert metrics.value("worker_failures_total", stage="infer") == 1


@run_async
async def test_failed_recovery_makes_worker_unhealthy_and_rejects_work():
    engine = FakeEngine()
    engine.fail_calls = 1
    async with serving(engine) as (scheduler, worker, _):
        lane = scheduler.open_lane(FILE)
        engine.warmup_error = RuntimeError("still broken")
        with pytest.raises(APIError):
            await (await lane.submit(wav(1)))
        await wait_until(lambda: worker.state == EngineState.FAILED)
        assert not worker.healthy and not worker.ready
        with pytest.raises(APIError) as e:
            await lane.submit(wav(2))
        assert (e.value.status, e.value.code) == (503, "model_not_ready")
        with pytest.raises(APIError) as e:
            scheduler.open_lane(FILE)
        assert e.value.code == "model_not_ready"


@run_async
async def test_hung_call_marks_worker_unready_and_unhealthy():
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine, infer_hang_seconds=0.1) as (scheduler, worker, metrics):
        await scheduler.open_lane(FILE).submit(wav(1))
        await call_started(engine)
        await asyncio.sleep(0.15)
        assert worker.hung and not worker.ready and not worker.healthy
        assert "gateway_worker_hung 1.0" in metrics.render()
        engine.gate.set()
        await wait_until(lambda: worker.ready)


@run_async
async def test_lane_refused_while_loading_and_load_failure_is_terminal():
    engine = FakeEngine()
    engine.load_gate.clear()
    cfg = config()
    metrics = Metrics()
    worker = EngineWorker(engine, cfg.infer_hang_seconds, metrics)
    scheduler = Scheduler(worker, cfg, metrics)
    worker.start()
    scheduler.start()
    with pytest.raises(APIError) as e:
        scheduler.open_lane(FILE)
    assert (e.value.status, e.value.code) == (503, "model_not_ready")
    assert worker.healthy  # loading is not a reason to restart
    engine.load_error = RuntimeError("weights missing")
    engine.load_gate.set()
    await wait_until(lambda: worker.state == EngineState.FAILED)
    assert not worker.healthy
    assert metrics.value("worker_failures_total", stage="load") == 1
    await scheduler.close(1)
    assert worker.stop(1)


def test_invalid_waveforms_never_reach_the_engine():
    import numpy as np

    from app.engine import check_waveform, MIN_FORWARD_SAMPLES

    for bad in (np.zeros(MIN_FORWARD_SAMPLES - 1, np.float32), np.zeros((2, 1600), np.float32),
                np.zeros(1600, np.float64), np.full(1600, np.nan, np.float32)):
        with pytest.raises(ValueError):
            check_waveform(bad)
    check_waveform(np.zeros(MIN_FORWARD_SAMPLES, np.float32))


@pytest.mark.parametrize("kind", [FILE, LIVE])
@run_async
async def test_chunk_limit_enforced_independently_of_queue_budget(kind):
    async with serving(FakeEngine(), max_chunk_seconds=1.0) as (scheduler, _, _):
        lane = scheduler.open_lane(kind)
        assert (await (await lane.submit(wav(1, 16000)))).text == "1"
        with pytest.raises(ValueError, match="max_chunk_seconds"):
            await lane.submit(wav(2, 16001))
        assert scheduler._jobs == 0 and scheduler._samples == 0


@run_async
async def test_running_deadline_expires_result_but_keeps_worker_and_budget():
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine) as (scheduler, _, metrics):
        lane = scheduler.open_lane(FILE)
        late = await lane.submit(wav(1), deadline=time.monotonic() + 0.1)
        await call_started(engine)
        with pytest.raises(APIError) as exc:
            await late
        assert (exc.value.status, exc.value.code) == (503, "request_timeout")
        following = await scheduler.open_lane(LIVE).submit(wav(2))
        await asyncio.sleep(0.05)
        assert scheduler._jobs == 2 and scheduler._samples == 3200
        assert engine.calls == [1] and not following.future.done()
        engine.gate.set()
        assert (await following).text == "2"
        assert engine.max_active == 1 and jobs(metrics, "expired") == 1
        assert scheduler._jobs == 0 and scheduler._samples == 0


@pytest.mark.parametrize("recovery", [False, True])
@run_async
async def test_blocked_warmup_marks_worker_unhealthy(recovery):
    engine = FakeEngine()
    if recovery:
        engine.fail_calls = 1
    else:
        engine.warmup_gate.clear()
    cfg = config(infer_hang_seconds=0.05)
    metrics = Metrics()
    worker = EngineWorker(engine, cfg.infer_hang_seconds, metrics)
    scheduler = Scheduler(worker, cfg, metrics)
    worker.start()
    scheduler.start()
    try:
        if recovery:
            await wait_until(lambda: worker.ready)
            engine.warmup_gate.clear()
            with pytest.raises(APIError):
                await (await scheduler.open_lane(FILE).submit(wav(1)))
        await wait_until(lambda: worker.hung)
        assert worker.state == (EngineState.RECOVERING if recovery else EngineState.LOADING)
        assert not worker.healthy and not worker.ready
        assert "gateway_worker_hung 1.0" in metrics.render()
        engine.warmup_gate.set()
        await wait_until(lambda: worker.ready)
        assert worker.healthy
    finally:
        engine.warmup_gate.set()
        await scheduler.close(1)
        assert worker.stop(1)


@run_async
async def test_release_audio_when_idle_runs_every_callback_once_after_running_job():
    """A realtime session registers several reservations (items, pending turn) on one lane."""
    engine = FakeEngine()
    engine.gate.clear()
    async with serving(engine) as (scheduler, _, _):
        lane = scheduler.open_lane(LIVE)
        job = await lane.submit(wav(1))
        await call_started(engine)
        lane.close()
        released = []
        lane.release_audio_when_idle(lambda: released.append("a"))
        lane.release_audio_when_idle(lambda: released.append("b"))
        await asyncio.sleep(0.05)
        assert released == [] and job.future.cancelled()  # the call still runs on the audio
        engine.gate.set()
        await wait_until(lambda: scheduler._jobs == 0)
        assert released == ["a", "b"]
        lane.release_audio_when_idle(lambda: released.append("c"))  # idle: at once
        assert released == ["a", "b", "c"]
