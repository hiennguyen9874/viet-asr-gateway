"""FastAPI gateway: lifespan (load/warm-up, drain), health/readiness, auth, request ids, error envelope.

Run with exactly one worker process (the single model instance lives in this process):
    uvicorn --factory app.main:create_app --workers 1 --no-access-log --timeout-graceful-shutdown 30
"""

import asyncio
import hashlib
import hmac
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.audio import AudioIngress
from app.config import Config, ConfigError
from app.engine import Engine, EngineWorker, SherpaEngine
from app.errors import (
    APIError,
    install_handlers,
    internal_error,
    invalid_api_key,
    model_not_ready,
    send_error,
    shutting_down,
)
from app.observability import configure_logging, log, Metrics, request_id
from app.realtime import install as install_realtime
from app.scheduler import Scheduler
from app.schemas import Status
from app.transcription import install as install_transcription

PUBLIC_PATHS = frozenset({"/health", "/ready"})  # probes; everything else needs an API key
WORKER_STOP_SECONDS = 5.0
LOG_PATHS = PUBLIC_PATHS | {"/metrics", "/v1/models", "/v1/audio/transcriptions", "/v1/realtime"}


def log_path(path: str) -> str:
    """Do not echo arbitrary client path components into metadata logs."""
    if path in LOG_PATHS:
        return path
    if path.startswith("/v1/models/"):
        return "/v1/models/{model}"
    return "<unmatched>"


def create_app(config: Config | None = None, engine: Engine | None = None) -> FastAPI:
    config = config or Config.from_env()
    configure_logging(config.log_level)
    if engine is None:
        if not config.models_dir.is_dir():
            raise ConfigError(
                f"GATEWAY_MODELS_DIR is not a directory (run scripts/stage_models.py): {config.models_dir}"
            )
        engine = SherpaEngine(
            config.models_dir,
            config.decoding_method,
            config.beam_size,
            config.num_threads,
            warmup_seconds=config.max_chunk_seconds,
        )

    if not config.spool_dir.is_dir():
        raise ConfigError(f"GATEWAY_SPOOL_DIR is not a directory: {config.spool_dir}")
    metrics = Metrics()
    metrics.process_resources()
    try:
        audio = AudioIngress(config, metrics)
    except FileNotFoundError as e:
        raise ConfigError(str(e)) from None
    worker = EngineWorker(engine, config.infer_hang_seconds, metrics)
    scheduler = Scheduler(worker, config, metrics)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # The model loads on the worker thread; /health answers meanwhile and /ready turns true after warm-up.
        audio.start()
        worker.start()
        scheduler.start()
        log(logging.INFO, "server.started")
        try:
            yield
        finally:
            log(logging.INFO, "server.draining")
            drained = await scheduler.close(config.shutdown_drain_seconds)
            stopped = await asyncio.to_thread(worker.stop, WORKER_STOP_SECONDS)
            audio.stop()
            level = logging.INFO if drained and stopped else logging.ERROR
            # A worker stuck in a call cannot be interrupted; it is a daemon thread and dies with the process.
            log(level, "server.stopped", drained=drained, worker_stopped=stopped)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.config, app.state.metrics, app.state.worker, app.state.scheduler = (
        config,
        metrics,
        worker,
        scheduler,
    )
    app.state.audio = audio
    install_handlers(app)
    install_transcription(app, config, audio, scheduler, worker, metrics)
    install_realtime(app, config, audio, scheduler, worker, metrics)
    app.add_middleware(GatewayMiddleware, api_keys=config.api_keys, metrics=metrics)

    @app.get("/health", response_model=Status)
    async def health():
        """Liveness: fails only when a restart is needed (load/recovery failed, or an inference call hangs)."""
        if not worker.healthy:
            return APIError(
                503, "server_error", "unhealthy", "Inference worker needs a restart."
            ).response()
        return Status(status="ok")

    @app.get("/ready", response_model=Status)
    async def ready():
        """Readiness: model loaded and warmed up, worker not recovering or hung, not draining."""
        if scheduler.closing:
            return shutting_down().response()
        if not worker.ready:
            return model_not_ready().response()
        return Status(status="ready")

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    return app


class GatewayMiddleware:
    """Request id (`x-request-id` header + log context), API key check, access log, last-resort 500 envelope.

    Pure ASGI so it also guards WebSocket handshakes: a bad key is refused with HTTP 401 before accept.
    """

    def __init__(self, app: ASGIApp, api_keys: tuple[str, ...], metrics: Metrics):
        self._app, self._metrics = app, metrics
        self._digests = [hashlib.sha256(k.encode()).digest() for k in api_keys]

    def _authorized(self, scope: Scope) -> bool:
        header = next((v for k, v in scope["headers"] if k == b"authorization"), b"")
        scheme, _, key = header.decode("latin-1").partition(" ")
        if scheme.lower() != "bearer" or not key.strip():
            return False
        digest = hashlib.sha256(key.strip().encode("latin-1")).digest()
        # Compare against every key so timing does not reveal which (if any) matched.
        return sum(hmac.compare_digest(digest, d) for d in self._digests) > 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return
        rid = f"req_{uuid.uuid4().hex}"
        token = request_id.set(rid)
        started = time.monotonic()
        status: int | None = None

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] in ("http.response.start", "websocket.http.response.start"):
                status = message["status"]
                MutableHeaders(scope=message).append("x-request-id", rid)
            elif message["type"] == "websocket.accept":
                status = 101
                message.setdefault("headers", [])  # optional in ASGI
                MutableHeaders(scope=message).append("x-request-id", rid)
            await send(message)

        try:
            if scope["path"] not in PUBLIC_PATHS and not self._authorized(scope):
                await send_error(scope, send_with_id, invalid_api_key())
            else:
                await self._app(scope, receive, send_with_id)
        except Exception:
            log(logging.ERROR, "http.unhandled_error", exc_info=True, path=log_path(scope["path"]))
            if status is None:
                await send_error(scope, send_with_id, internal_error())
        finally:
            duration = time.monotonic() - started
            self._metrics.inc(
                "http_requests_total",
                "HTTP requests and WebSocket handshakes by status.",
                status=str(status),
            )
            # Path only: no query string, headers, body or client filename.
            log(
                logging.INFO,
                "http.request",
                method=scope.get("method", "WS"),
                path=log_path(scope["path"]),
                status=status,
                duration_ms=round(duration * 1000, 1),
            )
            request_id.reset(token)
