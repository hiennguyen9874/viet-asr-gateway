# syntax=docker/dockerfile:1
# Image of the viet-asr gateway: one Uvicorn worker = one recognizer = CPU only (sherpa-onnx).
# Models are staged on the host (scripts/stage_models.py) and mounted read-only at /models at runtime; no model,
# HF token or API key is ever part of the image. Build: docker build -t viet-asr-gateway:r1 .
#
# Base: Ubuntu 24.04 for FFmpeg 6.1.1 (the version the M2 decoder allowlist was tested with) and a uv-managed
# CPython 3.12.14 (the baseline's). The sherpa-onnx wheels bring their own onnxruntime; no GPU runtime is needed.
ARG BASE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

FROM ghcr.io/astral-sh/uv:0.12.9@sha256:8b940d3a9d65bed080436972241af2e21c84b5e8c9193f7014ed71479ee795ff AS uv

FROM ${BASE} AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/srv/.venv
RUN uv python install 3.12.14
WORKDIR /srv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --python 3.12.14

FROM ${BASE}
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg=7:6.1.1-3ubuntu5 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 gateway \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin gateway
COPY --from=builder /opt/python /opt/python
COPY --from=builder /srv/.venv /srv/.venv
COPY app/ /srv/app/
ENV PATH=/srv/.venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    GATEWAY_MODELS_DIR=/models HF_HUB_OFFLINE=1 HOME=/tmp
USER 10001:10001
WORKDIR /srv
EXPOSE 18001
# Liveness only (/health): 503 when the worker hangs or failed to load. Docker does not restart an unhealthy
# container by itself; see README "Recovery".
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18001/health', timeout=4)"]
# WebSocket stack set explicitly (`auto` silently disables /v1/realtime without the lib); 2 MiB per client message.
CMD ["uvicorn", "--factory", "app.main:create_app", "--host", "0.0.0.0", "--port", "18001", "--workers", "1", \
     "--no-access-log", "--timeout-graceful-shutdown", "30", "--ws", "websockets-sansio", "--ws-max-size", "2097152"]
