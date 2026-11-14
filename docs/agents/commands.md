# Commands

The runtime is CPU only (sherpa-onnx), so there are no `--extra` profiles: plain `uv sync` / `uv run`. The host is
shared: pick a free `<port>` (8000 is taken; the container binds 18001).

## Setup

```bash
uv sync                                                 # Python 3.12; sherpa-onnx pinned in uv.lock
uv run scripts/stage_models.py                          # pinned model -> ./models (--verify: offline hash check)
uv run scripts/fetch_fixtures.py                        # fixtures (--verify [--small-only]: hash check only)
uv run scripts/fetch_fixtures.py --eval                 # local eval set for baselines/probes (~550 MB download; --verify --eval)
uv run pre-commit install                               # autoflake, isort, ruff-format, ruff (ruff.toml)
```

## Tests

```bash
uv run pytest                                           # CPU tiers; tests/model skips without the staged model, container without docker
uv run pytest tests/model                               # real model on CPU: parity with tests/model/reference.py, API
(cd tests/contract/js && npm ci && npm test)            # JS SDK contract (tsc --strict + node --test)
uv run pytest tests/container                           # image gate (needs docker)
uv run pre-commit run --all-files                       # lint + format
```

The CPU tiers need `ffmpeg` on PATH. If a relocated venv's `pytest` script imports the wrong environment, use
`uv run python -m pytest` instead. Image gate details: docs/container-gate.vi.md.

## Local server and smoke

```bash
GATEWAY_API_KEYS=<key of >=16 chars> uv run \
    uvicorn --factory app.main:create_app --port <port> --workers 1 --no-access-log \
    --timeout-graceful-shutdown 30 --ws websockets-sansio --ws-max-size 2097152    # /ready after warm-up
OPENAI_API_KEY=<key> uv run scripts/smoke_server.py \
    --base-url http://127.0.0.1:<port> --out docs/baseline/server.json       # server on default settings
```

Listening page against a running gateway (writes `outputs/demo/index.html`; `outputs/` is gitignored):

```bash
OPENAI_API_KEY=<key> uv run scripts/demo_page.py --base-url http://127.0.0.1:<port> \
    [--add <audio file or dir>] [--only <case ids or categories>] [--out outputs/<name>]
```

If `uv run ... uvicorn` fails with `Failed to spawn`, the venv was created under another path (stale shebangs):
use `uv run python -m uvicorn ...` or recreate `.venv`.

One recognizer per process, always one worker. Start the server from a shell with no `GATEWAY_TEST_*` exported:
unknown `GATEWAY_*` variables abort startup, as do leftover variables matching `app/config.py:OLD_PREFIXES`.
The model is int8 only; there is no precision setting.

## Baselines and probes

Benchmark invocation (R4): docs/roadmap.md §1.

```bash
uv run scripts/probe_limits.py --out docs/baseline/limits_probe.json  # short-audio floor, chunk length -> max_chunk_seconds (~15 min)
uv run scripts/baseline_offline.py --out docs/baseline/offline.json  # decoding, threads -> defaults (~13 min); both need vi_long.wav + eval set
uv run scripts/probe_alignment.py --out docs/baseline/alignment_probe.json  # emission times -> cue constants (~1 min)
uv run pytest tests/model/test_server.py              # R2a cue gate; writes docs/baseline/r2a_gate.json when it passes
uv run pytest tests/model/test_realtime.py            # R3a realtime gate; writes docs/baseline/r3a_gate.json when it passes
# Run each gate file on its own with a clean tree: a report written by the other marks the commit -dirty.
```
