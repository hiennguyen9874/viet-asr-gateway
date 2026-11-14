# Code map

Each file's job, and the tests that cover it. Area rules live in [change-rules.md](change-rules.md).

## app/ (the only code shipped in the image)

```text
config.py         typed GATEWAY_* env config, validated at startup; no default API key
engine.py         SherpaEngine (sherpa-onnx offline transducer, viet-asr int8, CPU, local files only) + EngineWorker,
                    the single thread that runs the model; merge_byte_tokens (<0xNN> runs -> UTF-8 characters);
                    ChunkResult = text + tokens + token emission times (samples, relative to the chunk)
scheduler.py      bounded fair queue (lanes = requests/sessions, jobs = chunks) feeding the worker
audio.py          AudioIngress: multipart ingress (body limit before parser, bounded spool), sandboxed FFmpeg
                    `decode`, decoded-audio RAM budget (`hold`, `reserve`)
chunking.py       `split`: quiet-point splitter, contiguous spans; <=100 ms audio -> no chunk (empty transcript)
alignment.py      pure: token emission times -> words (absolute samples) -> cues; raises AlignmentError
formats.py        pure: verbose_json / srt / vtt serializers
main.py           FastAPI app factory: lifespan, /health, /ready, /metrics, auth, request ids
transcription.py  /v1/models, POST /v1/audio/transcriptions (raw ASGI: json/text/SSE/verbose_json/srt/vtt,
                    disconnect watch, send timeout); parse_params, Joiner (the only text joiner), run_chunks,
                    WORD_TIMESTAMPS release switch
realtime.py       GET /v1/realtime (raw ASGI WebSocket): handshake checks before accept, TurnBuffer
                    (PCM16 24 kHz -> streaming soxr 16 kHz), session.update/event validation, one LIVE lane per
                    session, FIFO items via split/run_chunks/Joiner, bounded sends
errors.py         OpenAI error envelope + status/code mapping, realtime error factories, send_error
schemas.py        pydantic wire schemas: error envelope, status body
observability.py  JSON logs + Prometheus-style metrics (metadata only)
```

## scripts/ (host tooling, never in the image)

```text
stage_models.py     download pinned assets (assets/manifest.json) into ./models or $GATEWAY_MODELS_DIR, verify SHA256
fetch_fixtures.py   build fixtures from FLEURS vi_vn (CC BY 4.0) at a pinned revision, verify SHA256
baseline_offline.py RTF/WER per decoding method, thread sweep -> default threads and queue bounds; metadata-only
                      JSON with transcript SHA256; checks the app/config.py defaults
smoke_server.py     stdlib smoke of a running gateway (local or container) against the offline baseline's transcript
                      hashes -> metadata-only JSON
benchmark.py        R4 closed-loop REST/SSE/WS/mixed ramp + bounded soak; metadata-only JSON, not a capacity claim
demo_page.py        listening page: FLEURS clips + synthetic hard/edge cases -> running gateway -> outputs/<dir>/
                      (audio/, results.json, static index.html with WER, word diff, clickable SRT cues)
probe_limits.py     short-audio floor + latency/peak RSS/WER vs chunk length on the real recognizer -> JSON
probe_alignment.py  token emission times: resolution, shift, delay at known silences, cues -> JSON; checks the
                      app/alignment.py constants
eval_alignment.py   R2b gate: words vs hand annotations in tests/fixtures/alignment/ (none exist yet)
scoring.py          dependency-free WER (lower case, Unicode punctuation removed on both sides), shared by the smoke
                      scripts, baselines and demo page
subtitles.py        dependency-free SRT/VTT parser, shared by smoke_server.py and tests/live_server.py
```

## tests/

| Path | Needs | Covers |
|---|---|---|
| `unit/` | CPU + ffmpeg | manifests/pins, config, byte-fallback token merge (`test_engine_tokens.py`), WER scoring, app over real Uvicorn with the fake engine, transcription params/joiner/end-to-end API (disconnect, SSE, drain, timed formats), alignment, subtitles, realtime (`test_realtime_units.py` pure, `test_realtime_api.py` real WS), observability, benchmark, capacity hardening |
| `scheduler/` | CPU | scheduler/worker gate with the fake engine: serialization, bounds, fairness, cancel |
| `audio/` | CPU + ffmpeg | formats, hostile inputs, decode bounds, multipart limits, temp cleanup |
| `contract/` | CPU (+ Node for `js/`) | `examples/` parsed by the pinned Python SDK (`test_python_sdk.py`) and JS SDK (`js/contract.test.ts`); `test_server_sdk.py` runs a real server against both SDKs (REST over http, Realtime over wss with a self-signed cert, JS via `js/server.test.ts`) |
| `model/` | CPU + staged models | real engine text/tokens/times vs the minimal sherpa-onnx reference `reference.py` (`test_engine.py`, beam/greedy, byte-fallback merge on `vi_bytes.wav`, 1,360-sample floor), worker + lanes, `test_server.py` (json/text/SSE and the R2a cue gate on the real model, writes `docs/baseline/r2a_gate.json`), `test_realtime.py` (R3a realtime gate: turns vs REST, FIFO, short turns, concurrent sessions, long turn; writes `docs/baseline/r3a_gate.json`). Marked `model`; skipped when the model is not staged |
| `container/` | docker + staged models | image gate; skips without docker |

Shared helpers:

- `fake_engine.py`: engine stand-in for CPU tests (`text_fn` for real decoded audio; tokens + times).
- `live_server.py`: `serve(...)` runs real Uvicorn (websockets-sansio, optional self-signed TLS via `tls_dir`)
  around `create_app`; also float-WAV/PCM24 audio helpers and the raw realtime client `WS`.
- `fixtures/`: small committed fixtures + `manifest.json` (hashes, references, license); `vi_bytes.wav` makes the
  model emit byte-fallback tokens.

## Other paths

- `assets/manifest.json`: single source of model pins (repo id, commit, size + SHA256 per file).
- `Dockerfile`, `.dockerignore`, `compose.yaml`: the image and how it runs.
- `docs/baseline/*.json`: saved reports (offline baseline, server smoke, container gate, alignment probe, R2a gate,
  R3a gate, limits probe).
- `models/`, `fixtures-large/`: staged assets and large fixtures (gitignored).
- `references/`: read-only upstream sources (git submodules).
