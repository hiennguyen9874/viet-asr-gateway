# Change rules by area

Read the section for each area you touch before editing it. File roles: [code-map.md](code-map.md).

## Inference path (`engine.py`, `scheduler.py`)

- `SherpaEngine` is the only code that touches sherpa-onnx: one `OfflineRecognizer`, files from the staged
  local directory only (never the Hub), one chunk per call, always the int8 weights of `viet-asr`. Settings:
  `GATEWAY_DECODING_METHOD`, `GATEWAY_BEAM_SIZE` (-> `max_active_paths`), `GATEWAY_NUM_THREADS`.
- `EngineWorker` is the single thread that calls the recognizer. Work reaches it as jobs (one per chunk) on
  `Scheduler` lanes (one per request or realtime session).
- Engine text, tokens and times must equal the minimal sherpa-onnx reference `tests/model/reference.py` for the same
  audio and decoding (`tests/model/test_engine.py`). Do not post-process the text. The only token decoding is
  `merge_byte_tokens`: runs of byte-fallback tokens (`<0xNN>`) become the UTF-8 characters sherpa-onnx already put in
  the text, keeping the first byte's time (`tests/fixtures/vi_bytes.wav`).
- A lane holds at most `GATEWAY_QUEUE_MAX_JOBS_PER_LANE` (2) queued chunks; long files are fed incrementally. LIVE
  lanes get bounded priority (`live_priority_ratio`).
- Cancel drops queued chunks and discards the running chunk's result. A running decode finishes; the worker and
  budgets are released only when the real call returns.

## Audio (`audio.py`, `chunking.py`)

- Decode client audio only with `AudioIngress.decode`; chunk only with `chunking.split`.
- FFmpeg gets fixed argv, the `file` protocol and the demuxer/decoder allowlists; client filenames, paths and URLs
  stay out of its arguments, out of disk paths and out of logs.
- The engine refuses fewer than 1,360 samples (`MIN_FORWARD_SAMPLES`, the recognizer's measured floor). Audio <= 100 ms
  (1,600 samples) yields `""` without a model call; `GATEWAY_MAX_CHUNK_SECONDS` defaults to 20 s. Both come from
  `scripts/probe_limits.py` (docs/baseline.vi.md §5); change them only with a new report.
- Every resource is budgeted: body limit before the multipart parser, spool bytes, decoded audio in RAM (`hold`,
  realtime pending audio via `reserve`). Exceeding one returns 413/429/503 in the OpenAI envelope.

## Timestamps (`alignment.py`, `formats.py`)

- Times come only from the token emission times of the same inference that produced the text, plus the chunk offset.
  On mismatch raise `AlignmentError` -> 500 `alignment_failed`.
- `EMISSION_SHIFT`, `LAST_WORD_SAMPLES`, `PAUSE_SAMPLES` and the cue limits come from `scripts/probe_alignment.py`
  (docs/baseline.vi.md §6), which exits 1 if they drift. Change them only with a new probe report, then rerun the R2a
  gate (`tests/model/test_server.py`) and commit both reports. Rerun both after changing sherpa-onnx or the model pin.
- `WORD_TIMESTAMPS = False` in `app/transcription.py` until the R2b gate passes (docs/roadmap.md §2); flip it in the
  commit that carries the gate report.

## Realtime (`realtime.py`)

- Run Uvicorn with `--ws websockets-sansio --ws-max-size 2097152`; `--ws auto` silently disables WebSockets.
- Auth, model and capacity checks happen before `accept`.
- One LIVE lane per session; committed items run FIFO through the same `split` / `run_chunks` / `Joiner` as REST.
- The JS realtime SDK connects only over `wss:`; tests use a self-signed cert via `serve(tls_dir=...)`.
- After changing realtime, chunking, the engine, sherpa-onnx or the model pin, rerun the R3a gate
  (`tests/model/test_realtime.py`) and commit `docs/baseline/r3a_gate.json`.

## API contract (`transcription.py`, `realtime.py`, `errors.py`)

- `tests/contract/examples/` plus the SDK tests are the source of truth; docs/api-contract.vi.md describes them.
- To change the wire format, update in this order: the examples, `tests/contract/test_python_sdk.py`,
  `tests/contract/js/contract.test.ts` (and `server.test.ts` when server behaviour changes), then the contract doc
  and capability matrix.
- Omit optional fields instead of sending `null`; the JS types reject `null` for several of them.
- Example files are byte-exact (SRT/VTT/SSE end with a blank line, `transcription.txt` has no trailing newline);
  pre-commit excludes them, so keep editors from normalising them too.

## Config (`config.py`)

- Every `GATEWAY_*` variable is a server setting, and unknown ones abort startup. Leftover `GIPFORMER_*` or `QUAN_*`
  variables (old prefixes) abort too, naming the variable. `GATEWAY_TEST_*` (JS server test) is therefore kept out of the server's environment.
- Each limit is finite and validated at startup. A new setting also gets a row in the README config table.
- `num_threads` and `queue_max_buffered_seconds` defaults come from `scripts/baseline_offline.py` (docs/baseline.vi.md
  §4), which exits 1 if they drift; change them only with a new `docs/baseline/offline.json`.

## Pins, dependencies, image

- Change a model pin only in `assets/manifest.json`, then run `scripts/stage_models.py --update-manifest`. The
  `sherpa-onnx` version in `pyproject.toml` and `uv.lock` must equal the manifest's `code` pin
  (`tests/unit/test_manifests.py`).
- The image contains only `app/` and the venv; models are mounted read-only, keys come from a Docker secret.
- No torch or CUDA dependency: the image is CPU only. After changing the base image, sherpa-onnx or FFmpeg pins:
  rerun unit + contract tests and the full container gate with no skips (docs/container-gate.vi.md). After changing
  sherpa-onnx or the model pin, also rerun `scripts/probe_limits.py` (it exits 1 if the floor moved),
  `scripts/baseline_offline.py` and `scripts/smoke_server.py` (it checks transcripts against the baseline's hashes).
- `docs/baseline/*.json` reports hold metadata only. Commit `docs/baseline/server-container.json` only from a
  full container gate run with no skips; host emulations of the image are diagnostics, not gate evidence.
