# Roadmap

Current release: `v0.1.0-rc.1` (pre-release). Details of what ships: [implementation-plan.vi.md](implementation-plan.vi.md).

## Shipped

| Release | Scope |
|---|---|
| R1 | `/v1/models`, `POST /v1/audio/transcriptions` (`json`, `text`, SSE), auth, OpenAI errors, limits, Docker (CPU, sherpa-onnx) |
| R2a | `verbose_json` (no `segments`), `srt`, `vtt` with timestamps from token emission times |
| R3a | Realtime transcription over WebSocket `/v1/realtime`, explicit commit (`turn_detection: null`) |

## Not implemented yet

Ordered by priority. A feature is enabled only after its gate passes with recorded evidence.

### 1. Production capacity and limits (R4)

Current defaults (`GATEWAY_QUEUE_MAX_LANES=32`, `GATEWAY_MAX_SESSIONS=16`, …) are finite safety budgets, **not measured
capacity**. `GATEWAY_NUM_THREADS=4` and `GATEWAY_QUEUE_MAX_BUFFERED_SECONDS=600` come from the single-stream CPU
baseline (docs/baseline.vi.md §4: RTF 0.04 at fp32, beam 4, 4 threads), not from load. Tooling exists
(`scripts/benchmark.py`); the load measurement does not.

- [ ] Benchmark on an **exclusive** CPU host (record CPU model, core count, `GATEWAY_NUM_THREADS`, precision and
  decoding method): REST short/long, SSE, WebSocket commits and mixed, ramping clients; ≥ 3 runs per level, ≥ 100
  successes per group for p95. Fix workload, p95 SLO and error budget before measuring.
- [ ] Soak ≥ 1 h at the chosen level behind real TLS; RSS headroom, no occupancy growth.
- [ ] Failure drills on the real deployment: SIGTERM drain, process/container kill and restart, inference-failure
  recovery, hung-worker watchdog restart.
- [ ] Set limits from the measurements, record a decision table (hardware, image digest, config, max tested load,
  selected limit) and a rollback digest.

```bash
uv run scripts/benchmark.py --engine cpu --hardware '<cpu>-<cores>c-exclusive' \
  --base-url http://127.0.0.1:18001 --clients 1,2,4,8,16,32 --requests-per-client 100 \
  --workload mixed --phase warm --server-pid <host pid> --out docs/baseline/capacity-mixed-warm.json
```

A benchmark report (exit 0) is not a capacity claim; client counts are closed-loop, not microphone sessions.
No load measurement exists for the CPU image yet; the single-stream numbers (RTF ≈ 0.04 for a 20 s chunk at the
defaults) bound one worker at roughly 25 s of audio per second of wall time.

### 2. Word timestamps (R2b)

`timestamp_granularities[]=word` returns 400 until this gate passes (`WORD_TIMESTAMPS = False` in
`app/transcription.py`). Alignment code and the evaluator (`scripts/eval_alignment.py`) exist. The emission-time probe
(docs/baseline.vi.md §6) puts the first word after a silence a median 75 ms early with quartiles -255 / +120 ms, so
raw emission times are unlikely to pass this gate without a better word-start estimate.

- [ ] Hand-annotated word boundaries in `tests/fixtures/alignment/` (Praat/Audacity, not another aligner):
  ≥ 200 words, ≥ 10 utterances, `dev`/`test` splits.
- [ ] Gate on `test`: ≥ 90 % of matched words within 100 ms, median error ≤ 50 ms (thresholds fixed in advance).
- [ ] Flip the switch in the commit that carries the report; add `words` to `verbose_json`.

### 3. Server VAD and partial transcripts (R3b)

- [ ] Acoustic VAD (pin model, check license and CPU cost) with `threshold`, `prefix_padding_ms`,
  `silence_duration_ms`; emit `input_audio_buffer.speech_started/stopped` and auto-commit. Do not use the
  transducer's token emissions as VAD. `semantic_vad` stays rejected.
- [ ] Optional: partial text before commit via a stable-prefix decoder (append-only deltas, defined final
  correction policy), only with full integration tests.

### 4. Scale and performance

Each item below changes the "one recognizer, one worker thread, one chunk per call" rule, so each needs its own gate:
identical text vs `docs/baseline/offline.json`, per-request deadlines and cancel kept, and a capacity measurement (§1).

- [ ] Batching (R4): decode chunks from different lanes in one `decode_streams` call; measure throughput and
  latency against single-chunk calls at the same thread count before enabling.
- [ ] Multiple recognizer instances (R4): several recognizers on separate worker threads in one container, sharing
  the core budget with `GATEWAY_NUM_THREADS`; needs a RAM budget per instance (~0.8 GB fp32, ~0.46 GB int8).
- [ ] Horizontal scale: several containers behind a load balancer with sticky WebSocket routing (works today; needs
  the §1 measurement to size).
- [ ] Optional GPU runtime: sherpa-onnx with the onnxruntime CUDA provider, only if CPU measurements fall short of the
  required capacity. Needs its own image, pins and a full gate run; the CPU image stays the default
  ([ADR 0001](adr/0001-transducer-on-cpu-replaces-quan-ctc-on-gpu.md)).

### 5. Smaller items

- [ ] `conversation.item.created` in Realtime if a real client needs it (currently `conversation.item.added`).
- [ ] G.711 `audio/pcmu` / `audio/pcma` Realtime input formats.
- [ ] Browser access: ephemeral client secrets or a token service (today: server-to-server Bearer only).
- [ ] Optional punctuation / inverse text normalization as a separate, versioned post-processor.
- [ ] Python 3.11 support (only 3.12 tested).

## Out of scope

Not provided by this model; would need other engines behind the gateway:

- Languages other than Vietnamese, language auto-detection, `/v1/audio/translations`.
- `/v1/audio/speech` (TTS), chat/responses, Realtime conversation, WebRTC/SIP.
- Diarization (`diarized_json`, `known_speaker_*`), `prompt`/keywords biasing, `include[]=logprobs`.
- `verbose_json.segments`: the SDK schema requires Whisper-only fields (`avg_logprob`, `no_speech_prob`, `tokens`)
  that this model cannot fill with real data.
- TLS, CORS and per-client rate limiting inside the gateway: provided by the infrastructure in front of it.
