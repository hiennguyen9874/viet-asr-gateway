# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow the release tags (`vX.Y.Z-rc.N`).

## [Unreleased] - targets v0.2.0-rc.1

Spec #1: replace the served model with viet-asr while keeping the API. Baselines, probes and gate reports are
re-measured on viet-asr (#5); the container gate still has to be run on a Docker host.

### Changed (breaking)

- Neutral identifiers (#2):
  - Config prefix `GIPFORMER_*` → `GATEWAY_*`. Any leftover `GIPFORMER_*` or `QUAN_*` variable aborts startup and the
    error names the variable.
  - Served model id `gipformer1.5-68M-rnnt` → `viet-asr` (`/v1/models`, REST `model`, Realtime `?model=` and
    `session.update`). The old id gets `model_not_found`; the opt-in `whisper-1` alias is unchanged.
  - Metric prefix `gipformer_` → `gateway_`. Logger, worker thread and task names no longer mention Gipformer.
  - Package `viet-asr-gateway` `0.2.0rc1`, image and compose service `viet-asr-gateway`, Docker secret
    `gateway_api_keys` (was `gipformer_api_keys`), Dockerfile user/group `gateway` (uid/gid 10001 unchanged).
  - JS server-test variables `GIPFORMER_TEST_*` → `GATEWAY_TEST_*`.
  - Engine class `GipformerEngine` → `SherpaEngine`.
- Served model viet-asr (#3): pin `dangvansam/viet-asr@46adacf` (int8 only, Apache-2.0) replaces Gipformer in
  `assets/manifest.json`. Transcripts are written form, verbatim from the model (casing, punctuation, digits).
  `GATEWAY_PRECISION` is removed and now aborts startup as an unknown variable. Byte-fallback tokens (`<0xNN>`) are
  merged into UTF-8 characters so tokens always spell the text. WER scoring lower-cases and strips Unicode
  punctuation on both sides. Engine parity is checked against a minimal sherpa-onnx reference
  (`tests/model/reference.py`) instead of Gipformer's `infer_onnx.py`.
  - New fixture `tests/fixtures/vi_bytes.wav` (FLEURS dev) exercises byte-fallback tokens.
  - R2a gate: the shift test compares cue edges only over the common leading words (viet-asr may change a word
    when audio is shifted); `LONG_MAX_WER` 0.10 → 0.15 (viet-asr: 0.08–0.13 on `vi_long` across decodings, threads
    and chunk lengths, #5).
  - Smoke and the image gate refuse `docs/baseline/offline.json` measured on another model pin.
- Cues end at sentence ends (#4): a word ending in `.`, `?`, `!` or `…` closes its `srt`/`vtt` cue (`,` `;` `:` do
  not); the pause, 7 s and 80-character rules are unchanged.
- Defaults and constants re-measured on viet-asr (#5; docs/baseline.vi.md):
  - `GATEWAY_NUM_THREADS` 4 → 2, `GATEWAY_MAX_CHUNK_SECONDS` 20 → 15, `GATEWAY_QUEUE_MAX_BUFFERED_SECONDS` 600 → 450.
  - Decoding stays `modified_beam_search`, beam 4: greedy loses 0.97 WER point on the eval set (rule: switch at
    ≤ 0.5). Known limitation: beam 4 emits a word or two on silence and noise; greedy returns `""`.
  - Cue timing: word starts move 200 ms later (`EMISSION_SHIFT` 0 → 3200, shrinking near a chunk end), last word
    of a chunk 0.28 → 0.12 s, pause 0.52 → 0.64 s.
  - New local eval set (`scripts/fetch_fixtures.py --eval`: 200 FLEURS vi_vn test clips + 24 seeded non-speech
    clips, gitignored, hash manifest). `probe_limits.py` and `baseline_offline.py` derive and check the chunk length
    and the decoding method; the model-tier gates run the default engine settings.

- Removed the two Gipformer reference submodules (#6); ADR 0002 supersedes the model choice of ADR 0001.
  Project/agent docs describe viet-asr; generated listening outputs are no longer tracked and stay local.
  Model object metadata now uses `dangvansam` and the pinned revision's timestamp (`1793867382`).
  - Listening demo regenerated on the default server: 33 cases, no request errors; WER normalizes both sides
    without changing displayed text. Three regression tests cover casing, punctuation and the digit limitation.
  - Validation: 484 CPU-tier tests and 24 model-tier tests passed; pre-commit passed.
  - Legacy-prefix literals remain only in the startup rejection guard and its regression tests, as required
    by spec #1; this is the documented exception to the old-name grep criterion.
  - Container report and `v0.2.0-rc.1` tag remain an operator handoff, not agent-generated evidence.

Migration: rename every `GIPFORMER_*` variable to `GATEWAY_*`, move `secrets/gipformer_api_keys` to
`secrets/gateway_api_keys`, switch clients to `model=viet-asr` and dashboards to `gateway_*` metrics.

## [0.1.0-rc.1]

Pre-release: gipformer1.5-68M-rnnt on CPU (sherpa-onnx) behind a scoped OpenAI API: REST/SSE (R1),
verbose_json/srt/vtt (R2a), Realtime with explicit commit (R3a). Details: docs/implementation-plan.vi.md.
