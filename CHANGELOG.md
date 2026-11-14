# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow the release tags (`vX.Y.Z-rc.N`).

## [Unreleased] - targets v0.2.0-rc.1

Spec #1: replace the served model with viet-asr while keeping the API. Baselines, probes and gate reports are
re-measured on viet-asr in a later ticket; until then smoke and the image gate refuse the old baseline.

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
    when audio is shifted); `LONG_MAX_WER` 0.10 → 0.15 (viet-asr measures 0.114 on `vi_long`). Both are re-measured
    with the new baseline (#5).
  - Smoke and the image gate refuse `docs/baseline/offline.json` measured on another model pin.

Migration: rename every `GIPFORMER_*` variable to `GATEWAY_*`, move `secrets/gipformer_api_keys` to
`secrets/gateway_api_keys`, switch clients to `model=viet-asr` and dashboards to `gateway_*` metrics.

## [0.1.0-rc.1]

Pre-release: gipformer1.5-68M-rnnt on CPU (sherpa-onnx) behind a scoped OpenAI API: REST/SSE (R1),
verbose_json/srt/vtt (R2a), Realtime with explicit commit (R3a). Details: docs/implementation-plan.vi.md.
