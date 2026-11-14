# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow the release tags (`vX.Y.Z-rc.N`).

## [Unreleased] - targets v0.2.0-rc.1

Spec #1: replace the served model with viet-asr while keeping the API. The gateway still serves Gipformer until the
model swap lands.

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

Migration: rename every `GIPFORMER_*` variable to `GATEWAY_*`, move `secrets/gipformer_api_keys` to
`secrets/gateway_api_keys`, switch clients to `model=viet-asr` and dashboards to `gateway_*` metrics.

## [0.1.0-rc.1]

Pre-release: gipformer1.5-68M-rnnt on CPU (sherpa-onnx) behind a scoped OpenAI API: REST/SSE (R1),
verbose_json/srt/vtt (R2a), Realtime with explicit commit (R3a). Details: docs/implementation-plan.vi.md.
