# Deploy gipformer1.5-68M-rnnt

Docker-only gateway serving gipformer1.5-68M-rnnt (Vietnamese RNN-T ASR, sherpa-onnx on CPU) through a scoped OpenAI API: REST/SSE (R1),
verbose_json/srt/vtt (R2a), Realtime with explicit commit (R3a). Pre-release `v0.1.0-rc.1`. TLS, CORS and
per-client rate limits belong to the operator's infrastructure in front of it.

## Agent docs

- Code map, for finding where something lives or which test covers it: docs/agents/code-map.md
- Commands, for setup, tests by tier, a local server, smoke, image gate and lint: docs/agents/commands.md
- Change rules, before editing the inference path, audio, timestamps, realtime, API contract, config, or
  pins/image: docs/agents/change-rules.md (read the section for each area you touch)

## Project docs (Vietnamese; each is the source of truth for its topic)

- Deploy, usage, config table, operations: README.md
- Architecture: docs/deployment-design.vi.md
- API payloads, error codes, capability matrix: docs/api-contract.vi.md
- Pins, measurements, frontend limits: docs/baseline.vi.md
- Image gate status and procedure: docs/container-gate.vi.md
- Shipped milestones, evidence, deviations, definition of done: docs/implementation-plan.vi.md
- Unshipped work (R2b word timestamps, R3b VAD, R4 capacity, scale): docs/roadmap.md

## Always

- The runtime is CPU only (sherpa-onnx): plain `uv sync` / `uv run`, no `--extra`.
- The host is shared: run local servers on a free port (8000 is taken).
- One model instance, one thread: only `app/engine.py` calls the model, fed through `Scheduler` lanes. Endpoints reach
  it via `run_chunks` + `Joiner`.
- A feature ships only after its gate passes with recorded evidence; until then it stays switched off.
- Treat `references/` (upstream submodules) as read-only: read it for behaviour; app code and images stay
  independent of it.
- Keep `HF_TOKEN`, API keys, audio and transcripts out of commits, logs and reports.
- A change is done when code, tests and the docs describing it agree.

## References

- references/gipformer (inference scripts, sherpa-onnx usage; `infer_onnx.py` is the parity reference)
- references/gipformer1.5-68M-rnnt (config, tokens, model card; weights are Git LFS pointers, stage them instead)
- references/viet-asr (upstream VietASR: C++ core, bindings, server, model card and benchmark; source for the
  replacement model `dangvansam/viet-asr`, which the gateway runs with its own sherpa-onnx engine, not this library)

## Agent skills

### Issue tracker

Issues live in GitHub Issues, via the `gh` CLI (no git remote is configured yet; add one before using it). See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/`, created lazily. See `docs/agents/domain.md`.
