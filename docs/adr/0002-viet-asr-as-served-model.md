---
status: accepted
---

# viet-asr replaces Gipformer as the served model

The gateway now serves `viet-asr` (`dangvansam/viet-asr@46adacf`, an 86M-parameter Zipformer2 transducer, int8, Apache-2.0) instead of `gipformer1.5-68M-rnnt`. It keeps the runtime of [ADR 0001](0001-transducer-on-cpu-replaces-quan-ctc-on-gpu.md): one sherpa-onnx `OfflineRecognizer` on CPU, one worker thread, the same `Scheduler` lanes, `run_chunks` and transcript joiner. Only the model files change. Gipformer is not kept as a second option; it stays reachable only through git history.

## Context

- **Written-form text.** Gipformer writes all upper case, without punctuation, with numbers spelled out. Clients had to add casing, punctuation and inverse text normalization themselves, and `srt`/`vtt`/`verbose_json` were hard to read. viet-asr writes casing, punctuation and digits itself.
- **WER.** On the upstream benchmark (raw output against written-form references) viet-asr beats Gipformer on every public Vietnamese test set (VIVOS 20.9 vs 29.9, VLSP 2023 27.9 vs 34.2).
- **License.** Apache-2.0 for the model, compatible with the gateway's distribution.
- **int8 only.** The Hub repo publishes int8 weights only (upstream: int8 within 0.15 WER of fp32). The precision setting goes away; fp32 or a self-made export is out of scope.
- Checked on the host before the switch: `from_transducer` loads it (80 fbank, `context_size=2`, 8,192 tokens with punctuation, digits and byte-fallback `<0xNN>`), tokens still start words with `▁`, emission times still step 40 ms, the recognizer floor is still 1,360 samples.

## Considered options

- **The upstream native library (`references/viet-asr`).** Rejected: it cuts audio with its own VAD, runs batches one after another behind a lock and downloads the model itself, which conflicts with the gateway's chunking, scheduler and staged read-only pins.
- **Our own decoder on onnxruntime.** Rejected: re-implements beam search and feature extraction that sherpa-onnx already provides and pins, with a new parity burden.
- **Keep both models selectable by config.** Rejected for the same reasons as in ADR 0001: two pins, two sets of gate evidence, two text styles behind one API.

## Consequences

- **Neutral names**, so the next model swap does not rename again: `GATEWAY_*` settings (leftover old-prefix variables abort startup), model id `viet-asr`, metrics `gateway_*`, image/package `viet-asr-gateway`, secret `gateway_api_keys`. Breaking; released as `v0.2.0-rc.1` after the container gate.
- **Gates rerun on the new model** (docs/baseline.vi.md): engine parity against a minimal sherpa-onnx reference (`tests/model/reference.py`) instead of Gipformer's `infer_onnx.py`; limits probe (15 s chunks), offline baseline (beam 4 kept: greedy loses 0.97 WER point, above the 0.5-point rule; 2 threads), alignment probe (200 ms emission shift, last word 120 ms, pause 640 ms), R2a and R3a gates, server smoke. WER is scored lower-cased without punctuation on both sides.
- **Sentence-end cue rule**: a word ending in `.`, `?`, `!` or `…` closes its cue; `,` `;` `:` never do. Pause, 7 s and 80-character rules are unchanged.
- **Transcripts are verbatim**: the server normalizes nothing. The only token decoding is merging byte-fallback runs into UTF-8 characters so tokens always spell the text.
- **Known gaps**: punctuation and capitals at chunk cuts; beam 4 emits a word or two on silence and noise (greedy is almost always empty: 23/24 clips); digits vs spelled-out numbers inflate WER against FLEURS; transcripts can depend on the thread count.
- This ADR replaces the model choice of ADR 0001. Its runtime decision (sherpa-onnx on CPU, one recognizer, one worker thread, no torch/CUDA) still holds.
