---
status: partially superseded by ADR 0002
---

# A transducer on CPU through sherpa-onnx replaces QUAN CTC on GPU

> **Partially superseded** by [ADR 0002](0002-viet-asr-as-served-model.md): the model choice (Gipformer, fp32 by
> default) is replaced by viet-asr int8. The runtime decision (sherpa-onnx on CPU, one recognizer on one worker
> thread, no torch/CUDA, GPU deferred) still holds.

The gateway serves one model: Gipformer 1.5 (`gipformer1.5-68M-rnnt`, a 68M-parameter Zipformer transducer, MIT) through sherpa-onnx on CPU, replacing QUAN-Transcribe-1.0 (CTC, Gemma license) on GPU. Gipformer's published Vietnamese WER is better on most benchmarks, its ONNX export runs well on CPU, and the image drops torch, CUDA and the GPU requirement. QUAN is not kept as a second option; it stays reachable only through git history.

## Considered options

- **Keep both models selectable by config.** Rejected: two loaders, two sets of pins and two CUDA/CPU stacks, and every gate would need two sets of evidence.
- **sherpa-onnx with CUDA (onnxruntime-gpu).** Deferred, not rejected: kept as an opt-in if CPU measurements fall short.
- **PyTorch with icefall, k2 and kaldifeat on CUDA.** Rejected: heaviest stack, hardest to pin, and the reference itself recommends the ONNX path for deployment.

## Consequences

- A transducer has no frame-synchronous CTC path, so word and cue times are rebuilt from token emission times, and the timed-format gate (R2a) and the realtime gate (R3a) must be passed again on the new model.
- Decoding defaults to modified beam search (beam 4) in fp32, following the reference; greedy search and int8 are configuration. This costs CPU time that the baseline has to measure before capacity defaults are set.
- Every `QUAN_*` setting, the model id and the project names change (a breaking change accepted while the release is still `v0.1.0-rc.1`).
- The engine stays a single recognizer on a single worker thread; batching and several recognizer instances are deferred to the capacity work (R4).
