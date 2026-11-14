# Viet ASR Gateway

A gateway that serves one Vietnamese speech-recognition model through a scoped OpenAI audio API (REST/SSE, timed formats, Realtime with explicit commit). This glossary fixes the words used across code, tests, docs and tickets.

## Language

### Model

**Served model**:
The single speech-recognition model the gateway exposes, `viet-asr`. There is exactly one at a time.
_Avoid_: backend, checkpoint

**Transducer**:
A model that emits tokens frame by frame and may emit several tokens, or none, on one frame. It has no frame-synchronous best path to read timing from.
_Avoid_: RNN-T model (when the family matters, say "transducer"), CTC path

**Token**:
One unit of the served model's subword vocabulary, which also holds punctuation and digits; a token that starts a word carries the word-start marker `▁`. The blank symbol is not a token.
_Avoid_: piece, unit, subword

**Engine**:
The only component that runs the served model, one chunk at a time, on one thread. Nothing else touches the model.
_Avoid_: model wrapper, recognizer (that is the library's object inside the engine)

**Decoding method**:
How the engine turns the model's output into tokens: greedy search or modified beam search.
_Avoid_: decoder (that is a network inside the transducer)

**Pin**:
The exact model revision plus per-file size and SHA256 that the gateway is allowed to run.
_Avoid_: version, lock

**Staging**:
Downloading the pinned files, verified, into a local directory that the gateway later mounts read-only. The gateway never fetches a model itself.
_Avoid_: provisioning, caching

### Audio and transcripts

**Transcript**:
The final text of one request or one realtime item in written form (casing, punctuation, numbers as digits), exactly as the served model writes it.
_Avoid_: result, output text

**Chunk**:
A contiguous span of audio, no longer than the maximum chunk length, given to the engine in one call. Chunks of one input cover it exactly, in order.
_Avoid_: segment, piece, slice

**Short audio**:
Audio at or below the short-audio floor. It yields an empty transcript and never reaches the engine.
_Avoid_: silence, empty audio

**Transcript joiner**:
The single place where chunk texts become one transcript: one space between non-empty chunk texts. Not the transducer's joiner network.
_Avoid_: merger, concatenator

**Word**:
A maximal run of non-whitespace characters of the transcript, including any punctuation attached to it. In Vietnamese this is a syllable, not a compound.
_Avoid_: syllable, term

### Timing

**Token emission time**:
The audio time at which the engine emitted a token. It is a start time only; a token has no end.
_Avoid_: token timestamp, frame index

**Alignment**:
Deriving word times and cue times from token emission times of the same inference that produced the transcript. It never guesses; a mismatch is an alignment error.
_Avoid_: forced alignment, synchronization

**Cue**:
A timed line of subtitles: an `srt` or `vtt` entry. `verbose_json` does not expose cues as `segments`. A cue ends at a pause, at a word ending a sentence (`.`, `?`, `!`, `…`), or at a length limit.
_Avoid_: caption, subtitle line, segment

**Word timestamps**:
Per-word times requested through the API. Switched off until their gate passes.
_Avoid_: word-level alignment

### Scheduling

**Lane**:
The ordered stream of jobs of one REST request or one realtime session. A **live lane** belongs to a realtime session and gets bounded priority over a **file lane**.
_Avoid_: queue, channel, stream

**Job**:
One chunk waiting for or running in the engine, belonging to a lane.
_Avoid_: task, work item

### Realtime

**Session**:
One realtime WebSocket connection and its live lane.
_Avoid_: connection, call

**Turn**:
The audio a session has received since its last commit and not yet committed.
_Avoid_: utterance, buffer contents

**Commit**:
The explicit client event that closes the current turn into an item. There is no automatic turn detection.
_Avoid_: end of speech, flush

**Item**:
An immutable committed turn. Items of a session are transcribed first in, first out.
_Avoid_: message, utterance

### Delivery

**Gate**:
A check with recorded evidence that a feature must pass before it ships; until then the feature stays switched off.
_Avoid_: acceptance test, milestone check

**Probe**:
A measurement on the real served model whose result sets a constant (a floor, a limit, a shift). Its report is evidence, not a test.
_Avoid_: benchmark (that measures capacity, which is not claimed)

**Report**:
Saved metadata-only evidence from a gate, probe or baseline: no audio, transcripts or secrets.
_Avoid_: log, results dump

**Reference**:
Read-only upstream source kept for studying behaviour. App code and images never depend on it.
_Avoid_: vendor, dependency
