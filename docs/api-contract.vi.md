# API contract — Gipformer gateway (profile OpenAI đã pin)

Kiến trúc: [deployment-design.vi.md](deployment-design.vi.md). Cách chạy: [README](../README.md). Chưa hỗ trợ: [roadmap.md](roadmap.md).

Tài liệu này chốt payload đang phục vụ: R1 (REST/SSE), R2a (`verbose_json`/`srt`/`vtt`) và R3a (Realtime explicit
commit). Mọi ví dụ nằm trong
`tests/contract/examples/` và được **parse bằng SDK thật** (`tests/contract/test_python_sdk.py`,
`tests/contract/js/contract.test.ts`). Nếu tài liệu và ví dụ khác nhau thì ví dụ + test là chuẩn.

## 1. Profile đã pin

| | Pin | Kiểm chứng |
|---|---|---|
| OpenAI Python SDK | `openai==3.31.0` = commit `86cae50785fb7b7bf2a5d2383370b214b304acd2` (đúng commit thiết kế tham chiếu) | Client SDK thật qua mock transport: encoding request, parse JSON/text/SSE/error; Realtime event validate bằng pydantic union |
| OpenAI JS SDK | `openai@7.33.0` (gitHead `dd51b11156f920c57dc2e4ad4162cf28a42fe8a1`), TypeScript 5.9.3, Node ≥ 23.6 | `tsc --strict` với `satisfies` trên type SDK + client SDK thật với custom `fetch` |
| Realtime | GA schema (`session.type = "transcription"`), không dùng beta `transcription_session.update` | |

**Chưa kiểm chứng:** hành vi thật của server OpenAI (thứ tự event, mã lỗi chính xác). Contract dựa trên type SDK và
tài liệu OpenAI, không phải traffic quan sát được. Mã lỗi `code` bên dưới là của gateway này.

## 2. Capability matrix

| Tính năng | Trạng thái | Release |
|---|---|---|
| `GET /v1/models`, `GET /v1/models/{id}` | Hỗ trợ | R1 |
| `POST /v1/audio/transcriptions` `response_format=json\|text` | Hỗ trợ | R1 |
| `stream=true` (SSE delta/done) | Hỗ trợ | R1 |
| `response_format=verbose_json\|srt\|vtt` (không stream) | Hỗ trợ; cue gate qua trên model thật (`docs/baseline/r2a_gate.json`) | R2a |
| `timestamp_granularities[]=segment` | 400 `unsupported_parameter`: `TranscriptionSegment` cần field chỉ Whisper có; segment chỉ có qua `srt`/`vtt` | — |
| `timestamp_granularities[]=word` | 400 `unsupported_parameter` cho tới khi qua alignment gate ([roadmap](roadmap.md)) | R2b |
| WS `/v1/realtime` transcription, `turn_detection: null`, append/commit/clear | Hỗ trợ; realtime gate qua trên model thật (`docs/baseline/r3a_gate.json`) | R3a |
| `server_vad` | 400 `unsupported_turn_detection` | R3b |
| `semantic_vad`, `noise_reduction`, `audio/pcmu`, `audio/pcma` | Từ chối | — |
| `prompt` không rỗng, `keywords`, `include[]=logprobs`, `diarized_json`, `known_speaker_*`, `chunking_strategy` | Từ chối 400 | — |
| Language khác `vi`, auto-detect, translations, speech, chat/responses, WebRTC, SIP, ephemeral client secrets | Không hỗ trợ | — |

Không có endpoint nào trả `logprobs`, `avg_logprob`, `no_speech_prob`, confidence hay token id kiểu Whisper.

## 3. Models

- Id: `viet-asr`. Alias `whisper-1` chỉ khi bật cấu hình (mặc định tắt); khi bật, `whisper-1` được chấp
  nhận ở mọi field `model` và được liệt kê trong `/v1/models` như một model riêng (cùng backend Gipformer, output khác
  Whisper; ghi rõ trong README). Mọi id khác: 404 `model_not_found`.
- Object: `{"id", "object": "model", "created", "owned_by": "g-group-ai-lab"}`. `created` = commit time của revision
  weights đã pin (`dd9227d…` → `1789615207`). Ví dụ: `examples/models.json`, `examples/model.json`.

## 4. `POST /v1/audio/transcriptions`

Auth: `Authorization: Bearer <key>`. Body `multipart/form-data`.

| Field | Quy tắc |
|---|---|
| `file` | Bắt buộc, đúng một part. Tên file client bị bỏ qua (không dùng làm path, không log). Định dạng nhận diện theo nội dung, không theo đuôi: Container allowlist: WAV, FLAC, MP3/MPGA, MP4/M4A, OGG, WEBM/Matroska. Decoder allowlist: PCM (`s16le`, `s24le`, `s32le`, `u8`, `f32le`, `f64le`, A-law, µ-law), FLAC, ALAC, MP1/MP2/MP3, AAC, Vorbis, Opus. Chấp nhận mọi tổ hợp container/decoder trong hai allowlist mà FFmpeg đọc được (kể cả OGG/FLAC và Matroska/AAC); không giới hạn theo cặp. Container/codec ngoài allowlist (ADTS `.aac`, MPEG-PS, playlist, ADPCM, …) → 400 `invalid_audio`. Mọi sample rate/số kênh được đưa về mono 16 kHz (downmix của FFmpeg) |
| `model` | Bắt buộc; thiếu → 400 `missing_required_parameter`; không phục vụ → 404 `model_not_found` (`param: "model"`) |
| `language` | Không gửi hoặc `vi`; khác → 400 `unsupported_language`. Không gửi **không** có nghĩa là auto-detect |
| `temperature` | Không gửi hoặc số bằng 0 (`0`, `0.0`, `-0`; decode tất định theo `GATEWAY_DECODING_METHOD`). Không phải số hữu hạn (rỗng, `abc`, `nan`, `inf`) → 400 `invalid_value`; số khác 0 → 400 `unsupported_parameter` |
| `prompt` | Không gửi hoặc rỗng; khác → 400 `unsupported_parameter` |
| `response_format` | `json` (mặc định), `text`, `verbose_json`, `srt`, `vtt`; khác → 400 `unsupported_response_format` |
| `stream` | Đúng chuỗi `true` hoặc `false` (cả hai SDK gửi như vậy); giá trị khác (`True`, `1`, rỗng) → 400 `invalid_value`. `true` với `response_format` không gửi, `json` hoặc `text` → SSE (§5); với `verbose_json`/`srt`/`vtt` → 400 `unsupported_parameter` (`param: "stream"`) |
| `include[]` | Field lặp, tên **có** `[]` (đã xác minh cả hai SDK gửi như vậy). Bất kỳ giá trị nào → 400 `unsupported_parameter` |
| `timestamp_granularities[]` | Field lặp, tên có `[]`; giá trị lặp được gộp. Giá trị ngoài {`word`, `segment`} → 400 `invalid_value`; có granularity mà format ≠ `verbose_json` → 400 `unsupported_parameter`; `segment` → 400 `unsupported_parameter`; `word` → 400 `unsupported_parameter` cho tới R2b (`param` đều là `timestamp_granularities[]`) |
| Field khác (kể cả `chunking_strategy`, `known_speaker_names[]`, `include` không có `[]`) | 400 `unsupported_parameter` (`param` = tên field, cắt 64 ký tự). Không âm thầm bỏ qua |

Mọi field trừ `file`, `include[]`, `timestamp_granularities[]` chỉ được gửi một lần; lặp → 400 `invalid_value`
(`param` = tên). Thứ tự kiểm tra cố định, trả lỗi đầu tiên (`app/transcription.py:parse_params`, test
`tests/unit/test_transcription_params.py`): (1) field lạ theo thứ tự gửi, (2) field lặp, (3) thiếu/rỗng `model`,
(4) `include[]`, (5) giá trị `timestamp_granularities[]`, (6) `language`, (7) `temperature`, (8) `prompt`,
(9) `response_format`, (10) `stream` (giá trị, rồi format stream được), (11) granularity chỉ với `verbose_json`,
(12) `segment`, `word` chưa bật, (13) model không phục vụ (404). Multipart/`file` được kiểm tra trước tất cả
(§4 giới hạn), decode audio sau tất cả.

Kết quả:

- `json` → `200 application/json`: `{"text": "...", "usage": {"type": "duration", "seconds": <float>}}`.
  `usage.seconds` = số mẫu audio sau decode / sample rate gốc (tính trước chunking), là đại lượng đo được.
  Không có `logprobs`/`languages`. Ví dụ `examples/transcription.json`.
- `text` → `200 text/plain; charset=utf-8`, body đúng bằng transcript, không thêm newline. SDK trả nguyên body.
- `verbose_json` → `200 application/json`: `{"text", "language": "vietnamese", "duration", "usage": {"type":
  "duration", "seconds"}}`, `duration == usage.seconds`. **Không có `segments`** (schema SDK bắt buộc
  `avg_logprob`, `no_speech_prob`, `compression_ratio`, `tokens` kiểu Whisper — Gipformer không có dữ liệu thật) và
  không có `words` cho tới R2b. Ví dụ `examples/transcription_verbose.json`; dạng R2b (có `words`, chưa phục vụ):
  `examples/transcription_verbose_words.json`.
- `srt` → `200 text/plain; charset=utf-8`: cue `n\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext\n\n`, đánh số từ 1; không có
  cue → body rỗng. `vtt` → cùng content type, `WEBVTT\n\n` rồi `HH:MM:SS.mmm --> HH:MM:SS.mmm\ntext\n\n`; không có
  cue → chỉ header. Cả hai SDK trả nguyên body (string). Ví dụ `examples/transcription.srt`, `examples/transcription.vtt`.
- Timestamps (`srt`/`vtt`, sau này `words`): lấy từ thời điểm phát token (token emission time) của chính lần inference sinh ra text, cộng offset
  tuyệt đối của chunk; không chạy thêm inference. **Từ** = dãy ký tự không phải khoảng trắng của transcript (âm tiết tiếng
  Việt, không phải từ ghép); start của từ = thời điểm phát token đầu tiên của nó; end = start của từ kế tiếp trong chunk,
  hoặc với từ cuối chunk là độ dài giới hạn `LAST_WORD_SAMPLES` (0.28 s) và không quá cuối chunk. Không bù độ trễ
  (`EMISSION_SHIFT = 0`). Thời điểm phát token theo bước 40 ms.
  Cue mới khi khoảng cách giữa end của từ trước và start của từ sau ≥ 0.52 s, hoặc khi thêm từ sẽ làm cue dài > 7 s
  hay > 80 ký tự. Các hằng số này lấy từ probe `scripts/probe_alignment.py` (baseline §6). Khoảng cách này chỉ có thể xuất hiện giữa các chunk; trong cùng chunk, end của từ trước bằng start
  của từ sau. Khoảng cách lớn giữa các lần phát token không chứng minh có khoảng lặng, nên không tự tách cue.
  Độ chính xác (đo ở baseline §6): từ đầu tiên sau một khoảng lặng thường được đóng dấu sớm, có khi tới ~1 s (nằm
  trong khoảng lặng); token đầu tiên của chunk thường được đóng dấu ngay đầu chunk, kể cả khi chunk bắt đầu bằng im lặng. Vì vậy cue
  có thể bắt đầu trước lời nói và một cue có thể phủ qua khoảng lặng nằm trong cùng chunk. Cue không bao giờ phủ qua
  điểm cắt chunk.
  Ngoại lệ: một từ tự nó vượt giới hạn được giữ nguyên trong cue riêng, không cắt từ. Giờ SRT/VTT là
  ms nguyên làm tròn; cue có thứ tự, không chồng nhau, nằm trong `[0, duration]`. Một từ bị điểm cắt chunk chia đôi
  (hiếm, chỉ khi 5 s cuối trước giới hạn không có điểm lặng) thành hai từ, nhất quán với text.
- Mọi format cho cùng transcript: `json.text == text == verbose_json.text == SSE done.text ==` các text cue nối bằng
  một dấu cách. Token chỉ được chứa `▁` ở đầu và không chứa ký tự khoảng trắng; `▁` đứng riêng vẫn đặt start của từ.
  Nếu token sai cấu trúc hoặc không khớp transcript, `srt`/`vtt` trả 500 `alignment_failed` (không bao giờ chia đều
  thời gian); `json`/`text`/`verbose_json`/SSE không chạy alignment nên không bị ảnh hưởng.
- Text: dạng viết tiếng Việt (chữ hoa/thường, dấu câu, số bằng chữ số), nguyên văn như model sinh ra, không hậu xử
  lý. Token byte fallback (`<0xNN>`) được engine ghép thành ký tự UTF-8 để token luôn ghép ra đúng text. Nhiều chunk nối bằng
  một dấu cách; chunk rỗng không sinh dấu cách.
- Silence/clip ngắn (`app/chunking.py`): audio decode hợp lệ nhưng ≤ 1,600 mẫu ở 16 kHz (≤ 100 ms), kể
  cả 0 mẫu → `text: ""` (`verbose_json` với `duration` thật, `srt` rỗng, `vtt` chỉ header), không gọi model (recognizer lỗi dưới 1,360 mẫu; xem baseline §5). Silence dài vẫn qua model
  và cho `""` (đã đo), không dùng ngưỡng biên độ.
- Chunking: audio dài hơn `GATEWAY_MAX_CHUNK_SECONDS` (mặc định 20 s, đo ở baseline §5) được cắt tại điểm yên lặng nhất trong 5 s cuối
  trước giới hạn; các chunk phủ liền toàn bộ audio (không bỏ silence, không bỏ đuôi), mỗi chunk > 100 ms. Text có
  thể khác một lời gọi `transcribe()` của reference trên cùng file vì điểm cắt khác.
- Giới hạn (cấu hình, xem `app/config.py`): toàn bộ body ≤ `GATEWAY_MAX_UPLOAD_BYTES` (25 MiB) → 413, kể cả upload
  chunked; ≤ 32 part, mỗi field không phải file ≤ 8 KiB (→ 413), field phải là UTF-8; audio sau decode ≤
  `GATEWAY_MAX_AUDIO_SECONDS` (1800 s) → 400 `audio_too_long`; decode quá `GATEWAY_DECODE_TIMEOUT_SECONDS` → 400
  `invalid_audio`; tổng upload/decode/queue/infer/gửi response quá `GATEWAY_REQUEST_TIMEOUT_SECONDS` → HTTP 503
  hoặc SSE error `request_timeout` (nếu headers đã gửi); hết
  slot decode, spool disk, lane (`GATEWAY_QUEUE_MAX_LANES`, kiểm tra trước khi decode) hoặc budget audio đã decode giữ
  trong RAM (`GATEWAY_MAX_DECODED_SECONDS`, mặc định 3600 s cho mọi request đang chạy) → 429 `server_busy`. Stream bị
  cụt được decode phần đọc được nếu FFmpeg đọc được. Khi client ngắt hoặc request hết hạn, reservation của
  toàn bộ waveform được giữ tới khi mọi job của lane hoàn tất (slice đang infer vẫn giữ backing array).

## 5. SSE (`stream=true`)

- Validate request **và decode audio xong trước khi gửi headers**; lỗi giai đoạn này trả HTTP lỗi như §7.
- Headers: `200`, `Content-Type: text/event-stream; charset=utf-8`, `Cache-Control: no-cache`,
  `X-Accel-Buffering: no`, không có `Content-Length` (chunked). JSON trong event là compact UTF-8 (không escape
  `\uXXXX`). Không có heartbeat: reverse proxy phải tắt buffering và có read timeout ≥
  `GATEWAY_REQUEST_TIMEOUT_SECONDS` (README §2).
- Mỗi event là một dòng `data: <json>` + dòng trống; **không** có trường `event:` và **không** gửi `[DONE]`
  (cả hai SDK coi `[DONE]` là tùy chọn; stream kết thúc bằng đóng kết nối sau `done`).
- `{"type": "transcript.text.delta", "delta": "..."}`: một event cho mỗi chunk đã decode xong với text khác rỗng,
  theo đúng thứ tự audio. Delta thứ hai trở đi bắt đầu bằng dấu cách. Không gửi `logprobs`, `segment_id`.
- `{"type": "transcript.text.done", "text": "..."}` đúng một lần ở cuối; `"".join(deltas) == done.text`.
  Không gửi `usage` (type SDK của `done.usage` chỉ có biến thể `tokens`; gateway không có token count).
- Audio rỗng/ngắn: không có delta, chỉ một `done` với `text: ""`.
- Lỗi sau headers: một event `data: {"error": {"message", "type", "param", "code"}}` (cùng envelope với HTTP) rồi đóng
  stream, **không** có `done`. Cả hai SDK ném `APIError` với `code` tương ứng (đã test). Ví dụ
  `examples/transcription_stream.sse`, `examples/transcription_stream_error.sse`.
- Client ngắt kết nối: dừng enqueue chunk tiếp, hủy chunk đang chờ, bỏ kết quả chunk đang chạy; không phát gì
  thêm. Cũng áp dụng cho JSON/text và khi đang decode (FFmpeg bị kill). Client không đọc: một lần gửi event quá
  `GATEWAY_SEND_TIMEOUT_SECONDS` (10 s) được coi như ngắt kết nối (đóng kết nối, không gửi `error`).
- Tối đa `GATEWAY_QUEUE_MAX_JOBS_PER_LANE` (2) chunk của một request nằm trong queue/đang chạy cùng lúc: model chạy
  trước client tối đa một cửa sổ đó, server không buffer thêm event. Admission chunk tiếp theo bị block không
  chặn việc gửi delta của chunk đã hoàn tất; deadline cũng áp dụng cho gửi delta/done.

## 6. Realtime WebSocket `/v1/realtime` (R3a)

Kết nối: `GET /v1/realtime` (WebSocket), `Authorization: Bearer <key>` (server-to-server). Query tùy chọn, mỗi
tham số tối đa một lần: `model` (Python SDK `client.realtime.connect(model=…)`) phải là model phục vụ, là model ban
đầu của session; `intent=transcription` (JS SDK `OpenAIRealtimeWS({intent: "transcription"})`). Mọi lỗi handshake là
HTTP response **trước accept** (có `x-request-id`, envelope §7), theo thứ tự: auth sai → 401; tham số query khác (kể
cả `call_id`) → 400 `unsupported_parameter`; tham số lặp → 400 `invalid_value`; `intent` khác → 400
`invalid_value`; model không phục vụ → 404 `model_not_found`; đang drain → 503 `shutting_down`; chưa ready → 503
`model_not_ready`; đủ `GATEWAY_MAX_SESSIONS` session hoặc hết lane → 429 `server_busy` (`Retry-After: 1`). Cả hai SDK
báo lỗi handshake (Python: `websockets.exceptions.InvalidStatus` với `response.status_code`; JS: event `error`
"Unexpected server response: <status>"). Không hỗ trợ subprotocol `openai-insecure-api-key.*`.

JS SDK luôn kết nối bằng `wss:` (kể cả `buildRealtimeURL` tùy chỉnh): client JS cần TLS do hạ tầng phía trước gateway
cung cấp. Một message client tối đa 2 MiB (Uvicorn `--ws-max-size 2097152`, ≈ 32 s audio mỗi append); lớn hơn →
server đóng socket với code 1009. Message dạng binary → `error` `invalid_event`.

Session sau khi accept (server gửi `session.created` ngay):

```json
{"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
  "transcription": {"model": "viet-asr", "language": "vi"}, "turn_detection": null}}}
```

Client events:

| Event | Quy tắc |
|---|---|
| `session.update` | Field vắng mặt hoặc `null` = giữ nguyên. Chấp nhận: `session.type` = `transcription` (bắt buộc); `format` = `{"type": "audio/pcm", "rate": 24000}` (cả hai field tùy chọn); `transcription.model` = model phục vụ; `transcription.language` = `vi`; `transcription.prompt` rỗng; `turn_detection`/`noise_reduction` = `null`. Từ chối (event `error`, session **không đổi gì**, kể cả khi phần khác hợp lệ), `param` = đường dẫn chấm (vd. `session.audio.input.turn_detection`): key lạ → `unknown_parameter`; sai kiểu JSON, `session.type` khác, format type/rate khác → `invalid_value`; `audio/pcmu`/`audio/pcma`, `noise_reduction`, `prompt` không rỗng, `keywords`, `languages`, `delay`, `include` → `unsupported_parameter`; `turn_detection` khác null → `unsupported_turn_detection`; language khác → `unsupported_language`; model không phục vụ → `model_not_found`. Chỉ có một format nên không có trường hợp "đổi format khi buffer còn audio". Thành công → `session.updated` với session đầy đủ |
| `input_audio_buffer.append` | `audio` (bắt buộc, `missing_required_parameter`; không phải chuỗi → `invalid_value`) = base64 PCM16 little-endian mono 24 kHz. Base64 sai (ký tự lạ, padding sai) → `error` `invalid_audio` (`param: "audio"`). Byte lẻ giữ lại cho packet sau. Buffer vượt `GATEWAY_MAX_TURN_SECONDS` (120 s) → `error` `input_audio_buffer_full`; hết budget RAM audio toàn server (`GATEWAY_MAX_DECODED_SECONDS`) → `error` `server_busy` (`type: "rate_limit_error"`). Mọi lỗi: **toàn bộ** append bị bỏ, buffer không đổi. Thành công: không có event (như OpenAI) |
| `input_audio_buffer.commit` | Buffer rỗng = 0 mẫu nguyên (một byte lẻ vẫn là rỗng) → `error` `input_audio_buffer_commit_empty`, không tạo item, buffer giữ nguyên. Đã có `GATEWAY_MAX_PENDING_ITEMS` (4) item chưa có `completed`/`failed` → `error` `too_many_pending_items`, buffer **giữ nguyên** (commit lại sau). Ngược lại: byte lẻ cuối bị bỏ, tạo item mới, buffer mới rỗng ngay (turn kế tiếp nhận được trong lúc item trước đang infer). Turn ≤ 100 ms (≤ 2,400 mẫu 24 kHz) → item bình thường, `completed` với `transcript: ""` và usage thật, không gọi model (như REST §4) |
| `input_audio_buffer.clear` | Xóa pending audio (kể cả byte lẻ), không ảnh hưởng item đã commit → `input_audio_buffer.cleared` |
| Type khác trong union client GA (`response.create`, `conversation.item.create`, `output_audio_buffer.clear`, …) | `error` `unsupported_event` (`param: "type"`). JSON sai / không phải object → `invalid_event`; thiếu `type` hoặc type ngoài union GA (kể cả beta `transcription_session.update`) → `invalid_event` (`param: "type"`); field lạ trong event → `unknown_parameter`; `event_id` không phải chuỗi → `invalid_value`. Lỗi event không đóng socket |

Audio: PCM16 / 32768 → float32, resample 24 → 16 kHz bằng soxr HQ dạng stream khi append; waveform commit **bằng
tuyệt đối** soxr one-shot trên cả turn, độc lập cách chia packet. Text có thể
khác REST trên cùng file (FFmpeg resampler, quantize PCM16, ranh giới turn do client chọn). Turn dài hơn
`GATEWAY_MAX_CHUNK_SECONDS` được chia chunk như REST (§4): nhiều delta, cùng một item.

Server events cho mỗi commit, theo thứ tự trong cùng item:

1. `input_audio_buffer.committed` `{event_id, item_id, previous_item_id}`
2. `conversation.item.added` `{event_id, previous_item_id, item}` với
   `item = {"id", "object": "realtime.item", "type": "message", "role": "user", "status": "completed", "content": [{"type": "input_audio"}]}`
3. 0..n `conversation.item.input_audio_transcription.delta` `{event_id, item_id, content_index: 0, delta}`: chỉ phát
   **sau commit**, một delta mỗi chunk nội bộ đã final; nối lại = transcript
4. Đúng một trong hai:
   `conversation.item.input_audio_transcription.completed` `{event_id, item_id, content_index: 0, transcript, usage: {"type": "duration", "seconds"}}`
   hoặc `conversation.item.input_audio_transcription.failed` `{event_id, item_id, content_index: 0, error: {type, code, message}}`

Quy tắc chung:

- `event_id` duy nhất trong session (`event_…`), `item_id` dạng `item_…`. `previous_item_id` = item commit trước
  đó trong cùng session hoặc `null`.
- `usage.seconds` = thời lượng audio đã commit (byte / 2 / 24000). Trường `usage` là **bắt buộc** trong type SDK
  của `completed`; không bao giờ phát `usage` dạng `tokens`.
- Server này xử lý item theo thứ tự commit (FIFO, item sau bắt đầu khi item trước terminal), nên terminal event đi
  theo thứ tự commit; client vẫn nên ghép bằng `item_id` (OpenAI không đảm bảo thứ tự). Delta/terminal của item trước
  có thể xen giữa `committed`/`added` của item sau. `completed.transcript` là kết quả authoritative.
- Deadline mỗi item = lúc commit + `GATEWAY_REQUEST_TIMEOUT_SECONDS`. `failed.error.code`: `inference_failed`,
  `request_timeout`, `server_busy` (queue đầy), `model_not_ready`, `shutting_down`, `internal_error`. Delta đã gửi
  không bị thu hồi; session tiếp tục sau `failed`.
- Đóng session: không có event client và không có item đang chờ trong `GATEWAY_SESSION_IDLE_TIMEOUT_SECONDS` →
  `error` `idle_timeout` rồi close 1000; quá `GATEWAY_MAX_SESSION_SECONDS` → `error` `session_expired` rồi close 1000
  (item chưa xong và pending audio bị bỏ). Client không đọc (một lần gửi quá `GATEWAY_SEND_TIMEOUT_SECONDS`) hoặc ngắt:
  session kết thúc không gửi gì thêm, chunk đang chờ bị hủy, kết quả chunk đang chạy bị bỏ. Server restart/shutdown:
  Uvicorn đóng mọi socket với 1012 ngay, item chưa `completed` bị mất — client phải kết nối lại và gửi lại audio.
- Không phát `conversation.item.done`, `input_audio_buffer.speech_started/stopped` (R3b), `response.*`,
  `rate_limits.updated`. Không có partial trước commit.
- `error` event: `{"type": "error", "event_id", "error": {"type", "code", "message", "param", "event_id"}}`; `param`
  có thể `null`; `error.event_id` = `event_id` của event client gây lỗi, chỉ khi đó là chuỗi printable ≤ 64 ký tự
  (ngược lại bỏ hẳn field). Mã `error` của gateway: `invalid_event`, `unsupported_event`, `unknown_parameter`,
  `missing_required_parameter`, `invalid_value`, `unsupported_parameter`, `unsupported_turn_detection`,
  `unsupported_language`, `model_not_found`, `invalid_audio`, `input_audio_buffer_full`,
  `input_audio_buffer_commit_empty`, `too_many_pending_items`, `server_busy`, `idle_timeout`, `session_expired`.
- **Không gửi `null`** cho field optional mà type JS không cho null: bỏ hẳn `noise_reduction` và
  `content[].transcript` (JS type check bắt được hai lỗi này; Python SDK thì chấp nhận null).

Lý do chọn `conversation.item.added` thay vì `conversation.item.created` (thiết kế ghi `created`): cả hai đều có
trong union GA của SDK; docstring GA của `conversation.item.added` nêu rõ trường hợp "input audio buffer is committed",
còn `created` là event thời beta. Client chỉ cần transcript nên dựa vào `committed` + `completed`; nếu client thật
cần `created`, thêm sau bằng integration test.

Ví dụ: `examples/realtime_client.jsonl`, `examples/realtime_server.jsonl`.

## 7. Lỗi HTTP

Envelope (ví dụ `examples/error.json`):

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "language", "code": "unsupported_language"}}
```

| HTTP | `type` | `code` |
|---|---|---|
| 400 | `invalid_request_error` | `missing_required_parameter`, `unsupported_parameter`, `invalid_value` (kể cả multipart sai/quá nhiều field), `unsupported_language`, `unsupported_response_format`, `invalid_audio` (`param: "file"`: không decode được, định dạng không cho phép, decode quá thời gian, sample không hữu hạn), `audio_too_long` (`param: "file"`) |
| 401 | `authentication_error` | `invalid_api_key` (thiếu hoặc sai) |
| 404 | `invalid_request_error` | `model_not_found`, `not_found` (route không tồn tại; chỉ sau khi auth hợp lệ, nếu không là 401) |
| 405 | `invalid_request_error` | `method_not_allowed` |
| 413 | `invalid_request_error` | `request_too_large` (body vượt giới hạn, kiểm tra trước multipart parse; hoặc một form field vượt 8 KiB) |
| 429 | `rate_limit_error` | `server_busy` (queue/concurrency/decode slot/spool disk/budget audio đã decode đầy); kèm `Retry-After` khi có ước lượng |
| 500 | `server_error` | `inference_failed`, `alignment_failed` (`srt`/`vtt`: token không khớp transcript), `internal_error` (lỗi không lường trước, không lộ chi tiết) |
| 503 | `server_error` | `model_not_ready` (đang load/recover, `Retry-After: 5`), `shutting_down` (đang drain), `request_timeout` (hết budget tổng của request ở upload, decode, chờ queue hoặc đang chạy; call đang chạy vẫn giữ worker/budget tới khi kết thúc) |

Mọi response (kể cả lỗi) có header `x-request-id: req_…`; Python SDK đưa giá trị này vào `request_id` của exception; JS SDK dùng `requestID`.

Endpoint vận hành (không thuộc profile OpenAI): `GET /health` (liveness, không cần auth; 503 `unhealthy` khi
cần restart process), `GET /ready` (readiness, không cần auth; 200 `{"status": "ready"}` hoặc 503 như trên),
`GET /metrics` (Prometheus text, cần API key).

Không bao giờ trả FastAPI 422 `{"detail": …}`. Message không chứa transcript, audio, token hay đường dẫn nội bộ.
SDK ánh xạ status → class (`BadRequestError`, `AuthenticationError`, `NotFoundError`, …) và đọc `code`/`param`/`type`
từ envelope. `examples/service_errors.json` được kiểm thử bằng cả Python và JS SDK cho các status/code dịch vụ,
`Retry-After` và request ID (gồm `invalid_audio`, `audio_too_long`, 413 cho form field). API key phải có ít nhất 16 ký tự ASCII printable, không có khoảng trắng.

## 8. Kiểm tra contract

```bash
uv run pytest tests/contract tests/unit
(cd tests/contract/js && npm ci && npm test)
```

`tests/contract/test_server_sdk.py` chạy gateway thật (Uvicorn + FFmpeg, fake engine): payload có cùng key/type với
`examples/`, Python SDK parse được (Realtime: `client.realtime.connect`, mọi server event validate bằng union
pydantic), và chạy `js/server.test.ts` (JS SDK) với `GATEWAY_TEST_BASE_URL`, `GATEWAY_TEST_API_KEY`, `GATEWAY_TEST_AUDIO_DIR`,
cùng `GATEWAY_TEST_WSS_BASE_URL` + `GATEWAY_TEST_CA` (gateway TLS cert tự ký cho `OpenAIRealtimeWS`); `npm test` đơn lẻ bỏ
qua các test đó.

Khi đổi version SDK: cập nhật pin trong `pyproject.toml`/`tests/contract/js/package.json`, chạy lại cả hai, ghi lại
thay đổi schema ở tài liệu này. Mọi thay đổi contract: cập nhật `tests/contract/examples/`, cả hai SDK test, rồi tài
liệu này. Bỏ field optional thay vì gửi `null`.
