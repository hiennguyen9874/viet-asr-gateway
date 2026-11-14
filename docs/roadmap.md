# Roadmap

Mục tiêu pre-release: `v0.2.0-rc.1` (viet-asr int8); **chưa gắn tag**, chờ [container gate](container-gate.vi.md).
Phần đã đạt trên model mới: [implementation-plan.vi.md](implementation-plan.vi.md).

## Phạm vi đã có

| Release | Phạm vi |
|---|---|
| R1 | `/v1/models`, `POST /v1/audio/transcriptions` (json/text/SSE), auth, lỗi OpenAI, giới hạn, Docker CPU |
| R2a | `verbose_json` (không segments), `srt`/`vtt` từ thời điểm phát token, cue cuối câu/pause/giới hạn |
| R3a | Realtime transcription qua WS `/v1/realtime`, commit tường minh (`turn_detection: null`) |

## Chưa triển khai

Theo thứ tự ưu tiên. Chỉ bật tính năng sau khi gate đạt và có bằng chứng.

### 1. Capacity và giới hạn production (R4)

Mặc định `GATEWAY_QUEUE_MAX_LANES=32`, `GATEWAY_MAX_SESSIONS=16`, … là ngân sách an toàn hữu hạn, **không phải
capacity đã đo**. `GATEWAY_NUM_THREADS=2`, `GATEWAY_QUEUE_MAX_BUFFERED_SECONDS=450` đến từ baseline một stream
(viet-asr int8, beam 4, RTF eval 0.057; baseline §4). Tooling `scripts/benchmark.py` có, số đo tải chưa có.

- [ ] Benchmark trên CPU **riêng** (ghi CPU, core, `GATEWAY_NUM_THREADS`, decoding, model int8): REST ngắn/dài,
  SSE, WS commit, tải hỗn hợp; tăng client; ≥3 lượt mỗi mức, ≥100 thành công mỗi nhóm cho p95. Chốt workload,
  SLO p95 và ngân sách lỗi trước khi đo.
- [ ] Soak ≥1 giờ ở mức chọn, sau TLS thật; có RAM dự phòng, occupancy không tăng dần.
- [ ] Diễn tập trên deployment thật: SIGTERM drain, kill/restart container, recovery sau inference lỗi,
  watchdog restart khi worker treo.
- [ ] Chọn giới hạn theo số đo, ghi bảng quyết định (hardware, image digest, config, tải thử tối đa, giới hạn
  chọn) và digest rollback.

```bash
uv run python scripts/benchmark.py --engine cpu --hardware '<cpu>-<cores>c-exclusive' \
  --base-url http://127.0.0.1:18001 --clients 1,2,4,8,16,32 --requests-per-client 100 \
  --workload mixed --phase warm --server-pid <host pid> --out docs/baseline/capacity-mixed-warm.json
```

Report exit 0 không tự là tuyên bố capacity: số client là closed-loop, không phải số microphone session.
RTF chunk 15 s ≈0.058 chỉ cho cận một worker khoảng 17 s audio mỗi giây thực.

### 2. Word timestamps (R2b)

`timestamp_granularities[]=word` vẫn trả 400 (`WORD_TIMESTAMPS = False`). Alignment và evaluator
`scripts/eval_alignment.py` có sẵn. Probe (baseline §6), sau dịch 200 ms, đặt từ đầu sau khoảng lặng median 75 ms
sớm, tứ phân vị −255 / +120 ms; khó đạt gate nếu không cải thiện ước lượng start.

- [ ] Annotate tay ranh giới từ trong `tests/fixtures/alignment/` (Praat/Audacity, không dùng aligner khác):
  ≥200 từ, ≥10 câu, chia dev/test.
- [ ] Gate trên test: ≥90% từ khớp trong 100 ms, median lỗi ≤50 ms (ngưỡng chốt trước).
- [ ] Bật switch trong commit có report; thêm words vào verbose_json.

### 3. Server VAD và partial (R3b)

- [ ] VAD âm học (pin model, kiểm license và CPU), threshold/prefix_padding_ms/silence_duration_ms; phát
  speech_started/stopped và auto-commit. Không dùng thời điểm phát token làm VAD; semantic_vad vẫn bị từ chối.
- [ ] Nếu cần partial trước commit: decoder stable-prefix (delta chỉ nối thêm, luật sửa kết quả cuối xác định),
  phải có integration test đầy đủ. Model đã tự viết dấu câu/chữ số, server không thêm chuẩn hoá text.

### 4. Scale và hiệu năng

Các mục sau đổi quy tắc một recognizer/một worker/một chunk mỗi call, nên đều cần gate riêng: text bằng
`docs/baseline/offline.json`, giữ deadline/cancel mỗi request, và đo capacity (§1).

- [ ] Batching: decode chunk từ nhiều lane bằng một decode_streams; so throughput/latency với call từng chunk ở
  cùng số thread trước khi bật.
- [ ] Nhiều recognizer trên worker riêng trong một container, chia ngân sách core với `GATEWAY_NUM_THREADS`;
  ngân sách RAM mỗi instance (~0.45 GB peak, viet-asr int8, chunk 15 s).
- [ ] Scale ngang: nhiều container sau load balancer, sticky WS (làm được hiện tại, cần §1 để định cỡ).
- [ ] Runtime GPU tùy chọn: sherpa-onnx với CUDA provider, chỉ khi CPU không đủ. Cần image, pins và full gate riêng;
  CPU vẫn mặc định ([ADR 0001](adr/0001-transducer-on-cpu-replaces-quan-ctc-on-gpu.md)).

### 5. Việc nhỏ hơn

- [ ] conversation.item.created nếu client thật cần (hiện phát conversation.item.added).
- [ ] Audio G.711 audio/pcmu / audio/pcma cho Realtime.
- [ ] Truy cập browser: ephemeral client secret hoặc token service (hiện chỉ server-to-server Bearer).
- [ ] Python 3.11 (mới thử 3.12).

## Ngoài phạm vi

Cần engine khác hoặc hạ tầng phía trước, không phải tính năng của model này:

- Ngôn ngữ khác tiếng Việt, tự phát hiện ngôn ngữ, translations.
- TTS, chat/responses, Realtime hội thoại, WebRTC/SIP.
- Diarization, prompt/keyword biasing, logprobs.
- verbose_json.segments: SDK bắt buộc field Whisper-only mà model không có dữ liệu thật.
- TLS, CORS, rate limit theo client trong gateway (thuộc hạ tầng phía trước).
