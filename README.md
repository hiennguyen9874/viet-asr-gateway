# viet-asr gateway

Server nhận dạng tiếng nói tiếng Việt cho [viet-asr](https://huggingface.co/dangvansam/viet-asr)
(Zipformer2 transducer 86M, int8, sherpa-onnx, chạy trên CPU), triển khai bằng Docker, với một **tập con** API OpenAI:

- `GET /v1/models`, `GET /v1/models/{id}`
- `POST /v1/audio/transcriptions`: `json`, `text`, SSE (`stream=true`), `verbose_json`, `srt`, `vtt`
- `GET /v1/realtime`: Realtime transcription qua WebSocket, commit tường minh (`turn_detection: null`)

Mục tiêu pre-release: `v0.2.0-rc.1`; tag chỉ được gắn sau khi [container gate](docs/container-gate.vi.md) đạt.

Không phải tương thích OpenAI đầy đủ. Payload, mã lỗi và capability matrix: [docs/api-contract.vi.md](docs/api-contract.vi.md).

Giới hạn cần biết:

- Output là text dạng viết, nguyên văn như model sinh ra (chữ hoa/thường, dấu câu, số bằng chữ số); server không
  chuẩn hoá. Chỉ `language=vi`; không có logprobs.
- `srt`/`vtt` có thời gian từ thời điểm model phát từng token (bước 40 ms; cue có thể bắt đầu sớm tới ~1 s sau khoảng
  lặng, xem [baseline §6](docs/baseline.vi.md)). `verbose_json` không có `segments`; word timestamps chưa bật.
- Realtime: transcript chỉ có sau `input_audio_buffer.commit`; không có partial, không có VAD.
- Alias `whisper-1` (tắt mặc định, `GATEWAY_WHISPER_ALIAS=true`) chỉ để client cũ chạy được: vẫn là model viet-asr. Mọi model id khác trả 404 `model_not_found`.

Tài liệu: [kiến trúc](docs/deployment-design.vi.md) · [API contract](docs/api-contract.vi.md) ·
[baseline và pins](docs/baseline.vi.md) · [kiểm định image](docs/container-gate.vi.md) ·
[trạng thái](docs/implementation-plan.vi.md) · [roadmap](docs/roadmap.md)

## 1. Yêu cầu

- Linux, Docker Engine + Compose v2, CPU x86-64 (không cần GPU, CUDA hay torch).
- [uv](https://docs.astral.sh/uv/) trên host, chỉ để tải model (staging) và chạy smoke test.
- Model Apache-2.0 (chỉ có int8); chạy offline từ thư mục `./models` mount read-only. Lý do chọn model:
  [ADR 0002](docs/adr/0002-viet-asr-as-served-model.md).

## 2. Triển khai

```bash
# 1. Tải model đã pin vào ./models và kiểm tra SHA256 (một lần, cần mạng)
uv run scripts/stage_models.py          # HF_TOKEN=<token> nếu repo yêu cầu
uv run scripts/stage_models.py --verify # kiểm tra lại, không cần mạng

# 2. API key: một key mỗi dòng, ≥ 16 ký tự, đọc được bởi UID 10001 trong container
mkdir -p secrets
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/gateway_api_keys
sudo chown 10001 secrets/gateway_api_keys && sudo chmod 0400 secrets/gateway_api_keys

# 3. Build + chạy (CPU)
docker compose up -d --build

# 4. Chờ sẵn sàng (load + warm-up)
curl -s 127.0.0.1:18001/ready                                   # {"status":"ready"}

# 5. Thử nhanh với file mẫu có sẵn trong repo (tests/fixtures/vi_a.wav, 8.9 s, FLEURS, CC BY 4.0)
curl -s 127.0.0.1:18001/v1/audio/transcriptions \
    -H "Authorization: Bearer $(sudo head -1 secrets/gateway_api_keys)" \
    -F model=viet-asr -F file=@tests/fixtures/vi_a.wav
# {"text":"Các sản phẩm thủ công có thể được coi là đồ cổ, mặc dù ...","usage":{...}}

# 6. Smoke test (tùy chọn): mọi format + Realtime, so với baseline
OPENAI_API_KEY=$(sudo cat secrets/gateway_api_keys | head -1) \
    uv run scripts/smoke_server.py --base-url http://127.0.0.1:18001
```

Image (`viet-asr-gateway:r1`): Ubuntu 24.04 (pin digest), Python 3.12.14, FFmpeg 6.1.1, user `10001`, chỉ
chứa `app/` và venv. Không có model, `HF_TOKEN` hay API key trong image; server không truy cập Hub.
`compose.yaml` chạy:

- model mount `./models:/models:ro`, key qua Docker secret `/run/secrets/gateway_api_keys`;
- rootfs read-only, tmpfs `/tmp` 512 MiB (spool upload), `HF_HUB_OFFLINE=1`;
- port chỉ bind `127.0.0.1:18001`; `restart: unless-stopped`; `stop_grace_period: 70s`.

**Một container = một recognizer = một worker thread**, luôn một Uvicorn worker. Mở rộng: chạy nhiều container
(mỗi container một port host) sau load balancer; WebSocket cần sticky routing.

**TLS:** gateway không làm TLS, rate limit theo client hay CORS. Khi expose ra ngoài localhost, đặt sau hạ tầng
HTTPS của bạn với: SSE không buffer, hỗ trợ WebSocket upgrade, read timeout ≥ `GATEWAY_REQUEST_TIMEOUT_SECONDS`
(600 s), body ≥ `GATEWAY_MAX_UPLOAD_BYTES`. JS SDK Realtime chỉ kết nối `wss:`.

## 3. Gọi API

```bash
KEY=$(sudo cat secrets/gateway_api_keys | head -1)
curl -s 127.0.0.1:18001/v1/models -H "Authorization: Bearer $KEY"
curl -s 127.0.0.1:18001/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
    -F model=viet-asr -F file=@tests/fixtures/vi_a.wav
curl -sN 127.0.0.1:18001/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
    -F model=viet-asr -F stream=true -F file=@tests/fixtures/vi_a.wav
curl -s 127.0.0.1:18001/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
    -F model=viet-asr -F response_format=srt -F file=@tests/fixtures/vi_a.wav   # text, vtt, verbose_json
```

OpenAI Python SDK (đã kiểm thử `openai==3.31.0`; JS `openai@7.33.0`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18001/v1", api_key="<API key>")
with open("audio.mp3", "rb") as f:
    print(client.audio.transcriptions.create(file=f, model="viet-asr").text)
with open("audio.mp3", "rb") as f:
    for event in client.audio.transcriptions.create(file=f, model="viet-asr", stream=True):
        print(event.type, getattr(event, "delta", None) or getattr(event, "text", ""))
```

Realtime (`openai[realtime]`): PCM16 little-endian mono 24 kHz, base64; nhiều commit trên một socket, mỗi commit là
một item với các `delta` (một mỗi chunk ≤ 15 s) rồi `completed`.

```python
import base64

with client.realtime.connect(model="viet-asr") as conn:
    conn.recv()                                            # session.created
    for packet in pcm16_24k_packets:                       # bytes, ví dụ 20–200 ms mỗi packet
        conn.input_audio_buffer.append(audio=base64.b64encode(packet).decode())
    conn.input_audio_buffer.commit()
    for event in conn:
        if event.type == "conversation.item.input_audio_transcription.completed":
            print(event.transcript)
            break
```

Audio: WAV, FLAC, MP3, MP4/M4A (AAC/ALAC), OGG (Vorbis/Opus/FLAC), WEBM/Matroska, nhận diện theo nội dung. Audio dài
được cắt thành chunk ≤ 15 s tại điểm yên lặng; SSE gửi một delta mỗi chunk. Audio ≤ 100 ms → `""` (không gọi model). `srt`/`vtt`/
`verbose_json` không stream (`stream=true` → 400). Lỗi dùng envelope OpenAI; danh sách mã lỗi: contract §7.

## 4. Cấu hình

Thêm biến `GATEWAY_*` vào `environment:` của service (ví dụ trong `compose.override.yaml`). Biến lạ, biến còn sót prefix của
bản cũ (`OLD_PREFIXES` trong `app/config.py`), giá trị ≤ 0 hoặc không nhất quán → container không start. Không còn
setting precision (model chỉ có int8). Đầy đủ: `app/config.py`.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GATEWAY_API_KEYS_FILE` / `GATEWAY_API_KEYS` | (bắt buộc) | Key Bearer (compose dùng file secret) |
| `GATEWAY_WHISPER_ALIAS` | `false` | Chấp nhận model id `whisper-1` |
| `GATEWAY_LOG_LEVEL` | `INFO` | Mức log |
| `GATEWAY_MODELS_DIR` | `models` (compose: `/models`) | Thư mục local đã stage; không tải model lúc chạy |
| `GATEWAY_DECODING_METHOD` | `modified_beam_search` | Hoặc `greedy_search`: nhanh hơn ~33%, WER kém ~1 điểm, nhưng gần như luôn trả `""` trên im lặng/nhiễu (beam 4 thường sinh vài chữ): [baseline §4](docs/baseline.vi.md) |
| `GATEWAY_BEAM_SIZE` | 4 | Số path của `modified_beam_search` (số nguyên > 0; bỏ qua khi greedy) |
| `GATEWAY_NUM_THREADS` | 2 | Số intra-op thread của recognizer duy nhất (số nguyên > 0). Đo: chunk 15 s mất 1.18 / 0.86 / 0.75 s ở 1 / 2 / 4 thread; từ 4 thread hiệu suất < 50%. Transcript có thể đổi theo số thread: [baseline §2, §4](docs/baseline.vi.md) |
| `GATEWAY_MAX_UPLOAD_BYTES` | 25 MiB | Cả body multipart → 413 |
| `GATEWAY_MAX_AUDIO_SECONDS` | 1800 | Thời lượng sau decode → 400 `audio_too_long` |
| `GATEWAY_MAX_DECODED_SECONDS` | 3600 | Audio đã decode giữ trong RAM của mọi request (≈ 64 KB/s) → 429 |
| `GATEWAY_MAX_SPOOL_BYTES` | 200 MiB | File tạm của upload đang nhận (nằm trong tmpfs 512 MiB) → 429 |
| `GATEWAY_SPOOL_DIR` | Thư mục tạm hệ thống (`/tmp` trong image) | Tạo thư mục upload riêng mode 0700 bên trong |
| `GATEWAY_DECODE_TIMEOUT_SECONDS` | 60 | Quá thời gian decode FFmpeg → 400 `invalid_audio` |
| `GATEWAY_MAX_CONCURRENT_DECODES` | 2 | Số decode đồng thời; hết slot → 429 |
| `GATEWAY_MAX_CHUNK_SECONDS` | 15 | Độ dài chunk tối đa (cũng là độ dài warm-up). Chunk dài nhất có latency ≤ 1 s ở 2 thread; WER 10–30 s gần như bằng nhau; không nên quá 30 s (độ dài huấn luyện của model): [baseline §5](docs/baseline.vi.md) |
| `GATEWAY_QUEUE_MAX_LANES` | 32 | Request/session đồng thời → 429 |
| `GATEWAY_QUEUE_MAX_JOBS` | 64 | Tổng job queue + đang chạy |
| `GATEWAY_QUEUE_MAX_JOBS_PER_LANE` | 2 | Cửa sổ chunk mỗi lane; phải ≤ tổng job |
| `GATEWAY_LIVE_PRIORITY_RATIO` | 3 | Tối đa 3 job LIVE mỗi job FILE khi cả hai cùng chờ |
| `GATEWAY_QUEUE_TIMEOUT_SECONDS` | 30 | Chờ chỗ trong queue tối đa → 429 |
| `GATEWAY_QUEUE_MAX_BUFFERED_SECONDS` | 450 | Audio trong queue + đang chạy → chờ chỗ (rồi 429). Chọn để queue đầy chạy hết trong ≤ 30 s ở RTF đo được (beam 4, 2 thread): [baseline §4](docs/baseline.vi.md). Giảm nếu CPU chậm hơn hoặc tăng `GATEWAY_BEAM_SIZE` |
| `GATEWAY_REQUEST_TIMEOUT_SECONDS` | 600 | Tổng upload + decode + queue + infer + gửi → 503 `request_timeout` |
| `GATEWAY_SEND_TIMEOUT_SECONDS` | 10 | Một lần gửi SSE/WebSocket; client không đọc → ngắt |
| `GATEWAY_MAX_SESSIONS` | 16 | Realtime session đồng thời (≤ lanes) → handshake 429 |
| `GATEWAY_MAX_SESSION_SECONDS` | 3600 | Tuổi tối đa một session → `session_expired` |
| `GATEWAY_SESSION_IDLE_TIMEOUT_SECONDS` | 120 | Session không hoạt động → `idle_timeout` |
| `GATEWAY_MAX_TURN_SECONDS` | 120 | Audio chưa commit của một session → `input_audio_buffer_full` |
| `GATEWAY_MAX_PENDING_ITEMS` | 4 | Item đã commit chưa xong → `too_many_pending_items` |
| `GATEWAY_INFER_HANG_SECONDS` | 120 | Một call inference lâu hơn → worker "hung", `/health` 503 (call dài nhất đo được: 1.3 s) |
| `GATEWAY_SHUTDOWN_DRAIN_SECONDS` | 30 | Chờ job đang chạy khi tắt |

Các mặc định là giới hạn an toàn hữu hạn, **chưa phải capacity đã đo** (xem [roadmap](docs/roadmap.md) R4). Đo
trên phần cứng của bạn trước khi tăng.

## 5. Vận hành

- **Readiness/liveness:** `/ready` 200 khi model đã load + warm-up và không đang drain/recover (dùng cho load
  balancer). `/health` 503 khi load/recovery lỗi hoặc một call inference treo quá `GATEWAY_INFER_HANG_SECONDS`.
  Docker không tự restart container chỉ vì `unhealthy`: dùng watchdog/orchestrator (autoheal, systemd, Kubernetes
  `livenessProbe`). Lỗi inference đơn lẻ trả 500 `inference_failed`, worker tự warm-up lại.
- **Dừng/cập nhật:** `docker compose stop` (SIGTERM) ngừng nhận kết nối mới, request đang chạy được làm xong (≤ 30 s),
  drain queue (≤ 30 s); log `server.stopped` có `drained`. WebSocket **không** được drain: đóng với code 1012, item
  chưa `completed` bị mất, client kết nối lại và gửi lại audio. Rolling update: bỏ instance khỏi load balancer trước.
- **Rollback:** giữ image digest của bản tốt, chạy lại digest đó với cùng `./models`; chờ `/ready`
  và chạy smoke trước khi mở traffic.
- **Xoay vòng key** (key chỉ đọc lúc khởi động): thêm key mới vào file, restart, chuyển client, bỏ key cũ, restart.
- **Log/metrics:** log JSON ra stderr (`docker compose logs`), chỉ metadata (`request_id`, thời lượng, outcome);
  không ghi audio, transcript, key, tên file. `/metrics` (Prometheus, cần API key; không expose public):
  `gateway_transcriptions_total`, `gateway_queue_wait_seconds`, `gateway_infer_seconds`, `gateway_ready`, `gateway_worker_hung`,
  `gateway_realtime_sessions`, `gateway_realtime_commit_to_completed_seconds`, …
- `HF_TOKEN` chỉ dùng lúc staging, không đưa vào container.

## 6. Kiểm thử

```bash
uv run pytest                                  # unit, scheduler, audio, contract; tests/model cần model đã stage
uv run pytest tests/container                  # gate image (cần docker)
```

Chi tiết gate image: [docs/container-gate.vi.md](docs/container-gate.vi.md). Lệnh phát triển: [docs/agents/commands.md](docs/agents/commands.md).
