# Kiến trúc triển khai gipformer1.5-68M-rnnt

Cách chạy: [README](../README.md). Payload: [api-contract.vi.md](api-contract.vi.md). Việc chưa làm: [roadmap.md](roadmap.md).

## 1. Lựa chọn

**sherpa-onnx (offline transducer recognizer, CPU, version đã pin) + một gateway FastAPI mỏng, đóng gói Docker.**
Lý do chọn Gipformer trên CPU thay cho model CTC trên GPU trước đây: [ADR 0001](adr/0001-transducer-on-cpu-replaces-quan-ctc-on-gpu.md).

- Dùng export ONNX chính thức của model qua sherpa-onnx; không torch, CUDA, icefall/k2, không vLLM/Triton/Redis/Celery.
- REST, SSE và Realtime WebSocket dùng chung một engine, mỗi giao thức có adapter riêng.
- Một container = một process = một recognizer = một worker thread, chạy trên CPU.
- Phạm vi: *OpenAI Audio Transcriptions và Realtime transcription trong phạm vi công bố*. Gipformer là ASR tiếng Việt;
  dịch, TTS, chat hay Realtime hội thoại cần engine khác, không thuộc gateway này.

## 2. Kiến trúc

```text
OpenAI SDK / curl / WebSocket client
                 |
     (HTTPS do hạ tầng của người triển khai)
                 |
   Docker container: Uvicorn, 1 worker, port 18001
   ├─ auth Bearer + request id + OpenAI error envelope          app/main.py, app/errors.py
   ├─ REST / SSE / verbose_json / srt / vtt                     app/transcription.py, app/formats.py
   ├─ Realtime transcription (explicit commit)                  app/realtime.py
   ├─ audio ingress: giới hạn body, spool, FFmpeg sandbox       app/audio.py
   ├─ chunking tại điểm yên lặng (≤ 15 s, phủ liền audio)       app/chunking.py
   ├─ scheduler: queue có giới hạn, công bằng giữa lane         app/scheduler.py
   └─ EngineWorker: thread duy nhất gọi model                   app/engine.py
                 |
   SherpaEngine: sherpa-onnx OfflineRecognizer (transducer)  app/engine.py
   load từ /models (read-only, đã pin + SHA256), không truy cập Hub
```

### Quy tắc cốt lõi

- **Mọi inference được serialize.** Một `OfflineRecognizer` duy nhất, chỉ `EngineWorker` (một thread) gọi nó, mỗi lần
  một chunk; song song nằm trong onnxruntime (`GATEWAY_NUM_THREADS` intra-op thread), nên CPU và RAM có giới hạn. Endpoint không gọi model hay worker trực tiếp; mỗi request/session là một *lane* trong
  scheduler, mỗi chunk là một *job*.
- **Event loop chỉ làm I/O.** FFmpeg chạy subprocess, inference chạy trên worker thread.
- **Fairness:** round-robin giữa lane; lane live (Realtime) được ưu tiên có hạn mức để file dài không chiếm worker.
  Mỗi request chỉ giữ tối đa 2 chunk trong queue; không enqueue sẵn toàn bộ file dài.
- **Mọi tài nguyên có giới hạn:** body upload, spool disk, audio đã decode trong RAM, số lane/session/job, thời
  lượng, deadline tổng mỗi request. Vượt → 413/429/503 với envelope OpenAI.
- **Hủy:** client ngắt → hủy chunk đang chờ, bỏ kết quả chunk đang chạy. Kernel đang chạy không bị ngắt; worker và
  budget chỉ được giải phóng khi call thật kết thúc.
- **Treo/lỗi:** call inference lâu hơn `GATEWAY_INFER_HANG_SECONDS` → `/health` 503 (supervisor restart). Lỗi đơn lẻ →
  500 và worker tự warm-up lại.

## 3. Luồng xử lý

### REST và SSE

1. Kiểm tra auth, giới hạn body (trước khi parse multipart), validate tham số theo thứ tự cố định.
2. Spool upload vào thư mục tạm riêng; FFmpeg (argv cố định, chỉ protocol `file`, allowlist demuxer/decoder, env
   tối thiểu) decode về mono 16 kHz float32, dừng ngay khi vượt thời lượng cho phép.
3. `chunking.split`: chunk ≤ `GATEWAY_MAX_CHUNK_SECONDS`, cắt tại cửa sổ 50 ms yên lặng nhất trong 5 s cuối; các chunk
   phủ liền audio. Audio ≤ 100 ms → `""`, không gọi model (recognizer lỗi dưới 1,360 mẫu; baseline §5).
4. `run_chunks` đưa từng chunk qua lane; `Joiner` nối text bằng một dấu cách (chunk rỗng không sinh dấu cách).
5. SSE: headers chỉ gửi sau khi decode xong; một `delta` mỗi chunk, một `done` ở cuối; lỗi sau headers → event
   `error`, không có `done`.

### Timestamps (`srt`/`vtt`)

Lấy từ thời điểm phát token (token emission time) của chính lần inference sinh ra text (`app/alignment.py`), cộng
offset tuyệt đối của chunk; không chạy thêm inference. Bước 40 ms, không có độ trễ cố định cần bù; độ dài từ cuối chunk
và ngưỡng pause đo bằng `scripts/probe_alignment.py` (`docs/baseline/alignment_probe.json`, baseline §6). Nếu token
không khớp transcript → 500 `alignment_failed`; không bao giờ chia đều thời gian.

### Realtime

`GET /v1/realtime`: kiểm tra auth/model/capacity **trước** accept. Mỗi session giữ một lane LIVE. Audio PCM16 24 kHz
base64 được resample stream sang 16 kHz (soxr); commit tạo item bất biến, buffer mới nhận turn kế tiếp ngay. Item
xử lý FIFO qua cùng `split`/`run_chunks`/`Joiner` như REST. Gửi có giới hạn thời gian; idle/session timeout.

## 4. Model và dữ liệu

- `SherpaEngine.load` nạp `viet-asr/int8/{encoder,decoder,joiner}.int8.onnx` và `tokens.txt` từ thư mục đã stage
  (luôn int8, không có setting precision), chỉ đường dẫn local, không truy cập Hub. Text, token và thời điểm phát token
  giống hệt script tham chiếu sherpa-onnx tối giản `tests/model/reference.py` trên cùng audio và setting
  (`tests/model/test_engine.py`, beam/greedy). Kết quả mỗi chunk: text + token + thời điểm phát token (mẫu, tính từ
  đầu chunk). sherpa-onnx đã ghép token byte fallback `<0xNN>` trong text nhưng trả token thô, nên engine ghép mỗi
  chuỗi token byte liền nhau thành ký tự UTF-8 (giữ thời điểm của byte đầu); fixture `vi_bytes.wav` kiểm việc này.
- Pins duy nhất: `assets/manifest.json` (repo, commit, size, SHA256 từng file); `scripts/stage_models.py` tải và
  kiểm tra. Chi tiết: [baseline.vi.md](baseline.vi.md).
- Decode `modified_beam_search` beam 4 (mặc định, hoặc `greedy_search`), không KenLM. Output giữ nguyên dạng viết
  của model: chữ hoa/thường, dấu câu, số bằng chữ số.
- Peak RSS đo được (viet-asr int8, chunk 15 s, 2 thread): ~0.45 GB (baseline §4, §5).

## 5. Đóng gói

- Image multi-stage, chỉ CPU: builder dùng `uv sync --frozen --no-dev` (sherpa-onnx kèm onnxruntime; không torch,
  CUDA hay GPU runtime); runtime Ubuntu 24.04 (pin digest) + FFmpeg 6.1.1 (pin apt), chỉ copy `app/` và venv.
  Compose không reserve GPU.
- Non-root UID 10001, rootfs read-only, tmpfs `/tmp`, model mount read-only, API key qua Docker secret.
  `HF_HUB_OFFLINE=1`; image chạy được với `--network none`.
- `HEALTHCHECK` trên `/health`; `stop_grace_period: 70s` đủ cho Uvicorn 30 s + drain 30 s + dừng worker 5 s.
- Uvicorn luôn `--workers 1 --ws websockets-sansio --ws-max-size 2097152` (`auto` có thể âm thầm tắt WebSocket).

## 6. Bảo mật

- Không có API key mặc định; key ≥ 16 ký tự, không được trùng `HF_TOKEN`. `HF_TOKEN` chỉ dùng lúc staging.
- Không bao giờ log audio, transcript, key hay tên file; tên file client không dùng làm path, không truyền cho FFmpeg.
- Realtime chỉ server-to-server (Bearer header). Không có ephemeral client secret cho trình duyệt.
- Gateway không làm TLS, CORS hay rate limit theo client: thuộc hạ tầng phía trước.
