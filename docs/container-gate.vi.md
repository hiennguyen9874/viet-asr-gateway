# Kiểm định Docker image

Gate phải đạt trước khi phát hành một image: `tests/container/test_image.py`. Cách triển khai: [README](../README.md).

## 1. Trạng thái

| Image | Runtime | Kết quả | Bằng chứng |
|---|---|---|---|
| `viet-asr-gateway` (CPU) | sherpa-onnx (pin trong `uv.lock`), không torch/CUDA | **Gate Docker đạt trên working tree hiện tại: 6 passed, không skip**; chưa là bằng chứng phát hành từ cây sạch | Report cục bộ: `/tmp/gipformer-server-container.json` (`source_dirty: true`) |

Kiểm tra triển khai hiện tại: `docker compose up -d --no-build` chạy image `viet-asr-gateway:r1`,
`/ready` trả 200, Docker báo healthy; UID `10001:10001`, rootfs read-only, không có GPU device requests.
Smoke trên service thật tại `http://127.0.0.1:18001` đạt mọi format và Realtime trên 5 fixtures.
Key nằm trong `secrets/gateway_api_keys` (gitignored, UID 10001, mode 0400); không ghi giá trị vào report.

Suite `tests/unit tests/scheduler tests/audio tests/contract`: 443 passed, 1 skipped, 1 failed.
Test lỗi `test_client_that_stops_reading_is_dropped` không quan sát được `send_timeout` trong 30 s;
chạy riêng vẫn lỗi. Không coi suite này đã đạt; cần điều tra riêng trước khi phát hành.
Chạy lại gate từ cây sạch trước khi commit report phát hành.

Kiểm tra thay thế trước đây (chỉ chẩn đoán, **không thay gate**, không commit report): venv dựng như builder
(`uv sync --frozen --no-dev`, Python 3.12.14, ~166 MB, không có `torch`/`transformers`) + chỉ `app/`, model copy
read-only, key qua file, `HF_HUB_OFFLINE=1`, chạy trên host không có network namespace riêng. Các script của gate
(`smoke_server.py`, auth/lỗi/overload/isolation, SIGTERM khi đang stream → `done` + `drained: true`, restart → ready)
đều đạt, key không xuất hiện trong log. Gate Docker thật nay đã chạy như kết quả ở bảng trên.

## 2. Chạy gate

Cần Docker, model đã stage (`./models`, `scripts/stage_models.py`) và fixtures (`scripts/fetch_fixtures.py`, gồm
`fixtures-large/vi_long.wav`). Không cần GPU.

```bash
uv run pytest tests/container -v      # build viet-asr-gateway:test rồi chạy cả gate; phải pass hết, không skip
```

Test chỉ skip khi không có `docker`; thiếu model hoặc `vi_long.wav` là fail. Smoke ghi report vào
`docs/baseline/server-container.json` (chỉ metadata: thời gian, số cue, hash transcript, image id, commit; không có
transcript, audio hay key). Commit report cùng bản cập nhật bảng §1 khi gate pass trên cây sạch
(`source_dirty: false`).

## 3. Nội dung gate

| Test | Kiểm tra |
|---|---|
| Image | Build; chạy UID 10001; `/srv` chỉ có `app/` và `.venv`; không có assets, references, models, tests, `.git`, fixtures; Python 3.12.14, FFmpeg 6.1.1; `import sherpa_onnx` được, không có `torch`/`transformers` |
| Metadata | `docker inspect`/`history` không chứa `HF_TOKEN`, `GATEWAY_API_KEYS*` hay giá trị `HF_TOKEN` |
| Offline + smoke | Container `--network none`, rootfs read-only, model mount read-only, key qua file secret (như compose) → `/ready`; `scripts/smoke_server.py` trên 5 fixture: json/text/SSE/verbose_json/SRT/VTT và Realtime, hash transcript khớp `docs/baseline/offline.json` |
| Auth/lỗi/overload/isolation | 401, model 404, audio lỗi 400, 429 khi hết lane; hai request đồng thời không trộn transcript; key không có trong log |
| Shutdown + restart | SIGTERM khi đang stream 6 × `vi_long.wav` → SSE `done`, exit 0, `drained: true`; restart → ready |

`compose.yaml` không có cấu hình GPU; Dockerfile/compose không nhắc CUDA/torch (`tests/unit/test_manifests.py`).

Khi đổi base image, sherpa-onnx, FFmpeg pins, Dockerfile hoặc compose: chạy lại unit/contract và toàn bộ gate, không
skip.
