# Kiểm định Docker image

Gate phải đạt trước khi phát hành image: `tests/container/test_image.py`. Cách triển khai: [README](../README.md).

## 1. Trạng thái

| Image | Runtime | Kết quả | Bằng chứng |
|---|---|---|---|
| `viet-asr-gateway` (CPU, viet-asr int8) | sherpa-onnx (pin trong `uv.lock`), không torch/CUDA | **Chưa chạy container gate trên model mới**; host agent không có Docker | `docs/baseline/server-container.json` chưa có |

Report offline, R2a, R3a và smoke local đã đo lại trên viet-asr (xem [baseline](baseline.vi.md)).
Kết quả Docker của model cũ không chứng minh image hiện tại; mô phỏng image trên host chỉ dùng chẩn đoán.
Không gắn tag `v0.2.0-rc.1` trước full gate không skip.

## 2. Checklist bàn giao cho người chạy (#6)

Cần Docker Engine đang chạy, model đã stage và fixtures gồm `fixtures-large/vi_long.wav`. Không cần GPU.

- [ ] Commit các thay đổi code/test/docs; bắt đầu từ cây sạch (`git status --porcelain` rỗng).
- [ ] Kiểm model/fixtures và build image (exit 0, không lỗi cú pháp):

  ```bash
  uv sync
  uv run python scripts/stage_models.py --verify
  uv run python scripts/fetch_fixtures.py --verify
  docker build -t viet-asr-gateway:test .
  ```

- [ ] Chạy **toàn bộ gate**, không skip:

  ```bash
  uv run python -m pytest tests/container -v -rs
  ```

  Suite tự build `viet-asr-gateway:test`; chỉ skip khi không có Docker, thiếu model/long fixture là fail.
  Smoke ghi `docs/baseline/server-container.json`: metadata, hash transcript, image id, commit, `source_dirty`;
  không có transcript, audio hay key. Phải có `source_dirty: false`, mọi hash khớp baseline viet-asr.
- [ ] Kiểm report và kết quả đầy đủ; cập nhật bảng §1, commit report cùng tài liệu. Không commit report của lần
  chạy skip/fail hoặc mô phỏng trên host.
- [ ] Người chạy xác nhận gate đạt vào issue #6, sau đó gắn tag:

  ```bash
  git tag -a v0.2.0-rc.1 -m "viet-asr gateway v0.2.0-rc.1"
  ```

- [ ] Chỉ đóng #6 khi đã có report và tag. Agent không tự commit report container, không tự gắn tag.

## 3. Nội dung gate

| Test | Kiểm tra |
|---|---|
| Image | Build; UID 10001; `/srv` chỉ có `app/` và `.venv`; không assets, references, models, tests, `.git`, fixtures; Python 3.12.14, FFmpeg 6.1.1; import sherpa-onnx được, không torch/transformers |
| Metadata | `docker inspect`/`history` không chứa `HF_TOKEN`, `GATEWAY_API_KEYS*` hay giá trị secret |
| Offline + smoke | `--network none`, rootfs/model mount read-only, key qua file secret → ready; smoke trên 5 fixtures: json/text/SSE/verbose_json/SRT/VTT + Realtime; hash khớp `docs/baseline/offline.json` |
| Auth/lỗi/overload/isolation | 401, model 404, audio lỗi 400, 429 khi hết lane; request đồng thời không lẫn transcript; key không trong log |
| Shutdown + restart | SIGTERM khi stream 6 × `vi_long.wav` → SSE done, exit 0, `drained: true`; restart → ready |

`compose.yaml` không cấu hình GPU; Dockerfile/compose không nhắc CUDA/torch (`tests/unit/test_manifests.py`).
Khi đổi base image, sherpa-onnx, FFmpeg pins, Dockerfile hoặc compose: chạy lại unit/contract và full gate, không skip.
