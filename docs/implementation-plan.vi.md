# Trạng thái triển khai viet-asr gateway

Kiến trúc: [deployment-design.vi.md](deployment-design.vi.md). Việc còn lại: [roadmap.md](roadmap.md).
Mục tiêu pre-release: `v0.2.0-rc.1`; **chưa gắn tag**, chờ container gate.

## 1. Release

| Release | Nội dung | Trạng thái |
|---|---|---|
| R1 — REST/SSE | models, transcription `json`/`text`, SSE, auth, lỗi OpenAI, giới hạn, Docker CPU | Đạt trên server local (`docs/baseline/server.json`); image viet-asr **chưa qua container gate** |
| R2a — Rich output | `verbose_json` (không `segments`), `srt`, `vtt` với timestamps từ thời điểm phát token | Đạt trên model mới (`docs/baseline/r2a_gate.json`) |
| R2b — Word timestamps | `timestamp_granularities[]=word` | Chưa bật: chờ bộ annotate tay |
| R3a — Realtime explicit commit | WS `/v1/realtime`, PCM16 24 kHz, append/commit/clear | Đạt trên model mới (`docs/baseline/r3a_gate.json`) |
| R3b — Server VAD / partial | VAD âm học, partial trước commit | Chưa làm |
| R4 — Capacity | Giới hạn production đo trên CPU riêng; batching, nhiều recognizer, scale ngang | Tooling có, chưa đo tải |

## 2. Milestone và lịch sử

| Milestone | Kết quả | Kiểm chứng |
|---|---|---|
| M0 — Contract + baseline | Pins model/sherpa-onnx/SDK, engine offline, fixtures, số đo | [baseline.vi.md](baseline.vi.md), `tests/model/test_engine.py`, `tests/unit/test_manifests.py`, `tests/contract/` |
| M1 — Service + scheduler | Config typed, lifespan/health/ready, error envelope, single worker, queue công bằng có giới hạn, logs/metrics | `tests/scheduler/`, `tests/unit/test_app.py`, `tests/unit/test_config.py` |
| M2 — Audio ingress | Giới hạn body trước parser, spool có budget, FFmpeg sandbox, chunking tại điểm yên lặng, policy clip ngắn | `tests/audio/`, `tests/unit/test_chunking.py`, `docs/baseline/limits_probe.json` |
| M3 — REST/SSE + image | Endpoint, joiner, disconnect/send timeout, Dockerfile/compose CPU | `tests/unit/test_transcription_*`, `tests/contract/test_server_sdk.py`, `tests/model/test_server.py` |
| M4 (R2a) — Rich output | Alignment token emission time → cue, serializers; probe + cue gate | `tests/unit/test_alignment.py`, `docs/baseline/alignment_probe.json`, `docs/baseline/r2a_gate.json` |
| M5 — Realtime | Handshake trước accept, session.update, turn buffer + soxr stream, item FIFO, giới hạn session | `tests/unit/test_realtime_*.py`, `tests/model/test_realtime.py`, `docs/baseline/r3a_gate.json` |
| M6 — Chuyển sang Gipformer (lịch sử) | Thay model CTC trên GPU bằng gipformer1.5-68M-rnnt trên CPU (sherpa-onnx), prefix `GIPFORMER_*` | [ADR 0001](adr/0001-transducer-on-cpu-replaces-quan-ctc-on-gpu.md); bằng chứng cũ chỉ còn trong git |
| M7 (một phần) — Hardening | `scripts/benchmark.py`, OOM/hang/recovery giả lập, soak CPU có giới hạn, redaction log | `tests/unit/test_benchmark.py`, `tests/unit/test_capacity_hardening.py` |
| M8 — Chuyển sang viet-asr | Một model int8, text dạng viết, tên trung lập, byte fallback, cue cuối câu, đo lại defaults/gates, ADR 0002, docs/demo; **chờ container gate và tag** | Spec #1, issues #2–#6, [ADR 0002](adr/0002-viet-asr-as-served-model.md), §3–4 |

## 3. Gate và bằng chứng (viet-asr, CPU)

Mọi report hiện tại đo trên `dangvansam/viet-asr@46adacf` int8, sherpa-onnx 1.13.8, chỉ metadata.
Mặc định: beam 4, 2 intra-op thread, chunk 15 s, queue buffer 450 s.

| Gate | Trạng thái | Bằng chứng |
|---|---|---|
| Parity engine vs sherpa-onnx tối giản | Đạt: beam/greedy, text/token/time, ghép byte fallback UTF-8 | `tests/model/reference.py`, `tests/model/test_engine.py` |
| Clip ngắn + độ dài chunk | Đạt: sàn cứng 1,360 mẫu; policy ≤100 ms trả rỗng; chunk mặc định 15 s | `docs/baseline/limits_probe.json`, [baseline §5](baseline.vi.md) |
| Baseline offline | Đạt: WER eval beam 17.28%, greedy 18.25% → giữ beam (luật 0.5 điểm); 2 thread, buffer 450 s | `docs/baseline/offline.json`, [baseline §4](baseline.vi.md) |
| Server smoke (local) | Đạt: mọi format + Realtime; hash khớp offline; `vi_long` WER 8.1% | `docs/baseline/server.json` (commit `7f46ff5`) |
| R2a cue gate | Đạt: shift 200 ms, từ cuối 120 ms, pause 640 ms; cue cuối câu, 7 s / 80 ký tự | `docs/baseline/alignment_probe.json` (`d9149a7`), `docs/baseline/r2a_gate.json` (`a0b957f`) |
| R3a realtime gate | Đạt: transcript item bằng REST cùng waveform, FIFO, nhiều session, turn dài | `docs/baseline/r3a_gate.json` (commit `6d2abeb`) |
| Container gate (image CPU) | **Chưa chạy trên viet-asr**: host agent không có Docker | `docs/baseline/server-container.json` chưa có; [checklist bàn giao](container-gate.vi.md) |
| CPU suite | Đạt: 484 passed (unit + scheduler + audio + contract, gồm Python/JS SDK trên HTTP/WSS thật), sau cập nhật model metadata | `uv run python -m pytest tests/unit tests/scheduler tests/audio tests/contract` |

Issue #6 **chỉ đóng** khi người chạy xác nhận full container gate không skip, commit report từ cây sạch và gắn
tag `v0.2.0-rc.1`. Không dùng kết quả Docker của model cũ làm bằng chứng cho M8.

Demo nghe thử đã tạo lại qua server mặc định: **33 case, 0 lỗi request, 28 case chấm WER**;
`outputs/demo/index.html` và audio/transcript chỉ ở local, không còn được track. Demo chuẩn hoá cả hai phía khi chấm,
nhưng giữ nguyên text để hiển thị. Tier model chạy lại: **24 passed, không skip** (giữ các report đã ghi ở #5).

## 4. Lệch và giới hạn được chấp nhận

- `verbose_json` không có `segments`: schema SDK bắt buộc field kiểu Whisper (`avg_logprob`, `no_speech_prob`,
  `tokens`, …) mà model không có dữ liệu thật. Cue chỉ có ở `srt`/`vtt`.
- Realtime phát `conversation.item.added` (GA), không phải `conversation.item.created` (beta); chỉ infer khi commit,
  qua cùng splitter/joiner với REST, không dùng VAD của thư viện upstream.
- Sàn clip ngắn giữ 100 ms (1,600 mẫu) dù sàn cứng 1,360 mẫu: clip rất ngắn ít cho text hữu ích.
- Chunk 15 s được chọn theo latency ≤1 s và WER trong 0.5 điểm so với chunk tốt nhất (quét tới 30 s). Giữ dấu câu/chữ
  hoa ở vết cắt chunk; không hậu xử lý transcript.
- Giới hạn cue 7 s / 80 ký tự là quy ước phụ đề. Probe suy ra shift 200 ms, từ cuối 120 ms, pause 640 ms; timing vẫn
  có thể sớm ~0.8 s sau khoảng lặng. Word timestamps chưa được annotate/gate nên vẫn tắt.
- Tên model cũ chỉ còn trong lịch sử/ADR/changelog, ngoại trừ `app/config.py:OLD_PREFIXES` và regression test tương
  ứng: bắt buộc giữ literal để từ chối deployment migrate dở theo spec #1. Không che chuỗi để làm sạch grep.
- Beam 4 sinh 1–2 từ trên 24/24 clip im lặng/nhiễu đã thử, nhưng greedy kém 0.97 điểm WER nên không thành mặc định.
- Chấm WER hạ chữ thường, bỏ dấu câu Unicode hai phía; chưa đồng nhất chữ số và số đọc. Transcript phụ thuộc số
  intra-op thread; hash baseline chỉ chuẩn cho cấu hình mặc định. Số đo host dùng chung không phải capacity.

## 5. Definition of done

Mỗi tính năng phát hành có code + test + tài liệu, cleanup/cancel được kiểm thử, không log dữ liệu nhạy cảm, pins
cố định, capability matrix cập nhật và gate có bằng chứng. Tính năng chưa qua gate thì không bật.
