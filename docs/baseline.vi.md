# Baseline inference và pins — gipformer1.5-68M-rnnt

Số đo nền của model (ngoài gateway), các pin tái lập và giới hạn frontend. Đây là smoke/baseline trên vài fixture,
**không phải benchmark năng lực phục vụ** và không phải đánh giá WER.

## 1. Pins

| Thành phần | Pin |
|---|---|
| Model `g-group-ai-lab/gipformer1.5-68M-rnnt` | Hub revision `dd9227dcd8705c13f33bdbe59728d546ab94480f`: encoder/decoder/joiner fp32 + int8, `tokens.txt`, `config.json`, `README.md` (không stage `model.pt`, `bpe.model`) |
| Runtime `sherpa-onnx` | `1.13.8` (cùng `sherpa-onnx-core`; `pyproject.toml`, `uv.lock`) |
| Fixtures `google/fleurs` | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` (vi_vn dev, CC BY 4.0) |
| Python | 3.12.14 (image) |
| OpenAI SDK (contract) | Python `openai==3.31.0`, JS `openai@7.33.0` |

`assets/manifest.json` là nguồn pin duy nhất cho model (repo, revision, size, SHA256 từng file) và giữ version
sherpa-onnx; `tests/unit/test_manifests.py` kiểm tra version đó khớp `pyproject.toml` và `uv.lock`. Pin lấy từ Hub, không
từ submodule `references/gipformer1.5-68M-rnnt` (weights trong đó là con trỏ Git LFS). Đổi pin: sửa manifest rồi
chạy `scripts/stage_models.py --update-manifest`.

**License:** model MIT; repo không phân phối weights. Repo trên Hub không bị gate (không cần `HF_TOKEN`);
`stage_models.py` vẫn đọc `HF_TOKEN` nếu có.

## 2. Engine offline

`app/engine.SherpaEngine` tạo một `sherpa_onnx.OfflineRecognizer.from_transducer` từ file local đã stage (không
truy cập Hub), 16 kHz, 80 fbank, `GATEWAY_DECODING_METHOD`/`GATEWAY_BEAM_SIZE` như reference. Parity
(`tests/model/test_engine.py`, CPU, cần model đã stage): text giống hệt `references/gipformer/infer_onnx.py` cho
fp32/int8 × `modified_beam_search`/`greedy_search`; thời điểm phát token tăng dần, nằm trong chunk và khớp text.

## 3. Fixtures

`scripts/fetch_fixtures.py` tạo lại tất định từ FLEURS và kiểm tra SHA256 (`tests/fixtures/manifest.json`).

| File | Nội dung |
|---|---|
| `tests/fixtures/vi_a.wav`, `vi_b.wav` | 2 clip FLEURS khác nội dung (8.88 s, 8.46 s) + transcript tham chiếu |
| `tests/fixtures/silence_2s.wav` | 2 s số 0 |
| `tests/fixtures/short_200ms.wav` | 200 ms cắt giữa câu của `vi_a` |
| `fixtures-large/vi_long.wav` (gitignored) | 10 câu nối nhau, 106.62 s (đi qua splitter) |

FLEURS vi_vn không có speaker id: `vi_a`/`vi_b` chắc chắn khác nội dung, không chắc khác người nói.

## 4. Baseline offline — gipformer1.5-68M-rnnt (CPU)

Đo bằng `scripts/baseline_offline.py` (sherpa-onnx 1.13.8, model `dd9227dc`, Xeon Silver 4216 × 64 core, **host dùng
chung**: load average 5.5 → 16 trong lúc đo). Mỗi setting một process mới; engine dùng y như gateway: warm-up 20 s số
0, fixture là một chunk, `vi_long` cắt bằng `chunking.split` (20 s, 7 chunk) và nối bằng `Joiner`. Median 5 lần
(`vi_long`: 3 lần). Report chỉ có số đo, số đếm và SHA256 của transcript:
[`baseline/offline.json`](baseline/offline.json); hash của setting mặc định là chuẩn so sánh cho smoke server và
container gate.

**(a) Precision × decoding** (4 thread). RTF = latency / thời lượng audio.

| Setting | Load | RSS peak | RTF chunk 20 s | RTF `vi_long` | WER `vi_a` / `vi_b` / `vi_long` |
|---|---:|---:|---:|---:|---|
| **fp32, `modified_beam_search` (beam 4)** | 1.6 s | 792 MB | **0.041** | **0.040** | 0% / 4.0% / **4.0%** |
| fp32, `greedy_search` | 1.6 s | 796 MB | 0.030 | 0.031 | 0% / 4.0% / 4.4% |
| int8, `modified_beam_search` | 1.5 s | 462 MB | 0.032 | 0.031 | 0% / 4.0% / 4.4% |
| int8, `greedy_search` | 1.5 s | 465 MB | 0.022 | 0.023 | 0% / 4.0% / 4.4% |

Mọi setting tất định qua các lần lặp. `silence_2s` và `short_200ms` cho text khác rỗng (model tự phát token trên số 0
hay speech cắt ngắn; gateway không ép `""` với audio > 100 ms). Mỗi lời gọi có sàn nhỏ: 200 ms audio mất ~20 ms
(RTF ~0.1). WER thô so với transcript FLEURS (lower-case), chỉ để phát hiện model hỏng: trên 3 clip, beam 4 hơn greedy
và fp32 hơn int8 đúng một từ của `vi_long`; int8 giảm RAM ~40%, greedy nhanh hơn ~25–30%.

**(b) Số thread** (fp32, beam 4; chunk 20 s là lời gọi dài nhất gateway thực hiện). Hiệu suất = latency 1 thread /
(số thread × latency). Transcript giống hệt nhau ở mọi số thread.

| Thread | Latency chunk 20 s (median) | Hiệu suất | RTF `vi_long` |
|---:|---:|---:|---:|
| 1 | 1.72 s | 1.00 | 0.086 |
| 2 | 1.08 s | 0.79 | 0.053 |
| **4** | **0.81 s** | **0.53** | **0.041** |
| 8 | 0.63 s | 0.34 | 0.033 |
| 16 | 0.63 s | 0.17 | 0.036 |

**(c) Mặc định suy ra** (script exit 1 nếu `app/config.py` lệch):

- `GATEWAY_NUM_THREADS = 4`: số thread lớn nhất còn hiệu suất ≥ 0.5. Từ 8 thread trở lên mỗi core thêm chỉ nhanh
  thêm vài phần trăm cho một stream, và lấy core của FFmpeg decode và event loop. Lần đo trước trong cùng ngày cho
  0.55 / 0.35 ở 4 / 8 thread; trên host khác nên đo lại.
- `GATEWAY_QUEUE_MAX_BUFFERED_SECONDS` 900 → **600**: một chunk đã vào queue có thể phải chờ toàn bộ audio đã buffer
  trước nó; thời gian đó phải ≤ `GATEWAY_QUEUE_TIMEOUT_SECONDS` (30 s). Với RTF xấu nhất của mặc định (0.045, max
  qua các lần lặp) giới hạn là 668 s (lần đo trước: 721 s); 600 s chờ tối đa ~27 s. 900 s cũ (mặc định trước khi đo) sẽ là ~40 s.
- `GATEWAY_INFER_HANG_SECONDS = 120` giữ nguyên: ≥ 20 × lời gọi chậm nhất đo được (1.8 s, chunk 20 s ở 1 thread).
- Giữ nguyên, kiểm tra với số đo: một job LIVE chờ tối đa một chunk FILE đang chạy (≤ 0.9 s); file dài nhất (1,800 s)
  chạy một mình mất ~81 s, nên `GATEWAY_REQUEST_TIMEOUT_SECONDS` 600 s đủ cho ~7 file dài tối đa cùng chia worker.
  Lanes (32), sessions (16), jobs (64, 2 mỗi lane) là giới hạn an toàn khi rảnh; queue buffer ở trên là giới hạn chặn.

Đây là số đo một stream offline trên host dùng chung, **không phải năng lực phục vụ** (R4, [roadmap.md](roadmap.md) §1).

**Server smoke** (`scripts/smoke_server.py`, server local với cấu hình mặc định; report chỉ metadata
[`baseline/server.json`](baseline/server.json)): `json`/`text`/SSE/`verbose_json`/`srt`/`vtt` khớp nhau trên 5
fixture, hash transcript 4 fixture một chunk khớp (a), `vi_long` WER 4.0%; Realtime một socket 5 commit, WER bằng
REST. REST `vi_a` (8.88 s) ≈ 0.45 s, Realtime commit → completed ≈ 0.39 s; `vi_long` ≈ 4.4 s, delta đầu sau 0.7 s.

## 5. Clip ngắn và độ dài chunk — gipformer1.5-68M-rnnt (CPU)

Đo bằng `scripts/probe_limits.py` trên recognizer thật (sherpa-onnx 1.13.8, Xeon Silver 4216, 4 thread). Report chỉ
có số đo và số đếm: [`baseline/limits_probe.json`](baseline/limits_probe.json).

**Input ngắn nhất.** Với cả fp32/int8 × `modified_beam_search`/`greedy_search`, cả số 0 lẫn noise:

| Số mẫu 16 kHz | Gọi recognizer trực tiếp |
|---:|---|
| 0 … 1,359 | **RuntimeError** của onnxruntime (conv đầu vào của encoder cần ≥ 7 frame fbank) |
| ≥ 1,360 (85 ms) | Chạy được (kiểm tra mọi độ dài 1,360 … 1,759 và lưới tới 3 s) |

**Speech ngắn** (40 đoạn cắt từ `vi_a`/`vi_b` mỗi độ dài, fp32 beam 4): số đoạn có text khác rỗng là 1/40 ở 85 ms,
1/40 ở 100 ms, 20/40 ở 150 ms, 27/40 ở 200 ms, 39/40 ở 500 ms.

**Policy (từ report):**

- `MIN_FORWARD_SAMPLES = 1360` (`app/engine.py`): sàn an toàn cứng, đúng bằng giá trị đo; ngắn hơn không bao giờ tới
  model (`ValueError`). `tests/model/test_engine.py` kiểm tra 1,359 lỗi và 1,360/1,361 chạy được trên model thật.
- Audio ≤ 1,600 mẫu (100 ms) → `""` không gọi model (`app/chunking.py`, giữ nguyên): 85–100 ms gần như không bao giờ
  cho text, và sàn này cách sàn cứng 240 mẫu. Mọi chunk > 1,600 mẫu nên luôn ≥ 1,360.
- Silence dài vẫn qua model (không dùng ngưỡng biên độ). Splitter giữ cả phần đuôi ngắn.

**Độ dài chunk** (fp32, beam 4, 4 thread; mỗi độ dài một process mới, warm-up bằng độ dài đó như gateway, decode
speech `vi_long.wav` cắt/lặp tới độ dài đó, median 3 lần). WER: `vi_long.wav` (106.6 s, 272 từ) cắt bằng
`chunking.split` ở giới hạn đó.

| Chunk | Latency median | RTF | Peak RSS | WER `vi_long` |
|---:|---:|---:|---:|---:|
| 5 s | 0.21 s | 0.041 | 759 MB | 9.6% |
| 10 s | 0.39 s | 0.039 | 759 MB | 4.8% |
| 15 s | 0.58 s | 0.039 | 759 MB | 5.1% |
| **20 s** | **0.77 s** | **0.038** | **759 MB** | **4.0%** |
| 30 s | 1.23 s | 0.041 | 846 MB | 4.8% |
| 45 s | 1.98 s | 0.044 | 1,317 MB | 5.5% |
| 60 s | 2.85 s | 0.047 | 2,063 MB | 5.1% |
| 90 s | 4.65 s | 0.052 | 5,118 MB | 6.3% |
| 120 s | 7.28 s | 0.061 | 8,253 MB | 6.6% |
| 180 s | 14.45 s | 0.080 | 17,814 MB | 6.6% |

RSS sau load ≈ 387 MB; peak lúc load (759 MB) là peak của cả process tới 20 s. Trên 30 s, RAM tăng siêu tuyến tính
(attention theo độ dài) và RTF tăng; dưới 10 s, WER tăng do cắt giữa câu nhiều hơn.

**Mặc định `GATEWAY_MAX_CHUNK_SECONDS = 20`**: chunk dài nhất mà peak RSS vẫn bằng peak lúc load, latency mỗi
chunk < 1 s (cũng là thời gian tối đa một job LIVE phải chờ một chunk FILE đang chạy), WER thấp nhất trong bảng. WER
trên một file 272 từ chỉ để loại các giá trị tệ, không phân biệt được 10–30 s.

## 6. Thời điểm phát token và cue — gipformer1.5-68M-rnnt (CPU)

Đo bằng `scripts/probe_alignment.py` (sherpa-onnx 1.13.8, fp32, `modified_beam_search` beam 4, chunk 20 s). Report chỉ
có số đo và số đếm: [`baseline/alignment_probe.json`](baseline/alignment_probe.json).

| Đo | Kết quả |
|---|---|
| (a) Độ phân giải | Mọi thời điểm là bội của 640 mẫu (**40 ms**), ở cả fp32/int8 × beam/greedy |
| (b) Dịch audio (thêm 10 ms … 1 s số 0 phía trước, token không đổi) | ≥ 92% token dịch đúng lượng đó trong ±1 bước; ngoại lệ là token đầu (và có khi token thứ hai), luôn ở thời điểm 0 |
| (c) Khoảng lặng 1 s chèn tại 27 điểm yên lặng giữa câu: token đầu sau khoảng lặng − cuối khoảng lặng | median −75 ms, tứ phân vị −255 / +120 ms, p10 −957 ms; 17/27 nằm **trong** khoảng lặng |
| (c) Đầu khoảng lặng − token cuối trước nó | median 265 ms, p75 345 ms, p95 1,157 ms |
| (d) Khoảng cách giữa hai lần bắt đầu từ (trong câu) | median 280 ms, p90 520 ms, p95 880 ms |
| (e) Cue (`vi_long`, 106.6 s, 7 chunk) | 20 cue, dài nhất 6.97 s, nhiều nhất 80 ký tự; tốc độ đọc median 11 ký tự/s |

Model phát token gần đầu âm tiết nhưng không có độ trễ cố định: độ lệch đổi dấu giữa các tứ phân vị và từ đầu sau
khoảng lặng hay được đóng dấu sớm tới ~1 s. Encoder nhìn toàn bộ chunk nên thời điểm phát không bám chặt âm thanh.

**Hằng số (`app/alignment.py`; probe exit 1 nếu lệch):**

- `EMISSION_SHIFT = 0`: chỉ bù khi tứ phân vị 25/75 của (c) cùng dấu; ở đây khác dấu.
- `LAST_WORD_SAMPLES = 4480` (0.28 s): median của "đầu khoảng lặng − token cuối" (265 ms), làm tròn lên bội 40 ms.
- `PAUSE_SAMPLES = 8320` (0.52 s): p90 của (d), làm tròn lên bội 40 ms (dài hơn nhịp nói bình thường). Chỉ tác dụng
  giữa các chunk (trong chunk, end của từ = start của từ sau).
- `MAX_CUE_SAMPLES = 7 s`, `MAX_CUE_CHARS = 80`: quy ước phụ đề (tối đa 7 s trên màn hình, hai dòng 40 ký tự), không
  phải đặc tính model; trên speech thật giới hạn ký tự thường cắt trước.

**R2a cue gate** (`tests/model/test_server.py`, server thật trên model thật; report chỉ metadata
[`baseline/r2a_gate.json`](baseline/r2a_gate.json)): `verbose_json`/`srt`/`vtt` khớp `json` trên mọi fixture; cue có thứ
tự, trong audio, trong giới hạn; composite `vi_a | 1 s | vi_b | 1 s | vi_a | 1 s | vi_b` (37.7 s, 2 chunk, cắt trong
khoảng lặng giữa): không cue nào phủ qua khoảng lặng chứa điểm cắt, cue trước kết thúc 0.66 s trước nó, cue sau bắt
đầu 0.02 s trước khi lời nói tiếp; dịch audio 0.25/0.5/0.75 s dịch mọi mép cue (trừ start của từ đầu) trong ±20 ms;
`vi_long` dạng m4a WER 4.0%, 7 chunk, 20 cue.

Giới hạn: cue có thể bắt đầu trước lời nói (tới ~1 s) và phủ qua khoảng lặng nằm trong cùng một chunk. Không có mốc
thời gian từng từ do người annotate, nên độ chính xác từng từ chưa được đo (R2b, [roadmap.md](roadmap.md) §2).

## 7. Realtime trên model thật — gipformer1.5-68M-rnnt (CPU)

**R3a realtime gate** (`tests/model/test_realtime.py`, server thật trên model thật, fp32, beam 4, 4 thread; report chỉ
metadata [`baseline/r3a_gate.json`](baseline/r3a_gate.json)). Fixture 16 kHz resample lên PCM16 24 kHz (soxr HQ), gửi
từng packet 20 ms như client live:

- Mỗi turn (`vi_a`, `vi_b`, `silence_2s`, `short_200ms`, nhiều commit trên một socket) có transcript bằng REST trên
  đúng waveform 16 kHz đã commit, một delta mỗi chunk; WER `vi_a` 0%, `vi_b` 4%, bằng REST trên file gốc (ngưỡng:
  REST + 2 điểm). Digital silence cho text của chính model (như REST, không ép `""`).
- Bốn item commit liền nhau không chờ kết quả (speech, turn 100 ms, speech, turn 105 ms): `committed` nối
  `previous_item_id` đúng, delta/`completed` theo thứ tự commit, text bằng REST; turn 100 ms (1,600 mẫu 16 kHz) →
  `""` không gọi model, turn 105 ms → đúng một lời gọi model (tổng 3 lời gọi cho 4 item).
- Bốn session đồng thời, mỗi session hai item (`vi_a`/`vi_b`): không lẫn kết quả.
- Turn 37.7 s (> `GATEWAY_MAX_CHUNK_SECONDS` 20 s): một item, 2 delta, bằng REST.

## 8. Tái tạo

Baseline §4 (Gipformer, CPU, ~3.5 phút; cần model đã stage và `fixtures-large/vi_long.wav`; exit 1 nếu mặc định
trong `app/config.py` lệch số đo hoặc decode không tất định), rồi smoke trên server local cấu hình mặc định (lệnh ở
docs/agents/commands.md):

```bash
uv sync
uv run scripts/stage_models.py
uv run scripts/fetch_fixtures.py
uv run scripts/baseline_offline.py --out docs/baseline/offline.json
OPENAI_API_KEY=<key> uv run scripts/smoke_server.py --base-url http://127.0.0.1:<port> --out docs/baseline/server.json
```

Probe §5 (Gipformer, CPU, ~6 phút; cần model đã stage và `fixtures-large/vi_long.wav`; exit 1 nếu sàn đo được khác
`MIN_FORWARD_SAMPLES` hoặc sàn clip ngắn thấp hơn nó):

```bash
uv run scripts/probe_limits.py --out docs/baseline/limits_probe.json
```

Probe và gate §6 (Gipformer, CPU, ~1 phút mỗi lệnh; cần model đã stage và `fixtures-large/vi_long.wav`). Probe exit 1
nếu hằng số trong `app/alignment.py` khác giá trị suy ra; gate ghi `r2a_gate.json` khi mọi test gate pass:

```bash
uv run scripts/probe_alignment.py --out docs/baseline/alignment_probe.json
uv run pytest tests/model/test_server.py
```

Gate §7 (Gipformer, CPU, ~20 s; cần model đã stage); ghi `r3a_gate.json` khi mọi test gate pass:

```bash
uv run pytest tests/model/test_realtime.py
```

## 9. Reports trong `docs/baseline/`

| File | Nội dung |
|---|---|
| `offline.json` | Gipformer: RTF/WER theo precision × decoding, số thread, mặc định suy ra; hash transcript là chuẩn cho smoke (§4) |
| `server.json` | Gipformer: smoke gateway local trên CPU, mọi format + Realtime, chỉ metadata (§4) |
| `server-container.json` | Gipformer: smoke trong container gate, chỉ metadata; chưa có, gate chưa chạy ([container-gate.vi.md](container-gate.vi.md)) |
| `alignment_probe.json` | Gipformer: độ phân giải, dịch audio và độ trễ của thời điểm phát token; cue (§6) |
| `r2a_gate.json` | Gipformer: R2a cue gate trên server với model thật, chỉ metadata (§6) |
| `r3a_gate.json` | Gipformer: R3a realtime gate trên server với model thật, chỉ metadata (§7) |
| `limits_probe.json` | Gipformer: input ngắn nhất, speech ngắn, latency/RSS/WER theo độ dài chunk (§5) |

Chưa đo: LM/KenLM, Python 3.11, cold load khi page cache lạnh, tải đồng thời (xem [roadmap.md](roadmap.md)).
