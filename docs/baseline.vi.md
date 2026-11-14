# Baseline inference và pins — viet-asr

Số đo nền của model (ngoài gateway), các pin tái lập, giới hạn frontend và các gate trên model thật. Đây là số đo
một stream trên một host CPU dùng chung, **không phải benchmark năng lực phục vụ**.

## 1. Pins

| Thành phần | Pin |
|---|---|
| Model `dangvansam/viet-asr` | Hub revision `46adacf22d379e3d6b860d5d08e9fb02cbc122ea` (Apache-2.0): `int8/encoder.int8.onnx`, `int8/decoder.int8.onnx`, `int8/joiner.int8.onnx`, `int8/tokens.txt`, `README.md` (chỉ int8) |
| Runtime `sherpa-onnx` | `1.13.8` (cùng `sherpa-onnx-core`; `pyproject.toml`, `uv.lock`) |
| Fixtures `google/fleurs` | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` (vi_vn dev cho fixture, vi_vn test cho tập eval; CC BY 4.0) |
| Python | 3.12.14 (image) |
| OpenAI SDK (contract) | Python `openai==3.31.0`, JS `openai@7.33.0` |

`assets/manifest.json` là nguồn pin duy nhất cho model (repo, revision, size, SHA256 từng file) và giữ version
sherpa-onnx; `tests/unit/test_manifests.py` kiểm tra version đó khớp `pyproject.toml` và `uv.lock`. Đổi pin: sửa
manifest rồi chạy `scripts/stage_models.py --update-manifest`. Repo trên Hub không bị gate (không cần `HF_TOKEN`).

## 2. Engine offline

`app/engine.SherpaEngine` tạo một `sherpa_onnx.OfflineRecognizer.from_transducer` từ file int8 local đã stage (không
truy cập Hub), 16 kHz, 80 fbank, `GATEWAY_DECODING_METHOD`/`GATEWAY_BEAM_SIZE`/`GATEWAY_NUM_THREADS`. Parity
(`tests/model/test_engine.py`, CPU, cần model đã stage): text, token và thời điểm phát token giống hệt script tham
chiếu sherpa-onnx tối giản `tests/model/reference.py` cho `modified_beam_search` và `greedy_search`; token byte fallback
(`<0xNN>`) được ghép thành ký tự UTF-8 (`tests/fixtures/vi_bytes.wav`).

**Transcript phụ thuộc số thread.** Với cùng audio và decoding, output int8 có thể khác nhau theo
`GATEWAY_NUM_THREADS` (thứ tự cộng trong onnxruntime): trên `vi_long`, 1 và 2 thread cho cùng transcript, 4 thread
khác một hash, 8/16 thread khác WER (8.1% → 10.3%). Mỗi số thread vẫn tất định qua các lần lặp. Hash trong
`offline.json` chỉ là chuẩn so sánh cho cấu hình mặc định (2 thread).

## 3. Fixtures và tập eval

`scripts/fetch_fixtures.py` tạo lại tất định từ FLEURS và kiểm tra SHA256 (`tests/fixtures/manifest.json`).

| File | Nội dung |
|---|---|
| `tests/fixtures/vi_a.wav`, `vi_b.wav` | 2 clip FLEURS khác nội dung (8.88 s, 8.46 s) + transcript tham chiếu |
| `tests/fixtures/silence_2s.wav` | 2 s số 0 |
| `tests/fixtures/short_200ms.wav` | 200 ms cắt giữa câu của `vi_a` |
| `tests/fixtures/vi_bytes.wav` | 6.5 s cắt từ một clip FLEURS, làm viet-asr phát token byte fallback (`<0xC3><0x9A>` = "Ú") |
| `fixtures-large/vi_long.wav` (gitignored) | 10 câu nối nhau, 106.62 s (đi qua splitter) |

FLEURS vi_vn không có speaker id: `vi_a`/`vi_b` chắc chắn khác nội dung, không chắc khác người nói.

**Tập eval cục bộ** (`scripts/fetch_fixtures.py --eval`, gitignored trong `fixtures-large/eval/`; manifest chỉ có hash
và số mẫu: `tests/fixtures/eval_manifest.json`; `--verify --eval` kiểm offline):

- 200 clip FLEURS vi_vn **test** (bản sao nguyên văn): bản ghi đầu tiên theo thứ tự tên file của mỗi sentence id,
  tổng 43.3 phút, 6,116 từ tham chiếu. Tham chiếu (cột `transcription`) nằm trong `fixtures-large/eval/references.json`,
  hash của file được ghi trong manifest. Cột này giữ chữ số (39/200 câu có chữ số), nên sau chuẩn hoá nó trùng cột
  viết `raw_transcription` trên cả 200 clip.
- 24 clip không có tiếng nói, sinh tất định (seed cố định, PCM_16): số 0; nhiễu trắng và hồng ở −60/−45/−30 dBFS RMS;
  "room tone" (nhiễu hồng −65 dBFS + hum 50 Hz −60 dBFS); mỗi loại dài 2/5/10 s.

**Chấm WER** (`scripts/scoring.py`): hạ chữ thường và bỏ dấu câu Unicode ở cả hai phía; WER của tập eval là WER corpus
(tổng lỗi / tổng từ tham chiếu). Lệch giữa chữ số và số đọc bằng chữ không được xử lý.

## 4. Baseline offline — viet-asr (CPU)

Đo bằng `scripts/baseline_offline.py` (sherpa-onnx 1.13.8, model `46adacf2`, Xeon Silver 4216 × 64 core, **host dùng
chung**: load average 17 → 29 trong lúc đo). Mỗi setting một process mới; engine dùng y như gateway: warm-up 15 s số
0, fixture là một chunk, `vi_long` và mỗi clip eval cắt bằng `chunking.split` (15 s) và nối bằng `Joiner`. Median 5 lần
(`vi_long`: 3 lần; tập eval: 1 lượt). Report chỉ có số đo, số đếm và SHA256 của transcript:
[`baseline/offline.json`](baseline/offline.json); hash của setting mặc định là chuẩn so sánh cho smoke server và
container gate.

**(a) Decoding** (2 thread). RTF = latency / thời lượng audio.

| Setting | Load | RSS peak | RTF chunk 15 s | RTF eval | WER eval (200 clip) | Clip im lặng/nhiễu có chữ | WER `vi_a` / `vi_b` / `vi_long` |
|---|---:|---:|---:|---:|---:|---:|---|
| **`modified_beam_search` (beam 4)** | 2.1 s | 454 MB | **0.058** | **0.057** | **17.28%** | **24/24** | 0% / 16% / **8.1%** |
| `greedy_search` | 1.7 s | 456 MB | 0.039 | 0.039 | 18.25% | 1/24 | 0% / 16% / 10.3% |

Mọi setting tất định qua các lần lặp. Beam 4 sinh 1–2 từ trên **mọi** clip không có tiếng nói (kể cả số 0, ví dụ
"Không." trên `silence_2s`); greedy chỉ sinh 1 từ trên một clip số 0. `short_200ms` cho text ở cả hai decoding.
WER eval cao một phần do tên riêng nước ngoài và chữ số/số đọc (§3).

**Luật chọn decoding** (`choose_decoding`): nếu WER eval của greedy kém beam ≤ 0,5 điểm tuyệt đối thì mặc định là
`greedy_search`. Đo được greedy kém **0,97 điểm** (18.25% so với 17.28%) → **giữ `modified_beam_search`, beam 4**.
Hạn chế đã biết: trên im lặng/nhiễu, beam 4 thường trả vài chữ thay vì `""`. Client cần `""` trên im lặng có thể đặt
`GATEWAY_DECODING_METHOD=greedy_search` (nhanh hơn ~33%, WER kém ~1 điểm).

**(b) Số thread** (beam 4; chunk 15 s là lời gọi dài nhất gateway thực hiện). Hiệu suất = latency 1 thread / (số
thread × latency).

| Thread | Latency chunk 15 s (median) | Hiệu suất | RTF `vi_long` |
|---:|---:|---:|---:|
| 1 | 1.18 s | 1.00 | 0.079 |
| **2** | **0.86 s** | **0.69** | **0.056** |
| 4 | 0.75 s | 0.40 | 0.050 |
| 8 | 0.69 s | 0.21 | 0.049 |
| 16 | 0.79 s | 0.09 | 0.055 |

**(c) Mặc định suy ra** (script exit 1 nếu `app/config.py` lệch):

- `GATEWAY_DECODING_METHOD = modified_beam_search`, `GATEWAY_BEAM_SIZE = 4`: luật 0,5 điểm ở (a).
- `GATEWAY_NUM_THREADS` 4 → **2**: số thread lớn nhất còn hiệu suất ≥ 0.5. 4 thread chỉ nhanh hơn 2 thread ~15% trên
  một stream (hiệu suất 0.40) và lấy core của FFmpeg decode và event loop. Các lần đo trước trong cùng ngày cho hiệu
  suất 4 thread 0.35–0.43; trên host khác nên đo lại.
- `GATEWAY_QUEUE_MAX_BUFFERED_SECONDS` 600 → **450**: một chunk đã vào queue có thể phải chờ toàn bộ audio đã buffer
  trước nó; thời gian đó phải ≤ `GATEWAY_QUEUE_TIMEOUT_SECONDS` (30 s). Với RTF xấu nhất của mặc định (0.0595, max qua
  các lần lặp) giới hạn là 504 s (một lần đo trước: 492 s); 450 s chờ tối đa ~27 s.
- `GATEWAY_INFER_HANG_SECONDS = 120` giữ nguyên: ≥ 20 × lời gọi chậm nhất đo được (1.27 s, chunk 15 s ở 1 thread).
- Giữ nguyên, kiểm tra với số đo: một job LIVE chờ tối đa một chunk FILE đang chạy (≤ 0.9 s); file dài nhất (1,800 s)
  chạy một mình mất ~107 s, nên `GATEWAY_REQUEST_TIMEOUT_SECONDS` 600 s đủ cho ~5 file dài tối đa cùng chia worker.
  Lanes (32), sessions (16), jobs (64, 2 mỗi lane) là giới hạn an toàn khi rảnh; queue buffer ở trên là giới hạn chặn.

Đây là số đo một stream offline trên host dùng chung, **không phải năng lực phục vụ** (R4, [roadmap.md](roadmap.md) §1).

**Server smoke** (`scripts/smoke_server.py`, server local với cấu hình mặc định; report chỉ metadata
[`baseline/server.json`](baseline/server.json)): `json`/`text`/SSE/`verbose_json`/`srt`/`vtt` khớp nhau trên 5
fixture, hash transcript của 4 fixture một chunk khớp (a), `vi_long` WER 8.1% (9 chunk, 21 cue); Realtime một socket 5
commit, WER ≤ REST + 2 điểm (`vi_long` 8.5% so với 8.1%). REST `vi_a` (8.88 s) ≈ 0.68 s, Realtime commit → completed
≈ 0.52 s; `vi_long` ≈ 7.4 s, delta SSE đầu sau 1.07 s.

## 5. Clip ngắn và độ dài chunk — viet-asr (CPU)

Đo bằng `scripts/probe_limits.py` trên recognizer thật (sherpa-onnx 1.13.8, Xeon Silver 4216, mặc định: beam 4, 2
thread). Report chỉ có số đo và số đếm: [`baseline/limits_probe.json`](baseline/limits_probe.json).

**Input ngắn nhất.** Với cả `modified_beam_search` và `greedy_search`, cả số 0 lẫn noise:

| Số mẫu 16 kHz | Gọi recognizer trực tiếp |
|---:|---|
| 0 … 1,359 | **RuntimeError** của onnxruntime (conv đầu vào `encoder_embed` cần ≥ 7 frame fbank) |
| ≥ 1,360 (85 ms) | Chạy được (kiểm tra mọi độ dài 1,360 … 1,759 và lưới tới 3 s) |

**Speech ngắn** (40 đoạn cắt từ `vi_a`/`vi_b` mỗi độ dài, beam 4): số đoạn có text khác rỗng là 1/40 ở 85 ms,
0/40 ở 100 ms, 16/40 ở 150 ms, 18/40 ở 200 ms, 20/40 ở 300 ms, 33/40 ở 500 ms.

**Policy (từ report, giữ nguyên):**

- `MIN_FORWARD_SAMPLES = 1360` (`app/engine.py`): sàn an toàn cứng, đúng bằng giá trị đo; ngắn hơn không bao giờ tới
  model (`ValueError`). `tests/model/test_engine.py` kiểm tra 1,359 lỗi và 1,360/1,361 chạy được trên model thật.
- Audio ≤ 1,600 mẫu (100 ms) → `""` không gọi model (`app/chunking.py`): 85–100 ms gần như không bao giờ cho text, và
  sàn này cách sàn cứng 240 mẫu. Mọi chunk > 1,600 mẫu nên luôn ≥ 1,360.
- Silence dài vẫn qua model (không dùng ngưỡng biên độ). Splitter giữ cả phần đuôi ngắn.

**Độ dài chunk** (tối đa 30 s, độ dài huấn luyện của model). Latency/RSS: mỗi độ dài một process mới, warm-up bằng độ
dài đó như gateway, decode speech `vi_long.wav` cắt/lặp tới độ dài đó, median 3 lần. WER: 200 clip eval nối thành 20
bản ghi 10 câu (0.3 s số 0 giữa các câu, ~2.2 phút mỗi bản ghi), và `vi_long.wav` (272 từ), cắt bằng `chunking.split`
ở giới hạn đó.

| Chunk | Latency median / max | RTF | Peak RSS | WER eval (số chunk) | WER `vi_long` |
|---:|---:|---:|---:|---:|---:|
| 5 s | 0.30 / 0.31 s | 0.060 | 374 MB | 21.34% (1,062) | 14.7% |
| 10 s | 0.62 / 0.64 s | 0.062 | 374 MB | 17.79% (356) | 9.2% |
| **15 s** | **0.87 / 0.91 s** | **0.058** | **426 MB** | **17.84% (218)** | **8.1%** |
| 20 s | 1.17 / 1.19 s | 0.058 | 457 MB | 17.89% (157) | 12.9% |
| 25 s | 1.57 / 1.68 s | 0.063 | 589 MB | 18.02% (126) | 10.3% |
| 30 s | 1.91 / 1.96 s | 0.064 | 636 MB | 18.03% (106) | 9.6% |

RSS sau load ≈ 207 MB, peak lúc load 374 MB. Từ 10 s tới 30 s WER eval chỉ chênh 0,24 điểm (trong nhiễu của một tập
200 clip); dưới 10 s WER tăng rõ do cắt giữa câu nhiều hơn.

**Vết cắt chunk.** Model luôn kết thúc một chunk như một câu: ở mọi độ dài ≥ 10 s, **100%** chunk trước vết cắt kết
thúc bằng dấu câu, và 88–92% chunk sau bắt đầu bằng chữ hoa, kể cả khi vết cắt nằm giữa câu (có tiếng nói ở cả hai phía
trong cùng một clip). Ở 15 s: 115/206 vết cắt nằm giữa câu, cả 115 sinh dấu câu, 101 sinh chữ hoa. Chunk dài hơn giảm
số vết cắt (218 chunk ở 15 s, 106 ở 30 s) nhưng không đổi tỉ lệ; đây là hạn chế đã biết, server không sửa text.

**Mặc định `GATEWAY_MAX_CHUNK_SECONDS` 20 → 15** (luật `derive`, exit 1 nếu `app/config.py` lệch): trong các độ dài có
latency max ≤ 1 s (thời gian tối đa một job LIVE chờ một chunk FILE đang chạy) và peak RSS ≤ 1.25 × peak lúc load, lấy
độ dài **dài nhất** có WER eval cách WER tốt nhất ≤ 0,5 điểm (ít vết cắt nhất). 5/10/15 s đạt giới hạn; tốt nhất 17.79%
(10 s); 15 s (17.84%) hoà → chọn 15 s. 20 s có latency 1.19 s ở 2 thread nên bị loại.

## 6. Thời điểm phát token và cue — viet-asr (CPU)

Đo bằng `scripts/probe_alignment.py` (sherpa-onnx 1.13.8, `modified_beam_search` beam 4, 2 thread, chunk 15 s). Report
chỉ có số đo và số đếm: [`baseline/alignment_probe.json`](baseline/alignment_probe.json).

| Đo | Kết quả |
|---|---|
| (a) Độ phân giải | Mọi thời điểm là bội của 640 mẫu (**40 ms**), ở cả beam và greedy |
| (b) Dịch audio (thêm 10 ms … 1 s số 0 phía trước) | Token không đổi ở 15/16 lần dịch; khi đó ≥ 92% token dịch đúng lượng đó trong ±1 bước; token đầu luôn ở thời điểm 0 |
| (c) Khoảng lặng 1 s chèn tại 27 điểm yên lặng giữa câu: token đầu sau khoảng lặng − cuối khoảng lặng | median **−205 ms**, tứ phân vị −375 / −65 ms (cùng dấu), p10 −773 ms; 22/27 nằm **trong** khoảng lặng; text không đổi ở 19/27 |
| (c) Đầu khoảng lặng − token cuối trước nó | median 305 ms, p75 405 ms, p95 1,136 ms |
| (d) Khoảng cách giữa hai lần bắt đầu từ (trong câu) | median 280 ms, p90 640 ms, p95 920 ms |
| (e) Cue (`vi_long`, 106.6 s, 9 chunk) | 21 cue, dài nhất 7.0 s, nhiều nhất 80 ký tự; tốc độ đọc median 12 ký tự/s |

viet-asr phát token **sớm** hơn lời nói một khoảng khá ổn định (khác Gipformer, có độ lệch đổi dấu): tứ phân vị của
(c) cùng âm, nên luật bù độ trễ có hiệu lực. Encoder nhìn toàn bộ chunk nên thời điểm phát vẫn không bám chặt âm thanh:
từ đầu sau khoảng lặng có khi vẫn sớm tới ~0.8 s sau khi bù.

**Hằng số (`app/alignment.py`; probe exit 1 nếu lệch):**

- `EMISSION_SHIFT` 0 → **3200** (+0.2 s): −median của (c) làm tròn tới bội 40 ms, vì tứ phân vị 25/75 cùng dấu. Gần
  cuối chunk độ dịch giảm dần (tới 0) để từ cuối vẫn đủ `LAST_WORD_SAMPLES` trong chunk.
- `LAST_WORD_SAMPLES` 4480 → **1920** (0.12 s): median của "đầu khoảng lặng − token cuối" (305 ms) trừ độ dịch 200 ms,
  làm tròn lên bội 40 ms.
- `PAUSE_SAMPLES` 8320 → **10240** (0.64 s): p90 của (d), làm tròn lên bội 40 ms (dài hơn nhịp nói bình thường). Chỉ
  tác dụng giữa các chunk (trong chunk, end của từ = start của từ sau).
- `MAX_CUE_SAMPLES = 7 s`, `MAX_CUE_CHARS = 80`: quy ước phụ đề (tối đa 7 s trên màn hình, hai dòng 40 ký tự), không
  phải đặc tính model; trên speech thật cả hai đều chạm. Cue còn kết thúc sau từ tận cùng bằng `.`, `?`, `!`, `…`.

**R2a cue gate** (`tests/model/test_server.py`, server thật trên model thật với engine mặc định; report chỉ metadata
[`baseline/r2a_gate.json`](baseline/r2a_gate.json)): `verbose_json`/`srt`/`vtt` khớp `json` trên mọi fixture; cue có
thứ tự, trong audio, trong giới hạn (gồm luật cuối câu); `vi_a | im lặng | vi_b` (24.46 s, khoảng lặng 8.88–16.0 s kéo
tới quá giới hạn chunk 1 s, 2 chunk, cắt trong khoảng lặng): không cue nào phủ qua khoảng lặng chứa điểm cắt, cue trước
kết thúc 0.32 s trước nó, cue sau bắt đầu 0.83 s trước khi lời nói tiếp (trong khoảng lặng, sau điểm cắt); dịch audio
0.25/0.5/0.75 s dịch mọi mép cue so sánh được (14 mép, trừ start của từ đầu) trong ±30 ms; `vi_long` dạng m4a WER 8.8%,
9 chunk, 21 cue.

Giới hạn: cue có thể bắt đầu trước lời nói (tới ~0.8 s) và phủ qua khoảng lặng nằm trong cùng một chunk. Không có mốc
thời gian từng từ do người annotate, nên độ chính xác từng từ chưa được đo (R2b, [roadmap.md](roadmap.md) §2).

## 7. Realtime trên model thật — viet-asr (CPU)

**R3a realtime gate** (`tests/model/test_realtime.py`, server thật trên model thật, beam 4, 2 thread, chunk 15 s;
report chỉ metadata [`baseline/r3a_gate.json`](baseline/r3a_gate.json)). Fixture 16 kHz resample lên PCM16 24 kHz
(soxr HQ), gửi từng packet 20 ms như client live:

- Mỗi turn (`vi_a`, `vi_b`, `silence_2s`, `short_200ms`, nhiều commit trên một socket) có transcript bằng REST trên
  đúng waveform 16 kHz đã commit, một delta mỗi chunk; WER `vi_a` 0%, `vi_b` 16%, bằng REST trên file gốc (ngưỡng:
  REST + 2 điểm). Digital silence và `short_200ms` cho text của chính model (beam 4; như REST, không ép `""`).
- Bốn item commit liền nhau không chờ kết quả (speech, turn 100 ms, speech, turn 105 ms): `committed` nối
  `previous_item_id` đúng, delta/`completed` theo thứ tự commit, text bằng REST; turn 100 ms (1,600 mẫu 16 kHz) →
  `""` không gọi model, turn 105 ms → đúng một lời gọi model (tổng 3 lời gọi cho 4 item).
- Bốn session đồng thời, mỗi session hai item (`vi_a`/`vi_b`): không lẫn kết quả.
- Turn 37.7 s (> `GATEWAY_MAX_CHUNK_SECONDS` 15 s): một item, 3 delta, bằng REST.

## 8. Khoảng trống đã biết

- **Chữ số vs số đọc:** WER chuẩn hoá không đối chiếu "50" với "năm mươi"; mọi lệch kiểu này tính là lỗi. Tham chiếu
  FLEURS vi_vn phần lớn đã viết chữ số (§3), nên ảnh hưởng nhỏ nhưng chưa đo riêng.
- **Vết cắt chunk:** chunk trước vết cắt luôn kết thúc bằng dấu câu và chunk sau thường mở đầu bằng chữ hoa, kể cả
  giữa câu (§5). Server không sửa text; chunk dài hơn chỉ giảm số vết cắt.
- **Im lặng/nhiễu:** với beam 4 (mặc định), model sinh 1–2 từ trên mọi clip không có tiếng nói đã thử (§4); greedy
  gần như luôn trả `""`. Không có VAD hay ngưỡng biên độ (R3b).
- **Số thread:** transcript có thể đổi theo `GATEWAY_NUM_THREADS` (§2); hash baseline chỉ đúng với mặc định.
- **Tên riêng nước ngoài** chiếm phần lớn lỗi trên `vi_long`.

## 9. Tái tạo

Tập eval (một lần, ~550 MB tải về, gitignored):

```bash
uv sync
uv run scripts/stage_models.py
uv run scripts/fetch_fixtures.py
uv run scripts/fetch_fixtures.py --eval
```

Probe §5 (~15 phút; exit 1 nếu sàn đo được khác `MIN_FORWARD_SAMPLES`, sàn clip ngắn thấp hơn nó, hoặc
`max_chunk_seconds` mặc định khác giá trị suy ra), baseline §4 (~13 phút; exit 1 nếu mặc định trong `app/config.py`
lệch số đo hoặc decode không tất định), probe §6 (~1 phút; exit 1 nếu hằng số trong `app/alignment.py` khác giá trị suy
ra). Probe §5 và baseline §4 dùng nhau (decoding/thread ↔ độ dài chunk): đổi một mặc định thì chạy lại cả hai tới khi
cả hai exit 0.

```bash
uv run scripts/probe_limits.py --out docs/baseline/limits_probe.json
uv run scripts/baseline_offline.py --out docs/baseline/offline.json
uv run scripts/probe_alignment.py --out docs/baseline/alignment_probe.json
```

Gate §6/§7 (~1–2 phút mỗi lệnh; chạy riêng từng file để report không bị đánh dấu `-dirty` bởi report kia) và smoke trên
server local cấu hình mặc định (lệnh ở docs/agents/commands.md):

```bash
uv run python -m pytest tests/model/test_server.py     # ghi r2a_gate.json khi mọi test gate pass
uv run python -m pytest tests/model/test_realtime.py   # ghi r3a_gate.json khi mọi test gate pass
OPENAI_API_KEY=<key> uv run scripts/smoke_server.py --base-url http://127.0.0.1:<port> --out docs/baseline/server.json
```

## 10. Reports trong `docs/baseline/`

| File | Nội dung |
|---|---|
| `offline.json` | viet-asr: RTF/WER/tỉ lệ sinh chữ trên im lặng theo decoding, số thread, mặc định suy ra; hash transcript là chuẩn cho smoke (§4) |
| `server.json` | viet-asr: smoke gateway local trên CPU, mọi format + Realtime, chỉ metadata (§4) |
| `server-container.json` | Smoke trong container gate, chỉ metadata; chưa có, gate chưa chạy trên viet-asr ([container-gate.vi.md](container-gate.vi.md)) |
| `alignment_probe.json` | viet-asr: độ phân giải, dịch audio và độ trễ của thời điểm phát token; cue (§6) |
| `r2a_gate.json` | viet-asr: R2a cue gate trên server với model thật, chỉ metadata (§6) |
| `r3a_gate.json` | viet-asr: R3a realtime gate trên server với model thật, chỉ metadata (§7) |
| `limits_probe.json` | viet-asr: input ngắn nhất, speech ngắn, latency/RSS/WER/vết cắt theo độ dài chunk (§5) |

Chưa đo: LM/KenLM, Python 3.11, cold load khi page cache lạnh, tải đồng thời (xem [roadmap.md](roadmap.md)).
