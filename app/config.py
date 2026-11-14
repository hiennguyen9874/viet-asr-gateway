"""Typed gateway configuration, read once from `GATEWAY_*` environment variables.

Every limit is finite and validated at startup; an invalid or unknown variable aborts startup (`ConfigError`).
There is no default API key. Gateway API keys are separate from `HF_TOKEN` (staging secret), and a key equal to
`HF_TOKEN` is rejected so the two cannot be mixed up.
"""

import dataclasses
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16000  # model input rate; queue budgets are counted in 16 kHz samples
PREFIX = "GATEWAY_"
# Pre-rename settings (Gipformer era, then the CTC era): a leftover one means a half-migrated deployment.
OLD_PREFIXES = ("GIPFORMER_", "QUAN_")
DECODING_METHODS = ("modified_beam_search", "greedy_search")
MIN_API_KEY_LENGTH = 16
MIN_CHUNK_SECONDS = 1.0


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    api_keys: tuple[str, ...] = dataclasses.field(repr=False)
    models_dir: Path = Path("models")
    decoding_method: str = "modified_beam_search"  # or greedy_search
    beam_size: int = 4  # active paths of modified_beam_search; ignored by greedy_search
    num_threads: int = 4  # intra-op threads of the one recognizer; measured: docs/baseline.vi.md §4
    whisper_alias: bool = False  # opt-in alias `whisper-1` (docs/api-contract.vi.md §3)
    log_level: str = "INFO"

    # Upload and decode (M2)
    max_upload_bytes: int = 25 * 1024 * 1024  # whole request body, checked before multipart parsing
    max_spool_bytes: int = 200 * 1024 * 1024  # temp disk held by all uploads in flight
    spool_dir: Path = Path(tempfile.gettempdir())  # a private 0700 subdirectory is created inside
    max_audio_seconds: float = (
        1800.0  # decoded duration; FFmpeg is stopped as soon as it is exceeded
    )
    max_chunk_seconds: float = 20.0  # measured: docs/baseline.vi.md §5
    decode_timeout_seconds: float = 60.0
    max_concurrent_decodes: int = 2
    max_decoded_seconds: float = (
        3600.0  # decoded audio held in RAM by all requests in flight (429 above)
    )

    # Scheduler. A lane is one REST request or one realtime session.
    queue_max_lanes: int = 32
    queue_max_jobs: int = 64
    queue_max_jobs_per_lane: int = 2
    queue_max_buffered_seconds: float = (
        600.0  # audio held by queued + running jobs; measured: docs/baseline.vi.md §4
    )
    queue_timeout_seconds: float = 30.0  # max wait for queue capacity before 429
    live_priority_ratio: int = 3  # live jobs dispatched per file job when both are waiting
    request_timeout_seconds: float = 600.0  # total upload/decode/queue/infer budget of one request
    infer_hang_seconds: float = 120.0  # one inference call longer than this marks the worker hung

    # Realtime sessions (M5)
    max_sessions: int = 16
    max_session_seconds: float = 3600.0
    max_turn_seconds: float = 120.0
    max_pending_items: int = 4
    session_idle_timeout_seconds: float = 120.0
    send_timeout_seconds: float = (
        10.0  # one SSE/WebSocket send; a slower client counts as disconnected
    )

    shutdown_drain_seconds: float = 30.0

    @property
    def queue_max_buffered_samples(self) -> int:
        return int(self.queue_max_buffered_seconds * SAMPLE_RATE)

    @property
    def max_chunk_samples(self) -> int:
        return int(self.max_chunk_seconds * SAMPLE_RATE)

    @property
    def max_audio_samples(self) -> int:
        return int(self.max_audio_seconds * SAMPLE_RATE)

    def __post_init__(self) -> None:
        if not self.api_keys:
            raise ConfigError(
                "GATEWAY_API_KEYS (or GATEWAY_API_KEYS_FILE) must define at least one key"
            )
        for key in self.api_keys:
            if len(key) < MIN_API_KEY_LENGTH or not all("!" <= c <= "~" for c in key):
                raise ConfigError(
                    f"API keys must be >= {MIN_API_KEY_LENGTH} printable ASCII characters without spaces"
                )
        if len(set(self.api_keys)) != len(self.api_keys):
            raise ConfigError("duplicate API key")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ConfigError(
                f"GATEWAY_LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR, got {self.log_level!r}"
            )
        if self.decoding_method not in DECODING_METHODS:
            raise ConfigError(
                f"GATEWAY_DECODING_METHOD must be one of {', '.join(DECODING_METHODS)}, got {self.decoding_method!r}"
            )
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.type in (int, float) and not (math.isfinite(value) and value > 0):
                raise ConfigError(
                    f"{PREFIX}{f.name.upper()} must be a finite number > 0, got {value!r}"
                )
        if self.max_chunk_seconds > self.max_audio_seconds:
            raise ConfigError("GATEWAY_MAX_CHUNK_SECONDS must be <= GATEWAY_MAX_AUDIO_SECONDS")
        if self.max_chunk_seconds < MIN_CHUNK_SECONDS:
            # The splitter needs room to cut at a quiet point and keep every chunk above the short-clip floor.
            raise ConfigError(f"GATEWAY_MAX_CHUNK_SECONDS must be >= {MIN_CHUNK_SECONDS}")
        if self.max_decoded_seconds < self.max_audio_seconds:
            raise ConfigError("GATEWAY_MAX_DECODED_SECONDS must be >= GATEWAY_MAX_AUDIO_SECONDS")
        if self.max_spool_bytes < self.max_upload_bytes:
            raise ConfigError("GATEWAY_MAX_SPOOL_BYTES must be >= GATEWAY_MAX_UPLOAD_BYTES")
        # A single job must fit the queue budget, otherwise it could never be admitted.
        for name in ("max_chunk_seconds", "max_turn_seconds"):
            if getattr(self, name) > self.queue_max_buffered_seconds:
                raise ConfigError(
                    f"{PREFIX}{name.upper()} must be <= GATEWAY_QUEUE_MAX_BUFFERED_SECONDS"
                )
        if self.queue_max_jobs_per_lane > self.queue_max_jobs:
            raise ConfigError("GATEWAY_QUEUE_MAX_JOBS_PER_LANE must be <= GATEWAY_QUEUE_MAX_JOBS")
        if self.max_sessions > self.queue_max_lanes:
            raise ConfigError("GATEWAY_MAX_SESSIONS must be <= GATEWAY_QUEUE_MAX_LANES")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        env = os.environ if env is None else env
        fields = {f.name: f for f in dataclasses.fields(cls) if f.name != "api_keys"}
        known = {PREFIX + name.upper() for name in fields} | {
            PREFIX + "API_KEYS",
            PREFIX + "API_KEYS_FILE",
        }
        obsolete = sorted(k for k in env if k.startswith(OLD_PREFIXES))
        if obsolete:
            raise ConfigError(
                f"obsolete configuration variables (renamed to {PREFIX}*): {', '.join(obsolete)}"
            )
        unknown = sorted(k for k in env if k.startswith(PREFIX) and k not in known)
        if unknown:
            raise ConfigError(f"unknown configuration variables: {', '.join(unknown)}")

        kwargs: dict[str, object] = {"api_keys": _read_api_keys(env)}
        for name, f in fields.items():
            raw = env.get(PREFIX + name.upper())
            if raw is not None:
                kwargs[name] = _parse(PREFIX + name.upper(), raw.strip(), f.type)
        return cls(**kwargs)


def _read_api_keys(env: Mapping[str, str]) -> tuple[str, ...]:
    inline, path = env.get(PREFIX + "API_KEYS"), env.get(PREFIX + "API_KEYS_FILE")
    if inline is not None and path is not None:
        raise ConfigError("set only one of GATEWAY_API_KEYS and GATEWAY_API_KEYS_FILE")
    if path is not None:
        try:
            text = Path(path).read_text()
        except OSError as e:
            raise ConfigError(f"cannot read GATEWAY_API_KEYS_FILE: {e.strerror}") from None
        keys = tuple(
            line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
        )
    else:
        keys = tuple(k.strip() for k in (inline or "").split(",") if k.strip())
    hf_token = env.get("HF_TOKEN")
    if hf_token and hf_token.strip() in keys:
        raise ConfigError("an API key equals HF_TOKEN; gateway keys must be separate secrets")
    return keys


def _parse(name: str, raw: str, type_: object) -> object:
    try:
        if type_ is bool:
            if raw.lower() in ("1", "true", "yes", "on"):
                return True
            if raw.lower() in ("0", "false", "no", "off"):
                return False
            raise ValueError
        if type_ is int:
            return int(raw)
        if type_ is float:
            return float(raw)
        if type_ is Path:
            return Path(raw)
        return raw.upper() if name == PREFIX + "LOG_LEVEL" else raw
    except ValueError:
        raise ConfigError(f"{name}: invalid value {raw!r}") from None
