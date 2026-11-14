"""Startup configuration: required keys, finite limits, consistency, no silent typos."""

from pathlib import Path

import pytest

from app.config import Config, ConfigError

KEY = "k" * 20


def test_defaults_with_one_key():
    cfg = Config.from_env({"GATEWAY_API_KEYS": KEY})
    assert cfg.api_keys == (KEY,) and cfg.whisper_alias is False
    assert cfg.queue_max_buffered_samples == int(cfg.queue_max_buffered_seconds * 16000)
    assert KEY not in repr(cfg)
    assert (cfg.decoding_method, cfg.beam_size, cfg.num_threads) == ("modified_beam_search", 4, 4)


def test_parses_typed_values():
    cfg = Config.from_env({
        "GATEWAY_API_KEYS": f" {KEY} , {'j' * 20} ", "GATEWAY_WHISPER_ALIAS": "true", "GATEWAY_QUEUE_MAX_JOBS": "8",
        "GATEWAY_MAX_CHUNK_SECONDS": "20.5", "GATEWAY_MODELS_DIR": "/srv/models", "GATEWAY_LOG_LEVEL": "debug",
        "GATEWAY_SPOOL_DIR": "/var/spool/gateway",
        "GATEWAY_DECODING_METHOD": "greedy_search", "GATEWAY_BEAM_SIZE": "8", "GATEWAY_NUM_THREADS": "2",
    })
    assert (cfg.decoding_method, cfg.beam_size, cfg.num_threads) == ("greedy_search", 8, 2)
    assert cfg.api_keys == (KEY, "j" * 20) and cfg.whisper_alias is True and cfg.queue_max_jobs == 8
    assert cfg.max_chunk_seconds == 20.5 and cfg.models_dir == Path("/srv/models") and cfg.log_level == "DEBUG"
    assert cfg.max_chunk_samples == 328000 and cfg.spool_dir == Path("/var/spool/gateway")


def test_api_keys_file(tmp_path):
    f = tmp_path / "keys"
    f.write_text(f"# comment\n{KEY}\n\n{'j' * 20}\n")
    assert Config.from_env({"GATEWAY_API_KEYS_FILE": str(f)}).api_keys == (KEY, "j" * 20)
    with pytest.raises(ConfigError):
        Config.from_env({"GATEWAY_API_KEYS_FILE": str(tmp_path / "missing")})


@pytest.mark.parametrize("env", [
    {},  # no public default key
    {"GATEWAY_API_KEYS": " , "},
    {"GATEWAY_API_KEYS": "short"},
    {"GATEWAY_API_KEYS": "é" * 20},
    {"GATEWAY_API_KEYS": "界" * 20},
    {"GATEWAY_API_KEYS": "a" * 19 + "\u007f"},
    {"GATEWAY_API_KEYS": f"{KEY},{KEY}"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_API_KEYS_FILE": "/x"},
    {"GATEWAY_API_KEYS": KEY, "HF_TOKEN": KEY},  # gateway key must not be the HF token
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_QUEUE_MAX_JOB": "4"},  # typo
    {"GATEWAY_API_KEYS": KEY, "QUAN_QUEUE_MAX_JOBS": "4"},  # leftover pre-rename variable
    {"GATEWAY_API_KEYS": KEY, "QUAN_GPU": "0"},
    {"GATEWAY_API_KEYS": KEY, "GIPFORMER_QUEUE_MAX_JOBS": "4"},  # leftover pre-rename variable
    {"GIPFORMER_API_KEYS": KEY},  # old prefix only: no key under GATEWAY_*
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_DECODING_METHOD": "ctc"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_BEAM_SIZE": "0"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_BEAM_SIZE": "four"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_NUM_THREADS": "0"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_DEVICE": "cuda"},  # removed setting
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_QUEUE_MAX_JOBS": "many"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_QUEUE_MAX_JOBS": "0"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_MAX_AUDIO_SECONDS": "inf"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_MAX_AUDIO_SECONDS": "nan"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_DECODE_TIMEOUT_SECONDS": "-1"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_WHISPER_ALIAS": "maybe"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_LOG_LEVEL": "TRACE"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_MAX_CHUNK_SECONDS": "60", "GATEWAY_QUEUE_MAX_BUFFERED_SECONDS": "30"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_MAX_TURN_SECONDS": "1000", "GATEWAY_QUEUE_MAX_BUFFERED_SECONDS": "900"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_QUEUE_MAX_JOBS_PER_LANE": "10", "GATEWAY_QUEUE_MAX_JOBS": "5"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_MAX_SESSIONS": "40"},
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_MAX_CHUNK_SECONDS": "0.5"},  # splitter needs room above the short-clip floor
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_MAX_DECODED_SECONDS": "100"},  # a max-length file could never be held
    {"GATEWAY_API_KEYS": KEY, "GATEWAY_MAX_UPLOAD_BYTES": "1000", "GATEWAY_MAX_SPOOL_BYTES": "999"},
])
def test_invalid_config_fails_startup(env):
    with pytest.raises(ConfigError) as e:
        Config.from_env(env)
    assert KEY not in str(e.value)


@pytest.mark.parametrize("value", ["int8", "fp32"])
def test_removed_precision_setting_fails_startup(value):
    # The engine always loads the int8 weights; the old setting is now an unknown variable.
    with pytest.raises(ConfigError, match="unknown configuration variables: GATEWAY_PRECISION$"):
        Config.from_env({"GATEWAY_API_KEYS": KEY, "GATEWAY_PRECISION": value})


@pytest.mark.parametrize("name", ["GIPFORMER_API_KEYS", "GIPFORMER_MODELS_DIR", "QUAN_GPU"])
def test_obsolete_prefix_names_the_blocked_variable(name):
    with pytest.raises(ConfigError, match=f"obsolete configuration variables .*: {name}$"):
        Config.from_env({"GATEWAY_API_KEYS": KEY, name: "x"})
