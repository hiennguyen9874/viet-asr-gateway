"""CPU lane: pins are exact commits and the committed fixtures match their manifest."""

import json
import re
import tomllib
from pathlib import Path

from scripts.fetch_fixtures import MANIFEST as FIXTURES_MANIFEST
from scripts.fetch_fixtures import verify

ROOT = Path(__file__).resolve().parents[2]
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def test_asset_manifest_pins_commits_and_hashes():
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    assert list(manifest["models"]) == ["viet-asr"]
    for entry in manifest["models"].values():
        assert COMMIT.match(entry["revision"])
        for meta in entry["files"].values():
            assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256"]) and meta["size"] > 0


def test_manifest_lists_only_the_files_the_gateway_needs():
    entry = json.loads((ROOT / "assets/manifest.json").read_text())["models"]["viet-asr"]
    assert (entry["repo_id"], entry["revision"], entry["dir"], entry["license"]) == (
        "dangvansam/viet-asr", "46adacf22d379e3d6b860d5d08e9fb02cbc122ea", "viet-asr", "apache-2.0")
    onnx = {f"int8/{part}.int8.onnx" for part in ("encoder", "decoder", "joiner")}
    assert set(entry["files"]) == onnx | {"int8/tokens.txt", "README.md"}


def test_pyproject_and_lockfile_pin_sherpa_onnx_to_manifest_version():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    version = json.loads((ROOT / "assets/manifest.json").read_text())["code"]["sherpa-onnx"]["version"]
    locked = {p["name"]: p["version"] for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    for name in ("sherpa-onnx", "sherpa-onnx-core"):
        assert f"{name}=={version}" in deps
        assert locked[name] == version


def test_no_torch_or_cuda_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "optional-dependencies" not in project["project"]
    assert not any(d.split("=")[0].split("@")[0].strip() in ("torch", "transformers", "quan-transcribe")
                   for d in project["project"]["dependencies"])
    locked = {p["name"] for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    assert not locked & {"torch", "transformers", "quan-transcribe"}


def test_docker_files_have_no_cuda_concerns():
    for name in ("Dockerfile", "compose.yaml"):
        text = (ROOT / name).read_text().lower()
        assert "cuda" not in text and "nvidia" not in text and "torch" not in text


def test_small_fixtures_match_manifest():
    assert verify(json.loads(FIXTURES_MANIFEST.read_text()), include_large=False) == []
