"""Stage pinned model assets into a local directory and verify them against assets/manifest.json.

    uv run scripts/stage_models.py                    # download pinned revisions, verify SHA256
    uv run scripts/stage_models.py --verify           # offline: only check files already staged
    uv run scripts/stage_models.py --update-manifest  # (maintainers) record hashes after changing a pin

Pins (repo id + commit) live in the manifest. Files are fetched with `revision=<commit>`, never a branch. HF_TOKEN
is read from the environment by huggingface_hub when a repo needs it; it is never printed or written to disk here.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "assets" / "manifest.json"
DEFAULT_DEST = Path(os.environ.get("GATEWAY_MODELS_DIR", ROOT / "models"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def download(name: str, entry: dict, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"[{name}] {entry['repo_id']}@{entry['revision']} -> {dest}", flush=True)
    snapshot_download(
        entry["repo_id"],
        revision=entry["revision"],
        repo_type=entry.get("repo_type", "model"),
        allow_patterns=sorted(entry["files"]),
        local_dir=dest,
    )


def check(name: str, entry: dict, dest: Path, update: bool) -> list[str]:
    errors = []
    for rel, meta in sorted(entry["files"].items()):
        path = dest / rel
        if not path.is_file():
            errors.append(f"[{name}] missing {path}")
            continue
        size, digest = path.stat().st_size, sha256(path)
        if update:
            entry["files"][rel] = {"size": size, "sha256": digest}
        elif (size, digest) != (meta.get("size"), meta.get("sha256")):
            errors.append(f"[{name}] {rel}: got size={size} sha256={digest}, manifest {meta}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="staging root (default $GATEWAY_MODELS_DIR or ./models)",
    )
    ap.add_argument(
        "--verify", action="store_true", help="do not download; only verify staged files"
    )
    ap.add_argument(
        "--update-manifest",
        action="store_true",
        help="write sizes/hashes of staged files to the manifest",
    )
    ap.add_argument("--only", action="append", help="limit to these manifest model names")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    errors = []
    for name, entry in manifest["models"].items():
        if args.only and name not in args.only:
            continue
        dest = args.dest / entry["dir"]
        if not args.verify:
            download(name, entry, dest)
        errors += check(name, entry, dest, args.update_manifest)

    if args.update_manifest and not errors:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"updated {MANIFEST.relative_to(ROOT)}")
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    if not errors:
        print(f"OK: staged assets match {MANIFEST.relative_to(ROOT)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
