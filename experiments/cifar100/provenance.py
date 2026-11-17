"""
Unified Provenance, Checksum, and Run-Lineage Logger for CIFAR-100 Experiments.

Every data-generation, training, evaluation, and summarization script under
`experiments/cifar100/` MUST use this module to:
  1. Compute SHA256 checksums of all input and output files;
  2. Embed a `_provenance` metadata dictionary inside every saved `.json` / `.pt` dict;
  3. Append an immutable JSON record to `results/cifar100/PROVENANCE_LOG.jsonl`;
  4. Refresh `results/cifar100/cifar100_artifacts.sha256` for one-command `sha256sum -c` verification.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_LOG_PATH = PROJECT_ROOT / "results" / "cifar100" / "PROVENANCE_LOG.jsonl"
SHA256_MANIFEST_PATH = PROJECT_ROOT / "results" / "cifar100" / "cifar100_artifacts.sha256"


def sha256_file(path: os.PathLike | str, chunk_size: int = 1 << 20) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: os.PathLike | str, chunk_size: int = 1 << 20) -> str:
    p = Path(path)
    h = hashlib.md5()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def rel_path(path: os.PathLike | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def get_git_state() -> Dict[str, Any]:
    try:
        head = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        dirty_raw = (
            subprocess.check_output(["git", "status", "-s"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .splitlines()
        )
        return {"commit": head, "dirty_count": len(dirty_raw), "dirty_preview": dirty_raw[:30]}
    except Exception as exc:
        return {"commit": "unknown", "error": str(exc)}


def record_provenance(
    stage: str,
    run_id: str,
    script_path: os.PathLike | str,
    inputs: Iterable[os.PathLike | str] = (),
    outputs: Iterable[os.PathLike | str] = (),
    config: Optional[Dict[str, Any]] = None,
    metrics_summary: Optional[Dict[str, Any]] = None,
    notes: str = "",
    argv: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a provenance dictionary, append it to PROVENANCE_LOG.jsonl, and refresh the SHA256 manifest."""
    import torch

    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    input_hashes = {rel_path(p): sha256_file(p) for p in inputs if Path(p).exists()}
    output_hashes = {rel_path(p): sha256_file(p) for p in outputs if Path(p).exists()}

    record: Dict[str, Any] = {
        "timestamp_utc": now_utc,
        "stage": stage,
        "run_id": run_id,
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "script": rel_path(script_path),
        "argv": argv if argv is not None else sys.argv,
        "git": get_git_state(),
        "config": config or {},
        "inputs_sha256": input_hashes,
        "outputs_sha256": output_hashes,
        "metrics_summary": metrics_summary or {},
        "notes": notes,
    }

    PROVENANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = PROVENANCE_LOG_PATH.parent / ".provenance.lock"
    with lock_path.open("w", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            with PROVENANCE_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            refresh_sha256_manifest()
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    return record


def refresh_sha256_manifest() -> int:
    """Scan canonical CIFAR-100 artifact directories and write `results/cifar100/cifar100_artifacts.sha256`."""
    tracked_files: List[Path] = []

    raw_tar = PROJECT_ROOT / "data" / "core" / "cifar100" / "cifar-100-python.tar.gz"
    if raw_tar.exists():
        tracked_files.append(raw_tar)

    for folder_rel in [
        "docs/cifar100",
        "experiments/cifar100_pilot",
        "experiments/cifar100",
        "data/core/cifar100_splits",
        "checkpoints/cifar100",
        "results/cifar100",
    ]:
        folder = PROJECT_ROOT / folder_rel
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or p.name.endswith(".pyc") or p.name.startswith("."):
                continue
            if p.resolve() == SHA256_MANIFEST_PATH.resolve():
                continue
            tracked_files.append(p)

    SHA256_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for p in tracked_files:
        lines.append(f"{sha256_file(p)}  {rel_path(p)}")
    SHA256_MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


if __name__ == "__main__":
    cnt = refresh_sha256_manifest()
    print(f"[Provenance] Refreshed {SHA256_MANIFEST_PATH} with {cnt} tracked files.")
