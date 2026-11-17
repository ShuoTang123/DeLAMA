#!/usr/bin/env python3
"""
Download, SHA256-verify, and extract DeLAMA GitHub Release v1.0.0 assets
(model checkpoints, preprocessed LiDAR floorplan arrays, and per-episode raw JSON logs)
into the local repository checkout.

Usage:
    python3 scripts/download_assets.py
    python3 scripts/download_assets.py --components checkpoints data
    python3 scripts/download_assets.py --from-dir /path/to/local/release_assets
    python3 scripts/download_assets.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "checkpoints" / "assets_manifest.json"
CHUNK_BYTES = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def verify_key_checkpoints(manifest: dict) -> Tuple[int, int]:
    """Verify SHA256 hashes of extracted key checkpoints present locally."""
    key_ckpts = manifest.get("key_checkpoints", {})
    ok_count = 0
    missing_count = 0
    for rel, info in sorted(key_ckpts.items()):
        fp = ROOT / rel
        if not fp.exists():
            missing_count += 1
            print(f"  [MISSING] {rel}")
            continue
        got = sha256_file(fp)
        want = info["sha256"]
        if got != want:
            raise AssertionError(f"SHA256 mismatch on {rel}: got {got}, expected {want}")
        ok_count += 1
        print(f"  [OK] {rel:<64s} sha256={got[:16]}...")
    return ok_count, missing_count


def download_file(url: str, dst: Path, expected_bytes: int) -> None:
    headers = {"User-Agent": "DeLAMA-Asset-Downloader/1.0"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, dst.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or expected_bytes or 0)
            downloaded = 0
            last_pct = -10
            while chunk := resp.read(CHUNK_BYTES):
                out.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(100 * downloaded / total)
                    if pct >= last_pct + 20 or downloaded == total:
                        print(f"    {downloaded / (1024**2):6.2f} / {total / (1024**2):6.2f} MB ({pct:3d}%)")
                        last_pct = pct
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} while downloading {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error while downloading {url}: {exc.reason}") from exc


def safe_extract_tar_gz(archive_path: Path, target_root: Path) -> int:
    root_resolved = target_root.resolve()
    count = 0
    with tarfile.open(archive_path, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe archive member path blocked: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Unexpected link inside release archive blocked: {member.name}")
            dest_resolved = (root_resolved / member_path).resolve()
            if dest_resolved != root_resolved and root_resolved not in dest_resolved.parents:
                raise RuntimeError(f"Path escape blocked: {member.name}")
        for member in members:
            if sys.version_info >= (3, 12):
                tf.extract(member, path=root_resolved, filter="data")
            else:
                tf.extract(member, path=root_resolved)
            if member.isfile():
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify DeLAMA Release v1.0.0 assets.")
    parser.add_argument(
        "--components",
        nargs="+",
        default=["all"],
        choices=["all", "checkpoints", "data", "results"],
        help="Which asset bundles to install (default: all).",
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help="Optional local directory containing pre-downloaded .tar.gz release assets.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify SHA256 checksums of already-extracted checkpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if target files appear present.",
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if args.verify_only:
        print("=== Verifying Extracted Key Checkpoints ===")
        ok_n, miss_n = verify_key_checkpoints(manifest)
        if miss_n > 0:
            raise SystemExit(f"[FAIL] {miss_n} key checkpoint(s) missing; run scripts/download_assets.py first.")
        print(f"[PASS] Verified all {ok_n} key checkpoints.")
        return

    selected = list(manifest["archives"].keys()) if "all" in args.components else args.components

    # Quick skip check when not --force
    probe_paths = {
        "checkpoints": ROOT / "checkpoints" / "cifar100" / "p1_stage2_collab_phi_d128.pt",
        "data": ROOT / "data" / "mapping" / "lidar_simulation_final" / "train_mapset.npy",
        "results": ROOT / "results" / "cifar100" / "raw_runs" / "exp1_p1_group2_N6_100ep.json",
    }

    with tempfile.TemporaryDirectory(prefix="delama_assets_") as tmpdir:
        tmp_path = Path(tmpdir)
        for comp in selected:
            info = manifest["archives"][comp]
            fname = info["filename"]
            want_sha = info["sha256"]
            size_mb = info["size_bytes"] / (1024 ** 2)

            if not args.force and probe_paths[comp].exists():
                print(f"[SKIP] {comp} already present ({probe_paths[comp].relative_to(ROOT)}). Pass --force to re-extract.")
                continue

            if args.from_dir is not None:
                local_tar = args.from_dir / fname
                if not local_tar.exists():
                    raise FileNotFoundError(f"Missing local archive: {local_tar}")
                print(f"[{comp}] Reading local archive {local_tar} ({size_mb:.2f} MB)...")
                archive_to_use = local_tar
            else:
                dl_tar = tmp_path / fname
                print(f"[{comp}] Downloading {info['url']} ({size_mb:.2f} MB)...")
                download_file(info["url"], dl_tar, info["size_bytes"])
                archive_to_use = dl_tar

            got_sha = sha256_file(archive_to_use)
            if got_sha != want_sha:
                raise AssertionError(
                    f"Archive SHA256 mismatch for {fname}: got {got_sha}, expected {want_sha}"
                )
            print(f"  [OK] SHA256 verified ({got_sha[:16]}...), extracting into {ROOT}...")
            n_files = safe_extract_tar_gz(archive_to_use, ROOT)
            print(f"  [OK] Extracted {n_files} files for component '{comp}'.")

    if "checkpoints" in selected:
        print("\n=== Verifying Extracted Key Checkpoints ===")
        ok_n, miss_n = verify_key_checkpoints(manifest)
        if miss_n > 0:
            raise AssertionError(f"Missing {miss_n} checkpoints after extraction.")
        print(f"\n[PASS] Installed and verified {ok_n} key checkpoints.")


if __name__ == "__main__":
    main()
