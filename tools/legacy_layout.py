#!/usr/bin/env python3
"""Rebuild the original experiment layout from a downloaded release, so the scripts in code/ run unchanged.

    huggingface-cli download sumleo/RLCDAlignBench --repo-type dataset --local-dir hf
    python tools/legacy_layout.py --release hf --out work
    python work/code/restore_layout.py
    python work/code/run_jev.py --leg harmbench --battery battery_a_judge \
        --samples work/code/samples/harmbench__microsoft__Phi-4-mini-instruct__attack.jsonl --rescore

Files are symlinked (copied where symlinks are unavailable) and checked against the sha256 in file_map.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True, help="local copy of the Hugging Face dataset")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-verify", action="store_true", help="skip sha256 checks")
    args = ap.parse_args()

    n = bad = 0
    with (args.release / "file_map.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = args.release / row["release"]
            if not args.no_verify and hashlib.sha256(src.read_bytes()).hexdigest() != row["sha256"]:
                print(f"sha256 mismatch: {row['release']}")
                bad += 1
                continue
            dst = args.out / row["original"]
            if row["original"].startswith("results/metrics/"):
                # run_jev.py --rescore rewrites these, so copy them to keep the release untouched
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            else:
                link_or_copy(src, dst)
            n += 1
    shutil.copytree(REPO / "code", args.out / "code", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "samples", "results"))
    print(f"{n} files linked into {args.out}, {bad} mismatches")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
