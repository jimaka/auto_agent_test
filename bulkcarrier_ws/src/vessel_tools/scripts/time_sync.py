#!/usr/bin/env python3
"""Run manifest-driven dataset pipeline (bag ingest, sync, split, export)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from vessel_tools.pipeline import process_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process dataset manifest: bags -> aligned NPZ splits"
    )
    parser.add_argument("--manifest", required=True, help="Dataset manifest YAML")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root for relative paths (default: auto-detect)",
    )
    parser.add_argument(
        "--backend",
        choices=["rosbag", "rosbags", "auto"],
        default="auto",
        help="Bag reader backend",
    )
    args = parser.parse_args()

    backend = None if args.backend == "auto" else args.backend
    out = process_manifest(args.manifest, repo_root=args.repo_root, backend=backend)
    print(f"[vessel_tools] Pipeline complete: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
