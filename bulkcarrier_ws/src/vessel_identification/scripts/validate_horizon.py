#!/usr/bin/env python3
"""Validate N-step open-loop prediction on hold-out data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from vessel_identification.validate import validate_checkpoint, write_tube_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--N", type=int, default=None, help="Horizon (default: from checkpoint)")
    parser.add_argument("--tube-out", default=None, help="Optional tube_tightening.csv path")
    args = parser.parse_args()

    summary, metrics, _, cfg = validate_checkpoint(
        Path(args.checkpoint),
        Path(args.data),
        split=args.split,
        horizon=args.N,
    )
    print(json.dumps(summary, indent=2))

    if args.tube_out:
        write_tube_csv(Path(args.tube_out), metrics["x_rmse_per_step"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
