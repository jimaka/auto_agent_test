#!/usr/bin/env python3
"""Validate N-step open-loop prediction stability on hold-out data."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--N", type=int, default=30)
    args = parser.parse_args()
    print(
        f"[vessel_identification] validate_horizon: ckpt={args.checkpoint} N={args.N}"
    )
    print("TODO: report z/x RMSE and export tube tightening table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
