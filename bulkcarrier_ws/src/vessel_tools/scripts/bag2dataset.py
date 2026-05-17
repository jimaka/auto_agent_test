#!/usr/bin/env python3
"""Convert ROS bags to resampled training datasets (Ts from config)."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="ROS bag -> processed dataset")
    parser.add_argument("--bag", required=True, help="Input .bag path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--ts", type=float, default=0.25, help="Resample period [s]")
    args = parser.parse_args()
    print(f"[vessel_tools] bag2dataset: bag={args.bag} out={args.out} Ts={args.ts}")
    print("TODO: implement topic sync and CSV/npz export")
    return 0


if __name__ == "__main__":
    sys.exit(main())
