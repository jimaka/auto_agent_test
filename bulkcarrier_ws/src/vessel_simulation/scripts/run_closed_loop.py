#!/usr/bin/env python3
"""SIL closed-loop: MMG plant + trajectory reference + controller interface stub."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="zigzag")
    parser.add_argument("--duration", type=float, default=600.0)
    args = parser.parse_args()
    print(f"[vessel_simulation] SIL: scenario={args.scenario} T={args.duration}s")
    print("TODO: wire MMG3DOF + tracking metrics (cross, psi, u, overshoot)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
