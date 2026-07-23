#!/usr/bin/env python3
"""Train NN encoder + global linear (A, B) Deep Koopman model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="koopman_default.yaml path")
    parser.add_argument("--data", required=True, help="Processed dataset directory")
    parser.add_argument("--out", default="runs/koopman", help="Training output dir")
    args = parser.parse_args()
    cfg = args.config or str(
        Path(__file__).resolve().parents[1] / "config" / "koopman_default.yaml"
    )
    print(f"[vessel_identification] train: config={cfg} data={args.data} out={args.out}")
    print("TODO: PyTorch training loop with N-step horizon loss")
    return 0


if __name__ == "__main__":
    sys.exit(main())
