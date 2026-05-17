#!/usr/bin/env python3
"""Train NN encoder + global linear (A, B) Deep Koopman model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from vessel_identification.config import KoopmanConfig  # noqa: E402
from vessel_identification.trainer import train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Deep Koopman model")
    parser.add_argument("--config", default=None, help="koopman_default.yaml path")
    parser.add_argument("--data", required=True, help="Processed dataset directory")
    parser.add_argument("--out", default="runs/koopman", help="Training output dir")
    args = parser.parse_args()

    cfg_path = args.config or str(
        Path(__file__).resolve().parents[1] / "config" / "koopman_default.yaml"
    )
    cfg = KoopmanConfig.from_yaml(cfg_path)
    out_dir = Path(args.out)
    data_dir = Path(args.data)

    if not data_dir.is_dir():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    ckpt = train_model(cfg, data_dir, out_dir)
    print(f"[vessel_identification] Training done. Best checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
