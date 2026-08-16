#!/usr/bin/env python3
"""Export deployment bundle: A.bin, B.bin, encoder.onnx, meta.yaml, norms."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--registry",
        required=True,
        help="vessel_control model_registry root",
    )
    parser.add_argument("--model-id", required=True, help="e.g. koopman_v20260517_001")
    args = parser.parse_args()
    out = f"{args.registry}/{args.model_id}"
    print(f"[vessel_identification] export_model: -> {out}")
    print("TODO: write ONNX, binaries, meta.yaml, checksums.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
