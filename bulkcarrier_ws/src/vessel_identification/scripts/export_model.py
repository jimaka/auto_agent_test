#!/usr/bin/env python3
"""Export deployment bundle: A.bin, B.bin, encoder.onnx, meta.yaml, norms."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from vessel_identification.export import export_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--registry",
        required=True,
        help="vessel_control model_registry root",
    )
    parser.add_argument("--model-id", required=True, help="e.g. koopman_v20260517_001")
    parser.add_argument("--data", default=None, help="Processed data dir for validation metrics")
    parser.add_argument("--description", default="")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    out = export_bundle(
        checkpoint=Path(args.checkpoint),
        registry=Path(args.registry),
        model_id=args.model_id,
        data_dir=Path(args.data) if args.data else None,
        description=args.description,
        opset=args.opset,
    )
    print(f"[vessel_identification] Exported model bundle to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
