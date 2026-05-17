#!/usr/bin/env python3
"""Convert PyTorch Koopman checkpoint encoder to ONNX (verified for vessel_control C++)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from vessel_identification.export import export_bundle  # noqa: E402
from vessel_identification.onnx_convert import convert_checkpoint_to_onnx  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoint to ONNX encoder (+ optional full deploy bundle)"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt / checkpoint")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for encoder.onnx (default: checkpoint parent/onnx_export)",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="If set with --model-id, also write full model_registry bundle",
    )
    parser.add_argument("--model-id", default=None, help="model_registry/<model-id>")
    parser.add_argument("--data", default=None, help="Processed dataset for full bundle validation")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--atol", type=float, default=1e-5, help="Max abs error Torch vs ORT")
    parser.add_argument(
        "--full-bundle",
        action="store_true",
        help="Export complete registry bundle (ONNX + A/B + meta.yaml)",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    if args.full_bundle or (args.registry and args.model_id):
        if not args.registry or not args.model_id:
            print("ERROR: --registry and --model-id required for full bundle", file=sys.stderr)
            return 1
        out = export_bundle(
            checkpoint=ckpt,
            registry=Path(args.registry),
            model_id=args.model_id,
            data_dir=Path(args.data) if args.data else None,
            opset=args.opset,
        )
        print(f"[convert_to_onnx] Full bundle exported to {out}")
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent / "onnx_export"
    result = convert_checkpoint_to_onnx(ckpt, out_dir, opset=args.opset, atol=args.atol)
    summary = {
        "onnx": str(result.onnx_path),
        "encoder_io": str(result.io_path),
        "max_abs_err": result.max_abs_err,
        "opset": result.opset,
        "input_name": result.input_name,
        "output_name": result.output_name,
        "nx": result.nx,
        "nz": result.nz,
    }
    print(json.dumps(summary, indent=2))
    print(f"[convert_to_onnx] Verified ONNX export OK (max_err={result.max_abs_err:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
