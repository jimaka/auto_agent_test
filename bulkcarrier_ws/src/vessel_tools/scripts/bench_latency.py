#!/usr/bin/env python3
"""Benchmark ONNX lift + OSQP solve budget (target < 300 ms end-to-end)."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="model_registry package path")
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--N", type=int, default=30)
    args = parser.parse_args()
    print(
        f"[vessel_tools] bench_latency: model={args.model_dir} nz={args.nz} N={args.N}"
    )
    print("TODO: load ONNX + build QP timing harness")
    return 0


if __name__ == "__main__":
    sys.exit(main())
