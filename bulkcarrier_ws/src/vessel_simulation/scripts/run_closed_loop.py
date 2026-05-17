#!/usr/bin/env python3
"""MMG SIL closed-loop benchmark: Koopman-MPC vs Nomoto baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from vessel_simulation.sil_runner import run_benchmark_suite  # noqa: E402


def main() -> int:
    sim_pkg = Path(__file__).resolve().parents[1]  # vessel_simulation
    bulk = sim_pkg.parents[1]  # bulkcarrier_ws

    parser = argparse.ArgumentParser(description="MMG SIL closed-loop benchmark")
    parser.add_argument("--scenario", default="straight", choices=["straight", "turn", "zigzag"])
    parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration [s]")
    parser.add_argument(
        "--config",
        default=str(sim_pkg / "config" / "mmg_default.yaml"),
        help="MMG / SIL config YAML",
    )
    parser.add_argument(
        "--model",
        default=str(bulk / "src" / "vessel_control" / "model_registry" / "koopman_test"),
        help="Koopman model registry directory",
    )
    parser.add_argument(
        "--out",
        default=str(bulk.parent / "runs" / "sil_benchmark"),
        help="Output directory for logs and report",
    )
    parser.add_argument(
        "--update-meta",
        action="store_true",
        help="Set sil_passed in model meta.yaml from Koopman result",
    )
    args = parser.parse_args()

    report = run_benchmark_suite(
        config_path=Path(args.config),
        model_dir=Path(args.model),
        scenario=args.scenario,
        duration_s=args.duration,
        out_dir=Path(args.out) / args.scenario,
    )
    print(json.dumps(report, indent=2))

    if args.update_meta:
        meta_path = Path(args.model) / "meta.yaml"
        if meta_path.is_file():
            text = meta_path.read_text(encoding="utf-8")
            passed = bool(report.get("koopman_passed", False))
            if "sil_passed:" in text:
                import re

                text = re.sub(r"sil_passed:\s*\w+", f"sil_passed: {str(passed).lower()}", text)
            else:
                text += f"\nvalidation:\n  sil_passed: {str(passed).lower()}\n"
            meta_path.write_text(text, encoding="utf-8")
            print(f"Updated {meta_path} sil_passed={passed}")

    return 0 if report.get("koopman_passed") else 1


if __name__ == "__main__":
    sys.exit(main())
